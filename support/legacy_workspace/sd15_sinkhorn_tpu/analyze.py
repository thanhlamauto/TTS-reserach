"""Prompt-level analysis; selection is restricted to TRAIN stages."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import atomic_json,digest
from .evaluate import summary_metrics
from .sd15_sampler import checkpoint_step


def csv_write(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for r in rows for k,v in r.items() if not isinstance(v,(dict,list))})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def bootstrap(values):
    v=np.asarray(values,dtype='float64')
    rng=np.random.default_rng(192071)
    draws=v[rng.integers(0,len(v),size=(10000,len(v)))].mean(1)
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def analyze(root,stage,partial=False,select=False):
    root=Path(root); directory=root/stage
    plans=[json.loads(p.read_text()) for p in sorted(directory.glob('plan_worker*.json'))]
    if not plans: raise RuntimeError('no stage plan')
    plan=plans[0]
    if any(p!=plan for p in plans): raise RuntimeError('worker designs disagree')
    expected=len(plan['prompts'])*len(plan['configs'])
    guided=[json.loads(p.read_text()) for p in sorted(directory.glob('trials/*/*/result.json'))]
    if len(guided)!=expected and not partial:
        raise RuntimeError(f'incomplete stage: {len(guided)}/{expected} guided rows; use --partial for descriptive status')
    if not guided: raise RuntimeError('no completed guided rows')
    cfg=plan['sampler']; m,n=cfg['n_reference'],cfg['n_source']; seed=plan['seed']; split=plan['split']
    present=sorted({r['prompt_index'] for r in guided})
    rows=list(guided); bon={}
    for index in present:
        bankdir=root/'banks'/split/f'seed{seed}'/f'm{m}_n{n}'/f'p{index:03d}'
        bank=json.loads((bankdir/'bank.json').read_text())
        s=bank['sets']['source']
        rows.append(dict(config_id='unguided_source',config={'method':'unguided_source'},prompt_index=index,
                         seed=seed,**summary_metrics(s),logical_total_sample_steps=n*cfg['steps'],
                         incremental_sample_steps=n*cfg['steps'],reward_evaluations=n,
                         incremental_reward_evaluations=n,wall_seconds=s['denoise_seconds']+s['decode_seconds']+s['reward_seconds']+s['clip_seconds']))
        for p in bankdir.glob('fresh_bon_*/result.json'):
            row=json.loads(p.read_text()); bon[(index,row['fresh_count'])]=row; rows.append(row)
    for r in guided:
        count=r['logical_total_sample_steps']//cfg['steps']
        matched=bon.get((r['prompt_index'],count))
        if matched:
            r['mean_gain_vs_fresh_bon']=r['mean_reward']-matched['mean_reward']
            r['best_gain_vs_fresh_bon']=r['best_reward']-matched['best_reward']
        r.update({f'config_{k}':v for k,v in r['config'].items()})
    csv_write(directory/'results.csv',rows)
    groups=defaultdict(list)
    for r in rows: groups[r['config_id']].append(r)
    summaries=[]
    for cid,rr in groups.items():
        out=dict(config_id=cid,method=rr[0]['config']['method'],n_prompts=len(rr))
        for metric in ('mean_reward','best_reward','clip_alignment','clip_diversity','reward_gain',
                       'paired_improved_fraction','sinkhorn_before','sinkhorn_after','target_ess','plan_entropy',
                       'row_error','column_error','relative_update_rms','bf16_update_rms','clipping_fraction',
                       'wall_seconds','shared_bank_seconds','logical_total_sample_steps','reward_evaluations',
                       'mean_gain_vs_fresh_bon','best_gain_vs_fresh_bon'):
            values=[r[metric] for r in rr if r.get(metric) is not None]
            if values:
                out[metric]=float(np.mean(values))
                if metric in ('reward_gain','mean_gain_vs_fresh_bon','best_gain_vs_fresh_bon'):
                    low,high=bootstrap(values); out[metric+'_ci_low']=low; out[metric+'_ci_high']=high
        if 'mode' in rr[0]['config']: out.update({f'config_{k}':v for k,v in rr[0]['config'].items()})
        summaries.append(out)
    summaries.sort(key=lambda r:r['mean_reward'],reverse=True)
    csv_write(directory/'summary.csv',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,6))
    colors={'A':'#2563eb','B':'#d97706','C':'#059669','nearest':'#9333ea','fresh_bon':'#dc2626','unguided_source':'#6b7280'}
    used=set()
    for row in summaries:
        method=row['method']; marker='*' if method=='fresh_bon' else ('s' if method=='unguided_source' else {'soft':'o','top':'^','contrast':'D'}.get(row.get('config_mode'),'o'))
        ax.scatter(row['clip_diversity'],row['mean_reward'],color=colors[method],s=140 if marker=='*' else (95 if row.get('config_eta',.1)>.1 else 50),
                   marker=marker,label=method if method not in used else None,alpha=.85)
        used.add(method)
    frontier=[]
    for r in summaries:
        dominated=any(s['mean_reward']>=r['mean_reward'] and s['clip_diversity']>=r['clip_diversity']
                      and (s['mean_reward']>r['mean_reward'] or s['clip_diversity']>r['clip_diversity']) for s in summaries)
        if not dominated: frontier.append(r)
    frontier.sort(key=lambda r:r['clip_diversity'])
    ax.plot([r['clip_diversity'] for r in frontier],[r['mean_reward'] for r in frontier],color='#334155',ls='--',lw=1)
    ax.set(xlabel='Mean pairwise CLIP cosine distance (higher = more diverse)',ylabel='Mean ImageReward',
           title=f'SD1.5 Sinkhorn: {stage}, {len(present)} prompts'+(' (PARTIAL)' if len(guided)!=expected else ''))
    from matplotlib.lines import Line2D
    first_legend=ax.legend(title='Method',loc='best'); ax.add_artist(first_legend)
    ax.legend(handles=[Line2D([],[],color='gray',marker=marker,ls='',label=mode) for mode,marker in [('soft','o'),('top','^'),('contrast','D')]],title='Reward mode',loc='lower left')
    ax.margins(x=.1,y=.1); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(directory/'reward_diversity_pareto.png',dpi=170); plt.close(fig)
    correlation=None
    if len(guided)>2:
        reduction=np.array([r['sinkhorn_before']-r['sinkhorn_after'] for r in guided])
        gains=np.array([r['reward_gain'] for r in guided])
        if reduction.std()>0 and gains.std()>0: correlation=float(np.corrcoef(reduction,gains)[0,1])
    accounting=dict(completed_guided=len(guided),expected_guided=expected,
                    unique_inference_sample_steps=sum(r.get('incremental_sample_steps',0) for r in rows)
                    +len(present)*m*cfg['steps'],
                    unique_reward_evaluations=sum(r.get('incremental_reward_evaluations',0) for r in rows)+len(present)*m,
                    note='Excludes warmup/replay/failed attempts; these are recorded separately in environment and replay files.',
                    pooled_correlation_sinkhorn_reduction_reward_gain=correlation)
    atomic_json(directory/'analysis.json',dict(stage=stage,split=split,accounting=accounting,summary=summaries,
                                              pareto_config_ids=[r['config_id'] for r in frontier]))
    lines=[f'# SD1.5 Sinkhorn — {stage}', '',f'Completed {len(guided)}/{expected} guided prompt/config rows on {len(present)} {split.upper()} prompts.',
           '', '| Configuration | Mean reward | Best reward | Gain vs source | Gain vs fresh BoN | CLIP diversity |',
           '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        label=r['method']
        if 'config_mode' in r: label+=f"/{r['config_mode']}, eta={r['config_eta']:g} ({r['config_id']})"
        fmt=lambda key:f"{r[key]:.4f}" if key in r else '—'
        lines.append('| '+' | '.join([label]+[fmt(k) for k in ('mean_reward','best_reward','reward_gain','mean_gain_vs_fresh_bon','clip_diversity')])+' |')
    lines+=['', '## Interpretation and accounting', '',
            'Fresh Best-of-N reports the top N images among independently drawn budget-matched samples; its best reward is the maximum across all fresh candidates.',
            'The unguided source has a smaller budget. A gain over source alone does not establish better test-time scaling.',
            'Per-trial budgets include full reference generation, full source generation, and the resumed guided suffix. Cached costs are also reported separately for the actual sweep.',
            'The 50% comparison uses 840 sample-steps versus 28 fresh DDIM30 images. Classifier-free guidance doubles UNet sample-branch evaluations.',
            'Smoke fresh14 is padded to 16 by the fixed batch4 sampler; this is disclosed and is not used for a scaling claim. At other checkpoints the integer-sample BoN budget may leave fewer than 30 logical sample-steps unused.',
            'CIs resample prompts (10,000 paired bootstrap draws). TRAIN scores are exploratory and subject to selection bias. One-prompt smoke intervals are descriptive only.',
            'CLIP distance is a feature-space diversity proxy. Contrastive B/C and feature-cost/raw-latent updates are heuristics. One intervention is used per trajectory.',
            f'Pooled exploratory correlation of Sinkhorn reduction and reward gain: {correlation}. This is not a causal or independent-sample test.', '',
            f'Accounting: `{json.dumps(accounting,sort_keys=True)}`']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if select:
        if split!='train' or stage=='smoke' or len(guided)!=expected:
            raise RuntimeError('selection requires a complete TRAIN comparison/focused stage')
        allowed=[r for r in summaries if r['method'] in ('A','B','C')]
        selected=[]
        for r in allowed:
            c=groups[r['config_id']][0]['config']
            # Select at most two; prefer distinct algorithms to answer mechanism questions.
            if not selected or c['method']!=selected[0]['method']: selected.append(c)
            if len(selected)==2: break
        if len(selected)<2:
            for r in allowed:
                c=groups[r['config_id']][0]['config']
                if c not in selected: selected.append(c)
                if len(selected)==2: break
        selection=dict(selection_split='train',stage=stage,configs=selected,
                       selection_rule='highest TRAIN mean reward, prefer distinct methods; otherwise two distinct configs',
                       source_analysis_sha256=digest(summaries),prompt_indices=present)
        atomic_json(root/'selection.json',selection)
    print(json.dumps(dict(stage=stage,accounting=accounting,top=summaries[:4]),indent=2))
    return summaries


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['smoke','compare','focused','heldout'],default='compare')
    p.add_argument('--partial',action='store_true'); p.add_argument('--select',action='store_true')
    a=p.parse_args(); analyze(a.output,a.stage,a.partial,a.select)


if __name__=='__main__': main()
