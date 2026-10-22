import argparse
import csv
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from sd15_sinkhorn_tpu.common import atomic_json
from sd15_sinkhorn_tpu.evaluate import summary_metrics
from .run import OLD


def read(p): return json.loads(Path(p).read_text())


def csv_write(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for row in rows for k,v in row.items() if not isinstance(v,(dict,list))})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)


@lru_cache(maxsize=None)
def image_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recycled(parts,n=8):
    scored={k:[] for k in ['rewards','clip_alignment','clip_embeddings']}
    seen=set(); selected_from=[]
    for label,scores,folder in parts:
        for i in range(len(scores['rewards'])):
            h=image_hash(str(Path(folder)/f'{i:03d}.png'))
            if h in seen: continue
            seen.add(h); selected_from.append(label)
            for k in scored: scored[k].append(scores[k][i])
    order=np.argsort(scored['rewards'],kind='stable')[-n:]
    return {**summary_metrics(scored,indices=order),
            'unique_candidates':len(seen),'selected_guided_images':sum(selected_from[i]=='guided' for i in order)}


def bootstrap(values):
    a=np.asarray(values,dtype='float64'); rng=np.random.default_rng(81031)
    means=a[rng.integers(len(a),size=(20000,len(a)))].mean(1)
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def rows_for(root,phase,split,partial=False):
    folder=Path(root)/'full'/phase; rows=[]
    steps=30 if phase=='30eta0' else 100
    expected=8 if split=='train' else 24
    if steps==30:
        stage='compare' if split=='train' else 'heldout'
        guided=list((OLD/stage).glob('trials/*/*/result.json'))
        guided=[p for p in guided if read(p)['config']['method'] in 'ABC']
        bank_base=OLD/'banks'/split/'seed20261015' if split=='train' else OLD/'banks'/split/'seed20361015'
        bank_dirs={i:bank_base/'m16_n8'/f'p{i:03d}' for i in range(expected)}
    else:
        guided=list((folder/split).glob('trials/*/*/result.json'))
        bank_dirs={i:folder/'banks'/split/f'p{i:03d}' for i in range(expected)}
    for p in sorted((folder/split/'bfs').glob('p*/*/result.json')):
        r=read(p)
        if len(r['events'])!=3 or not r['finite']: raise RuntimeError('invalid BFS record')
        for e in r['events']:
            assert len(e['parent_indices'])==r['n']
            assert all(0<=i<r['n'] for i in e['parent_indices'])
            assert np.isfinite(e['raw_rewards']).all()
        rows.append(dict(method=r['method'],family='BFS',prompt_index=r['prompt_index'],
                         mean_reward=r['mean_reward'],best_reward=r['best_reward'],clip_alignment=r['clip_alignment'],
                         clip_diversity=r['clip_diversity'],diffusion_sample_steps=r['diffusion_sample_steps'],
                         reward_evaluations=r['reward_evaluations'],wall_seconds=r['wall_seconds'],
                         unique_final_latents=r['unique_final_latents'],population_mean=r['all_population']['mean_reward'],
                         n=r['n'],output_policy='top8' if r['n']==28 else 'all8',source_file=str(p)))
    for i,bank_dir in bank_dirs.items():
        if not (bank_dir/'bank.json').exists():
            if partial: continue
            raise RuntimeError(f'missing bank {bank_dir}')
        bank=read(bank_dir/'bank.json'); source=bank['sets']['source']; ref=bank['sets']['reference']
        base_parts=[('source',source,bank_dir/'source/images'),('reference',ref,bank_dir/'reference/images')]
        rows.append(dict(method='source8',family='base',prompt_index=i,**summary_metrics(source),
                         diffusion_sample_steps=8*steps,reward_evaluations=8,output_policy='all8'))
        rows.append(dict(method='bank24_select',family='base',prompt_index=i,**recycled(base_parts),
                         diffusion_sample_steps=24*steps,reward_evaluations=24,output_policy='top8'))
        bon=bank_dir/'fresh_bon_28/result.json'
        if bon.exists():
            r=read(bon)
            rows.append(dict(method='fresh_bon28',family='BoN',prompt_index=i,mean_reward=r['mean_reward'],
                             best_reward=r['best_reward'],clip_alignment=r['clip_alignment'],clip_diversity=r['clip_diversity'],
                             diffusion_sample_steps=28*steps,reward_evaluations=28,wall_seconds=r['wall_seconds'],output_policy='top8'))
    for p in sorted(guided):
        r=read(p); c=r['config']; i=r['prompt_index']; bank_dir=bank_dirs[i]
        if not (bank_dir/'bank.json').exists(): continue
        bank=read(bank_dir/'bank.json')
        base=f"{c['method']}_{r['config_id']}"
        label=f"{c['method']} eps={c['epsilon']} {c['mode']} eta={c['eta']}"
        common=dict(prompt_index=i,family='Sinkhorn',epsilon=c['epsilon'],eta_transport=c['eta'],
                    mode=c['mode'],gamma_neg=c['gamma_neg'],original_config_id=r['config_id'],label=label,
                    diffusion_sample_steps=r['logical_total_sample_steps'],reward_evaluations=r['reward_evaluations'],
                    source_file=str(p))
        rows.append(dict(**common,method=base+'_guided',mean_reward=r['mean_reward'],best_reward=r['best_reward'],
                         clip_alignment=r['clip_alignment'],clip_diversity=r['clip_diversity'],output_policy='guided8'))
        parts=[('source',bank['sets']['source'],bank_dir/'source/images'),
               ('reference',bank['sets']['reference'],bank_dir/'reference/images'),('guided',r['scores'],p.parent/'images')]
        rows.append(dict(**common,method=base+'_recycle',**recycled(parts),output_policy='top8_all_available'))
    for r in rows: r.update(phase=phase,split=split)
    counts=defaultdict(int)
    for r in rows: counts[r['method']]+=1
    required=['bfs_manual_n8','bfs_manual_n28','bfs_graph_n28','source8','bank24_select','fresh_bon28']
    if not partial:
        for method in required: assert counts[method]==expected,(phase,split,method,counts[method],expected)
        n_guided=12 if split=='train' else 2
        assert len([r for r in rows if r['output_policy']=='guided8'])==expected*n_guided
        assert all(n==expected for n in counts.values())
    return rows


def analyze(root,partial=False):
    root=Path(root); out=root/'analysis'; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for phase,split in [('30eta0','train'),('30eta0','test'),('100eta1','test')]:
        rows+=rows_for(root,phase,split,partial)
    csv_write(out/'results.csv',rows)
    groups=defaultdict(list)
    for r in rows: groups[(r['phase'],r['split'],r['method'])].append(r)
    summaries=[]
    for (phase,split,method),rr in groups.items():
        result=dict(phase=phase,split=split,method=method,family=rr[0]['family'],n_prompts=len(rr),
                    diffusion_sample_steps=rr[0]['diffusion_sample_steps'],reward_evaluations=rr[0]['reward_evaluations'],
                    output_policy=rr[0]['output_policy'],label=rr[0].get('label',method))
        for k in ['mean_reward','best_reward','clip_alignment','clip_diversity','wall_seconds','unique_final_latents','population_mean','selected_guided_images']:
            values=[r[k] for r in rr if k in r]
            if values: result[k]=float(np.mean(values))
        summaries.append(result)
    summaries.sort(key=lambda r:(r['phase'],r['split'],-r['best_reward']))
    csv_write(out/'summary.csv',summaries)
    paired=[]
    for phase,split in [('30eta0','train'),('30eta0','test'),('100eta1','test')]:
        methods=[m for p,s,m in groups if p==phase and s==split]
        for bfs in ['bfs_manual_n28','bfs_graph_n28']:
            for other in methods:
                if other==bfs or not ('_guided' in other or '_recycle' in other or other=='fresh_bon28'): continue
                a={r['prompt_index']:r for r in groups.get((phase,split,bfs),[])}
                b={r['prompt_index']:r for r in groups[(phase,split,other)]}
                indices=sorted(set(a)&set(b))
                if not indices: continue
                rec=dict(phase=phase,split=split,A=bfs,B=other,n_prompts=len(indices))
                for metric in ['best_reward','mean_reward','clip_diversity','clip_alignment']:
                    diff=np.array([a[i][metric]-b[i][metric] for i in indices])
                    low,high=bootstrap(diff)
                    rec.update({metric+'_delta':float(diff.mean()),metric+'_ci_low':low,metric+'_ci_high':high})
                    if metric=='best_reward': rec.update(wins=int((diff>1e-7).sum()),ties=int((abs(diff)<=1e-7).sum()),losses=int((diff< -1e-7).sum()))
                rec['budget_matched']=all(a[i]['diffusion_sample_steps']==b[i]['diffusion_sample_steps'] for i in indices)
                paired.append(rec)
    csv_write(out/'paired_comparisons.csv',paired)
    recipes=[]
    aa={r['prompt_index']:r for r in groups.get(('100eta1','test','bfs_manual_n8'),[])}
    for (phase,split,method),rr in groups.items():
        if phase!='30eta0' or split!='test': continue
        if not ('_guided' in method or '_recycle' in method or method=='fresh_bon28'): continue
        bb={r['prompt_index']:r for r in rr}; indices=sorted(set(aa)&set(bb))
        if not indices: continue
        delta=np.array([aa[i]['best_reward']-bb[i]['best_reward'] for i in indices])
        lo,hi=bootstrap(delta)
        recipes.append(dict(A='100eta1_bfs_manual_n8',B='30eta0_'+method,n_prompts=len(indices),
                            A_NFE=800,B_NFE=840,protocol_matched=False,exact_budget_matched=False,
                            best_reward_delta=float(delta.mean()),ci_low=lo,ci_high=hi))
    csv_write(out/'recipe_comparisons.csv',recipes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for ax,phase,title in zip(axes,['30eta0','100eta1'],['DDIM30 eta0: 840 sample-steps','DDIM100 eta1: 2,800 sample-steps']):
        for r in summaries:
            if r['phase']!=phase or r['split']!='test': continue
            m=r['method']
            if m=='bank24_select': continue
            color,marker=('#64748b','s')
            label=m
            if r['family']=='Sinkhorn':
                marker='o' if r['output_policy']=='guided8' else '^'; color='#2563eb'
                eps=r['label'].split('eps=')[1].split()[0]
                label='A eps='+eps+(' guided' if marker=='o' else ' recycle')
            elif m=='fresh_bon28': color,marker='#dc2626','*'; label='Fresh BoN28'
            elif m=='bfs_manual_n28': color,marker='#059669','D'; label='BFS manual N28'
            elif m=='bfs_graph_n28': color,marker='#7c3aed','D'; label='BFS graph N28'
            elif m=='bfs_manual_n8': color,marker='#d97706','D'; label='BFS manual N8 (lower budget)'
            elif m=='source8': label='Source N8 (lower budget)'
            ax.scatter(r['clip_diversity'],r['mean_reward'],s=110 if marker=='*' else 65,color=color,marker=marker,label=label,alpha=.85)
        ax.set(title=title,xlabel='CLIP image diversity',ylabel='Mean reward of output set')
        ax.grid(alpha=.2); ax.margins(.12)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('BFS versus frozen Sinkhorn — target 24 paired follow-up prompts'+(' (PARTIAL)' if partial else ''))
    fig.tight_layout(rect=[0,.15,1,.94]); fig.savefig(out/'reward_diversity.png',dpi=180); plt.close(fig)
    result=dict(complete=not partial,summary=summaries,paired=paired,recipe_comparisons=recipes)
    atomic_json(out/'analysis.json',result)
    lines=['# BFS versus Sinkhorn — đo trực tiếp trên TPU'+(' (PARTIAL, chưa kết luận pha chưa đủ dữ liệu)' if partial else ''), '',
           'Các policy được giữ cố định trước lượt chạy: BFS manual gamma=0.024 tại indices [20,40,80] của DDIM100; graph [40,60,90], tau [8,8,32]; hai cấu hình A đã đóng băng ở thí nghiệm trước. Không chọn hyperparameter từ kết quả so sánh mới.', '',
           'Đã kiểm tra kết quả cũ: policy_verification_compiler_v1/audit_summary.csv có MANUAL8 selected IR=1.0604434 và graph selected IR=1.0514580. seed_coverage_compiler_v2 dùng pool/seed khác có MANUAL=0.9994291. Các số lịch sử không phải kết quả trên 24 prompt hiện tại và không được gộp vào bảng dưới.', '',
           'Best/prompt là chỉ số lịch sử của BFS: max terminal ImageReward, sau đó lấy trung bình qua prompt. Mean/top8 và CLIP diversity là chỉ số bổ sung. N28 BFS và BoN chọn top8; guided Sinkhorn chỉ có tám ảnh cuối. Vì reference/source đã sinh đầy đủ và chấm reward, báo cáo thêm Sinkhorn recycle: chọn top8 từ reference+source+guided, khử ảnh trùng bitwise, không thêm diffusion hoặc reward calls. Bank-only top8 là control ngân sách thấp hơn.', '']
    for phase,title in [('30eta0','DDIM30 eta0'),('100eta1','DDIM100 eta1')]:
        lines += [f'## {title} — 24 prompt follow-up', '',
                  '| Method | n prompt | Diffusion NFE/prompt | Reward eval/prompt | Mean/top8 | Best/prompt | CLIP diversity |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in summaries:
            if r['phase']!=phase or r['split']!='test': continue
            label=r['label']+' '+r['output_policy'] if r['family']=='Sinkhorn' else r['method']
            lines.append(f"| {label} | {r['n_prompts']} | {r['diffusion_sample_steps']} | {r['reward_evaluations']} | {r['mean_reward']:.5f} | {r['best_reward']:.5f} | {r['clip_diversity']:.5f} |")
        lines += ['', '| Paired contrast: A − B | Best/prompt Δ | 95% prompt-bootstrap CI | W/T/L |', '|---|---:|---:|---:|']
        for r in paired:
            if r['phase']!=phase or r['split']!='test': continue
            lines.append(f"| {r['A']} − {r['B']} | {r['best_reward_delta']:+.5f} | [{r['best_reward_ci_low']:+.5f}, {r['best_reward_ci_high']:+.5f}] | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines += ['', '## Recipe BFS thường dùng: 800 NFE so với Sinkhorn 840 NFE', '',
              'BFS manual N8 DDIM100 eta1 được đặt cạnh recipe Sinkhorn DDIM30 eta0. Cùng prompt/seed nhưng khác scheduler regime, và BFS dùng ít hơn 40 NFE (4,76%). Đây là so sánh recipe thực dụng; không dùng để quy riêng chênh lệch cho algorithm.', '',
              '| Contrast | n prompt | Best/prompt Δ | CI 95% |', '|---|---:|---:|---:|']
    for r in recipes:
        lines.append(f"| {r['A']} − {r['B']} | {r['n_prompts']} | {r['best_reward_delta']:+.5f} | [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] |")
    lines += ['', '## Diễn giải protocol và giới hạn', '',
              '- Mỗi bảng so sánh cùng prompt, seed, model/revision, BF16 UNet/VAE, FP32 DDIM arithmetic và stateless per-step noise. Các particle đầu của BFS N8/N28 và source Sinkhorn dùng chung initial stream. Fresh BoN dùng stream độc lập như protocol cũ.',
              '- BFS dùng trực tiếp StaticGraphBFSAdapter/PaperBFSAdapter và SSP từ official source: Max-history bắt đầu từ 0, temperature rồi softmax, đúng N offspring, event RNG cũ. Vòng denoising dùng chung sampler mới với Sinkhorn; đây không phải byte-identical replay pipeline BF16 scheduler lịch sử.',
              '- DDIM30 transfer completed-update progress: ceil((old_index+1)*30/100)-1. Manual thành [6,12,24], graph thành [12,18,27]; giữ nguyên tau tại các event gốc. Không dùng lại gamma trên chỉ số 30 bước để vô tình thay temperature.',
              '- DDIM eta0 không tạo jitter sau resampling. Con của cùng ancestor có thể trùng terminal image; mean reward cao có thể đi kèm collapse diversity. DDIM100 eta1 được chạy riêng để kiểm tra chế độ stochastic branching. Không dùng chênh lệch giữa hai bảng để suy ra hiệu ứng riêng của algorithm.',
              '- N28 BFS cùng diffusion budget với Sinkhorn M16/N8: 840 NFE ở DDIM30, 2800 ở DDIM100. BFS N8 có budget thấp hơn (240/800). BFS có 112 reward evaluations ở N28, Sinkhorn 32 và BoN28 là 28: chưa matched verifier compute. Số CFG sample-branch evaluations gấp đôi diffusion NFE.',
              '- Hyperparameter của A chuyển nguyên từ TRAIN DDIM30 sang DDIM100; không có tuning mới cho eta1. Hai BFS policy được lấy từ các thí nghiệm lịch sử độc lập. Các prompt follow-up đã từng được đánh giá trong thí nghiệm trước, nên đây là mở rộng comparator, không phải một TEST mới chưa từng nhìn.',
              '- CI dùng 20.000 lần paired prompt bootstrap; không coi ảnh chung prompt là quan sát độc lập. Nhiều contrast chưa điều chỉnh family-wise error; không chọn winner từ TEST. TRAIN gồm đủ A/B/C của sweep trước và được xuất riêng trong CSV.',
              '- Trường recycle dùng các terminal reference/source vốn đã có trong ngân sách; không tính lại reward. Không dùng decoded x0 tại event BFS làm ảnh terminal.', '',
              'Files: results.csv, summary.csv, paired_comparisons.csv, reward_diversity.png, analysis.json; raw events/latents/images ở full/. Dữ liệu sd15-sinkhorn-v2 và toàn bộ BFS cũ được giữ nguyên.', '']
    (out/'REPORT_VI.md').write_text('\n'.join(lines))
    print(json.dumps(dict(rows=len(rows),summaries=len(summaries),paired=len(paired),complete=not partial),indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--partial',action='store_true')
    a=p.parse_args(); analyze(a.output,a.partial)


if __name__=='__main__': main()
