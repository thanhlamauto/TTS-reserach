"""CPU analysis of the frozen paired diagnostic. Never performs model inference."""
from pathlib import Path
import argparse,csv,hashlib,json,math,statistics
import numpy as np
from paired_pilot import read,write,policies


def interval(x, draws, alpha=.05):
    return np.quantile(np.asarray(x)[draws].mean(1),[alpha/2,1-alpha/2]).tolist()


def variance(x):
    P,S=x.shape;W=float(x.var(axis=1,ddof=1).mean());B=float(x.mean(1).var(ddof=1)-W/S)
    return dict(within_seed_variance=W,between_prompt_variance_untruncated=B,
                between_prompt_variance_forecast=max(0.,B),seed_share_at_four_seeds=(W/S)/(max(0.,B)+W/S) if W else 0.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    plan=read(root/'preregistered.json');ps=policies(plan);P,S,C=6,4,6
    assert read(root/'correctness.json')['passed']
    files=list((root/'full_policies').glob('*.json'));assert len(files)==P*S*C
    refs=list((root/'references/A').glob('*.json'));assert len(refs)==P*S
    edge_ids=list(plan['edges']['edges']);E=len(edge_ids);assert E==16
    vals=np.empty((P,S,C));pred=np.empty_like(vals);actual=np.empty_like(vals)
    V=np.zeros((C,E));role={role:k for k,c in enumerate(plan['candidates']) for role in c['roles']}
    for k,c in enumerate(plan['candidates']):
        for edge in plan['edges']['paths'][c['semantic_sha256']]:V[k,edge_ids.index(edge)]=1
    seen=set();edge_rows=[];raw_hashes={}
    def record(path):
        raw_hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)
    for i,pid in enumerate(plan['prompts']):
        for j,seed in enumerate(plan['seeds']):
            rr=record(root/'references/A'/f'p{pid}-s{seed}.json');d=[]
            for eid in edge_ids:
                edge=plan['edges']['edges'][eid]
                job=dict(src=edge['src'],dst=edge['dst'],tau=edge['multiplier'],mid=None,mtau=None)
                key=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:20]
                path=root/'observations/A'/f'p{pid}-s{seed}'/(key+'.json');er=record(path)
                assert er['job']==job and er['reference_reward']==rr['reward']
                assert er['regret']==rr['reward']-er['reward']
                d.append(er['regret']);edge_rows.append(dict(prompt=pid,seed=seed,edge=eid,regret=er['regret']))
            for k,pol in enumerate(ps):
                r=record(root/'full_policies'/f'p{pid}-s{seed}-{pol["id"]}.json')
                key=(pid,seed,pol['id']);assert key not in seen;seen.add(key)
                assert r['policy']==pol and r['NFE']==800 and r['reward']==max(r['scores'])
                assert r['initial_latents_sha256']==rr['initial_latents_sha256']
                assert len(r['events'])==3 and [e['sampling_index'] for e in r['events']]==pol['resampling_steps']
                assert r['counters']['actual_verifier_candidate_scores']==32 and r['counters']['actual_verifier_batch_calls']==4
                vals[i,j,k]=r['reward'];actual[i,j,k]=rr['reward']-r['reward'];pred[i,j,k]=V[k]@np.array(d)
    assert len(edge_rows)==384 and all(np.isfinite(x).all() for x in [vals,pred,actual])
    residual=actual-pred
    rng=np.random.default_rng(plan['stat_seed']);draws=rng.integers(P,size=(plan['bootstrap'],P))
    summaries=[]
    for k,c in enumerate(plan['candidates']):
        summaries.append(dict(policy_id=ps[k]['id'],roles=c['roles'],mean_IR=float(vals[:,:,k].mean()),
             predicted_regret_mean=float(pred[:,:,k].mean()),actual_regret_mean=float(actual[:,:,k].mean()),
             residual_mean=float(residual[:,:,k].mean()),residual_CI95=interval(residual[:,:,k].mean(1),draws),
             residual_Bonferroni_approx_CI=interval(residual[:,:,k].mean(1),draws,.05/C),
             residual_variance=variance(residual[:,:,k])))
    contrasts=[]
    z=statistics.NormalDist().inv_cdf(1-.05/(2*3));zp=statistics.NormalDist().inv_cdf(.8)
    for ar,br in plan['primary_contrasts']:
        aa,bb=vals[:,:,role[ar]],vals[:,:,role[br]];d=aa-bb;x=d.mean(1);v=variance(d)
        sd=float(x.std(ddof=1));mde=(z+zp)*sd/math.sqrt(P)
        covariance=float(np.mean([np.cov(u,v,ddof=1)[0,1] for u,v in zip(aa,bb)]))
        contrasts.append(dict(A=ar,B=br,delta=float(x.mean()),CI95=interval(x,draws),
              family_Bonferroni_approx_CI=interval(x,draws,.05/3),**v,within_prompt_paired_policy_covariance=covariance,
              approximate_MDE80=mde,approximate_prompts_for_epsilon002=math.ceil(((z+zp)*sd/.02)**2),
              interpretation='descriptive historical-TRAIN/fresh-seed diagnostic; not final confirmation or a valid sequential certificate'))
    # LOPO: one held-out prompt including every seed. These folds are dependent.
    folds=[]
    for i,pid in enumerate(plan['prompts']):
        train=[j for j in range(P) if j!=i]
        graph=min(range(C),key=lambda k:(pred[train,:,k].mean(),plan['candidates'][k]['semantic_sha256']))
        direct=min(range(C),key=lambda k:(-vals[train,:,k].mean(),plan['candidates'][k]['semantic_sha256']))
        folds.append(dict(prompt=pid,graph_policy=ps[graph]['id'],direct_policy=ps[direct]['id'],
                          graph_IR=float(vals[i,:,graph].mean()),direct_IR=float(vals[i,:,direct].mean()),
                          direct_minus_graph=float(vals[i,:,direct].mean()-vals[i,:,graph].mean())))
    from scipy.stats import spearmanr
    rho=float(spearmanr(-pred.mean((0,1)),vals.mean((0,1))).statistic)
    work=[read(p) for p in (root/'work_ledgers').glob('*.json')]
    totals={k:sum(w[k] for w in work) for k in ['actual_denoising_NFE','actual_verifier_candidate_scores','actual_verifier_batch_calls']}
    decision=dict(status=['UNDERIDENTIFIED','UNDERPOWERED'] if any(c['approximate_MDE80']>.02 for c in contrasts) else ['UNDERIDENTIFIED'],
                  next_design='NO NEW OBJECTIVE: diagnose the completed same-row panel; do not auto-expand or nominate a winner',
                  selected_finalist=None,confirmation_opened=False,independent_compiler_replicates=0,
                  equal_offline_budget_comparison=False,production_change=False)
    result=dict(passed=True,prompts=P,seeds=S,policies=C,full_policy_rows=144,reference_rows=24,edge_rows=384,
                summaries=summaries,primary_contrasts=contrasts,LOPO=folds,
                graph_full_policy_Spearman=rho,actual_new_work_including_gates=totals,
                uncertainty='prompt-cluster approximate/descriptive bootstrap; six independent prompts, not24',decision=decision)
    write(root/'RESULTS.json',result);write(root/'RAW_HASHES.json',raw_hashes)
    with (root/'same_row_edge_observations.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(edge_rows[0]));w.writeheader();w.writerows(edge_rows)
    with (root/'same_row_policy_residuals.csv').open('w') as f:
        fields=['prompt','seed','policy_id','reward','predicted_regret','actual_regret','residual'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,pid in enumerate(plan['prompts']):
            for j,seed in enumerate(plan['seeds']):
                for k,pol in enumerate(ps):w.writerow(dict(zip(fields,[pid,seed,pol['id'],vals[i,j,k],pred[i,j,k],actual[i,j,k],residual[i,j,k]])))
    lines=['# Covered-bank paired TPU diagnostic','',
           'Engineering panel complete:6 historically explored TRAIN prompts x4 fresh seeds,6 exact policies and16 union edges per prompt-seed. This is development only; no final confirmation or independent compilation replicate. Fixed epsilon=.02IR.','',
           '| Policy roles | Terminal mean IR | Predicted regret | Actual regret | Residual mean | Descriptive residual95%CI |',
           '|---|---:|---:|---:|---:|---|']
    for s in summaries:lines.append(f'| {", ".join(s["roles"])} | {s["mean_IR"]:.6f} | {s["predicted_regret_mean"]:.6f} | {s["actual_regret_mean"]:.6f} | {s["residual_mean"]:.6f} | {s["residual_CI95"]} |')
    lines+=['','Primary paired contrasts (mean4 seeds inside each prompt first):','```json',json.dumps(contrasts,indent=2),'```','',
            f'Surrogate/full-policy Spearman over6 policies: {rho:.4f}. This is a small dependent policy set, not a validated global rank guarantee.',
            'LOPO direct versus graph is an out-of-fold predictive diagnostic. The algorithms use different profiling work; no matched-budget compiler superiority or stability claim follows.',
            'Graph-only, historical verification output, exact Manual/Transfer/Fine and direct full-policy data are retained. A future formal compiler comparison must allocate equal offline budgets and independent selection/confirmation sets.',
            '',f'Actual measured new denoising/verifier work including correctness: {json.dumps(totals)}. CPU analysis, compile/warmup/retries and full slice time are tracked separately; no money estimate is inferred.',
            '',f'Decision: {json.dumps(decision)}','',
            'Secondary independent neural metrics: NOT_EVALUATED. ImageReward and every other neural forward in this run execute on the existing TPU; CPU is limited to preprocessing, SSP, bookkeeping and statistics. No GPU, Modal, VALIDATION, TEST or reserved confirmation pool was used.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'COMPLETE.json',dict(passed=True,raw_full_rows=len(seen),raw_edges=len(edge_rows),decision=decision))
    print(json.dumps(dict(status='complete',full_rows=144,edge_rows=384,Spearman=rho,decision=decision)))


if __name__=='__main__':main()
