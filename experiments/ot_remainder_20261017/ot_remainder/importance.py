"""Complete the importance-resampling control without altering frozen toy rows."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .r3 import sample,reward,metrics,numerical_gates

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'COMPLETE.json').exists():return
    begin=time.time();numerical_gates()
    old=Path(cfg['old_root']);oldcfg=json.loads((old/'preregistered.json').read_text())
    for key in ['seed','populations','replicates','tilt','steps','ridge']:
        assert cfg[key]==oldcfg[key],key
    prior=[json.loads(x) for x in (old/'rows.jsonl').read_text().splitlines()]
    rows=[]
    for n in cfg['populations']:
        for rep in range(cfg['replicates']):
            for si,scenario in enumerate(['linear','quadratic','four_mode','four_mode_missing_rare']):
                x=sample(n,scenario,np.random.default_rng(np.random.SeedSequence([cfg['seed'],n,rep,si])))
                oldnone=next(r for r in prior if (r['n'],r['replicate'],r['scenario'],r['method'])==(n,rep,scenario,'none'))
                assert metrics(x,scenario,0)==oldnone['initial_metrics']
                r=reward(x,scenario);logw=cfg['tilt']*r;w=np.exp(logw-logw.max());w/=w.sum()
                for method in ['importance_multinomial','importance_systematic']:
                    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],n,rep,si,2901]))
                    u=rng.random(n) if method.endswith('multinomial') else (np.arange(n)+rng.random())/n
                    ids=np.searchsorted(np.cumsum(w),u,side='right').clip(0,n-1);y=x[ids]
                    rows.append(dict(n=n,replicate=rep,scenario=scenario,method=method,
                        initial_metrics=metrics(x,scenario,0),terminal_metrics=metrics(y,scenario,cfg['tilt']),
                        reward_calls=2*n,terminal_mean_reward=float(reward(y,scenario).mean()),
                        ess=float(1/(w@w)),parents=ids.tolist(),unique_parents=len(set(ids.tolist()))))
    combined=prior+rows;summary=[]
    for n in cfg['populations']:
        for scenario in ['linear','quadratic','four_mode','four_mode_missing_rare']:
            for method in sorted({r['method'] for r in combined}):
                sub=[r for r in combined if (r['n'],r['scenario'],r['method'])==(n,scenario,method)]
                fields=['mean_error','covariance_error'] if scenario in ['linear','quadratic'] else ['mass_tv','missing_rare_failure']
                summary.append(dict(n=n,scenario=scenario,method=method,**{k:float(np.mean([r['terminal_metrics'][k] for r in sub])) for k in fields}))
    (root/'new_rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    (root/'all_rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in combined))
    (root/'preregistered.json').write_text(json.dumps(cfg,indent=2)+'\n')
    result=dict(passed=True,new_rows=len(rows),reused_frozen_rows=len(prior),all_rows=len(combined),
        summary=summary,old_initial_population_reproduction=True,
        old_nonlinear_gate_unchanged=json.loads((old/'COMPLETE.json').read_text())['nonlinear_gate'],
        additional_analytic_reward_evaluations=sum(r['reward_calls'] for r in rows),
        elapsed_seconds=time.time()-begin,neural_nfe=0,tpu_hours=0,gpu_hours=0,
        limitations=['Finite self-normalized importance resampling, not exact target samples',
                    'Same initial missing-rare pool cannot recover absent modes by resampling'])
    (root/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
