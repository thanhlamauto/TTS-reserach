"""Toy-only weak Poisson transport. No reward gradients or image models."""
import argparse,json,time
from pathlib import Path
import numpy as np


def basis(x,centers,bandwidth=1.5,linear=False):
    n,d=x.shape
    phi=x.copy();grad=np.broadcast_to(np.eye(d),(n,d,d)).copy()
    if linear:return phi,grad
    diff=x[:,None,:]-centers[None,:,:]
    kernel=np.exp(-np.sum(diff**2,axis=-1)/(2*bandwidth**2))
    return np.concatenate([phi,kernel],1),np.concatenate([grad,-diff*kernel[:,:,None]/bandwidth**2],1)


def velocity(x,r,centers,ridge=1e-3,linear=False):
    phi,grad=basis(x,centers,linear=linear)
    G=np.einsum('nld,nmd->lm',grad,grad)/len(x)
    b=np.mean((r-r.mean())[:,None]*phi,0)
    a=np.linalg.solve(G+ridge*np.eye(len(b)),b)
    v=np.einsum('l,nld->nd',a,grad)
    return v,float(np.linalg.norm(G@a-b)/(np.linalg.norm(b)+1e-12)),float(np.linalg.cond(G+ridge*np.eye(len(b))))


CENTERS=np.array([[-2.5,-2.5],[-2.5,2.5],[2.5,-2.5],[2.5,2.5]])
PRIOR=np.array([.4,.3,.25,.05]);MODE_REWARD=np.array([0,.5,1,2.])


def reward(x,scenario):
    if scenario=='linear':return x@np.array([.7,-.4])
    if scenario=='quadratic':return -.5*np.sum((x-np.array([1.,-.5]))**2,1)
    ids=np.argmin(np.sum((x[:,None,:]-CENTERS[None,:,:])**2,2),1)
    return MODE_REWARD[ids]


def sample(n,scenario,rng):
    if scenario in ['linear','quadratic']:return rng.normal(size=(n,2))
    p=PRIOR.copy()
    if scenario=='four_mode_missing_rare':p[-1]=0;p/=p.sum()
    ids=rng.choice(4,size=n,p=p)
    return CENTERS[ids]+.35*rng.normal(size=(n,2))


def metrics(x,scenario,tilt):
    if scenario in ['linear','quadratic']:
        mean=tilt*np.array([.7,-.4]) if scenario=='linear' else tilt/(1+tilt)*np.array([1.,-.5])
        cov=np.eye(2) if scenario=='linear' else np.eye(2)/(1+tilt)
        empirical=(x-x.mean(0)).T@(x-x.mean(0))/len(x)
        return dict(mean_error=float(np.linalg.norm(x.mean(0)-mean)),covariance_error=float(np.linalg.norm(empirical-cov)),
                    mean=x.mean(0).tolist(),covariance=empirical.tolist())
    ids=np.argmin(np.sum((x[:,None,:]-CENTERS[None,:,:])**2,2),1)
    mass=np.bincount(ids,minlength=4)/len(x)
    target=PRIOR*np.exp(tilt*MODE_REWARD);target/=target.sum()
    return dict(mode_mass=mass.tolist(),target_mass=target.tolist(),mass_tv=float(np.abs(mass-target).sum()/2),
                missing_rare_failure=bool(mass[-1]==0))


def numerical_gates():
    rng=np.random.default_rng(910);x=rng.normal(size=(7,2));c=rng.normal(size=(5,2))
    phi,g=basis(x,c);fd=np.zeros_like(g);h=1e-5
    for k in range(2):
        plus=x.copy();minus=x.copy();plus[:,k]+=h;minus[:,k]-=h
        fd[:,:,k]=(basis(plus,c)[0]-basis(minus,c)[0])/(2*h)
    error=float(np.max(np.abs(g-fd)))
    # Cubature has exact standard Gaussian mean/covariance for linear reward.
    z=np.array([[2.,0],[-2.,0],[0,2.],[0,-2.],[0,0],[0,0],[0,0],[0,0]])
    a=np.array([.7,-.4]);v,_,_=velocity(z,z@a,c,ridge=0,linear=True)
    linear_error=float(np.max(np.abs(v-a)))
    if error>1e-8 or linear_error>1e-12:raise RuntimeError('analytic gradient/linear weak equation failed')
    return dict(passed=True,basis_fd_max_abs=error,linear_gaussian_cubature_velocity_error=linear_error)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'COMPLETE.json').exists():return
    start=time.time();gates=numerical_gates();rows=[]
    scenarios=['linear','quadratic','four_mode','four_mode_missing_rare']
    methods=['none','linear','kernel','shuffled']
    for n in cfg['populations']:
        for rep in range(cfg['replicates']):
            for si,scenario in enumerate(scenarios):
                initial=sample(n,scenario,np.random.default_rng(np.random.SeedSequence([cfg['seed'],n,rep,si])))
                # Fixed centers, never fitted to terminal errors or reward labels.
                centers=initial[np.linspace(0,n-1,min(16,n)).astype(int)].copy()
                for method in methods:
                    x=initial.copy();residual=[];conditions=[]
                    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],n,rep,si,1337]))
                    for step in range(cfg['steps']):
                        r=reward(x,scenario)
                        if method=='shuffled':r=r[rng.permutation(n)]
                        if method!='none':
                            v,res,cond=velocity(x,r,centers,cfg['ridge'],method=='linear')
                            x=x+(cfg['tilt']/cfg['steps'])*v
                            residual.append(res);conditions.append(cond)
                        if not np.isfinite(x).all():raise RuntimeError('nonfinite toy particles')
                    terminal=reward(x,scenario)
                    row=dict(n=n,replicate=rep,scenario=scenario,method=method,
                        initial_metrics=metrics(initial,scenario,0),terminal_metrics=metrics(x,scenario,cfg['tilt']),
                        reward_calls=n*(cfg['steps']+1),terminal_mean_reward=float(terminal.mean()),
                        average_relative_weak_residual=None if not residual else float(np.mean(residual)),
                        max_ridge_condition=None if not conditions else max(conditions))
                    rows.append(row)
    summary=[]
    for n in cfg['populations']:
        for scenario in scenarios:
            for method in methods:
                subset=[r for r in rows if r['n']==n and r['scenario']==scenario and r['method']==method]
                fields=['mean_error','covariance_error'] if scenario in ['linear','quadratic'] else ['mass_tv','missing_rare_failure']
                summary.append(dict(n=n,scenario=scenario,method=method,**{k:float(np.mean([r['terminal_metrics'][k] for r in subset])) for k in fields}))
    def value(s,m,k):return next(x[k] for x in summary if x['n']==64 and x['scenario']==s and x['method']==m)
    nonlinear_gate=(value('quadratic','kernel','covariance_error')<=.8*value('quadratic','linear','covariance_error') and
                    value('quadratic','kernel','covariance_error')<value('quadratic','shuffled','covariance_error') and
                    value('four_mode','kernel','mass_tv')<value('four_mode','linear','mass_tv') and
                    value('four_mode','kernel','mass_tv')<value('four_mode','shuffled','mass_tv'))
    result=dict(passed=True,numerical_gates=gates,rows=len(rows),summary=summary,nonlinear_gate=bool(nonlinear_gate),
        next_stage='TOY_ONLY_NO_AUTOMATIC_IMAGE_RUN',total_reward_calls=sum(r['reward_calls'] for r in rows),
        elapsed_seconds=time.time()-start,actual_billing_usd=None,gpu_hours=0,tpu_hours=0,
        limitations=['Finite empirical Poisson solution, not exact terminal tilted diffusion',
                    'Piecewise constant mode reward has exact reweighted prior-mode masses up to negligible nearest-center tail overlap',
                    'Missing-rare experiment deliberately omits that mode at initialization; no mode-creation guarantee',
                    'Numerical weak residual does not imply final moment or mode-mass accuracy'])
    (root/'rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    (root/'preregistered.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (root/'COMPLETE.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
