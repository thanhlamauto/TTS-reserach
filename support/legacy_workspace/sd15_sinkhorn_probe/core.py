import hashlib
import numpy as np
import torch

METHODS=['none','A','mean_matched','minus_A','shuffle_matched','random_matched']


def rms(x):
    return x.float().flatten(1).square().mean(1).sqrt()


def conditional(p):
    return p.double()/p.double().sum(1,keepdim=True)


def bary(p,z):
    return (conditional(p)@z.double().flatten(1)).reshape(-1,*z.shape[1:])


def plan_metrics(p,a,w):
    p,a,w=p.double(),a.double(),w.double()
    rows=conditional(p); positive=w>0
    kl=(rows[:,positive]*(rows[:,positive].clamp_min(1e-300).log()-w[positive].log())).sum(1)
    return dict(row_mass_error_max=float((p.sum(1)-a).abs().max()),
                col_mass_error_max=float((p.sum(0)-w).abs().max()),
                row_dependency_kl=float((a*kl).sum()))


def field_metrics(v):
    v=v.double(); common=v.mean(0,keepdim=True)
    total=float(v.square().mean()); common_energy=float(common.square().mean())
    return dict(rms=total**.5,common_energy_fraction=common_energy/total if total else 0.,
                specific_energy_fraction=float((v-common).square().mean())/total if total else 0.)


def diagnostics(d):
    c=d['C'].double(); interaction=c-c.mean(0,keepdim=True)-c.mean(1,keepdim=True)+c.mean()
    x=d['X'].float(); actual=d['X_after'].float()-x
    proposal=d['delta_postclip_fp32'].float(); mask=proposal!=0
    identity=((d['w'].double()-d['u'].double())[:,None]*d['Z'].double().flatten(1)).sum(0)
    raw=bary(d['P_target'],d['Z'])-bary(d['P_self'],d['Z'])
    return dict(prompt_index=d['meta']['prompt_index'],epsilon=d['epsilon'],
        target_plan=plan_metrics(d['P_target'],d['a'],d['w']),
        self_plan=plan_metrics(d['P_self'],d['a'],d['u']),
        interaction_std=float(interaction.std(unbiased=False)),
        interaction_std_over_epsilon=float(interaction.std(unbiased=False)/d['epsilon']),
        cost_mean=float(c.mean()),cost_std=float(c.std(unbiased=False)),
        raw_field=field_metrics(raw),used_field=field_metrics(d['direction_used']),
        actual_field=field_metrics(actual),target_ess=float(1/d['w'].double().square().sum()),
        barycentric_mean_identity_rms_error=float((raw.flatten(1).mean(0)-identity).square().mean().sqrt()),
        clip_fraction=float((rms(d['delta_preclip'])>torch.as_tensor(d['clip_rms'])+1e-7).float().mean()),
        actual_update_rms=float(actual.square().mean().sqrt()),
        actual_update_over_latent_rms=float(actual.square().mean().sqrt()/x.square().mean().sqrt()),
        actual_update_rms_by_source=rms(actual).tolist(),
        cast_nonzero_proposal_lost_fraction=float(((actual==0)&mask).sum()/mask.sum()) if mask.any() else 0.,
        cast_relative_l2_error=float((actual-proposal).norm()/proposal.norm()) if proposal.norm() else 0.)


def solve64(c,a,w,epsilon,tol=1e-12,max_iter=10000):
    c,a,w=c.double(),a.double(),w.double()
    a=a/a.sum(); w=w/w.sum(); active=w>0
    aa,ww=a,w[active]; cc=c[:,active]
    la,lw=aa.log(),ww.log(); lk=la[:,None]+lw[None,:]-cc/epsilon
    u,v=torch.zeros_like(aa),torch.zeros_like(ww)
    for i in range(max_iter):
        u=la-torch.logsumexp(lk+v[None,:],1)
        v=lw-torch.logsumexp(lk+u[:,None],0)
        if i%10==9:
            p=(lk+u[:,None]+v[None,:]).exp()
            err=max(float((p.sum(1)-aa).abs().max()),float((p.sum(0)-ww).abs().max()))
            if err<tol: break
    else: raise RuntimeError('float64 shuffled control solver did not converge')
    result=torch.zeros_like(c); result[:,active]=p
    return result,dict(iterations=i+1,marginal_error=err,tolerance=tol)


def matched_latent(x,direction,target_rms):
    direction=direction.float(); norm=rms(direction)
    if ((norm<1e-12)&(target_rms>1e-10)).any():
        raise RuntimeError('zero control direction cannot match nonzero A')
    delta=direction*(target_rms/norm.clamp_min(1e-12))[:,None,None,None]
    y=(x.float()+delta).to(x.dtype)
    actual=rms(y.float()-x.float())
    ratio=torch.where(target_rms>1e-10,actual/target_rms.clamp_min(1e-12),torch.ones_like(actual))
    return y,ratio.tolist()


def controls(d,seed):
    assert d['meta']['operator_variant']=='raw_A'
    x=d['X']; actual=d['X_after'].float()-x.float(); target=rms(actual)
    if float(target.max())<1e-10: raise RuntimeError('A is a no-op; diagnose before running controls')
    g=torch.Generator().manual_seed(seed)
    permutation=torch.randperm(len(d['w']),generator=g); shuffled=d['w'][permutation]
    p,info=solve64(d['C'],d['a'],shuffled,d['epsilon'])
    pself,info_self=solve64(d['C'],d['a'],d['u'],d['epsilon'])
    mean=((d['w'].double()-d['u'].double())[:,None]*d['Z'].double().flatten(1)).sum(0)
    mean=mean.reshape(1,*x.shape[1:]).repeat(len(x),1,1,1)
    directions=dict(mean_matched=mean,minus_A=-actual,
                    shuffle_matched=bary(p,d['Z'])-bary(pself,d['Z']),
                    random_matched=torch.randn(x.shape,generator=g))
    latents={'none':x.clone(),'A':d['X_after'].clone()}; ratios={}; cosines={}
    for name,direction in directions.items():
        latents[name],ratios[name]=matched_latent(x,direction,target)
        v=(latents[name].float()-x.float()).flatten(1); aa=actual.flatten(1)
        cosines[name]=((v*aa).sum(1)/(v.norm(dim=1)*aa.norm(dim=1)).clamp_min(1e-12)).tolist()
    max_error=max(abs(v-1) for values in ratios.values() for v in values)
    if max_error>.02: raise RuntimeError(f'post-cast control RMS mismatch exceeds 2%: {max_error}')
    return dict(latents=latents,rms_ratios=ratios,actual_cosines_with_A=cosines,max_rms_relative_error=max_error,
                seed=seed,permutation=permutation.tolist(),shuffled_weights=shuffled.tolist(),
                shuffled_solver=info,self_solver64=info_self)


def noise_seed(root,prompt,replicate,particle,step):
    return int(np.random.SeedSequence([root,prompt,replicate,particle,step]).generate_state(1,dtype=np.uint64)[0])%(2**63-1)


def noise_tape(root,prompt,replicate,n=8,start=50,steps=100,shape=(4,64,64)):
    return torch.stack([torch.stack([torch.randn(shape,generator=torch.Generator().manual_seed(
        noise_seed(root,prompt,replicate,i,k)),dtype=torch.float32) for i in range(n)]) for k in range(start,steps)])


def tensor_sha(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
