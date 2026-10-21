"""Per-parent empirical covariance; never normalize a weak direction to its cap."""
import torch

def rms(x):
    return float(x.double().square().mean().sqrt())

def covariance(z, badness):
    z, b = z.double(), badness.double()
    return -((z-z.mean(0))*(b-b.mean()).reshape(-1,1,1,1)).sum(0)/(len(z)-1)

def direction(x, z, rewards, base_reward, cfg):
    b=(cfg['quality_threshold']-rewards.double()).clamp_min(0)
    v=covariance(z,b); va=covariance(z[:4],b[:4]); vb=covariance(z[4:],b[4:])
    cosine=float((va*vb).sum()/(va.norm()*vb.norm()).clamp_min(1e-30))
    r_q=float(torch.tensor([rms(s.float()-x.float()) for s in z]).double().quantile(.5))
    eligible=base_reward<cfg['quality_threshold']
    active=eligible and float(b.std())>cfg['badness_std_min'] and cosine>cfg['split_cosine_min'] and rms(v)>1e-12
    return v.float() if active else torch.zeros_like(x,dtype=torch.float32),dict(
        badness=b.tolist(),base_badness=max(0.,cfg['quality_threshold']-base_reward),
        r_Q=r_q,badness_std=float(b.std()),split_cosine=cosine,
        ungated_covariance_rms=rms(v),active=bool(active),parent_below_quality=bool(eligible),
        flat_badness=bool(float(b.std())<=cfg['badness_std_min']))

def capped_state(x, v, cap, eta):
    update=v.float()*eta
    update*=min(1.,cap/max(rms(update),1e-30))
    # Check the REAL BF16 state too, rather than only the requested FP32 update.
    for _ in range(12):
        y=(x.float()+update).to(torch.bfloat16); actual=rms(y.float()-x.float())
        if actual<=cap+1e-10:return y,dict(requested_rms=rms(update),actual_rms=actual,cap=cap)
        update*=.99*cap/actual
    raise RuntimeError('BF16 trust-bound guard failed')

def matched_random(x,z,target,generator):
    if target==0:return x.clone(),dict(actual_rms=0.,relative_match_error=0.)
    centered=z.float()-z.float().mean(0)
    coefficients=torch.randn(len(z),generator=generator)
    v=(centered*coefficients[:,None,None,None]).sum(0);v/=max(rms(v),1e-30)
    lo,hi=0.,target*2+1e-6;best=None
    for _ in range(50):
        scale=(lo+hi)/2;y=(x.float()+scale*v).to(torch.bfloat16);actual=rms(y.float()-x.float())
        if actual<=target and (best is None or target-actual<best[0]):best=(target-actual,y,actual)
        if actual>target:hi=scale
        else:lo=scale
    assert best is not None and (target-best[2])/target<.02,'Actual random displacement not matched'
    return best[1],dict(actual_rms=best[2],relative_match_error=(target-best[2])/target,
                       geometry='independent isotropic coefficients in centered local ensemble span')

def tests():
    g=torch.Generator().manual_seed(9876);x=torch.randn((4,8,8),generator=g).to(torch.bfloat16)
    z=x.float()+.2*torch.randn((8,4,8,8),generator=g)
    a=torch.randn(x.shape,generator=g);b=(z*a).flatten(1).sum(1)
    v=covariance(z,b)
    assert float((v*a).sum())<0,'Covariance sign is wrong for linear badness'
    assert rms(covariance(z,torch.zeros(8)))==0
    y,log=capped_state(x,v.float(),.05,1.)
    assert log['actual_rms']<=.05 and log['actual_rms']>0
    rand,receipt=matched_random(x,z,log['actual_rms'],g)
    assert receipt['relative_match_error']<.02
    tiny,tl=capped_state(x,torch.ones_like(x).float()*1e-12,.1,1.)
    assert torch.equal(x,tiny),'Tiny covariance must not be normalized into a forced step'
    cfg=dict(quality_threshold=0.,badness_std_min=1e-6,split_cosine_min=0.)
    zero,info=direction(x,z,torch.ones(8),1.,cfg)
    assert not info['active'] and rms(zero)==0
    return dict(passed=True,negative_covariance_sign=True,flat_badness_noop=True,
                good_parent_noop=True,actual_bf16_cap=True,matched_random=True,weak_direction_not_amplified=True)
