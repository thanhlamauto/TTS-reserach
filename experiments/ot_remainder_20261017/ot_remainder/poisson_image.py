"""Single fixed-proxy weak-Poisson correction in an orthonormal low-pass subspace."""
import numpy as np
import torch

def project(x):
    # 4 channels x 4 x 4 nonoverlapping 16x16 DC blocks. Rows have norm one.
    return x.reshape(len(x),4,4,16,4,16).sum(axis=(3,5)).reshape(len(x),64)/16.

def transpose(y):
    return np.repeat(np.repeat(y.reshape(len(y),4,4,4)/16.,16,axis=2),16,axis=3)

def basis(y,centers,h,linear=False):
    n,d=y.shape;phi=y.copy();grad=np.broadcast_to(np.eye(d),(n,d,d)).copy()
    if linear:return phi,grad
    z=y[:,None,:]-centers[None,:,:];k=np.exp(-np.sum(z*z,axis=-1)/(2*h*h))
    return np.concatenate([phi,k],1),np.concatenate([grad,-z*k[:,:,None]/(h*h)],1)

def gates():
    rng=np.random.default_rng(7251);x=rng.normal(size=(3,4,64,64));y=rng.normal(size=(3,64))
    adjoint=abs(float((project(x)*y).sum()-(x*transpose(y)).sum()))
    orth=float(np.max(np.abs(project(transpose(y))-y)))
    z=rng.normal(size=(3,64));c=z[:2];phi,g=basis(z,c,8.);h=1e-5
    e=np.zeros_like(z);e[:,7]=h
    fd=(basis(z+e,c,8.)[0]-basis(z-e,c,8.)[0])/(2*h)
    err=float(np.max(np.abs(fd-g[:,:,7])))
    assert adjoint<1e-10 and orth<1e-12 and err<1e-8
    return dict(adjoint_error=adjoint,projection_orthogonality_error=orth,basis_fd_error=err,passed=True)

def edit(state,rewards,method,bound,seed):
    x=state.float().numpy().astype(np.float64);y=project(x);n=len(x)
    centers=y[np.linspace(0,n-1,min(16,n)).astype(int)].copy()
    dist=np.sqrt(np.sum((y[:,None,:]-y[None,:,:])**2,axis=-1));positive=dist[dist>0]
    bandwidth=max(1e-6,float(np.median(positive)))
    r=np.asarray(rewards,dtype=np.float64).copy()
    if method=='shuffled':r=r[np.random.default_rng(seed).permutation(n)]
    phi,grad=basis(y,centers,bandwidth,method=='linear')
    G=np.einsum('nld,nmd->lm',grad,grad)/n
    b=np.mean((r-r.mean())[:,None]*phi,axis=0)
    a=np.linalg.solve(G+.001*np.eye(len(b)),b)
    v=np.einsum('l,nld->nd',a,grad);delta=.5*transpose(v)
    rms=np.sqrt(np.mean(delta*delta,axis=(1,2,3)));scale=np.minimum(1.,bound/np.maximum(rms,1e-30))
    # Quantization may violate a float64 clipping bound. Bisect only violators;
    # no equal-length normalization of particles or additional guidance step.
    out=torch.tensor(x+delta*scale[:,None,None,None]).to(torch.bfloat16)
    actual=(out.float().numpy().astype(np.float64)-x)
    ar=np.sqrt(np.mean(actual*actual,axis=(1,2,3)))
    bad=ar>bound+1e-12
    lo=np.zeros(n);hi=scale.copy()
    for _ in range(30):
        mid=(lo+hi)/2
        trial=torch.tensor(x+delta*mid[:,None,None,None]).to(torch.bfloat16)
        tr=np.sqrt(np.mean((trial.float().numpy()-x)**2,axis=(1,2,3)))
        lo=np.where(tr<=bound,mid,lo);hi=np.where(tr>bound,mid,hi)
    scale=np.where(bad,lo,scale)
    out=torch.tensor(x+delta*scale[:,None,None,None]).to(torch.bfloat16)
    actual=out.float().numpy().astype(np.float64)-x
    ar=np.sqrt(np.mean(actual*actual,axis=(1,2,3)));assert float(ar.max())<=bound+1e-12
    achieved=project(actual)/.5
    weak=np.einsum('nld,nd->l',grad,achieved)/n
    target_norm=np.linalg.norm(b)+1e-12
    log=dict(method=method,bound=bound,bandwidth=bandwidth,ridge=.001,tilt=.5,
        proxy_fixed=True,reward_gradient=False,raw_velocity_rms=rms.tolist(),actual_state_rms=ar.tolist(),
        active_particles=int(np.sum(ar>0)),clip_scales=scale.tolist(),
        relative_weak_residual_before_clip=float(np.linalg.norm(G@a-b)/target_norm),
        relative_weak_residual_after_clip_quantization=float(np.linalg.norm(weak-b)/target_norm),
        ridge_condition=float(np.linalg.cond(G+.001*np.eye(len(b)))),basis_count=len(b),
        note='Empirical fixed proxy tilt; clipping/quantization changes the weak equation; no exact terminal-tilt claim')
    return out,log
