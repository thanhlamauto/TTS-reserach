"""One intervention only; contrastive B/C and pooled updates are heuristics."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
import torch.nn.functional as F
from .sinkhorn import checked, coupling, barycenter, divergence, reference_scale, features, quadratic


@dataclass(frozen=True)
class Config:
    method: str = 'A'
    mode: str = 'soft'
    checkpoint: float = 0.5
    eta: float = 0.1
    epsilon: float = 0.1
    beta: float = 1.
    top_fraction: float = 0.25
    gamma_neg: float = 0.
    trust_region: float = 0.1
    cost: str = 'raw'
    lowpass: int = 1
    jitter: float = 0.

    def record(self):
        return asdict(self)


def targets(rewards, mode, beta, fraction):
    r = checked(rewards)
    if r.ndim != 1 or len(r) < 2 or not 0 < fraction <= 0.5:
        raise ValueError('at least two rewards and top_fraction in (0, .5] required')
    standard = (r-r.mean())/r.std(unbiased=False).clamp_min(1e-6)
    count = min(len(r)//2, max(1, math.ceil(len(r)*fraction)))
    # Stable ranking makes tied rewards reproducible; positive/negative sets disjoint.
    order = torch.argsort(r, stable=True)
    good, bad = order[-count:], order[:count]
    w, neg = torch.zeros_like(r), torch.zeros_like(r)
    neg[bad] = 1/count
    if mode == 'soft':
        w = torch.softmax(beta*standard, 0)
    elif mode in ('top', 'contrast'):
        w[good] = 1/count
    else:
        raise ValueError(mode)
    return w, neg, good.tolist(), bad.tolist()


def intervene(x, z, rewards, cfg, seed):
    x, z = checked(x), checked(z)
    if cfg.method not in ('A','B','C','nearest') or cfg.eta < 0 or cfg.trust_region <= 0 or cfg.jitter < 0:
        raise ValueError('invalid intervention configuration')
    scale = reference_scale(z, cfg.cost)
    w, neg, good, bad = targets(rewards, cfg.mode, cfg.beta, cfg.top_fraction)
    plan = coupling(x, z, w, cfg.epsilon, scale, cfg.cost)
    target = barycenter(plan, z)
    rng = np.random.default_rng(seed)
    ancestors = []
    if cfg.method == 'A':
        self_plan = coupling(x, z, torch.full_like(w, 1/len(w)), cfg.epsilon, scale, cfg.cost)
        direction = target-barycenter(self_plan, z)
    elif cfg.method == 'B':
        direction = target-x
    elif cfg.method == 'C':
        rows = plan.p / plan.p.sum(1, keepdim=True)
        for p in rows.numpy():
            p = p.astype(np.float64); p /= p.sum()
            ancestors.append(int(rng.choice(len(w), p=p)))
        direction = z[ancestors]-x
    else:
        costs = quadratic(features(x, cfg.cost), features(z[good], cfg.cost))
        ancestors = [good[i] for i in costs.argmin(1).tolist()]
        direction = z[ancestors]-x
    if cfg.mode == 'contrast' and cfg.gamma_neg:
        negative = barycenter(coupling(x,z,neg,cfg.epsilon,scale,cfg.cost),z)
        direction += cfg.gamma_neg*(target-negative)
    if cfg.lowpass > 1:
        direction = F.interpolate(F.avg_pool2d(direction,cfg.lowpass),size=(64,64),mode='bilinear',align_corners=False)
    update = cfg.eta*direction
    latent_scale = z.square().mean().sqrt().clamp_min(1e-8)
    if cfg.method == 'C' and cfg.jitter:
        update += torch.from_numpy(rng.standard_normal(x.shape).astype('float32'))*cfg.jitter*latent_scale
    rms = update.square().flatten(1).mean(1).sqrt()
    limit = cfg.trust_region*latent_scale
    factors = (limit/rms.clamp_min(1e-12)).clamp_max(1)
    update *= factors[:,None,None,None]
    y = checked(x+update)
    before = divergence(x,z,w,cfg.epsilon,scale,cfg.cost)
    after = divergence(y,z,w,cfg.epsilon,scale,cfg.cost)
    diag = dict(target_ess=float(1/w.square().sum()),plan_entropy=plan.entropy,
                row_error=plan.row_error,column_error=plan.column_error,iterations=plan.iterations,
                sinkhorn_before=before,sinkhorn_after=after,cost_scale=scale,
                latent_scale=float(latent_scale),update_rms=float(update.square().mean().sqrt()),
                relative_update_rms=float(update.square().mean().sqrt()/latent_scale),
                clipping_fraction=float((factors<1).float().mean()),
                unique_ancestors=len(set(ancestors)),ancestors=ancestors,good=good,bad=bad,
                heuristic_feature_update=cfg.cost=='pooled',heuristic_contrast=cfg.mode=='contrast' and cfg.method!='A')
    return y, diag
