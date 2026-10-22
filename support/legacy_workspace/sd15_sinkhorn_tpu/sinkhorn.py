"""Small balanced OT problems, deliberately solved on CPU in float32."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


def checked(x):
    x = torch.as_tensor(x, device='cpu', dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError('NaN/Inf in OT input')
    return x


def features(x, cost='raw'):
    x = checked(x)
    if x.ndim != 4 or x.shape[1:] != (4, 64, 64):
        raise ValueError(f'expected [N,4,64,64], got {tuple(x.shape)}')
    if cost == 'pooled':
        x = F.avg_pool2d(x, 8)
    elif cost != 'raw':
        raise ValueError(cost)
    return x.flatten(1)


def quadratic(x, z):
    # Avoid cancellation in the self diagonals of a Gram matrix.
    return torch.stack([0.5 * (z - row).square().mean(1) for row in x])


def reference_scale(z, cost='raw'):
    f = features(z, cost)
    c = quadratic(f, f)
    off = c[~torch.eye(len(z), dtype=torch.bool)]
    return max(float(off.median()) if len(off) else 1., 1e-8)


@dataclass
class Plan:
    p: torch.Tensor
    objective: float
    row_error: float
    column_error: float
    entropy: float
    iterations: int


def solve(c, a=None, w=None, epsilon=0.1, tol=2e-6, max_iter=2000):
    c = checked(c)
    n, m = c.shape
    a = torch.full((n,), 1/n) if a is None else checked(a)
    w = torch.full((m,), 1/m) if w is None else checked(w)
    if epsilon <= 0 or (a < 0).any() or (w < 0).any():
        raise ValueError('epsilon > 0 and nonnegative weights required')
    if not torch.isclose(a.sum(), torch.tensor(1.)) or not torch.isclose(w.sum(), torch.tensor(1.)):
        raise ValueError('weights must sum to one')
    # Exact zero target masses (top mode) are removed from the dual problem.
    ia, iw = a > 0, w > 0
    aa, ww, cc = a[ia], w[iw], c[ia][:, iw]
    la, lw = aa.log(), ww.log()
    lk = la[:, None] + lw[None, :] - cc / epsilon
    u, v = torch.zeros_like(aa), torch.zeros_like(ww)
    for it in range(1, max_iter+1):
        u = la - torch.logsumexp(lk + v[None, :], 1)
        v = lw - torch.logsumexp(lk + u[:, None], 0)
        if it % 10 == 0:
            pp = (lk + u[:, None] + v[None, :]).exp()
            if max(float((pp.sum(1)-aa).abs().max()), float((pp.sum(0)-ww).abs().max())) < tol:
                break
    lp = lk + u[:, None] + v[None, :]
    pp = lp.exp()
    p = torch.zeros_like(c)
    p[ia.nonzero().flatten()[:, None], iw.nonzero().flatten()[None, :]] = pp
    re, ce = float((p.sum(1)-a).abs().max()), float((p.sum(0)-w).abs().max())
    if not torch.isfinite(p).all() or max(re, ce) > 1e-4:
        raise RuntimeError(f'Sinkhorn failed: row={re}, column={ce}')
    # KL(P || a tensor-product w), with total masses both one.
    kl = (pp * (lp-la[:, None]-lw[None, :])).sum()
    return Plan(p, float((pp*cc).sum()+epsilon*kl), re, ce,
                float(-(pp*lp).sum()), it)


def coupling(x, z, w, epsilon, scale, cost='raw'):
    return solve(quadratic(features(x, cost), features(z, cost))/scale,
                 w=w, epsilon=epsilon)


def barycenter(plan, z):
    rows = plan.p / plan.p.sum(1, keepdim=True)
    return (rows @ checked(z).flatten(1)).reshape(-1, *z.shape[1:])


def divergence(x, z, w, epsilon, scale, cost='raw'):
    fx, fz = features(x, cost), features(z, cost)
    cross = solve(quadratic(fx, fz)/scale, w=w, epsilon=epsilon).objective
    xx = solve(quadratic(fx, fx)/scale, epsilon=epsilon).objective
    zz = solve(quadratic(fz, fz)/scale, a=w, w=w, epsilon=epsilon).objective
    return cross - 0.5*(xx+zz)
