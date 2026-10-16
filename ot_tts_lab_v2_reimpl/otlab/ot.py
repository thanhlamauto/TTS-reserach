"""Full KL-regularized transport with an envelope derivative and weighted SW2."""
import torch


def masses(w, n, device):
    if w is None:
        w = torch.full((n,), 1 / n, dtype=torch.float64, device=device)
    w = torch.as_tensor(w, dtype=torch.float64, device=device).detach()
    if w.shape != (n,) or not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("invalid masses")
    return w / w.sum()


def quadratic(x, y):
    return (x[:, None, :] - y[None, :, :]).square().sum(-1)


@torch.no_grad()
def solve(cost, a=None, b=None, epsilon=0.1, tol=1e-10, max_iter=10000):
    if cost.ndim != 2 or not torch.isfinite(cost).all() or epsilon <= 0 or max_iter < 10 or tol <= 0:
        raise ValueError("invalid transport problem")
    c = cost.detach().double()
    a, b = masses(a, len(c), c.device), masses(b, c.shape[1], c.device)
    ia, ib = a > 0, b > 0
    aa, bb = a[ia], b[ib]
    cc = c[ia][:, ib]
    loga, logb = aa.log(), bb.log()
    kernel = loga[:, None] + logb[None, :] - cc / float(epsilon)
    u, v = torch.zeros_like(aa), torch.zeros_like(bb)
    for it in range(max_iter):
        u = loga - torch.logsumexp(kernel + v[None, :], dim=1)
        v = logb - torch.logsumexp(kernel + u[:, None], dim=0)
        if it % 10 == 9:
            logp = kernel + u[:, None] + v[None, :]
            p = logp.exp()
            err = max(float((p.sum(1) - aa).abs().max()), float((p.sum(0) - bb).abs().max()))
            if err <= tol:
                break
    else:
        raise RuntimeError(f"Sinkhorn failed to converge, marginal error={err}")
    full = torch.zeros_like(c)
    full[ia[:, None] & ib[None, :]] = p.reshape(-1)
    # KL(P || a tensor b), with equal total mass; the -P+ab terms cancel.
    kl = (p * (logp - loga[:, None] - logb[None, :])).sum()
    value = (p * cc).sum() + float(epsilon) * kl
    return full, value, dict(iterations=it + 1, marginal_error=err, epsilon=float(epsilon))


def regularized_ot(cost, a=None, b=None, epsilon=0.1):
    p, value, _ = solve(cost, a, b, epsilon)
    # Exact forward value; plan held constant only for the envelope derivative.
    correction = (cost.double() * p).sum()
    return value + correction - correction.detach()


def sinkhorn_divergence(x, y, a=None, b=None, epsilon=0.1):
    cross = regularized_ot(quadratic(x, y), a, b, epsilon)
    xx = regularized_ot(quadratic(x, x), a, a, epsilon)
    yy = regularized_ot(quadratic(y, y), b, b, epsilon)
    return cross - 0.5 * (xx + yy)


def epsilon_scale(x, y, kappa=1.0):
    c = quadratic(x.detach().double(), y.detach().double())
    centered = c - c.mean(0, keepdim=True) - c.mean(1, keepdim=True) + c.mean()
    # Explicit numerical floor handles coincident features without division by zero.
    return float(kappa * centered.std(unbiased=False).clamp_min(1e-6))


def weighted_w2_1d(x, y, a=None, b=None):
    if x.ndim != 1 or y.ndim != 1 or not len(x) or not len(y):
        raise ValueError("1D atoms required")
    a, b = masses(a, len(x), x.device), masses(b, len(y), y.device)
    xs, ix = torch.sort(x)
    ys, iy = torch.sort(y)
    a, b = a[ix], b[iy]
    ca, cb = a.cumsum(0), b.cumsum(0)
    la, lb = ca - a, cb - b
    overlap = (torch.minimum(ca[:, None], cb[None, :]) -
               torch.maximum(la[:, None], lb[None, :])).clamp_min(0)
    return (overlap * (xs[:, None] - ys[None, :]).square()).sum()


def projections(dim, count=32, seed=20261016):
    g = torch.Generator().manual_seed(seed)
    directions = torch.randn(count, dim, generator=g, dtype=torch.float64)
    return torch.nn.functional.normalize(directions, dim=1)


def sliced_w2(x, y, a=None, b=None, directions=None):
    if directions is None:
        raise ValueError("fixed projection directions must be supplied")
    d = directions.to(device=x.device, dtype=x.dtype)
    return torch.stack([weighted_w2_1d(x @ v, y @ v, a, b) for v in d]).mean()
