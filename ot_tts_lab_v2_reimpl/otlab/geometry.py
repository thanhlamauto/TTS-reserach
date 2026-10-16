"""Orthonormal block coordinates and an actual, post-cast latent trust ball."""
import torch
import torch.nn.functional as F


def rms(x):
    return x.double().flatten(1).square().mean(1).sqrt()


def block_down(x, factor=4):
    if x.ndim != 4 or factor < 1 or x.shape[-1] % factor or x.shape[-2] % factor:
        raise ValueError("block projection requires divisible NCHW dimensions")
    return F.avg_pool2d(x, factor) * factor


def block_up(z, factor=4):
    return z.repeat_interleave(factor, -2).repeat_interleave(factor, -1) / factor


def lowpass(x, factor=4):
    return block_up(block_down(x, factor), factor)


def clip_rms(delta, bound):
    limits = torch.as_tensor(bound, dtype=torch.float64, device=delta.device).expand(len(delta))
    if (limits < 0).any() or not torch.isfinite(limits).all():
        raise ValueError("invalid trust bound")
    scale = (limits / rms(delta).clamp_min(1e-30)).clamp_max(1)
    return delta * scale.to(delta.dtype).reshape(-1, 1, 1, 1)


@torch.no_grad()
def trusted_state(x, delta_xt, bound):
    if x.shape != delta_xt.shape or not torch.isfinite(delta_xt).all():
        raise ValueError("invalid latent edit")
    limits = torch.as_tensor(bound, dtype=torch.float64, device=x.device).expand(len(x))
    delta = clip_rms(delta_xt.float(), limits)
    lo, hi = torch.zeros(len(x), device=x.device), torch.ones(len(x), device=x.device)
    # Rounding can push a nominally feasible FP32 update outside the ball.
    initial = (x.float() + delta).to(x.dtype)
    safe = rms(initial.float() - x.float()) <= limits
    lo[safe] = 1
    for _ in range(24):
        mid = (lo + hi) / 2
        candidate = (x.float() + mid[:, None, None, None] * delta).to(x.dtype)
        okay = rms(candidate.float() - x.float()) <= limits
        lo = torch.where(okay, mid, lo)
        hi = torch.where(okay, hi, mid)
    result = (x.float() + lo[:, None, None, None] * delta).to(x.dtype)
    actual = rms(result.float() - x.float())
    if (actual > limits).any():
        raise RuntimeError("post-cast trust projection failed")
    return result, {"actual_rms": actual.cpu().tolist(), "bound": limits.cpu().tolist(),
                    "cast_scale": lo.cpu().tolist(), "active": (actual > 0).cpu().tolist()}
