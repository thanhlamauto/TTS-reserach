"""Deterministic feature grouping and a quality-weighted positive target."""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def cluster(features, max_groups=4):
    f = F.normalize(features.double(), dim=-1)
    if not len(f) or max_groups < 1:
        raise ValueError("invalid clustering input")
    centers = [0]
    while len(centers) < min(max_groups, len(f)):
        nearest = torch.cdist(f, f[centers]).square().min(1).values
        nearest[centers] = -1
        j = int(nearest.argmax())
        if nearest[j] <= 1e-12:
            break
        centers.append(j)
    c = f[centers]
    for _ in range(20):
        labels = torch.cdist(f, c).argmin(1)
        updated = torch.stack([f[labels == g].mean(0) if (labels == g).any() else c[g]
                               for g in range(len(c))])
        updated = F.normalize(updated, dim=1)
        if torch.equal(updated, c):
            break
        c = updated
    labels = torch.cdist(f, c).argmin(1)
    # Compress empty groups; capacity uses the actual group count.
    return torch.unique(labels, sorted=True, return_inverse=True)[1]


@torch.no_grad()
def target_weights(reward, groups, fraction=0.5, beta=1.0, balance=0.5):
    if reward.ndim != 1 or not torch.isfinite(reward).all() or not 0 < fraction <= 1 or not 0 <= balance <= 1:
        raise ValueError("invalid target configuration")
    k = max(1, math.ceil(len(reward) * fraction))
    ids = torch.argsort(reward, descending=True, stable=True)[:k]
    selected = reward[ids].double()
    # Explicit design choice where the supplied prose does not fix reward units.
    z = (selected - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-8)
    quality = torch.softmax(float(beta) * z, dim=0)
    labels = groups[ids]
    balanced = torch.zeros_like(quality)
    unique = labels.unique()
    for g in unique:
        mask = labels == g
        balanced[mask] = quality[mask] / quality[mask].sum() / len(unique)
    w = torch.zeros(len(reward), dtype=torch.float64, device=reward.device)
    w[ids] = (1 - balance) * quality + balance * balanced
    return w, ids
