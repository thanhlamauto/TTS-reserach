"""Two-pass feature VJP and projected clean-latent optimization, no UNet graph."""
import torch

from .geometry import block_down, block_up, clip_rms, trusted_state
from .ot import epsilon_scale, projections, sinkhorn_divergence, sliced_w2
from .target import cluster, target_weights


def feature_vjp(encode, clean, objective, microbatch=1):
    with torch.no_grad():
        feature = torch.cat([encode(clean[i:i + microbatch]) for i in range(0, len(clean), microbatch)])
    feature = feature.detach().requires_grad_(True)
    loss = objective(feature)
    cotangent = torch.autograd.grad(loss, feature)[0].detach()
    pieces = []
    for i in range(0, len(clean), microbatch):
        leaf = clean[i:i + microbatch].detach().requires_grad_(True)
        replay = encode(leaf)
        # Exact replay is required: stochastic encoders invalidate this VJP.
        if not torch.allclose(replay.detach(), feature[i:i + microbatch].detach(), atol=1e-6, rtol=1e-5):
            raise RuntimeError("feature recomputation is not deterministic")
        pieces.append(torch.autograd.grad(replay, leaf, grad_outputs=cotangent[i:i + microbatch])[0])
    gradient = torch.cat(pieces)
    if not torch.isfinite(gradient).all() or not torch.isfinite(loss):
        raise RuntimeError("nonfinite loss/VJP")
    return gradient, float(loss.detach())


def proximal_edit(cp, encode, loss="sinkhorn", bound=0.025, factor=4, steps=2,
                  kappa=1.0, slices=32, projection_seed=20261016, shuffle=False,
                  target_beta=1.0, balance=0.5, microbatch=1):
    if steps < 1 or loss not in {"sinkhorn", "sliced", "mean"} or bound <= 0:
        raise ValueError("invalid proximal configuration")
    x, u0, bank = cp["source_xt"], cp["source_clean"].detach(), cp["bank_clean"].detach()
    with torch.no_grad():
        target = torch.cat([encode(bank[i:i + microbatch]) for i in range(0, len(bank), microbatch)]).detach()
        source = torch.cat([encode(u0[i:i + microbatch]) for i in range(0, len(u0), microbatch)]).detach()
    weights, good = target_weights(cp["bank_reward"], cluster(target), beta=target_beta, balance=balance)
    permutation = torch.arange(len(weights))
    if shuffle:
        permutation = torch.randperm(len(weights), generator=torch.Generator().manual_seed(projection_seed + 1))
        weights = weights[permutation]
    epsilon = epsilon_scale(source, target, kappa)
    dirs = projections(source.shape[-1], slices, projection_seed)

    def objective(features):
        if loss == "mean":
            return (features.mean(0) - (target * weights.to(target.dtype)[:, None]).sum(0)).square().sum()
        if loss == "sliced":
            return sliced_w2(features, target, b=weights, directions=dirs)
        return sinkhorn_divergence(features, target, b=weights, epsilon=epsilon)

    z = torch.zeros_like(block_down(u0, factor))
    clean_radius = bound / float(cp["signal_scale"])
    losses, grad_norms = [], []
    for _ in range(steps):
        clean = u0 + block_up(z, factor)
        grad, value = feature_vjp(encode, clean, objective, microbatch)
        gz = block_down(grad, factor)
        norm = gz.flatten(1).norm(dim=1)
        normalized = gz / norm.clamp_min(1e-20)[:, None, None, None]
        # One fixed radius/steps update in RMS units, then projection. This
        # engineering step size is preregistered, not copied from unavailable code.
        scale = (u0[0].numel() ** 0.5) * clean_radius / steps
        z = z - normalized * scale
        z = block_down(clip_rms(block_up(z, factor), clean_radius), factor).detach()
        losses.append(value)
        grad_norms.append(norm.detach().tolist())
    delta_xt = float(cp["signal_scale"]) * block_up(z, factor)
    entry, trust = trusted_state(x, delta_xt, bound)
    with torch.no_grad():
        final_features = torch.cat([encode((u0 + block_up(z, factor))[i:i + microbatch])
                                    for i in range(0, len(u0), microbatch)])
        final_loss = float(objective(final_features))
    return dict(entry_states=entry, losses=losses + [final_loss], gradient_norms=grad_norms,
                epsilon=epsilon, target_weights=weights.tolist(), target_indices=good.tolist(),
                target_permutation=permutation.tolist(), projection_seed=projection_seed,
                slices=slices, trust=trust)
