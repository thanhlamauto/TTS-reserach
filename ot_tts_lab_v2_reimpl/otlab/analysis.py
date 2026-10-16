"""Method-specific pools, bank-first ties, prompt-block paired inference."""
import numpy as np
import torch


def terminal_metrics(source_reward, bank_reward, features=None, baseline=None):
    s, b = source_reward.double(), bank_reward.double()
    joined = torch.cat([b, s])
    ids = torch.argsort(joined, descending=True, stable=True)[:8]
    out = dict(source_mean=float(s.mean()), source_p10=float(torch.quantile(s, .1)),
               pool_best=float(joined.max()), pool_top8_mean=float(joined[ids].mean()),
               top8_source_contribution=int((ids >= len(b)).sum()), top8_indices=ids.tolist())
    if baseline is not None:
        ref = terminal_metrics(baseline, bank_reward)
        for key in ["source_mean", "source_p10", "pool_best", "pool_top8_mean"]:
            out[key + "_gain"] = out[key] - ref[key]
    if features is not None:
        f = torch.nn.functional.normalize(features.double(), dim=1)
        mask = ~torch.eye(len(f), dtype=torch.bool, device=f.device)
        out["source_dino_cosine_diversity"] = float((1 - f @ f.T)[mask].mean()) if mask.any() else 0.
    return out


def prompt_bootstrap(rows, method, control, field, count=20000, seed=20261016):
    # First pair by the full conditional unit, then average within prompt.
    units = {}
    for row in rows:
        key = tuple(row[k] for k in ["prompt_id", "bank_id", "prefix_id", "replicate"])
        if row["method"] in (method, control):
            bucket = units.setdefault(key, {})
            if row["method"] in bucket:
                raise ValueError("duplicate analysis unit")
            bucket[row["method"]] = row[field]
    by_prompt = {}
    for key, unit in units.items():
        if set(unit) != {method, control}:
            raise ValueError("unpaired analysis unit")
        by_prompt.setdefault(key[0], []).append(unit[method] - unit[control])
    if not by_prompt:
        raise ValueError("no paired prompts")
    x = np.array([np.mean(by_prompt[k]) for k in sorted(by_prompt)])
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), size=(count, len(x)))].mean(1)
    return dict(delta=float(x.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                prompts=len(x), conditional_units=len(units),
                inference="exploratory prompt-block bootstrap; unadjusted",
                one_prompt_is_quality_evidence=False if len(x) == 1 else None)
