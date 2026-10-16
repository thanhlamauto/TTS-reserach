"""Discrete edit proposals and exact soft-capacity assignment for small N."""
import math
import torch

from .geometry import lowpass, trusted_state
from .target import cluster


@torch.no_grad()
def propose(cp, source_features, bank_features, bound=0.025, factor=4, fraction=0.5):
    x, u, bank, reward = (cp[k] for k in ("source_xt", "source_clean", "bank_clean", "bank_reward"))
    if source_features.shape[0] != len(x) or bank_features.shape[0] != len(bank):
        raise ValueError("feature batch mismatch")
    count = max(1, math.ceil(len(bank) * fraction))
    good = torch.argsort(reward, descending=True, stable=True)[:count]
    groups = cluster(bank_features[good])
    distance = torch.cdist(source_features.double(), bank_features[good].double())
    states = x[:, None].repeat(1, 3, 1, 1, 1)
    active = torch.zeros(len(x), 3, dtype=torch.bool)
    active[:, 0] = True
    reference = torch.full((len(x), 3), -1, dtype=torch.long)
    audits = []
    for i in range(len(x)):
        first = int(distance[i].argmin())
        options = [first]
        eligible = groups != groups[first]
        if eligible.any():
            candidates = torch.where(eligible, distance[i], torch.inf)
            options.append(int(candidates.argmin()))
        for slot, j in enumerate(options, 1):
            edit = lowpass((bank[good[j]] - u[i])[None], factor) * float(cp["signal_scale"])
            state, audit = trusted_state(x[i:i + 1], edit, bound)
            if not torch.equal(state[0], x[i]):
                states[i, slot], active[i, slot], reference[i, slot] = state[0], True, good[j]
            audits.append({"source": i, "slot": slot, **audit})
    return dict(entry_states=states, active=active, reference_indices=reference, trust=audits)


def utilities(proxy, active):
    proxy = proxy.double()
    if proxy.ndim != 2 or proxy.shape != active.shape or not active[:, 0].all():
        raise ValueError("invalid proxy/active shape or absent no-op")
    if not torch.isfinite(proxy[active]).all():
        raise ValueError("nonfinite active proxy reward")
    gain = proxy - proxy[:, :1]
    # One common scale preserves the objective's utility units across sources.
    scale = proxy[active].std(unbiased=False).clamp_min(1e-8)
    feasible = active & (gain >= 0)
    feasible[:, 0] = True
    return gain / scale, feasible


def assignment(utility, labels, feasible, penalty=0.1, mode="ot", capacity=None):
    if mode not in {"ot", "greedy", "independent"} or penalty < 0:
        raise ValueError("invalid allocation configuration")
    n, k = utility.shape
    if labels.shape != utility.shape or feasible.shape != utility.shape or not feasible[:, 0].all():
        raise ValueError("invalid assignment input")
    if not torch.isfinite(utility[feasible]).all() or (labels[feasible] < 0).any():
        raise ValueError("invalid active utility/group")
    group_ids = sorted(set(labels[feasible].tolist()))
    remap = {v: i for i, v in enumerate(group_ids)}
    group_count = len(group_ids)
    cap = math.ceil(n / group_count) if capacity is None else capacity
    if cap < 0:
        raise ValueError("invalid capacity")
    occupancy = lambda counts: sum(max(v - cap, 0) for v in counts)
    if mode == "independent":
        penalty = 0.0
    if mode in {"independent", "greedy"}:
        counts, choices, value = [0] * group_count, [], 0.0
        for i in range(n):
            scores = []
            for j in range(k):
                if not feasible[i, j]:
                    continue
                g = remap[int(labels[i, j])]
                extra = int(counts[g] >= cap)
                scores.append((float(utility[i, j]) - penalty * extra, -j, g))
            best, negj, g = max(scores)
            choices.append(-negj)
            counts[g] += 1
            value += best
        return choices, value
    # Dynamic programming over occupancy states; lexicographic ties favor no-op.
    dp = {(0,) * group_count: (0.0, ())}
    for i in range(n):
        nxt = {}
        for counts, (value, choices) in dp.items():
            for j in range(k):
                if not feasible[i, j]:
                    continue
                updated = list(counts)
                updated[remap[int(labels[i, j])]] += 1
                updated = tuple(updated)
                candidate = (value + float(utility[i, j]), choices + (j,))
                old = nxt.get(updated)
                if old is None or candidate[0] > old[0] or (candidate[0] == old[0] and candidate[1] < old[1]):
                    nxt[updated] = candidate
        dp = nxt
    best = min(((value - penalty * occupancy(counts), choices) for counts, (value, choices) in dp.items()),
               key=lambda p: (-p[0], p[1]))
    return list(best[1]), best[0]


@torch.no_grad()
def select(proxy, features, active, mode="ot", penalty=0.1, max_groups=4):
    u, feasible = utilities(proxy, active)
    labels = torch.full(active.shape, -1, dtype=torch.long)
    labels[active] = cluster(features[active], max_groups)
    choice, value = assignment(u, labels, feasible, penalty, mode)
    return dict(choices=choice, objective=value, labels=labels.tolist(), feasible=feasible.tolist())
