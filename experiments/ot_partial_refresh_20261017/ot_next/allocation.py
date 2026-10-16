"""Coverage allocation with a FREE donor marginal; no Sinkhorn iterations."""
import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    y = np.exp(x - x.max(axis=axis, keepdims=True))
    return y / y.sum(axis=axis, keepdims=True)


def semi_relaxed(alpha, distance, logits, gamma=1., epsilon=1., reference=None):
    alpha, distance, logits = map(lambda x: np.asarray(x, dtype=np.float64), (alpha, distance, logits))
    if (distance.shape != (len(alpha), len(logits)) or not np.isfinite(distance).all()
            or not np.isfinite(logits).all() or not np.isfinite(alpha).all()
            or np.any(alpha < 0) or not np.isclose(alpha.sum(), 1) or epsilon <= 0 or gamma < 0):
        raise ValueError('invalid semi-relaxed problem')
    u = np.ones(len(logits)) / len(logits) if reference is None else np.asarray(reference, dtype=float)
    if u.shape != logits.shape or np.any(u <= 0) or not np.isclose(u.sum(), 1):
        raise ValueError('donor reference must be positive and normalized')
    coupling = alpha[:, None] * softmax(np.log(u)[None, :] + (logits[None, :] - gamma * distance) / epsilon)
    return coupling, coupling.sum(axis=0)


def residual_resample(weights, count, uniforms):
    w = np.asarray(weights, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if np.any(w < 0) or not np.isfinite(w).all() or not np.isclose(w.sum(), 1) or count < 1:
        raise ValueError('invalid residual resampling inputs')
    expected = count * w
    counts = np.floor(expected).astype(int)
    residual_count = count - counts.sum()
    if len(uniforms) < residual_count or np.any(uniforms < 0) or np.any(uniforms >= 1):
        raise ValueError('insufficient keyed uniforms')
    if residual_count:
        residual = np.maximum(expected - counts, 0)
        cdf = np.cumsum(residual / residual.sum())
        cdf[-1] = 1.
        draws = np.searchsorted(cdf, uniforms[:residual_count], side='right')
        counts += np.bincount(draws, minlength=len(w))
    # Deterministic donor-order assignment to ascending replaced child slots.
    parents = np.repeat(np.arange(len(w)), counts)
    assert len(parents) == count
    return parents, counts


def ess(weights):
    return float(1 / np.square(weights).sum())


def reward_logits(rewards, target_ess=6., floor=1e-6):
    r = np.asarray(rewards, dtype=np.float64)
    spread = float(r.max() - r.min())
    if spread <= floor:
        return np.zeros_like(r), dict(reason='near_constant', inverse_temperature=0., achieved_ess=float(len(r)))
    z = (r - r.max()) / spread
    # Bounded inverse temperature avoids amplifying almost-tied top donors.
    low, high = 0., 64.
    for _ in range(70):
        mid = (low + high) / 2
        if ess(softmax(mid * z)) > target_ess:
            low = mid
        else:
            high = mid
    inverse = (low + high) / 2
    return inverse * z, dict(reason='target_ess_or_cap', inverse_temperature=inverse,
                            achieved_ess=ess(softmax(inverse * z)), reward_range=spread)


def grouping(features, donor_ids, survivor_ids, merge_distance, max_groups=4):
    f = np.asarray(features, dtype=np.float64)
    f = f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-12)
    if not np.isfinite(f).all() or not 0 < merge_distance <= 2:
        raise ValueError('invalid normalized features/merge distance')
    all_dist = np.maximum(0., 1 - f @ f.T)
    donors = np.asarray(donor_ids, dtype=int)
    medoids = [int(donors[0])]
    while len(medoids) < min(max_groups, len(donors)):
        distances = all_dist[np.ix_(donors, medoids)].min(axis=1)
        j = int(np.argmax(distances))
        if distances[j] <= merge_distance:
            break
        medoids.append(int(donors[j]))
    # PAM-like within-cluster medoid refinement, fixed deterministic tie rule.
    for _ in range(20):
        labels = all_dist[np.ix_(donors, medoids)].argmin(axis=1)
        updated = []
        for g, old in enumerate(medoids):
            members = donors[labels == g]
            updated.append(int(members[np.argmin(all_dist[np.ix_(members, members)].sum(axis=1))]) if len(members) else old)
        if updated == medoids:
            break
        medoids = updated
    # Refinement can make centers too similar: merge explicitly afterwards.
    kept = []
    for medoid in medoids:
        if not kept or all_dist[medoid, kept].min() > merge_distance:
            kept.append(medoid)
    medoids = kept
    labels = all_dist[:, medoids].argmin(axis=1)
    occupancy = np.bincount(labels[survivor_ids], minlength=len(medoids))
    demand = np.maximum(len(f) / len(medoids) - occupancy, 0)
    if demand.sum() <= 0:
        raise RuntimeError('positive replacement slots must yield positive demand')
    raw_distance = all_dist[np.ix_(medoids, donors)]
    positive = all_dist[np.ix_(donors, donors)]
    positive = positive[positive > 1e-12]
    scale = max(float(np.median(positive)) if len(positive) else 1., 1e-6)
    return dict(medoids=medoids, labels=labels, occupancy=occupancy, demand=demand,
                alpha=demand / demand.sum(), normalized_distance=raw_distance / scale,
                distance_scale=scale)


def allocate(rewards, features, merge_distance, gamma=1., uniforms=None):
    rewards = np.asarray(rewards, dtype=float)
    if rewards.shape != (24,) or not np.isfinite(rewards).all():
        raise ValueError('R1 has exactly24 finite reward proxies')
    ranked = np.argsort(-rewards, kind='stable')
    survivors = np.sort(ranked[:18])
    replaced = np.sort(ranked[18:])
    donors = ranked[:12]
    group = grouping(features, donors, survivors, merge_distance)
    logits, calibration = reward_logits(rewards[donors])
    weights = {'partial_reward': softmax(logits)}
    hard = np.zeros(12)
    donor_labels = group['labels'][donors]
    for g, mass in enumerate(group['alpha']):
        members = donor_labels == g
        if not members.any():
            raise RuntimeError('medoid group has no donor')
        hard[members] += mass * softmax(logits[members])
    weights['partial_group'] = hard
    coupling, weights['partial_ot'] = semi_relaxed(group['alpha'], group['normalized_distance'], logits, gamma)
    _, zero = semi_relaxed(group['alpha'], group['normalized_distance'], logits, 0.)
    if not np.allclose(zero, weights['partial_reward'], atol=1e-14, rtol=1e-14):
        raise RuntimeError('gamma0 reward-only identity failed')
    if uniforms is None:
        raise ValueError('pre-keyed common rounding uniforms required')
    result = {}
    for method in ['none', 'partial_reward', 'partial_group', 'partial_ot']:
        parent = np.arange(24)
        counts = np.zeros(12, dtype=int)
        if method != 'none':
            choice, counts = residual_resample(weights[method], 6, uniforms)
            parent[replaced] = donors[choice]
        result[method] = dict(parent_ids=parent.tolist(), offspring_counts=counts.tolist(),
                              donor_weights=weights[method].tolist() if method != 'none' else None,
                              ess=ess(weights[method]) if method != 'none' else None,
                              occupancy_after=np.bincount(group['labels'][parent], minlength=len(group['medoids'])).tolist())
    common = dict(survivors=survivors.tolist(), replaced=replaced.tolist(), donors=donors.tolist(),
                  logits=logits.tolist(), reward_calibration=calibration,
                  groups={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in group.items()},
                  coupling=coupling.tolist(), gamma_zero_recovers_reward=True,
                  rounding_uniforms=np.asarray(uniforms).tolist())
    return common, result


def sibling_pairs(parents, replaced):
    pairs = []
    for donor in sorted(set(parents[i] for i in replaced)):
        children = sorted(i for i in replaced if parents[i] == donor)
        pairs.extend(list(zip(children[::2], children[1::2])))
    return pairs
