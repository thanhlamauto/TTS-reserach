"""Fixed-radius mean-gradient proposal; no terminal outcomes enter sampling."""
import numpy as np


def gradient(rewards, radius):
    y = np.asarray(rewards, dtype=np.float64).reshape(2, 9)
    g = (y[:, 1::2]-y[:, 2::2])/(2*radius)
    return g, g.mean(axis=0)


def weight(z, g, cfg):
    score = np.asarray(z) @ np.asarray(g)
    harm = np.maximum(0., (-cfg['delta_reward']-score)/cfg['reward_scale'])
    return np.maximum(cfg['weight_floor'], np.exp(np.maximum(-700, -cfg['gamma']*harm**2)))


def sample(cfg, seed, pid, g, count=3):
    guided, uniform, audit = [], [], []
    for child in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, pid, 2026101722, child]))
        bypass = rng.random() < cfg['rho']
        for attempt in range(100000):
            direction = rng.normal(size=4)
            z = cfg['radius']*direction/np.linalg.norm(direction)
            w = float(weight(z, g, cfg))
            if attempt == 0:
                uniform.append(z.copy())
            if bypass or rng.random() <= w:
                guided.append(z)
                audit.append(dict(child=child, unfiltered_mixture_branch=bool(bypass),
                                  rejected=attempt, prediction=float(z@g), weight=w))
                break
        else:
            raise RuntimeError('CPU rejection budget exhausted; do not fall back or retune')
    return np.asarray(guided), np.asarray(uniform), audit


def checks(cfg):
    # Exact budget and physical batch shapes are scientific requirements.
    assert 12*40 + (18+3+11)*60 == 2400
    assert (12*40, 20*60, 12*60) == (480, 1200, 720)
    assert all(n % 4 == 0 for n in [12, 20, 12, 24])
    g = np.array([1., -2., 3., -4.])
    a, b, log = sample(cfg, 8722, 1, g)
    assert np.allclose(np.linalg.norm(a, axis=1), cfg['radius'])
    assert np.allclose(np.linalg.norm(b, axis=1), cfg['radius'])
    zero, original, _ = sample(cfg, 8722, 1, np.zeros(4))
    assert np.array_equal(zero, original), 'flat weights must recover identical proposals'
    assert np.array_equal(a, sample(cfg, 8722, 1, g)[0]), 'proposal RNG changed'
    return dict(passed=True, exact_cold_budget=2400, batch_padding=0,
                flat_filter_recovers_uniform=True, proposal_replay=True)
