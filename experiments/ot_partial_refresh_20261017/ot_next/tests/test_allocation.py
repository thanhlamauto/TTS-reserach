import unittest
import numpy as np
from ot_next.allocation import allocate, semi_relaxed, softmax, residual_resample, reward_logits, grouping, sibling_pairs
from ot_next.identity import compatible_scheduler_metadata


class AllocationTests(unittest.TestCase):
    def test_only_ignored_default_metadata_may_reorder(self):
        old = dict(beta_start=.00085, _use_default_values=['a','b'])
        same = dict(beta_start=.00085, _use_default_values=['b','a'])
        changed = dict(beta_start=.00086, _use_default_values=['b','a'])
        wrong_default = dict(beta_start=.00085, _use_default_values=['b','c'])
        self.assertTrue(compatible_scheduler_metadata(old,same))
        self.assertFalse(compatible_scheduler_metadata(old,changed))
        self.assertFalse(compatible_scheduler_metadata(old,wrong_default))
    def test_gamma_zero_and_row_constraint(self):
        alpha = np.array([.2, .8])
        d = np.array([[1, 9, 2], [2, 5, 1.]])
        logits = np.array([1., -1., .2])
        p, w = semi_relaxed(alpha, d, logits, gamma=0)
        np.testing.assert_allclose(p.sum(axis=1), alpha)
        np.testing.assert_allclose(w, softmax(logits), atol=1e-14)

    def test_free_columns_and_row_offset(self):
        d = np.array([[1., 2., 3.], [0., 4., 8.]])
        alpha = [.4, .6]
        p, w = semi_relaxed(alpha, d, [0, 0, 0])
        q, same = semi_relaxed(alpha, d + np.array([[7], [-2]]), [0, 0, 0])
        _, changed = semi_relaxed(alpha, d + [3, 0, 0], [0, 0, 0])
        np.testing.assert_allclose(p, q, atol=1e-15)
        self.assertGreater(np.max(abs(w - changed)), .1)
        self.assertFalse(np.allclose(w, [1/3]*3))

    def test_zero_mass_and_extreme_logits(self):
        p, w = semi_relaxed([0, 1], [[0, 1], [1, 0]], [1e6, -1e6])
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_array_equal(p[0], [0, 0])
        self.assertEqual(w.sum(), 1.)

    def test_resampling_counts_and_monte_carlo(self):
        rng = np.random.default_rng(42)
        mean = np.zeros(3)
        w = np.array([.57, .3, .13])
        for _ in range(6000):
            parents, counts = residual_resample(w, 6, rng.random(6))
            self.assertEqual(len(parents), 6)
            self.assertEqual(counts.sum(), 6)
            mean += counts / 6000
        np.testing.assert_allclose(mean, 6*w, atol=.025)

    def test_near_constant_reward_and_tied_ess(self):
        logits, audit = reward_logits(np.zeros(12))
        self.assertTrue(np.all(logits == 0))
        self.assertEqual(audit['reason'], 'near_constant')
        _, audit = reward_logits(np.linspace(-1, 1, 12))
        self.assertAlmostEqual(audit['achieved_ess'], 6.)
        self.assertLessEqual(reward_logits([1.]*9+[0.]*3)[1]['inverse_temperature'], 64)

    def test_identical_features_merge_to_one(self):
        f = np.ones((24, 4))
        g = grouping(f, np.arange(12), np.arange(18), .1)
        self.assertEqual(len(g['medoids']), 1)
        self.assertEqual(g['demand'].tolist(), [6.])

    def test_common_masks_and_donor_state_identity(self):
        rng = np.random.default_rng(20)
        f = rng.normal(size=(24, 8))
        common, methods = allocate(np.arange(24.), f, .1, uniforms=rng.random(6))
        for name, d in methods.items():
            np.testing.assert_array_equal(np.array(d['parent_ids'])[common['survivors']], common['survivors'])
            if name != 'none':
                self.assertTrue(set(np.array(d['parent_ids'])[common['replaced']]) <= set(common['donors']))
                self.assertEqual(sum(d['offspring_counts']), 6)
        self.assertEqual(methods['none']['parent_ids'], list(range(24)))

    def test_antithetic_pairing_precedes_noise(self):
        p = [0, 0, 0, 1, 1, 1]
        self.assertEqual(sibling_pairs(p, list(range(6))), [(0, 1), (3, 4)])


if __name__ == '__main__':
    unittest.main()
