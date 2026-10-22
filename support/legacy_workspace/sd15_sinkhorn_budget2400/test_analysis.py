import unittest
import numpy as np
from .analyze import prefix_top8


class BudgetSelectionTests(unittest.TestCase):
    def test_bon_uses_first24_not_best24_of28(self):
        scores=dict(rewards=list(range(24))+[1000]*4,
                    clip_alignment=[.5]*28,
                    clip_embeddings=np.eye(28).tolist())
        r=prefix_top8(scores)
        self.assertEqual(r['best_reward'],23)
        self.assertEqual(r['mean_reward'],19.5)
        self.assertEqual(r['n_images'],8)

    def test_missing_candidates_fail(self):
        with self.assertRaises(AssertionError):
            prefix_top8(dict(rewards=[1]*23,clip_alignment=[1]*23,
                            clip_embeddings=[[1,0]]*23))


if __name__=='__main__': unittest.main()
