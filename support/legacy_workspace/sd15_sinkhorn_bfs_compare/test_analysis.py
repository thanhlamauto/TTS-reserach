import math
from pathlib import Path
import tempfile
import unittest
from .analyze import recycled,bootstrap,image_hash


class TestAnalysis(unittest.TestCase):
    def test_recycle_deduplicates_and_selects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; guided=root/'guided'
            source.mkdir(); guided.mkdir()
            (source/'000.png').write_bytes(b'source0')
            (source/'001.png').write_bytes(b'source1')
            (guided/'000.png').write_bytes(b'source0')
            (guided/'001.png').write_bytes(b'new_guided')
            v=1/math.sqrt(2)
            a=dict(rewards=[1.,3.],clip_alignment=[.1,.3],clip_embeddings=[[1.,0.],[0.,1.]])
            b=dict(rewards=[1.,4.],clip_alignment=[.1,.4],clip_embeddings=[[1.,0.],[v,v]])
            r=recycled([('source',a,source),('guided',b,guided)],n=2)
            self.assertEqual(r['unique_candidates'],3)
            self.assertEqual(r['selected_guided_images'],1)
            self.assertEqual(r['mean_reward'],3.5)
            self.assertEqual(r['best_reward'],4.)
            self.assertAlmostEqual(r['clip_diversity'],1-v)
        image_hash.cache_clear()

    def test_constant_paired_bootstrap(self):
        lo,hi=bootstrap([.125]*24)
        self.assertEqual(lo,.125); self.assertEqual(hi,.125)


if __name__=='__main__': unittest.main()
