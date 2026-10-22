import unittest
import torch
from sd15_sinkhorn_tpu.steering import Config,intervene,targets
from sd15_sinkhorn_tpu.common import initial_noise,event_seed


class TestSteering(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        g=torch.Generator().manual_seed(4)
        self.x=torch.randn(4,4,64,64,generator=g)
        self.z=torch.randn(8,4,64,64,generator=g)
        self.r=torch.arange(8,dtype=torch.float32)

    def test_zero_beta(self):
        y,d=intervene(self.x,self.z,self.r,Config(beta=0),42)
        self.assertTrue(torch.equal(y,self.x)); self.assertEqual(d['update_rms'],0)

    def test_shapes_finite(self):
        for method in ('A','B','C','nearest'):
            for mode in ('soft','top','contrast'):
                for cost in ('raw','pooled'):
                    y,d=intervene(self.x,self.z,self.r,Config(method=method,mode=mode,cost=cost,gamma_neg=.2),4)
                    self.assertEqual(y.shape,self.x.shape); self.assertTrue(torch.isfinite(y).all())

    def test_trust(self):
        y,d=intervene(self.x,self.z,self.r,Config(method='B',eta=10,trust_region=.01),4)
        bound=d['latent_scale']*.01
        self.assertLessEqual(float((y-self.x).square().flatten(1).mean(1).sqrt().max()),bound+1e-6)
        self.assertEqual(d['clipping_fraction'],1.)

    def test_split(self):
        w,neg,good,bad=targets(self.r,'contrast',1,.25)
        self.assertEqual(good,[6,7]); self.assertEqual(bad,[0,1]); self.assertFalse(set(good)&set(bad))
        self.assertAlmostEqual(float(w.sum()),1); self.assertAlmostEqual(float(neg.sum()),1)

    def test_ancestor_reproducible(self):
        cfg=Config(method='C',jitter=.001)
        a,da=intervene(self.x,self.z,self.r,cfg,123)
        b,db=intervene(self.x,self.z,self.r,cfg,123)
        self.assertTrue(torch.equal(a,b)); self.assertEqual(da['ancestors'],db['ancestors'])

    def test_odd_bank_split(self):
        w,neg,good,bad=targets(torch.arange(3,dtype=torch.float32),'contrast',1,.5)
        self.assertEqual(good,[2]); self.assertEqual(bad,[0])
        self.assertFalse(set(good)&set(bad))

    def test_seeds_and_rewards(self):
        a,sa=initial_noise(42,3,0,4); b,sb=initial_noise(42,3,0,8); z,sz=initial_noise(42,3,1,4)
        self.assertTrue(torch.equal(a,b[:4])); self.assertEqual(sa,sb); self.assertNotEqual(sa,sz)
        self.assertFalse(torch.equal(a,z)); self.assertEqual(event_seed(42,3,15),event_seed(42,3,15))
        p=torch.tensor([4,2,7,0,1,3,5,6])
        w,*_=targets(self.r,'soft',1,.25); v,*_=targets(self.r[p],'soft',1,.25)
        self.assertTrue(torch.allclose(w[p],v))
