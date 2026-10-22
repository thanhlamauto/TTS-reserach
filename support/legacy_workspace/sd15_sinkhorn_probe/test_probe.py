import unittest
import torch
from .core import solve64,bary,field_metrics,plan_metrics,matched_latent,noise_tape
from .analyze import pool_metrics


class ProbeTests(unittest.TestCase):
    def test_additive_cost_has_source_independent_rows(self):
        c=torch.arange(8,dtype=torch.float64)[:,None]+torch.arange(16,dtype=torch.float64)[None,:]/10
        a=torch.ones(8)/8; w=torch.zeros(16); w[:4]=.25
        p,info=solve64(c,a,w,.1)
        self.assertLess(info['marginal_error'],1e-12)
        self.assertLess(abs(plan_metrics(p,a,w)['row_dependency_kl']),1e-12)
        self.assertTrue(torch.all(p[:,4:]==0))

    def test_barycentric_mean_identity_and_energy(self):
        g=torch.Generator().manual_seed(2); c=torch.rand(8,16,generator=g)
        z=torch.randn(16,2,3,3,generator=g); a=torch.ones(8)/8
        w=torch.zeros(16); w[:4]=.25; u=torch.ones(16)/16
        p,_=solve64(c,a,w,.1); q,_=solve64(c,a,u,.1)
        v=bary(p,z)-bary(q,z)
        target=((w-u).double()[:,None]*z.double().flatten(1)).sum(0)
        self.assertTrue(torch.allclose(v.flatten(1).mean(0),target,atol=1e-10,rtol=0))
        energy=field_metrics(v)
        self.assertAlmostEqual(energy['common_energy_fraction']+energy['specific_energy_fraction'],1,places=12)

    def test_norm_matching_after_cast_and_zero_direction(self):
        torch.manual_seed(42); x=torch.randn(8,4,64,64).bfloat16()
        y,ratios=matched_latent(x,torch.randn_like(x.float()),torch.full((8,),.1))
        self.assertEqual(y.dtype,x.dtype)
        self.assertLess(max(abs(r-1) for r in ratios),.02)
        with self.assertRaises(RuntimeError): matched_latent(x,torch.zeros_like(x.float()),torch.ones(8))

    def test_noise_particle_prefix_replicate_independence(self):
        a=noise_tape(3,0,0,n=3,start=50,steps=52,shape=(2,3,3))
        b=noise_tape(3,0,0,n=8,start=50,steps=52,shape=(2,3,3))
        c=noise_tape(3,0,1,n=3,start=50,steps=52,shape=(2,3,3))
        self.assertTrue(torch.equal(a,b[:,:3])); self.assertFalse(torch.equal(a,c))

    def test_pool_ties_favor_bank_and_never_merge_other_methods(self):
        bank=dict(rewards=[1.]*16,clip_alignment=[1.]*16,clip_embeddings=[[1.,0.]]*16)
        tied=dict(rewards=[1.]*8,clip_alignment=[1.]*8,clip_embeddings=[[0.,1.]]*8)
        r=pool_metrics(bank,tied)
        self.assertEqual(r['candidate_fraction_in_top8'],0)
        self.assertEqual(r['pool_top8_diversity'],0)
        better={**tied,'rewards':[2.]+[0.]*7}
        r=pool_metrics(bank,better)
        self.assertEqual(r['candidate_fraction_in_top8'],1/8)
        self.assertEqual(r['pool_best_reward'],2)


if __name__=='__main__': unittest.main()
