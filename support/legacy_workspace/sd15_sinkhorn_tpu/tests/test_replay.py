import unittest
import torch
from diffusers import DDIMScheduler
from sd15_sinkhorn_tpu.sd15_sampler import ddim_update,ddim_coefficients,checkpoint_step


class TestReplay(unittest.TestCase):
    def test_ddim_matches_diffusers(self):
        s=DDIMScheduler(clip_sample=False); s.set_timesteps(30)
        g=torch.Generator().manual_seed(42)
        x=torch.randn(2,4,8,8,generator=g); eps=torch.randn(x.shape,generator=g)
        for eta in (0.,1.):
            for t in s.timesteps:
                noise=torch.randn(x.shape,generator=g)
                expected=s.step(eps,int(t),x,eta=eta,variance_noise=noise).prev_sample
                actual=ddim_update(x,eps,ddim_coefficients(s,t,eta),noise)
                self.assertTrue(torch.allclose(actual,expected,atol=2e-6,rtol=2e-6))

    def test_schedule_replay(self):
        s=DDIMScheduler(clip_sample=False); s.set_timesteps(30)
        x=torch.randn(2,4,8,8,generator=torch.Generator().manual_seed(4))
        saved=None
        for k,t in enumerate(s.timesteps):
            x=ddim_update(x,x.tanh()*.1,ddim_coefficients(s,t,0)).to(torch.bfloat16)
            if k==14: saved=x.clone()
        for t in s.timesteps[15:]: saved=ddim_update(saved,saved.tanh()*.1,ddim_coefficients(s,t,0)).to(torch.bfloat16)
        self.assertTrue(torch.equal(x,saved)); self.assertEqual(checkpoint_step(.5,30),15)
