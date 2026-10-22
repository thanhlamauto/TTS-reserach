import unittest
import torch
from sd15_sinkhorn_tpu.sinkhorn import solve,divergence,reference_scale


class TestOT(unittest.TestCase):
    def test_marginals(self):
        c=torch.rand(8,16,generator=torch.Generator().manual_seed(42))*3
        w=torch.softmax(torch.linspace(-2,2,16),0)
        p=solve(c,w=w,epsilon=.1)
        self.assertLess(p.row_error,3e-6); self.assertLess(p.column_error,3e-6)

    def test_identical(self):
        x=torch.randn(4,4,64,64,generator=torch.Generator().manual_seed(3))
        d=divergence(x,x,torch.ones(4)/4,.1,reference_scale(x))
        self.assertLess(abs(d),1e-6)

    def test_zero_weights(self):
        p=solve(torch.rand(3,4),w=torch.tensor([.5,0,.5,0]))
        self.assertTrue((p.p[:,[1,3]]==0).all())

    def test_reject_nan(self):
        with self.assertRaises(ValueError): solve(torch.tensor([[float('nan')]]))
