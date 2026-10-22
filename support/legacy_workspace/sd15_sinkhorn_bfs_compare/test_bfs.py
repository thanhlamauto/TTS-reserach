import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from .bfs import policy,make_adapter


class TestBFS(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0,'/home/thanhlamtba31/static-bfs-dcs/official/text_to_image/fkd_diffusers')

    def test_mapping(self):
        self.assertEqual(policy('manual',100)['resampling_steps'],[20,40,80])
        self.assertEqual(policy('manual',30)['resampling_steps'],[6,12,24])
        self.assertEqual(policy('graph',30)['resampling_steps'],[12,18,27])
        self.assertEqual([t for _,t in policy('manual',30)['temperature_by_step']],
                         [t for _,t in policy('manual',100)['temperature_by_step']])

    def test_exact_ssp_and_history(self):
        from scripts.static_bfs_graph.adapter import event_seed
        from scripts.diffusion_classical_search_t2i.tpu_runner import _seed_numba
        import smc_utils
        p=policy('graph',100)
        raw=torch.tensor([-.5,.2,.4,1.])
        adapter=make_adapter(4,p,42,3,lambda x:raw)
        x=torch.arange(4.)[:,None]
        out,_=adapter.resample(sampling_idx=40,latents=x,x0_preds=x)
        scores=torch.maximum(raw*8,torch.zeros(4))
        weights=torch.softmax(scores,0).numpy().astype('float64')
        s=event_seed(42,3,40); np.random.seed(s); _seed_numba(s)
        expected=smc_utils.Resample_dict['ssp'](weights,4)
        self.assertEqual(adapter.events[-1]['parent_indices'],expected.tolist())
        self.assertTrue(torch.equal(out,x[expected])); self.assertEqual(len(out),4)
        self.assertTrue(torch.equal(adapter.previous,scores[expected]))
        old=adapter.previous.clone(); raw.fill_(-100)
        adapter.resample(sampling_idx=60,latents=out,x0_preds=out)
        self.assertEqual(adapter.events[-1]['particle_scores'],old.tolist())

    def test_zero_floor_and_determinism(self):
        p=policy('manual',30); x=torch.arange(8.)[:,None]
        a=make_adapter(8,p,20261015,0,lambda x:-torch.ones(8))
        b=make_adapter(8,p,20261015,0,lambda x:-torch.ones(8))
        for event in p['resampling_steps']:
            aa,_=a.resample(sampling_idx=event,latents=x,x0_preds=x)
            bb,_=b.resample(sampling_idx=event,latents=x,x0_preds=x)
            self.assertTrue(torch.equal(aa,bb)); self.assertTrue(torch.equal(aa,x))
            self.assertEqual(a.events[-1]['particle_scores'],[0.]*8)


if __name__=='__main__': unittest.main()
