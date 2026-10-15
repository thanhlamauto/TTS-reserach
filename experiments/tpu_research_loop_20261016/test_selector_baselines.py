import unittest
import numpy as np
from selector_baselines import cache_key,direct_full_policy_select,graph_only_select,full_policy_verify_select
class SelectorTests(unittest.TestCase):
 def test_baseline_can_beat_surrogate(self):
  h=['manual','transfer','fine'];x=np.tile([0.,.2,.5],(8,2,1));self.assertEqual(direct_full_policy_select(x,h)['policy_hash'],'fine')
  graph=graph_only_select(np.tile([.1,0,.3],(8,1)),np.eye(3),h);self.assertEqual(graph['policy_hash'],'transfer')
  self.assertEqual(full_policy_verify_select(x,h,['manual','transfer'])['policy_hash'],'transfer')
 def test_duplicates_rejected(self):
  with self.assertRaises(ValueError):direct_full_policy_select(np.zeros((3,2,2)),['x','x'])
 def test_cache_model_input_branch_isolation(self):
  base=dict(model_revision='m',scheduler_sha256='s',precision='bf16',code_sha256='c',policy_prefix=[20],input_sha256='p',initial_noise_sha256='n',rng_state_sha256='r',kind='prefix')
  a=cache_key(**base);self.assertNotEqual(a,cache_key(**{**base,'model_revision':'m2'}))
  self.assertNotEqual(a,cache_key(**{**base,'input_sha256':'p2'}))
  with self.assertRaises(ValueError):cache_key(**{**base,'kind':'suffix'})
  self.assertNotEqual(cache_key(**{**base,'kind':'suffix','branch_suffix':{'policy':[40,60],'rng':'a'}}),cache_key(**{**base,'kind':'suffix','branch_suffix':{'policy':[50,80],'rng':'b'}}))
 def test_shared_edges_cancel(self):
  rng=np.random.default_rng(1);d=rng.normal(size=(20,3));g=graph_only_select(d,np.array([[1,1,0],[1,0,1]]),['a','b']);diff=g['paired_candidate_cost_blocks'][:,0]-g['paired_candidate_cost_blocks'][:,1]
  np.testing.assert_allclose(diff,d[:,1]-d[:,2],rtol=1e-12,atol=1e-12)
if __name__=='__main__':unittest.main()
