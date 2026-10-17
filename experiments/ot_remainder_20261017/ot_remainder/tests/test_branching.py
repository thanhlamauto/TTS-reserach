import unittest
import numpy as np
from ot_remainder.branching import allocation,anchors


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.f=np.eye(8)
        self.records=[dict(root=i,direct_levels=[20,40],reward=float(i/2),feature=self.f[i]) for i in range(8)]
        self.u=np.linspace(.01,.99,8)

    def test_exact_counts_and_equal_exploration(self):
        for n in [4,8]:
            for method in ['uniform','reward_ucb','group','action_ot']:
                parents,log=allocation(method,20,n,self.records,self.f[:4],0,self.u)
                self.assertEqual(len(parents),n)
                self.assertEqual(log['exploration_slots'],n//4)
                self.assertEqual(sum(log['offspring_counts']),n)
                self.assertAlmostEqual(sum(log['weights']),1)

    def test_level_conditioning(self):
        extra=dict(root=0,direct_levels=[20],reward=10.,feature=self.f[0])
        _,old=allocation('reward_ucb',40,4,self.records,self.f[:4],0,self.u)
        _,new=allocation('reward_ucb',40,4,self.records+[extra],self.f[:4],0,self.u)
        self.assertEqual(old['n_direct'],new['n_direct'])
        self.assertEqual(old['weights'],new['weights'])

    def test_no_good_no_fake_groups(self):
        self.assertIsNone(anchors(self.f,-np.ones(8),0,.1))
        for method in ['uniform','reward_ucb','group','action_ot']:
            _,log=allocation(method,20,4,self.records,None,0,self.u)
            np.testing.assert_allclose(log['weights'],np.full(8,1/8))

    def test_rows_free_column(self):
        _,log=allocation('action_ot',20,8,self.records,self.f[:4],0,self.u)
        np.testing.assert_allclose(np.sum(log['coupling'],axis=1),log['alpha'])
        self.assertGreater(np.ptp(log['weights']),0)


if __name__=='__main__':unittest.main()
