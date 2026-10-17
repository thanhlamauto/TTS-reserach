import unittest
import numpy as np
from ot_next.output_selection import quality_diversity_indices,good_distinct


class OutputTests(unittest.TestCase):
    def test_quality_gate_then_diverse_selection_and_fixed_fill(self):
        f=np.array([[1.,0],[1.,0],[0,1],[-1,0],[0,-1]])
        ids=quality_diversity_indices([5,4,3,2,1],f,2.5,count=3)
        self.assertEqual(ids,[0,2,1])
        filled=quality_diversity_indices([5,4,3,2,1],f,9,count=3)
        self.assertEqual(filled,[0,1,2])

    def test_exact_nontransitive_diversity_graph(self):
        d=np.ones((4,4))-np.eye(4)
        d[0,1:]=.1;d[1:,0]=.1
        x=good_distinct([1]*4,d,[True]*4,0,.5)
        self.assertEqual(x['count'],3)
        self.assertEqual(x['indices'],[1,2,3])

    def test_unknown_bounds_and_quality(self):
        d=np.ones((3,3))-np.eye(3)
        self.assertEqual(good_distinct([1,-1,2],d,[True,True,None],0,.1)['count'],1)
        self.assertEqual(good_distinct([1,-1,2],d,[True,True,None],0,.1,True)['count'],2)
