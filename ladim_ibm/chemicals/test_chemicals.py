import numpy as np
from ladim_ibm.chemicals import gridforce


class Test_nearest_unmasked:
    def test_correct_when_all_unmasked(self):
        mask = np.zeros((4, 3))
        i = np.array([1, 1, 2])
        j = np.array([2, 3, 3])
        ii, jj = gridforce.nearest_unmasked(mask, i, j)
        assert ii.tolist() == i.tolist()
        assert jj.tolist() == j.tolist()
