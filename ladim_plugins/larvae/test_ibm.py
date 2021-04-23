from ladim_plugins.larvae import ibm
import numpy as np


class Test_sinkvel_egg:
    def test_changes_with_egg_diam(self):
        s = ibm.sinkvel_egg(
            mu_w=0.001, dens_w=1000, dens_egg=1001,
            diam_egg=np.array([0.0005, 0.001, 0.002]),
        )

        assert s.tolist() == [0.00013625, 0.0005423076178904741, 0.001424807617890474]

    def test_changes_with_egg_dens_when_large_egg(self):
        s = ibm.sinkvel_egg(
            mu_w=0.001, dens_w=1000, dens_egg=np.array([999, 1000, 1001]),
            diam_egg=0.002,
        )

        assert s.tolist() == [-0.001424807617890474, 0, 0.001424807617890474]

    def test_changes_with_egg_dens_when_small_egg(self):
        s = ibm.sinkvel_egg(
            mu_w=0.001, dens_w=1000, dens_egg=np.array([999, 1000, 1001]),
            diam_egg=0.0005,
        )

        assert s.tolist() == [-0.00013625, 0, 0.00013625]
