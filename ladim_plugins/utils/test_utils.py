from ladim_plugins.utils import light
import numpy as np


class Test_light:
    def test_changes_with_latitude(self):
        Eb = light(time='2000-01-01T12', lon=5, lat=[0, 60])
        assert Eb.tolist() == [1500.0520471376183, 1484.063211499949]

    def test_changes_with_time(self):
        Eb_1 = light(time='2000-01-01T12', lon=5, lat=60)
        Eb_2 = light(time='2000-01-01T00', lon=5, lat=60)
        assert [Eb_1, Eb_2] == [1484.063211499949, 1.15e-05]

    def test_changes_with_date(self):
        Eb_1 = light(time='2000-01-01T12', lon=5, lat=60)
        Eb_2 = light(time='2000-06-01T12', lon=5, lat=60)
        assert [Eb_1, Eb_2] == [1484.063211499949, 1502.4089323926253]

    def test_changes_with_depth(self):
        Eb = light(time='2000-01-01T12', lon=5, lat=60, depth=np.array([0, 5, 10]))
        assert Eb.tolist() == [1484.063211499949, 545.9563449096972, 200.84611506938265]
