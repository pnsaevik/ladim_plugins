from ladim_plugins.utils import light, density
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


class Test_density:
    def test_changes_with_temperature(self):
        rho = density(temp=np.array([0, 50]), salt=30)
        assert rho.tolist() == [1024.0715523751858, 1009.9641764883575]

    def test_changes_with_salinity(self):
        rho = density(temp=4, salt=np.array([10, 40]))
        assert rho.tolist() == [1007.9473603468945, 1031.7686242667996]
