import pytest
from ladim_plugins.utils import rasterize
import xarray as xr
import numpy as np


class Test_merge_ladim:
    @pytest.fixture(scope='class')
    def data(self):
        ladim_0 = xr.Dataset(
            data_vars=dict(
                instance_offset=5,
                release_time=xr.Variable(
                    dims='particle',
                    data=np.datetime64('2000') + np.arange(5) + np.timedelta64(1, 'D'),
                ),
                farmid=xr.Variable(dims='particle', data=[101, 102, 101, 102, 101]),
                pid=xr.Variable(
                    dims='particle_instance',
                    data=[1, 1, 2, 1, 2, 3, 2, 3, 4, 5],
                ),
                X=xr.Variable(
                    dims='particle_instance',
                    data=[1, 1.1, 2, 1.2, 2.1, 3, 2.2, 3.1, 4, 5],
                ),
                particle_count=xr.Variable(
                    dims='time',
                    data=[1, 2, 3, 4],
                ),
            ),
            coords=dict(
                time=np.datetime64('2000') + np.arange(4) + np.timedelta64(1, 'D'),
            ),
        )
        ladim_1 = ladim_0.isel(time=range(2), particle_instance=range(3))
        ladim_2 = ladim_0.isel(time=range(2, 4), particle_instance=range(3, 10))
        ladim_2['instance_offset'] += 3

        return [ladim_0, ladim_1, ladim_2]

    def test_correct_result(self, data):
        ladim_0, ladim_1, ladim_2 = data
        ladim_new = rasterize.merge_ladim([ladim_1, ladim_2])
        assert str(ladim_new) == str(ladim_0)

