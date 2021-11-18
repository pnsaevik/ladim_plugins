from ladim_plugins.utils import rasterize
import pytest
import xarray as xr
import numpy as np


class Test_ladim_iterator:
    @pytest.fixture(scope='class')
    def ladim_dset(self):
        return xr.Dataset(
            data_vars=dict(
                X=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
                Y=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
                Z=xr.Variable('particle_instance', [0, 1, 2, 3, 4, 5]),
                lon=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
                lat=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
                farm_id=xr.Variable('particle', [12345, 12346, 12347, 12348]),
                pid=xr.Variable('particle_instance', [0, 1, 2, 3, 1, 2]),
                particle_count=xr.Variable('particle_count', [4, 2]),
            ),
            coords=dict(
                time=np.array(['2000-01-02', '2000-01-03']).astype('datetime64[D]'),
            ),
        )

    def test_returns_one_dataset_per_timestep_when_multiple_datasets(self, ladim_dset):
        it = rasterize.ladim_iterator([ladim_dset, ladim_dset])
        dsets = list(it)
        assert len(dsets) == ladim_dset.dims['time'] * 2
