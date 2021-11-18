from ladim_plugins.utils import rasterize
import pytest
import xarray as xr
import numpy as np


@pytest.fixture(scope='module')
def ladim_dset():
    return xr.Dataset(
        data_vars=dict(
            X=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
            Y=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
            Z=xr.Variable('particle_instance', [0, 1, 2, 3, 4, 5]),
            lon=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
            lat=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
            instance_offset=xr.Variable((), 0),
            farm_id=xr.Variable('particle', [12345, 12346, 12347, 12348]),
            pid=xr.Variable('particle_instance', [0, 1, 2, 3, 1, 2]),
            particle_count=xr.Variable('time', [4, 2]),
        ),
        coords=dict(
            time=np.array(['2000-01-02', '2000-01-03']).astype('datetime64[D]'),
        ),
    )


@pytest.fixture(scope='class')
def ladim_dset2(ladim_dset):
    d = ladim_dset.copy(deep=True)
    d['instance_offset'] += d.dims['particle_instance']
    d = d.assign_coords(time=d.time + np.timedelta64(2, 'D'))
    return d


class Test_ladim_iterator:
    def test_returns_one_dataset_per_timestep_when_multiple_datasets(self, ladim_dset, ladim_dset2):
        it = rasterize.ladim_iterator([ladim_dset, ladim_dset2])
        dsets = list(it)
        assert len(dsets) == ladim_dset.dims['time'] + ladim_dset2.dims['time']

    def test_returns_correct_time_selection(self, ladim_dset):
        iterator = rasterize.ladim_iterator([ladim_dset])
        particle_count = [d.particle_count.values.item() for d in iterator]
        assert particle_count == ladim_dset.particle_count.values.tolist()

        iterator = rasterize.ladim_iterator([ladim_dset])
        time = [d.time.values.item() for d in iterator]
        assert time == ladim_dset.time.values.tolist()

    def test_returns_correct_instance_selection(self, ladim_dset):
        iterator = rasterize.ladim_iterator([ladim_dset])
        z = [d.Z.values.tolist() for d in iterator]
        assert z == [[0, 1, 2, 3], [4, 5]]

        iterator = rasterize.ladim_iterator([ladim_dset])
        pid = [d.pid.values.tolist() for d in iterator]
        assert pid == [[0, 1, 2, 3], [1, 2]]

    def test_broadcasts_particle_variables(self, ladim_dset):
        iterator = rasterize.ladim_iterator([ladim_dset])
        farm_id = [d.farm_id.values.tolist() for d in iterator]
        assert farm_id == [[12345, 12346, 12347, 12348], [12346, 12347]]

    def test_updates_instance_offset(self, ladim_dset):
        iterator = rasterize.ladim_iterator([ladim_dset])
        offset = [d.instance_offset.values.tolist() for d in iterator]
        assert offset == [0, 4]
