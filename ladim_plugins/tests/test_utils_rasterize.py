import pytest
from ladim_plugins.utils import rasterize
import xarray as xr
import numpy as np
import netCDF4 as nc
from uuid import uuid4


class Test_ladim_chunks:
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
                    data=[0, 0, 1, 0, 1, 2, 1, 2, 3, 4],
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

    def test_number_of_chunks_equals_number_of_time_slots_if_unlimited_rows(self, data):
        _, ladim_1, ladim_2 = data
        chunks = rasterize.ladim_chunks([ladim_1, ladim_2], ['X'])
        num_chunks = sum(1 for _ in chunks)
        assert num_chunks == len(ladim_1.time) + len(ladim_2.time)

    def test_number_of_chunks_increases_if_small_maxrow(self, data):
        _, ladim_1, ladim_2 = data
        chunks = rasterize.ladim_chunks([ladim_1, ladim_2], ['X'], max_rows=2)
        num_chunks = sum(1 for _ in chunks)
        assert num_chunks > len(ladim_1.time) + len(ladim_2.time)

    def test_chunks_of_merged_and_separated_datasets_are_equivalent(self, data):
        ladim_0, ladim_1, ladim_2 = data
        chunks_12 = rasterize.ladim_chunks([ladim_1, ladim_2], ['X'])
        chunks_0 = rasterize.ladim_chunks([ladim_0], ['X'])
        for chunk_0, chunk_12 in zip(chunks_0, chunks_12):
            assert chunk_0['X'].tolist() == chunk_12['X'].tolist()

    def test_returns_dict_of_ndarrays(self, data):
        _, ladim_1, ladim_2 = data
        varnames = ['X', 'farmid', 'time']
        chunk = next(rasterize.ladim_chunks([ladim_1, ladim_2], varnames))
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == set(varnames)
        assert all(isinstance(v, np.ndarray) for v in chunk.values())

    def test_chunks_have_correct_size(self, data):
        ladim_0, ladim_1, ladim_2 = data
        chunks = rasterize.ladim_chunks([ladim_1, ladim_2], ['X'])
        pcounts_expected = ladim_0.particle_count.values.tolist()
        pcounts_actual = [len(chunk['X']) for chunk in chunks]
        assert pcounts_actual == pcounts_expected

    def test_broadcasts_particle_vars(self, data):
        ladim_0, ladim_1, ladim_2 = data
        chunks = rasterize.ladim_chunks([ladim_1, ladim_2], ['farmid'])
        pcounts_expected = ladim_0.particle_count.values.tolist()
        pcounts_actual = [len(chunk['farmid']) for chunk in chunks]
        assert pcounts_actual == pcounts_expected

    def test_broadcasts_time_vars(self, data):
        ladim_0, ladim_1, ladim_2 = data
        chunks = rasterize.ladim_chunks([ladim_1, ladim_2], ['time'])
        pcounts_expected = ladim_0.particle_count.values.tolist()
        pcounts_actual = [len(chunk['time']) for chunk in chunks]
        assert pcounts_actual == pcounts_expected


class Test_init_raster:
    def test_adds_dimensions(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            bin_keys = ['x', 'y']
            bin_centers = [[1, 2, 3], [4, 5, 6, 7]]
            rasterize.init_raster(dset, bin_keys, bin_centers)
            assert dset.dimensions['x'].size == 3
            assert dset.dimensions['y'].size == 4

    def test_adds_coords(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            bin_keys = ['x', 'y']
            bin_centers = [[1, 2, 3], [4, 5, 6, 7]]
            rasterize.init_raster(dset, bin_keys, bin_centers)
            assert dset.variables['x'][:].tolist() == bin_centers[0]
            assert dset.variables['y'][:].tolist() == bin_centers[1]

    def test_adds_time_coord(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            bin_keys = ['time']
            bin_centers = [np.array(['2000', '2001', '2003']).astype('datetime64')]
            rasterize.init_raster(dset, bin_keys, bin_centers)
            assert dset.variables['time'][:].tolist() == [0, 1, 3]

    def test_adds_bounds(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            bin_keys = ['x', 'y']
            bin_centers = [[1., 2., 3.], [4., 5., 6., 7.]]
            rasterize.init_raster(dset, bin_keys, bin_centers)
            assert dset.variables['x_bounds'][:, 0].tolist() == [0.5, 1.5, 2.5]
            assert dset.variables['x_bounds'][:, 1].tolist() == [1.5, 2.5, 3.5]
            assert dset.variables['x'].bounds == 'x_bounds'

    def test_adds_integer_bounds_when_bincenter_is_integer(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            bin_keys = ['x']
            bin_centers = [[1, 2, 3]]
            rasterize.init_raster(dset, bin_keys, bin_centers)
            assert dset.variables['x_bounds'][:, 0].tolist() == [1, 2, 3]
            assert dset.variables['x_bounds'][:, 1].tolist() == [2, 3, 4]
            assert dset.variables['x'].bounds == 'x_bounds'

    def test_adds_bincount(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            bin_keys = ['x', 'y']
            bin_centers = [[1, 2, 3], [4, 5, 6, 7]]
            rasterize.init_raster(dset, bin_keys, bin_centers)
            assert dset.variables['bincount'].dimensions == tuple(bin_keys)
            assert np.sum(dset.variables['bincount'][:]) == 0

    def test_can_add_weights(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            bin_keys = ['x', 'y']
            bin_centers = [[1, 2, 3], [4, 5, 6, 7]]
            rasterize.init_raster(dset, bin_keys, bin_centers, weights=('super', ))
            assert dset.variables['super'].dimensions == tuple(bin_keys)

    def test_can_copy_dtype_from_ladim_dset(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset_ladim:
                dset_ladim.createVariable('z', np.int32, ())
                bin_keys = ['x', 'y']
                bin_centers = [[1, 2, 3], [4, 5, 6, 7]]
                rasterize.init_raster(
                    dset, bin_keys, bin_centers, weights=('z', ), dset_ladim=dset_ladim)
                assert dset.variables['z'].dtype == dset_ladim.variables['z'].dtype


class Test_update_raster:
    @pytest.fixture()
    def raster(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            dset.createDimension('X', 2)
            dset.createDimension('Y', 3)
            dset.createDimension('time', 4)
            dset.createDimension('bnd', 2)
            dset.createVariable('X', float, 'X')[:] = [2.5, 5.5]
            dset.createVariable('Y', float, 'Y')[:] = [30, 55, 70]
            dset.createVariable('time', float, 'time')[:] = [-1, 0, 1, 2]
            dset.createVariable('X_bounds', float, ('X', 'bnd'))[:] = [[1.5, 3.5], [3.5, 7.5]]
            dset.createVariable('Y_bounds', float, ('Y', 'bnd'))[:] = [[15, 45], [45, 65], [65, 75]]
            dset.createVariable('time_bounds', float, ('time', 'bnd'))[:] = [
                [-1.5, -.5], [-.5, .5], [.5, 1.5], [1.5, 2.5]]
            dset.variables['X'].bounds = 'X_bounds'
            dset.variables['Y'].bounds = 'Y_bounds'
            dset.variables['time'].bounds = 'time_bounds'
            dset.variables['time'].units = 'hours since 2000-01-01T00'
            dset.set_auto_maskandscale(False)
            yield dset

    @pytest.fixture()
    def chunk(self):
        return dict(
            time=np.array([np.datetime64('2000')] * 10),
            X=np.arange(10),
            Y=np.arange(10) * 10,
            W=np.ones(10) * 10,
        )

    def test_correct_when_single_bin_key(self, raster, chunk):
        raster.createVariable('bincount', np.int32, ('X', ))[:] = 0
        rasterize.update_raster(raster, chunk, ['X'])
        assert raster.variables['bincount'][:].tolist() == [2, 4]

    def test_correct_when_two_bin_keys(self, raster, chunk):
        raster.createVariable('bincount', np.int32, ('X', 'Y'))[:] = 0
        rasterize.update_raster(raster, chunk, ['X', 'Y'])
        assert raster.variables['bincount'][:].tolist() == [[2, 0, 0], [1, 2, 1]]

    def test_increments_raster_when_multiple_calls(self, raster, chunk):
        raster.createVariable('bincount', np.int32, ('X', ))[:] = 0
        rasterize.update_raster(raster, chunk, ['X'])
        double_first_val = 2 * raster.variables['bincount'][:]
        rasterize.update_raster(raster, chunk, ['X'])
        second_val = raster.variables['bincount'][:]
        assert double_first_val.tolist() == second_val.tolist()
