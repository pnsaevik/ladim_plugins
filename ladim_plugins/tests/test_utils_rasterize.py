import pytest
from ladim_plugins.utils import rasterize
import xarray as xr
import numpy as np
import netCDF4 as nc
from uuid import uuid4


@pytest.fixture(scope='module')
def multipart_ladim():
    ladim_0 = xr.Dataset(
        data_vars=dict(
            instance_offset=5,
            release_time=xr.Variable(
                dims='particle',
                data=np.datetime64('2000') + np.arange(5) * np.timedelta64(1, 'D'),
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
            mass=xr.Variable(
                dims='particle_instance',
                data=[10, 11, 20, 12, 21, 30, 22, 31, 40, 50],
            ),
            particle_count=xr.Variable(
                dims='time',
                data=[1, 2, 3, 4],
            ),
        ),
        coords=dict(
            time=np.datetime64('2000') + np.arange(4) * np.timedelta64(1, 'D'),
        ),
    )
    ladim_1 = ladim_0.isel(time=range(2), particle_instance=range(3))
    ladim_2 = ladim_0.isel(time=range(2, 4), particle_instance=range(3, 10))
    ladim_2['instance_offset'] += 3

    return [ladim_0, ladim_1, ladim_2]


@pytest.fixture(scope='module')
def multipart_ladim_nc(multipart_ladim):
    dsets = []
    for d in multipart_ladim:
        dset = nc.Dataset(uuid4().hex, memory=d.to_netcdf())
        dset.set_auto_maskandscale(False)
        dsets.append(dset)

    return dsets


class Object:
    def __init__(self, index_obj=None, **kwargs):
        self.index_obj = index_obj
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return self.index_obj[item]


class Test_csr_to_coo_chunks:
    def test_single_chunk_if_unrestricted(self):
        counts = [0, 5, 4, 0, 1]
        max_size = np.inf
        chunks = list(rasterize.csr_to_coo_chunks(counts, max_size))
        assert chunks == [(slice(0, 10), slice(0, 5))]

    def test_multiple_chunks_if_restricted(self):
        counts = [0, 5, 4, 0, 1]
        max_size = 5
        chunks = list(rasterize.csr_to_coo_chunks(counts, max_size))
        assert chunks == [
            (slice(0,  5), slice(0, 2)),
            (slice(5, 10), slice(2, 5)),
        ]

    def test_yields_oversized_chunks_if_maxsize_less_than_rowsize(self):
        counts = [0, 5, 4, 0, 1]
        max_size = 4
        chunks = list(rasterize.csr_to_coo_chunks(counts, max_size))
        assert chunks == [
            (slice(0,  0, None), slice(0, 1, None)),
            (slice(0,  5, None), slice(1, 2, None)),
            (slice(5,  9, None), slice(2, 4, None)),
            (slice(9, 10, None), slice(4, 5, None)),
        ]


class Test_ladim_to_sparse_chunks:
    @pytest.fixture(scope="class")
    def dset(self):
        return Object(
            dict(
                pid=Object(
                    np.array([0, 0, 1, 0, 1, 2, 1, 2, 3, 4]),
                    dimensions=('particle_instance', )),
                X=Object(
                    np.array([1, 1.1, 2, 1.2, 2.1, 3, 2.2, 3.1, 4, 5]),
                    dimensions=('particle_instance', )),
                particle_count=Object(
                    np.array([1, 2, 3, 4]),
                    dimensions=('time', )),
                farmid=Object(
                    np.array([101, 102, 101, 102, 101]),
                    dimensions=('particle', )),
            ),
            dimensions=dict(particle_instance=Object(size=10)),
        )

    def test_single_chunk_when_unlimited(self, dset):
        chunks = rasterize.ladim_chunks(dset, ['X'])
        chunk_values = [chunk['X'].tolist() for chunk in chunks]
        assert len(chunk_values) == 1
        assert chunk_values == [dset['X'][...].tolist()]

    def test_multiple_wholestep_chunks_when_limited(self, dset):
        chunks = rasterize.ladim_chunks(dset, ['X'], max_size=5)
        chunk_values = [chunk['X'].tolist() for chunk in chunks]
        assert len(chunk_values) == 3
        assert chunk_values == [
            dset['X'][:3].tolist(),
            dset['X'][3:6].tolist(),
            dset['X'][6:].tolist(),
        ]

    def test_oversized_chunks_when_strict_limitation(self, dset):
        chunks = rasterize.ladim_chunks(dset, ['X'], max_size=2)
        chunk_values = [chunk['X'].tolist() for chunk in chunks]
        assert len(chunk_values) == 4
        assert chunk_values == [
            dset['X'][:1].tolist(),   # Timestep 0
            dset['X'][1:3].tolist(),  # Timestep 1
            dset['X'][3:6].tolist(),  # Timestep 2, oversized
            dset['X'][6:].tolist(),   # Timestep 3, oversized
        ]

    def test_broadcasts_particle_vars(self, dset):
        chunks = rasterize.ladim_chunks(dset, ['farmid'])
        assert next(chunks)['farmid'].tolist() == [
            101, 101, 102, 101, 102, 101, 102, 101, 102, 101]

    def test_broadcasts_time_vars(self, dset):
        chunks = rasterize.ladim_chunks(dset, ['particle_count'])
        assert next(chunks)['particle_count'].tolist() == [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]


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
            assert dset.variables['time'][:].tolist() == [
                946684800000, 978307200000, 1041379200000]

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


class Test_dt64_to_num:
    def test_correct_when_hour_accuracy(self):
        dt64 = np.datetime64('2000') + np.arange(5) * np.timedelta64(1, 'h')
        num = rasterize.dt64_to_num(
            dates=dt64,
            units='minutes since 2000-01-01',
            calendar='standard',
        )
        assert num.tolist() == [0, 60, 120, 180, 240]

    def test_correct_when_nanosecond_accuracy(self):
        dt64 = np.datetime64('2000') + np.arange(5) * np.timedelta64(1000000000, 'ns')
        num = rasterize.dt64_to_num(
            dates=dt64,
            units='seconds since 2000-01-01',
            calendar='standard',
        )
        assert num.tolist() == [0, 1, 2, 3, 4]

    def test_correct_when_fractional_output(self):
        dt64 = np.datetime64('2000') + 6 * np.arange(5) * np.timedelta64(1, 'h')
        num = rasterize.dt64_to_num(
            dates=dt64,
            units='days since 2000-01-01',
            calendar='standard',
        )
        assert num.tolist() == [0, .25, .5, .75, 1]

    def test_correct_when_day_accuracy(self):
        dt64 = np.datetime64('2000') + np.arange(5) * np.timedelta64(1, 'D')
        num = rasterize.dt64_to_num(
            dates=dt64,
            units='hours since 2000-01-01',
            calendar='standard',
        )
        assert num.tolist() == [0, 24, 48, 72, 96]


class Test_adaptive_histogram:
    @staticmethod
    def assert_equals_histogramdd(sample, bins, **kwargs):
        expected = np.histogramdd(sample, bins, **kwargs)[0]
        actual = np.zeros_like(expected)
        subset, idx = rasterize.adaptive_histogram(sample, bins, **kwargs)
        actual[idx] += subset
        assert actual.tolist() == expected.tolist()

    def test_returns_truncated_histogram_if_subset_input(self):
        bins = [[0, 1, 2, 3, 4, 5], [0, 1, 2]]
        sample = [[1.5, 1.6, 1.7, 2.2, 2.3], [.1, .1, .1, .1, 1.2]]
        values, idx = rasterize.adaptive_histogram(sample, bins)
        assert idx == np.s_[1:3, 0:2]
        assert values.shape == (2, 2)

    def test_equals_histogramdd_if_full_input(self):
        bins = [[0, 1, 2, 3]]
        sample = [[.5, 1.5, 1.6, 1.7, 2.2, 2.3]]
        expected = np.histogramdd(sample, bins)[0]
        actual = rasterize.adaptive_histogram(sample, bins)[0]
        assert actual.tolist() == expected.tolist()

    def test_equals_histogramdd_if_full_input_and_weights(self):
        bins = [[0, 1, 2, 3]]
        sample = [[.5, 1.5, 1.6, 1.7, 2.2, 2.3]]
        self.assert_equals_histogramdd(sample, bins, weights=sample[0])

    def test_equals_histogramdd_if_subset_input(self):
        bins = [[0, 1, 2, 3]]
        sample = [[1.5, 1.6, 1.7]]
        self.assert_equals_histogramdd(sample, bins)

    def test_equals_histogramdd_if_on_bin_edges(self):
        bins = [[0, 1, 2, 3, 4, 5, 6]]
        sample = [[1.0, 2.0, 2.5, 2.6, 2.7, 3.0, 3.1, 3.2, 4.0]]
        self.assert_equals_histogramdd(sample, bins)

    def test_equals_histogramdd_if_outside_bin_edges(self):
        bins = [[2, 3, 4]]
        sample = [[1.0, 2.0, 2.5, 2.6, 2.7, 3.0, 3.1, 3.2, 4.0]]
        self.assert_equals_histogramdd(sample, bins)


class Test_chunked_histogram:
    @staticmethod
    def assert_equals_histogramdd(sample, bins, max_size=None, weights=None):
        expected = np.histogramdd(sample, bins, weights=weights)[0]
        actual = np.zeros_like(expected)
        chunks = rasterize.chunked_histogram(sample, bins, max_size, weights)
        for subset, idx in chunks:
            actual[idx] += subset
        assert actual.tolist() == expected.tolist()

    def test_single_chunk_if_unrestricted(self):
        bins = [[0, 1, 2, 3]]
        sample = [[.5, 1.5, 1.6, 1.7, 2.2, 2.3]]
        chunks = rasterize.chunked_histogram(sample, bins)
        assert len(list(chunks)) == 1

    def test_equals_histogramdd_if_unrestricted(self):
        bins = [[0, 1, 2, 3]]
        sample = [[.5, 1.5, 1.6, 1.7, 2.2, 2.3]]
        self.assert_equals_histogramdd(sample, bins)

    def test_multiple_chunks_if_restricted(self):
        bins = [[0, 1, 2, 3]]
        sample = [[.5, 1.5, 1.6, 1.7, 2.2, 2.3]]
        chunks = rasterize.chunked_histogram(sample, bins, max_size=2)
        assert len(list(chunks)) > 1

    def test_equals_histogramdd_if_restricted(self):
        bins = [[0, 1, 2, 3]]
        sample = [[.5, 1.5, 1.6, 1.7, 2.2, 2.3]]
        self.assert_equals_histogramdd(sample, bins, max_size=2)

    def test_returns_truncated_histogram_if_subset_input(self):
        bins = [[0, 1, 2, 3, 4, 5], [0, 1, 2]]
        sample = [[1.5, 1.6, 1.7, 2.2, 2.3], [.1, .1, .1, .1, 1.2]]
        values, idx = next(rasterize.chunked_histogram(sample, bins))
        assert idx == np.s_[1:3, 0:2]
        assert values.shape == (2, 2)

    def test_equals_histogramdd_if_full_input_and_weights(self):
        bins = [[0, 1, 2, 3]]
        sample = [[.5, 1.5, 1.6, 1.7, 2.2, 2.3]]
        self.assert_equals_histogramdd(sample, bins, weights=sample[0])

    def test_equals_histogramdd_if_on_bin_edges(self):
        bins = [[0, 1, 2, 3, 4, 5, 6]]
        sample = [[1.0, 2.0, 2.5, 2.6, 2.7, 3.0, 3.1, 3.2, 4.0]]
        self.assert_equals_histogramdd(sample, bins)

    def test_equals_histogramdd_if_outside_bin_edges(self):
        bins = [[2, 3, 4]]
        sample = [[1.0, 2.0, 2.5, 2.6, 2.7, 3.0, 3.1, 3.2, 4.2]]
        self.assert_equals_histogramdd(sample, bins)


class Test_extend_bins:
    @pytest.fixture()
    def dset(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as d:
            d.createDimension('bins', None)
            d.createDimension('bnd', 2)
            d.createVariable('bins', 'i', 'bins')[:] = [1, 3, 5]
            d.createVariable('bins_b', 'i', ('bins', 'bnd'))[:] = [[0, 2], [2, 4], [4, 6]]
            d['bins'].bounds = 'bins_b'
            d.set_auto_maskandscale(False)
            yield d

    def test_does_nothing_if_maxval_inside_bounds_when_explicit_bounds(self, dset):
        rasterize.extend_bins(dset, 'bins', 4)
        assert dset.dimensions['bins'].size == 3

    def test_does_nothing_if_maxval_inside_bounds_when_implicit_bounds(self, dset):
        del dset['bins'].bounds
        rasterize.extend_bins(dset, 'bins', 4)
        assert dset.dimensions['bins'].size == 3

    def test_expands_if_maxval_equals_right_bound_when_explicit_bounds(self, dset):
        rasterize.extend_bins(dset, 'bins', 6)
        assert dset.dimensions['bins'].size == 4
        assert dset['bins'][:].tolist() == [1, 3, 5, 7]
        assert dset['bins_b'][:].T.tolist() == [[0, 2, 4, 6], [2, 4, 6, 8]]

    def test_expands_if_maxval_equals_right_bound_when_implicit_bounds(self, dset):
        del dset['bins'].bounds
        rasterize.extend_bins(dset, 'bins', 6)
        assert dset.dimensions['bins'].size == 4
        assert dset['bins'][:].tolist() == [1, 3, 5, 7]

    def test_expands_if_large_maxval_when_explicit_bounds(self, dset):
        rasterize.extend_bins(dset, 'bins', 11)
        assert dset.dimensions['bins'].size == 6
        assert dset['bins'][:].tolist() == [1, 3, 5, 7, 9, 11]
        assert dset['bins_b'][:].T.tolist() == [[0, 2, 4, 6, 8, 10], [2, 4, 6, 8, 10, 12]]

    def test_expands_if_large_maxval_when_implicit_bounds(self, dset):
        del dset['bins'].bounds
        rasterize.extend_bins(dset, 'bins', 11)
        assert dset.dimensions['bins'].size == 6
        assert dset['bins'][:].tolist() == [1, 3, 5, 7, 9, 11]


class Test_update_raster:
    @pytest.fixture()
    def raster(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            dset.createDimension('X', 2)
            dset.createDimension('Y', 3)
            dset.createDimension('time', None)
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
            dset.variables['time'].calendar = 'standard'
            dset.set_auto_maskandscale(False)
            yield dset

    @pytest.fixture()
    def chunk(self):
        return dict(
            time=np.zeros(10),
            X=np.arange(10),
            Y=np.arange(10) * 10,
            W=np.ones(10) * 10,
        )

    def test_correct_when_single_bin_key(self, raster, chunk):
        raster.createVariable('bincount', np.int32, ('X', ))[:] = 0
        rasterize.update_raster(raster, chunk, ['X'])
        assert raster.variables['bincount'][:].tolist() == [2, 4]

    def test_correct_when_single_bin_key_and_weights(self, raster, chunk):
        raster.createVariable('W', np.float32, ('X', ))[:] = 0
        rasterize.update_raster(raster, chunk, ['X'], weight_var='W')
        assert raster.variables['W'][:].tolist() == [20, 40]

    def test_correct_when_two_bin_keys(self, raster, chunk):
        raster.createVariable('bincount', np.int32, ('X', 'Y'))[:] = 0
        rasterize.update_raster(raster, chunk, ['X', 'Y'])
        assert raster.variables['bincount'][:].tolist() == [[2, 0, 0], [1, 2, 1]]

    def test_correct_when_time_bin_key(self, raster, chunk):
        raster.createVariable('bincount', np.int32, ('time', ))[:] = 0
        rasterize.update_raster(raster, chunk, ['time'])
        assert raster.variables['bincount'][:].tolist() == [0, 10, 0, 0]

    def test_increments_raster_when_multiple_calls(self, raster, chunk):
        raster.createVariable('bincount', np.int32, ('X', ))[:] = 0
        rasterize.update_raster(raster, chunk, ['X'])
        double_first_val = 2 * raster.variables['bincount'][:]
        rasterize.update_raster(raster, chunk, ['X'])
        second_val = raster.variables['bincount'][:]
        assert double_first_val.tolist() == second_val.tolist()

    def test_expands_outer_dim_automatically(self, raster):
        chunk = dict(time=np.arange(10))
        assert raster['time'][:].tolist() == [-1, 0, 1, 2]
        raster.createVariable('bincount', np.int32, ('time', ))[:] = 0
        rasterize.update_raster(raster, chunk, ['time'])
        assert raster.variables['bincount'][:].tolist() == [0] + [1] * 10


class Test_parse_args:
    def test_standard_syntax(self):
        args = rasterize.parse_args(['myrasterfile.nc', 'myladimfile.nc'])
        assert args.ladim_file == ['myladimfile.nc']
        assert args.raster_file == 'myrasterfile.nc'

    def test_accepts_multiple_ladimfiles(self):
        args = rasterize.parse_args(['myrasterfile.nc', 'ladim1.nc', 'ladim2.nc'])
        assert args.ladim_file == ['ladim1.nc', 'ladim2.nc']
        assert args.raster_file == 'myrasterfile.nc'

    def test_fails_when_no_ladimfile(self):
        with pytest.raises(SystemExit):
            rasterize.parse_args(['myrasterfile.nc'])

    def test_expands_glob_patterns(self):
        import os
        import glob
        thisdir = os.path.dirname(__file__)
        pattern = os.path.join(thisdir, '*.py')
        args = rasterize.parse_args(['myrasterfile.nc', 'ladim1.nc', pattern])

        assert args.ladim_file[0] == 'ladim1.nc'
        assert len(args.ladim_file) == 1 + len(glob.glob(pattern))

    def test_accepts_max_size_argument(self):
        args = rasterize.parse_args(['myrasterfile.nc', 'ladim.nc', '--max_size', '1e10'])
        assert args.max_size == 1e10
        args_def = rasterize.parse_args(['myrasterfile.nc', 'ladim.nc'])
        assert args_def.max_size == 1e6


class Test_rasterize:
    @pytest.fixture()
    def raster_data(self):
        with nc.Dataset(uuid4().hex, 'w', diskless=True) as dset:
            dset.createDimension('X', 5)
            dset.createVariable('X', np.float32, 'X')[:] = [0, 1, 2, 3, 4]
            dset['X'].bounds = 'X_bounds'

            dset.createDimension('bnd', 2)
            v = dset.createVariable('X_bounds', np.float32, ('X', 'bnd'))
            v[:, 0] = [0, 1, 2, 3, 4]
            v[:, 1] = [1, 2, 3, 4, 6]

            dset.createDimension('farmid', 2)
            dset.createVariable('farmid', np.float32, 'farmid')[:] = [101, 102]
            dset['farmid'].bounds = 'farmid_bounds'

            v = dset.createVariable('farmid_bounds', np.float32, ('farmid', 'bnd'))
            v[:, 0] = [101, 102]
            v[:, 1] = [102, 103]

            dset.createDimension('time', 3)
            dset.createVariable('time', np.int32, 'time')[:] = [0, 1, 2]
            dset['time'].units = 'days since 2000-01-01'
            dset['time'].bounds = 'time_bounds'
            dset['time'].calendar = 'standard'

            v = dset.createVariable('time_bounds', np.int32, ('time', 'bnd'))
            v[:, 0] = [0, 1, 2]
            v[:, 1] = [1, 2, 3]

            yield dset

    def test_writes_to_bincount(self, raster_data, multipart_ladim_nc):
        raster_data.createVariable('bincount', np.int32, ('time', 'farmid', 'X'))[:] = 0
        rasterize.rasterize(raster_data, multipart_ladim_nc[1:])

        assert raster_data['bincount'][:].tolist() == [
            [[0, 2, 0, 1, 0], [0, 0, 1, 0, 0]],
            [[0, 1, 0, 1, 1], [0, 0, 2, 0, 1]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ]

    def test_writes_to_weights(self, raster_data, multipart_ladim_nc):
        raster_data.createVariable('bincount', np.int32, ('time', 'farmid', 'X'))[:] = 0
        raster_data.createVariable('mass', np.int32, ('time', 'farmid', 'X'))[:] = 0
        rasterize.rasterize(raster_data, multipart_ladim_nc[1:])

        assert raster_data['mass'][:].tolist() == [
            [[0, 22, 0, 30,  0], [0, 0, 21, 0,  0]],
            [[0, 11, 0, 31, 50], [0, 0, 42, 0, 40]],
            [[0,  0, 0,  0,  0], [0, 0,  0, 0,  0]],
        ]


class Test_sparse_to_dense_chunks:
    def test_single_chunk_if_unrestricted(self):
        coords = np.array([[1, 1, 3, 2, 3], [1, 2, 5, 4, 4]])
        vals = np.array([1, 2, 3, 4, 5])
        chunks = list(rasterize.sparse_to_dense_chunks(coords, vals))
        assert len(chunks) == 1

    def test_equals_numpy_indirection_when_single_chunk(self):
        coords = np.array([[1, 1, 3, 2, 3], [1, 2, 5, 4, 4]])
        vals = np.array([1, 2, 3, 4, 5])

        expected = np.zeros((4, 6))
        expected[tuple(coords)] = vals

        actual = np.zeros_like(expected)
        chunks = rasterize.sparse_to_dense_chunks(coords, vals)
        for dense, idx in chunks:
            actual[idx] += dense

        assert actual.tolist() == expected.tolist()

    def test_multiple_chunks_if_restricted(self):
        coords = np.array([[1, 1, 1, 2, 3], [1, 4, 5, 4, 2]])
        vals = np.array([1, 2, 3, 4, 5])
        chunks = list(rasterize.sparse_to_dense_chunks(coords, vals, max_size=4))
        assert len(chunks) > 1

    def test_equals_numpy_indirection_when_multiple_chunks(self):
        coords = np.array([[1, 1, 1, 2, 3], [1, 4, 5, 4, 2]])
        vals = np.array([1, 2, 3, 4, 5])

        expected = np.zeros((4, 6))
        expected[tuple(coords)] = vals

        actual = np.zeros_like(expected)
        chunks = rasterize.sparse_to_dense_chunks(coords, vals, max_size=4)
        for dense, idx in chunks:
            actual[idx] += dense

        assert actual.tolist() == expected.tolist()


class Test_sparse_histogram:
    def test_correct_when_unweighted_onedim(self):
        sample = [np.arange(10)]
        bins = [[0, 2, 6, 7]]
        coords, vals = rasterize.sparse_histogram(sample, bins)

        dense = np.zeros(tuple(len(b) - 1 for b in bins))
        dense[tuple(coords)] = vals
        assert dense.tolist() == [2, 4, 1]

    def test_correct_when_weighted_onedim(self):
        sample = [np.arange(10)]
        bins = [[0, 2, 6, 7]]
        weights = np.arange(10)
        coords, vals = rasterize.sparse_histogram(sample, bins, weights)

        dense = np.zeros(tuple(len(b) - 1 for b in bins))
        dense[tuple(coords)] = vals
        assert dense.tolist() == [1, 14, 6]

    def test_correct_when_unweighted_twodim(self):
        sample = [np.arange(10), 10*np.arange(10)]
        bins = [[2, 4, 6], [20, 40, 60, 80]]
        coords, vals = rasterize.sparse_histogram(sample, bins)

        dense = np.zeros(tuple(len(b) - 1 for b in bins))
        dense[tuple(coords)] = vals
        assert dense.tolist() == [[2, 0, 0], [0, 2, 0]]
