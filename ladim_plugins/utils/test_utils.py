from ladim_plugins import utils
import numpy as np
import xarray as xr
import pytest


class Test_light:
    def test_changes_with_latitude(self):
        Eb = utils.light(time='2000-01-01T12', lon=5, lat=[0, 60])
        assert Eb.tolist() == [1500.0520471376183, 1484.063211499949]

    def test_changes_with_time(self):
        Eb_1 = utils.light(time='2000-01-01T12', lon=5, lat=60)
        Eb_2 = utils.light(time='2000-01-01T00', lon=5, lat=60)
        assert [Eb_1, Eb_2] == [1484.063211499949, 1.15e-05]

    def test_changes_with_date(self):
        Eb_1 = utils.light(time='2000-01-01T12', lon=5, lat=60)
        Eb_2 = utils.light(time='2000-06-01T12', lon=5, lat=60)
        assert [Eb_1, Eb_2] == [1484.063211499949, 1502.4089323926253]

    def test_changes_with_depth(self):
        Eb = utils.light(time='2000-01-01T12', lon=5, lat=60, depth=np.array([0, 5, 10]))
        assert Eb.tolist() == [1484.063211499949, 545.9563449096972, 200.84611506938265]


class Test_density:
    def test_changes_with_temperature(self):
        rho = utils.density(temp=np.array([0, 50]), salt=30)
        assert rho.tolist() == [1024.0715523751858, 1009.9641764883575]

    def test_changes_with_salinity(self):
        rho = utils.density(temp=4, salt=np.array([10, 40]))
        assert rho.tolist() == [1007.9473603468945, 1031.7686242667996]


class Test_viscosity:
    def test_changes_with_temperature(self):
        mu = utils.viscosity(temp=np.array([0, 50]), salt=30)
        assert mu.tolist() == [0.0018605000000000002, 0.0009204999999999999]

    def test_changes_with_salinity(self):
        mu = utils.viscosity(temp=10, salt=np.array([0, 40]))
        assert mu.tolist() == [0.0013235000000000002, 0.0014155000000000003]


class Test_ladim_raster:
    @pytest.fixture(scope='class')
    def ladim_dset(self):
        return xr.Dataset(
            data_vars=dict(
                lon=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
                lat=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
                particle_count=xr.Variable('particle_count', [4, 2]),
            ),
            coords=dict(
                time=np.array(['2000-01-02', '2000-01-03']).astype('datetime64[D]'),
            ),
        )

    def test_returns_bincount_when_default_variables(self, ladim_dset):
        grid_dset = xr.Dataset(coords=dict(lat=[60, 61, 62], lon=[5, 6]))

        raster = utils.ladim_raster(ladim_dset, grid_dset)

        assert raster.bincount.values.tolist() == [
            [[2, 1], [0, 1], [0, 0]],
            [[1, 0], [0, 0], [0, 1]],
        ]

    def test_copies_wgs84_georeference_from_grid_dataset(self, ladim_dset):
        grid_dset = xr.Dataset(
            data_vars=dict(
                crs=xr.Variable(
                    dims=(), data=0,
                    attrs=dict(grid_mapping_name='latitude_longitude'),
                )
            ),
            coords=dict(
                lat=xr.Variable(
                    'lat', [60, 61, 62],
                    attrs=dict(standard_name='latitude'),
                ),
                lon=xr.Variable(
                    'lon', [5, 6],
                    attrs=dict(standard_name='longitude'),
                ),
            )
        )

        raster = utils.ladim_raster(ladim_dset, grid_dset)

        assert raster.bincount.attrs['grid_mapping'] == 'crs'
