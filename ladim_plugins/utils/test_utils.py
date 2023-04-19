from ladim_plugins import utils
import numpy as np
import xarray as xr
import pytest
import sqlite3


class Test_light:
    def test_changes_with_latitude(self):
        Eb = utils.light(time='2000-01-01T12', lon=5, lat=[0, 60])
        assert Eb.round(1).tolist() == [1500.1, 1484.1]

    def test_changes_with_time(self):
        Eb_1 = utils.light(time='2000-01-01T12', lon=5, lat=60).round(1)
        Eb_2 = utils.light(time='2000-01-01T00', lon=5, lat=60)
        assert [Eb_1, Eb_2] == [1484.1, 1.15e-05]

    def test_changes_with_date(self):
        Eb_1 = utils.light(time='2000-01-01T12', lon=5, lat=60).round(1)
        Eb_2 = utils.light(time='2000-06-01T12', lon=5, lat=60).round(1)
        assert [Eb_1, Eb_2] == [1484.1, 1502.4]

    def test_changes_with_depth(self):
        Eb = utils.light(time='2000-01-01T12', lon=5, lat=60, depth=np.array([0, 5, 10]))
        assert Eb.round(1).tolist() == [1484.1, 546.0, 200.8]


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

    @pytest.fixture(scope='class')
    def wgs84_dset(self):
        return xr.Dataset(
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

    @pytest.fixture(scope='class')
    def ortho_dset(self):
        return xr.Dataset(
            data_vars=dict(
                crs=xr.Variable(
                    dims=(), data=0,
                    attrs=dict(
                        grid_mapping_name='orthographic',
                        longitude_of_projection_origin=5,
                        latitude_of_projection_origin=61,
                    ),
                )
            ),
            coords=dict(
                y=xr.Variable(
                    'y', [-120000, 0, 120000],
                    attrs=dict(standard_name='projection_y_coordinate'),
                ),
                x=xr.Variable(
                    'x', [0, 60000],
                    attrs=dict(standard_name='projection_x_coordinate'),
                ),
            )
        )

    def test_adds_bin_edge_info_to_output(self, ladim_dset):
        grid_dset = xr.Dataset(coords=dict(lat=[60, 61, 62], lon=[5, 6]))
        raster = utils.ladim_raster(ladim_dset, grid_dset)
        assert raster.lat.attrs['bounds'] == 'lat_bounds'
        assert raster.lat_bounds.values.tolist() == [
            [59.5, 60.5], [60.5, 61.5], [61.5, 62.5],
        ]

    def test_returns_bincount_when_default_variables(self, ladim_dset):
        grid_dset = xr.Dataset(coords=dict(lat=[60, 61, 62], lon=[5, 6]))
        raster = utils.ladim_raster(ladim_dset, grid_dset)
        assert raster.bincount.values.tolist() == [
            [[2, 1], [0, 1], [0, 0]],
            [[1, 0], [0, 0], [0, 1]],
        ]

    def test_copies_wgs84_georeference_from_grid_dataset(self, ladim_dset, wgs84_dset):
        raster = utils.ladim_raster(ladim_dset, wgs84_dset)
        assert raster.bincount.attrs['grid_mapping'] == 'crs'

    def test_adds_area_info_if_not_present(self, ladim_dset, wgs84_dset):
        raster = utils.ladim_raster(ladim_dset, wgs84_dset)

        assert raster.cell_area.values.round(1).tolist() == [
            [6190827534.7, 6190827534.7],
            [6003046562.4, 6003046562.4],
            [5813411740.2, 5813411740.2],
        ]

    def test_can_use_weights(self, ladim_dset, wgs84_dset):
        ladim_dset['mass'] = xr.Variable(
            dims='particle_instance',
            data=np.arange(10, 16),
            attrs=dict(
                units='kg',
            )
        )

        raster = utils.ladim_raster(ladim_dset, wgs84_dset, weights=(None, 'mass', ))

        assert raster.bincount.values.tolist() == [
            [[2, 1], [0, 1], [0, 0]],
            [[1, 0], [0, 0], [0, 1]],
        ]

        assert raster.mass.values.tolist() == [
            [[21, 12], [0, 13], [0, 0]],
            [[14, 0], [0, 0], [0, 15]],
        ]

        assert raster.mass.attrs['units'] == 'kg'

    def test_can_compute_area_of_orthographic_gridfile(self, ladim_dset, ortho_dset):
        raster = utils.ladim_raster(ladim_dset, ortho_dset)
        assert raster.cell_area.values.tolist() == [[7200000000] * 2] * 3

    def test_can_convert_to_orthographic_projection(self, ladim_dset, ortho_dset):
        raster = utils.ladim_raster(ladim_dset, ortho_dset)

        assert raster.bincount.values.tolist() == [
            [[2, 1], [0, 1], [0, 0]],
            [[1, 0], [0, 0], [0, 1]],
        ]


class Test_converter_sqlite:
    @pytest.fixture(scope='class')
    def ladim_dset(self):
        return xr.Dataset(
            data_vars=dict(
                lon=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
                lat=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
                particle_count=xr.Variable('time', [4, 2]),
                instance_offset=xr.Variable((), 0),
                release_time=xr.Variable('particle', [0, 0, 0, 0]),
                pid=xr.Variable('particle_instance', [0, 1, 2, 3, 1, 2]),
            ),
            coords=dict(
                time=np.array([12345678, 12345679]),
            ),
        )

    def test_adds_tables(self, ladim_dset):

        conn = sqlite3.connect(':memory:')
        utils.to_sqlite(ladim_dset, conn)
        res = conn.execute("SELECT name FROM sqlite_schema WHERE type ='table' AND name NOT LIKE 'sqlite_%';")
        table_names = [n for (n, ) in res.fetchall()]

        assert 'particle' in table_names
        assert 'particle_instance' in table_names

    def test_adds_particle_columns(self, ladim_dset):
        con = sqlite3.connect(':memory:')
        utils.to_sqlite(ladim_dset, con)

        res = con.execute("PRAGMA table_info(particle)")
        col_names = {v[1] for v in res.fetchall()}
        assert col_names == {'release_time', 'pid'}

    def test_adds_instance_columns(self, ladim_dset):
        con = sqlite3.connect(':memory:')
        utils.to_sqlite(ladim_dset, con)

        res = con.execute("PRAGMA table_info(particle_instance)")
        col_names = {v[1] for v in res.fetchall()}
        assert col_names == {'time', 'lat', 'pid', 'lon'}
