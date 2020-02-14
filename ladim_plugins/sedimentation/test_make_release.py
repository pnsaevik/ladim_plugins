import numpy as np
from ladim_plugins.sedimentation import make_release
# noinspection PyPackageRequirements
import pandas as pd


class Test_main:
    def test_returns_dataframe_when_empty_config(self):
        config = dict()
        result = make_release.main(**config)
        assert isinstance(result, pd.DataFrame)

    def test_accepts_singular_location(self):
        config = dict(
            location=dict(
                lat=[0],
                lon=[0],
            ),
            num_particles=1,
        )
        result = make_release.main(**config)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_number_of_particles(self):
        config = dict(num_particles=10)
        result = make_release.main(**config)
        assert len(result) == 10


class Test_get_release_data:
    def test_is_homogeneous_table_when_point_input(self):
        data = make_release.get_release_data(
            date='2019-01-02T03:04:05', lat=60, lon=5, z=1,
            distribution='POINT (0 0)', num_particles=10)

        assert list(data.variables) == ['release_time', 'lat', 'lon', 'Z']
        assert list(data.dims) == ['particle']
        assert data.dims['particle'] == 10
        assert data.lat.values.tolist() == [60] * 10
        assert data.lon.values.tolist() == [5] * 10
        assert data.Z.values.tolist() == [1] * 10
        assert data.release_time.values.astype(str).tolist() == \
            ['2019-01-02T03:04:05.000000000'] * 10

    def test_varying_zcoord_when_tuple_z_input(self):
        data = make_release.get_release_data(
            date='2019-01-02T03:04:05', lat=60, lon=5, z=(1, 5),
            distribution='POINT (0 0)', num_particles=10)

        z = data.Z.values

        assert z.min() > 1
        assert z.max() < 5
        assert len(np.unique(z)) == len(z)

    def test_horizontal_distribution_when_polygon_input(self):
        data = make_release.get_release_data(
            date='2019-01-02T03:04:05', lat=60, lon=0, z=0, num_particles=1000,
            distribution='POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))')

        lat_m = (data.lat.values - 60) * 60 * 1852
        lon_m = data.lon.values * 60 * 1852 * 0.5

        assert -1 < lat_m.min() < -0.9
        assert 0.9 < lat_m.max() < 1
        assert -1 < lon_m.min() < -0.9
        assert 0.9 < lon_m.max() < 1


class Test_get_polygon_sample:
    def test_all_inside_when_triangle(self):
        coords = np.array([[5, 3], [5, 1], [6, 3]])
        x, y = make_release.get_polygon_sample(coords, 100)

        assert np.all(x >= 5)
        assert np.all(y <= 3)
        assert np.all(2 * (y - 1) >= (x - 5))

    def test_all_inside_when_rectangle(self):
        coords = np.array([[1, 10], [2, 10], [2, 12], [1, 12]])
        x, y = make_release.get_polygon_sample(coords, 100)

        assert np.all(x >= 1)
        assert np.all(x <= 2)
        assert np.all(y >= 10)
        assert np.all(y <= 12)

    def test_does_not_work_when_nonconvex_polygon(self):
        coords = np.array([[0, 0], [10, 0], [10, 10], [9, 1]])
        x, y = make_release.get_polygon_sample(coords, 100)
        is_inside_forbidden_area = (x < 9) & (y > 1)
        assert np.count_nonzero(is_inside_forbidden_area) > 0


class Test_get_concave_polygon_sample:
    def test_all_inside_when_triangle(self):
        coords = np.array([[5, 3], [5, 1], [6, 3]])
        x, y = make_release.get_concave_polygon_sample(coords, 100)

        assert np.all(x >= 5)
        assert np.all(y <= 3)
        assert np.all(2 * (y - 1) >= (x - 5))

    def test_all_inside_when_rectangle(self):
        coords = np.array([[1, 10], [2, 10], [2, 12], [1, 12]])
        x, y = make_release.get_concave_polygon_sample(coords, 100)

        assert np.all(x >= 1)
        assert np.all(x <= 2)
        assert np.all(y >= 10)
        assert np.all(y <= 12)

    def test_works_when_nonconvex_polygon(self):
        coords = np.array([[0, 0], [10, 0], [10, 10], [9, 1]])
        x, y = make_release.get_concave_polygon_sample(coords, 100)
        is_inside_forbidden_area = (x < 9) & (y > 1)
        assert np.count_nonzero(is_inside_forbidden_area) == 0
