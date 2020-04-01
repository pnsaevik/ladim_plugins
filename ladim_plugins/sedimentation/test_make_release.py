import numpy as np
from ladim_plugins.sedimentation import make_release
# noinspection PyPackageRequirements
import pandas as pd


class Test_main:
    def test_returns_dataframe_when_empty_config(self):
        config = dict()
        result = make_release.main(config)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_number_of_particles(self):
        config = dict(num_particles=10)
        result = make_release.main(config)
        assert len(result) == 10

    def test_correct_latlon_when_given(self):
        config = dict(num_particles=5, location=dict(lat=1, lon=2))
        result = make_release.main(config)
        assert result['lat'].values.tolist() == [1, 1, 1, 1, 1]
        assert result['lon'].values.tolist() == [2, 2, 2, 2, 2]

    def test_correct_latlon_when_given_string_loc(self):
        config = dict(
            num_particles=5, location=dict(
                lat="1° 30.0'", lon="1° 12.0' 36.0''",
            ),
        )
        result = make_release.main(config)
        assert result['lat'].values.tolist() == [1.5] * 5
        assert result['lon'].values.tolist() == [1.21] * 5

    def test_correct_time_when_given(self):
        config = dict(
            num_particles=5, start_time='2000-01-01', stop_time='2000-01-02')
        result = make_release.main(config)
        assert result['release_time'].values.astype(str).tolist() == [
            '2000-01-01T00:00:00.000000000',
            '2000-01-01T06:00:00.000000000',
            '2000-01-01T12:00:00.000000000',
            '2000-01-01T18:00:00.000000000',
            '2000-01-02T00:00:00.000000000',
        ]

    def test_location_when_polygon(self):
        config = dict(num_particles=5, location=dict(
            lat=[0, 0, 1], lon=[0, 1, 1],
        ))
        result = make_release.main(config)
        assert result['lat'].values.tolist() == [
            0.2082749619173354,
            0.5288949197529045,
            0.4319554389060677,
            0.07440336170733897,
            0.07103605819788694,
        ]
        assert result['lon'].values.tolist() == [
            0.6458941130666561,
            0.5624127887373075,
            0.8917730007820798,
            0.9636627605010293,
            0.6165584811742223,
        ]
        assert np.all(result['lon'].values >= result['lat'].values)

    def test_multiple_groups(self):
        config = [
            dict(num_particles=2, group_id=1),
            dict(num_particles=3, group_id=2),
        ]
        result = make_release.main(config)
        assert result['group_id'].values.tolist() == [1, 1, 2, 2, 2]


class Test_to_numeric_latlon:
    def test_correct_if_float(self):
        assert 1.23 == make_release.to_numeric_latlon(1.23)

    def test_correct_if_simple_string(self):
        assert 1.23 == make_release.to_numeric_latlon("1.23")

    def test_correct_if_minutes(self):
        assert 1.5 == make_release.to_numeric_latlon("1° 30.0'")
        assert 1.5 == make_release.to_numeric_latlon("1°30.0'")
        assert 1.5 == make_release.to_numeric_latlon(" 1 ° 30.0 ' ")
        assert 1.5 == make_release.to_numeric_latlon("  1 °30.0  '")

    def test_correct_if_min_and_sec(self):
        assert 1.21 == make_release.to_numeric_latlon("1° 12.0' 36.0''")

    def test_correct_if_min_and_sec_doublequote(self):
        assert 1.21 == make_release.to_numeric_latlon("1° 12.0' 36.0\"")


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


class Test_is_convex:
    def test_returns_true_if_clockwise_square(self):
        coords = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert make_release.is_convex(coords)

    def test_returns_true_if_counterclockwise_square(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert make_release.is_convex(coords)

    def test_returns_false_if_nonconvex_quadrilateral(self):
        coords = np.array([[0, 0], [1, 0], [.1, .1], [0, 1]])
        assert not make_release.is_convex(coords)
