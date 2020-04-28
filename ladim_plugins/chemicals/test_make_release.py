import numpy as np
from ladim_plugins.chemicals import make_release
import pandas as pd


class Test_main:
    def test_returns_dataframe(self):
        config = dict(
            location=dict(lat=1, lon=2, width=100000),
            depth=[0, 10],
            release_time='2000-01-01T00:00',
            num_particles=5,
            group_id=0,
        )
        result = make_release.main(config)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_number_of_particles(self):
        config = dict(
            location=dict(lat=1, lon=2, width=100000),
            depth=[0, 10],
            release_time='2000-01-01T00:00',
            num_particles=5,
            group_id=0,
        )
        result = make_release.main(config)
        assert len(result) == 5

    def test_correct_latlon(self):
        config = dict(
            location=dict(lat=1, lon=2, width=0),
            depth=[0, 10],
            release_time='2000-01-01T00:00',
            num_particles=5,
            group_id=0,
        )
        result = make_release.main(config)
        assert result['lat'].values.tolist() == [1, 1, 1, 1, 1]
        assert result['lon'].values.tolist() == [2, 2, 2, 2, 2]

    def test_correct_latlon_when_given_string_loc(self):
        config = dict(
            location=dict(
                lat="1° 30.0'",
                lon="1° 12.0' 36.0''",
                width=0,
            ),
            depth=[0, 10],
            release_time='2000-01-01T00:00',
            num_particles=5,
            group_id=0,
        )
        result = make_release.main(config)
        assert result['lat'].values.tolist() == [1.5] * 5
        assert result['lon'].values.tolist() == [1.21] * 5

    def test_correct_depth(self):
        config = dict(
            location=dict(lat=1, lon=2, width=100000),
            depth=[0, 10],
            release_time='2000-01-01T00:00',
            num_particles=5,
            group_id=0,
        )
        result = make_release.main(config)
        assert result['Z'].values.tolist() == [0, 2.5, 5, 7.5, 10]

    def test_location_when_width(self):
        config = dict(
            location=dict(lat=60, lon=0, width=100000),
            depth=[0, 0],
            release_time='2000-01-01T00:00',
            num_particles=5,
            group_id=0,
        )
        result = make_release.main(config)
        assert result['lat'].values.tolist() == [
            59.86883128075036,
            59.94388666278995,
            59.64776945642973,
            59.58313567847057,
            59.95907224908919,
        ]
        assert result['lon'].values.tolist() == [
            0.11207533991782959,
            0.83809605611991130,
            0.07219299663993050,
            -.69935533484709590,
            -.77068973051398990,
        ]

    def test_multiple_groups(self):
        config = [
            dict(
                location=dict(lat=1, lon=2, width=0),
                depth=[0, 10],
                release_time='2000-01-01T00:00',
                num_particles=2,
                group_id=1,
            ),
            dict(
                location=dict(lat=1, lon=2, width=0),
                depth=[0, 10],
                release_time='2000-01-01T00:00',
                num_particles=3,
                group_id=2,
            ),
        ]
        result = make_release.main(config)
        assert result['group_id'].values.tolist() == [1, 1, 2, 2, 2]

    def test_sample_config_file(self):
        from pkg_resources import resource_stream
        import yaml
        package = 'ladim_plugins.chemicals'
        np.random.seed(0)

        with resource_stream(package, 'release.yaml') as config_file:
            config = yaml.safe_load(config_file)

        result = make_release.main(config)
        assert result.to_dict('list') == {
            'release_time': [
                '2000-01-01 00:00',
                '2000-01-01 00:00',
                '2000-01-01 00:00',
                '2000-01-01 00:00',
                '2000-01-01 00:00'
            ],
            'lat': [
                59.99995380437174,
                59.99997982348463,
                61.00002017550346,
                61.000034318039795,
                60.99993441892962
            ],
            'lon': [
                4.0004254257913106,
                4.000277779481143,
                5.0004469904383075,
                5.0001709991110435,
                4.9998984997307225
            ],
            'Z': [0.0, 10.0, 0.0, 5.0, 10.0],
            'group_id': [1, 1, 2, 2, 2]
        }


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


class Test_triangulate:
    def test_if_triangle(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        triangles = make_release.triangulate(coords)
        assert triangles.tolist() == [[[0, 0], [0, 1], [1, 0]]]

    def test_if_clockwise_square(self):
        coords = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        triangles = make_release.triangulate(coords)
        assert triangles.tolist() == [
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 1], [1, 0]],
        ]

    def test_if_counterclockwise_square(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        triangles = make_release.triangulate(coords)
        assert triangles.tolist() == [
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [1, 1], [0, 1]],
        ]


class Test_triangle_areas:
    def test_if_single_triangle(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        area = make_release.triangle_areas(coords)
        assert area.tolist() == 0.5

    def test_if_multiple_triangles(self):
        coords = np.array([
            [[0, 0], [0, 1], [1, 0]],
            [[0, 0], [0, 2], [1, 0]],
        ])
        area = make_release.triangle_areas(coords)
        assert area.tolist() == [0.5, 1]


class Test_get_polygon_sample_convex:
    def test_all_inside_when_triangle(self):
        coords = np.array([[5, 3], [5, 1], [6, 3]])
        x, y = make_release.get_polygon_sample_convex(coords, 100)

        assert np.all(x >= 5)
        assert np.all(y <= 3)
        assert np.all(2 * (y - 1) >= (x - 5))

    def test_all_inside_when_rectangle(self):
        coords = np.array([[1, 10], [2, 10], [2, 12], [1, 12]])
        x, y = make_release.get_polygon_sample_convex(coords, 100)

        assert np.all(x >= 1)
        assert np.all(x <= 2)
        assert np.all(y >= 10)
        assert np.all(y <= 12)

    def test_does_not_work_when_nonconvex_polygon(self):
        coords = np.array([[0, 0], [10, 0], [10, 10], [9, 1]])
        x, y = make_release.get_polygon_sample_convex(coords, 100)
        is_inside_forbidden_area = (x < 9) & (y > 1)
        assert np.count_nonzero(is_inside_forbidden_area) > 0
