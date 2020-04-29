import numpy as np
from ladim_plugins.chemicals import make_release
import pandas as pd


class Test_main:
    def test_returns_dataframe(self):
        config = dict(
            location=dict(lat=1, lon=2, width=100000),
            depth=0,
            release_time='2000-01-01T00:00',
            num_particles=5,
            group_id=0,
        )
        result = make_release.main(config)
        assert isinstance(result, pd.DataFrame)

    def test_start_time_varies_when_range(self):
        config = dict(
            location=dict(lat=1, lon=2, width=100000),
            depth=0,
            release_time=['2000-01-01T00:00', '2000-01-02T00:00'],
            num_particles=5,
            group_id=0,
        )
        result = make_release.main(config)
        time = result.release_time.values
        assert len(np.unique(time)) == config['num_particles']

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
            location=dict(lat=1, lon=2),
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
                lon="1° 12.0' 36.0''"
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
        assert result['Z'].values.tolist() == [5.0, 10.0, 7.5, 0.0, 2.5]

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
            60.26228062814893,
            59.974021504109096,
            60.061176683164156,
            60.382640291567434,
            59.6143322738132,
        ]
        assert result['lon'].values.tolist() == [
            0.11207533991782959,
            0.8380960561199113,
            0.0721929966399305,
            -0.6993553348470959,
            0.08178697665959145,
        ]

    def test_location_when_polygon(self):
        config = dict(
            location=dict(
                lat=[0, 0, 1, 1],
                lon=[0, 1, 1, 0],
            ),
            depth=[0, 0],
            release_time='2000-01-01T00:00',
            num_particles=5,
            group_id=0,
        )
        result = make_release.main(config)
        assert result['lat'].values.tolist() == [
            0.7917250380826646,
            0.4711050802470955,
            0.5680445610939323,
            0.925596638292661,
            0.07103605819788694,
         ]
        assert result['lon'].values.tolist() == [
            0.5623808488506793,
            0.966482131015597,
            0.5401824381239879,
            0.11074060120630969,
            0.5455224229763354,
        ]

    def test_location_when_multipolygon(self):
        config = dict(
            location=dict(
                lat=[[0, 0, 1, 1], [10, 10, 11]],
                lon=[[0, 1, 1, 0], [10, 11, 11]],
            ),
            depth=[0, 0],
            release_time='2000-01-01T00:00',
            num_particles=5,
            group_id=0,
        )
        result = make_release.main(config)
        assert result['lat'].values.tolist() == [
            0.5623808488506793,
            10.033517868984402,
            0.5401824381239879,
            0.11074060120630969,
            0.45447757702366465,
        ]
        assert result['lon'].values.tolist() == [
            0.6458941130666561,
            10.471105080247096,
            0.8917730007820798,
            0.9636627605010293,
            0.6165584811742223,
        ]

    def test_location_when_file(self):
        from pkg_resources import resource_filename, cleanup_resources
        fname = resource_filename('ladim_plugins.chemicals', 'release_area.geojson')
        config = dict(
            location=dict(file=fname),
            depth=[0, 0],
            release_time='2000-01-01T00:00',
            num_particles=10,
            group_id=0,
        )
        result = make_release.main(config)

        cleanup_resources()

        assert result['lat'].values.tolist() == [
            4.2296566196845715, 4.328053483969628, 4.029523923346864,
            4.293874185420884, 4.18931048406682, 4.272949678970935,
            4.163571684849372, 10.167380154452061, 10.22184324905015,
            0.28467408823734275,
        ]
        assert result['lon'].values.tolist() == [
            2.7917250380826646, 2.4711050802470957, 2.431955438906068,
            2.925596638292661, 2.928963941802113, 2.087129299701541,
            2.979781602559674, 10.222711237402478, 10.699994927300079,
            0.4146619399905236,
        ]

    def test_multiple_groups(self):
        config = [
            dict(
                location=dict(lat=1, lon=2),
                depth=[0, 10],
                release_time='2000-01-01T00:00',
                num_particles=2,
                group_id=1,
            ),
            dict(
                location=dict(lat=1, lon=2),
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
                '2000-01-01T00:00',
                '2000-01-01T00:00',
                '2000-01-01T00:00',
                '2000-01-01T00:00',
                '2000-01-01T00:00'
            ],
            'lat': [
                59.99996568023897,
                60.000065584359625,
                61.00002805526148,
                61.00017610643909,
                61.000208421707285
            ],
            'lon': [
                4.0004254257913106,
                4.000277779481143,
                5.0004469904383075,
                5.0001709991110435,
                4.9998984997307225
            ],
            'Z': [10.0, 0.0, 0.0, 5.0, 10.0],
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
