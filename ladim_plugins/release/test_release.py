from ladim_plugins.release import make_release
from ladim_plugins.release import makrel
import pytest
import numpy as np


@pytest.fixture(scope='function')
def conf0() -> dict:
    return dict(
        date='2000-01-01 01:02:03',
        num=2,
        location=[5, 60]
    )


class Test_make_release:
    def test_returns_dict_of_lists(self, conf0):
        r = make_release(conf0)
        assert isinstance(r, dict)
        assert all(isinstance(v, list) for v in r.values())

    def test_returns_correct_keys(self, conf0):
        r = make_release(conf0)
        assert list(r.keys()) == ['release_time', 'lon', 'lat', 'Z']

    def test_returns_correct_number(self, conf0):
        r = make_release(conf0)
        assert all(len(v) == conf0['num'] for v in r.values())

    def test_accepts_stream(self, conf0):
        import io
        s = '{"date": "2000-01-01 01:02:03", "num": 2, "location": [5, 60]}'
        r1 = make_release(io.StringIO(s))
        r2 = make_release(conf0)
        assert r1 == r2

    def test_throws_FileNotFound_if_invalid_filename(self):
        with pytest.raises(FileNotFoundError):
            make_release("this_is_no_file_name")

    def test_can_add_attributes(self, conf0):
        conf0['attrs'] = dict(
            first=0,
            second=[1, 2],
            third=lambda num: np.arange(num) + 10,
            fourth="numpy.arange",
        )

        r = make_release(conf0)

        assert list(r.keys()) == ['release_time', 'lon', 'lat', 'Z', 'first',
                                  'second', 'third', 'fourth']
        assert r['first'] == [0, 0]
        assert r['second'] == [1, 2]
        assert r['third'] == [10, 11]
        assert r['fourth'] == [0, 1]

    def test_can_change_sortorder(self, conf0):
        conf0['attrs'] = dict(first=0, second=1, third=2)
        conf0['columns'] = ['lon', 'lat', 'first', 'third', 'Z']

        r = make_release(conf0)

        assert list(r.keys()) == conf0['columns']
        assert r['first'] == [0, 0]
        assert r['third'] == [2, 2]

    def test_accepts_numpy_date(self, conf0):
        r0 = make_release(conf0)
        conf0['date'] = np.datetime64(conf0['date'])
        r1 = make_release(conf0)
        assert r0 == r1

    def test_can_return_date_range(self, conf0):
        conf0['num'] = 7
        conf0['date'] = ['2000-01-01 01:00', '2000-01-01 02:00']
        r = make_release(conf0)
        assert r['release_time'] == [
            '2000-01-01T01:00:00',
            '2000-01-01T01:10:00',
            '2000-01-01T01:20:00',
            '2000-01-01T01:30:00',
            '2000-01-01T01:40:00',
            '2000-01-01T01:50:00',
            '2000-01-01T02:00:00',
        ]

    def test_can_set_depth(self, conf0):
        np.random.seed(0)

        r = make_release(conf0)
        assert r['Z'] == [0, 0]

        conf0['depth'] = 3
        r = make_release(conf0)
        assert r['Z'] == [3, 3]

        conf0['depth'] = [3, 6]
        conf0['num'] = 3
        r = make_release(conf0)
        assert r['Z'] == [6, 3, 4.5]

    def test_can_print_to_file(self, conf0):
        import io
        handle = io.StringIO()
        make_release(conf0, handle)

        handle.seek(0)
        assert handle.read().replace('\r', '') == (
            '2000-01-01T01:02:03\t5\t60\t0.0\n'
            '2000-01-01T01:02:03\t5\t60\t0.0\n'
        )

    def test_can_use_polygon_as_sampling_area(self, conf0):
        np.random.seed(0)

        conf0['num'] = 5
        conf0['location'] = [[1, 0, 1], [0, 0, 1]]

        r = make_release(conf0)

        assert r['lat'] == [
            0.3541058869333439,
            0.4375872112626925,
            0.10822699921792023,
            0.03633723949897072,
            0.3834415188257777,
        ]
        assert r['lon'] == [
            0.5623808488506793,
            0.966482131015597,
            0.5401824381239879,
            0.11074060120630969,
            0.45447757702366465,
        ]
        assert all(lon >= lat for lon, lat in zip(r['lon'], r['lat']))

    def test_can_use_multipolygon_as_sampling_area(self, conf0):
        lat = [[0, 0, 1, 1], [10, 10, 11]]
        lon = [[0, 1, 1, 0], [10, 11, 11]]
        conf0['location'] = [lon, lat]
        conf0['num'] = 5

        result = make_release(conf0)
        assert result['lat'] == [
            10.279093103873562,
            0.6788795301189603,
            10.194233074072574,
            0.7683521354018671,
            10.27429140657797,
        ]
        assert result['lon'] == [
            10.758615624322356,
            0.10590760718779213,
            10.473600419346658,
            0.4179802079248929,
            10.736918177128958,
        ]

    def test_can_use_geojson_as_sampling_area(self, conf0):
        geojson = """
{
    "type": "FeatureCollection",
    "name": "test",
    "crs": {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
    },
    "features": [
        {
            "type": "Feature",
            "properties": { },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [ [ [ 0, 0 ], [ 0, 1 ], [ 1, 1 ], [ 1, 0 ], [ 0, 0 ] ] ],
                    [ [ [ 10, 10 ], [ 11, 10 ], [ 11, 11], [ 10, 10 ] ] ]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": { },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [ [ [ 2, 4 ], [ 2, 5 ], [ 3, 5 ], [ 3, 4 ], [ 2, 4 ] ] ]
                ]
            }
        }
    ]
}
"""
        import io
        geojson_file = io.StringIO(geojson)

        conf0['location'] = geojson_file
        conf0['num'] = 10
        result = make_release(conf0)

        assert result['lat'] == [
            0.2881989094015015,
            0.8337908471860672,
            0.05999750359194067,
            0.16505472944482502,
            10.392169331284531,
            4.948557330140221,
            4.623566958688373,
            10.365725942042666,
            0.7238489893061836,
            10.347209682994508,
        ]
        assert result['lon'] == [
            0.6350588736035638,
            0.004700432322112369,
            0.4181496705614657,
            0.4143685882263688,
            10.917471828996119,
            2.6749527709916476,
            2.0384254264727346,
            10.690973619783644,
            0.6827982579307039,
            10.5688642009686,
        ]

    def test_can_use_offset_polygon_as_sampling_area(self, conf0):
        np.random.seed(0)

        conf0['location'] = dict(
            center=[0, 60],
            offset=[
                [-50000, 50000, 50000, -50000],
                [-50000, -50000, 50000, 50000],
            ]
        )
        conf0['num'] = 5

        result = make_release(conf0)

        assert result['lat'] == [
            60.26228062814893,
            59.974021504109096,
            60.061176683164156,
            60.382640291567434,
            59.6143322738132,
        ]
        assert result['lon'] == [
            0.11207533991782959,
            0.8380960561199113,
            0.0721929966399305,
            -0.6993553348470959,
            0.08178697665959145,
        ]

    def test_accepts_list_of_configs(self, conf0):
        r = make_release([conf0, conf0])
        assert len(r['release_time']) == 4


class Test_triangulate:
    def test_if_triangle(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        triangles = makrel.triangulate(coords)
        assert triangles.tolist() == [[[0, 0], [0, 1], [1, 0]]]

    def test_if_clockwise_square(self):
        coords = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        triangles = makrel.triangulate(coords)
        assert triangles.tolist() == [
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 1], [1, 0]],
        ]

    def test_if_counterclockwise_square(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        triangles = makrel.triangulate(coords)
        assert triangles.tolist() == [
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [1, 1], [0, 1]],
        ]


class Test_triangle_areas:
    def test_if_single_triangle(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        area = makrel.triangle_areas(coords)
        assert area.tolist() == 0.5

    def test_if_multiple_triangles(self):
        coords = np.array([
            [[0, 0], [0, 1], [1, 0]],
            [[0, 0], [0, 2], [1, 0]],
        ])
        area = makrel.triangle_areas(coords)
        assert area.tolist() == [0.5, 1]


class Test_get_polygon_sample_convex:
    def test_all_inside_when_triangle(self):
        coords = np.array([[5, 3], [5, 1], [6, 3]])
        x, y = makrel.get_polygon_sample_convex(coords, 100)

        assert np.all(x >= 5)
        assert np.all(y <= 3)
        assert np.all(2 * (y - 1) >= (x - 5))

    def test_all_inside_when_rectangle(self):
        coords = np.array([[1, 10], [2, 10], [2, 12], [1, 12]])
        x, y = makrel.get_polygon_sample_convex(coords, 100)

        assert np.all(x >= 1)
        assert np.all(x <= 2)
        assert np.all(y >= 10)
        assert np.all(y <= 12)

    def test_does_not_work_when_nonconvex_polygon(self):
        coords = np.array([[0, 0], [10, 0], [10, 10], [9, 1]])
        x, y = makrel.get_polygon_sample_convex(coords, 100)
        is_inside_forbidden_area = (x < 9) & (y > 1)
        assert np.count_nonzero(is_inside_forbidden_area) > 0


class Test_get_polygon_sample_nonconvex:
    def test_all_inside_when_triangle(self):
        coords = np.array([[5, 3], [5, 1], [6, 3]])
        x, y = makrel.get_polygon_sample_nonconvex(coords, 100)

        assert np.all(x >= 5)
        assert np.all(y <= 3)
        assert np.all(2 * (y - 1) >= (x - 5))

    def test_all_inside_when_rectangle(self):
        coords = np.array([[1, 10], [2, 10], [2, 12], [1, 12]])
        x, y = makrel.get_polygon_sample_nonconvex(coords, 100)

        assert np.all(x >= 1)
        assert np.all(x <= 2)
        assert np.all(y >= 10)
        assert np.all(y <= 12)

    def test_works_when_nonconvex_polygon(self):
        coords = np.array([[0, 0], [10, 0], [10, 10], [9, 1]])
        x, y = makrel.get_polygon_sample_nonconvex(coords, 100)
        is_inside_forbidden_area = (x < 9) & (y > 1)
        assert np.count_nonzero(is_inside_forbidden_area) == 0


class Test_is_convex:
    def test_returns_true_if_clockwise_square(self):
        coords = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert makrel.is_convex(coords)

    def test_returns_true_if_counterclockwise_square(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert makrel.is_convex(coords)

    def test_returns_false_if_nonconvex_quadrilateral(self):
        coords = np.array([[0, 0], [1, 0], [.1, .1], [0, 1]])
        assert not makrel.is_convex(coords)


class Test_metric_degree_conversion:
    def test_metric_to_degrees_correct_when_lat_60(self):
        dx = 1000
        dy = 2000
        lat = 60
        dlon, dlat = makrel.metric_diff_to_degrees(dx, dy, lat)
        assert (dlon, dlat) == (0.01796630568239042, 0.017981358739225493)

    def test_degrees_to_metric_correct_when_lat_60(self):
        dlon = 0.02
        dlat = 0.01
        lat = 60
        dx, dy = makrel.degree_diff_to_metric(dlon, dlat, lat)
        assert (dx, dy) == (1113.194907932736, 1112.2629991453834)

    def test_back_and_forth_is_identity_when_lat_60(self):
        dx = 1000
        dy = 2000
        lat = 60
        dlon, dlat = makrel.metric_diff_to_degrees(dx, dy, lat)
        dx2, dy2 = makrel.degree_diff_to_metric(dlon, dlat, lat)

        assert np.isclose(dx, dx2)
        assert np.isclose(dy, dy2)
