from ladim_plugins.release import make_release
from ladim_plugins.release import mkrel
import pytest
import numpy as np


@pytest.fixture()
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
        r = make_release(conf0)
        assert r['Z'] == [0, 0]

        conf0['depth'] = 3
        r = make_release(conf0)
        assert r['Z'] == [3, 3]

        conf0['depth'] = [3, 6]
        conf0['num'] = 3
        r = make_release(conf0)
        assert r['Z'] == [3, 4.5, 6]

    def test_can_print_to_file(self, conf0):
        import io
        handle = io.StringIO()
        conf0['file'] = handle
        make_release(conf0)

        handle.seek(0)
        assert handle.read().replace('\r', '') == (
            '2000-01-01T01:02:03\t5\t60\t0.0\n'
            '2000-01-01T01:02:03\t5\t60\t0.0\n'
        )


class Test_triangulate:
    def test_if_triangle(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        triangles = mkrel.triangulate(coords)
        assert triangles.tolist() == [[[0, 0], [0, 1], [1, 0]]]

    def test_if_clockwise_square(self):
        coords = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        triangles = mkrel.triangulate(coords)
        assert triangles.tolist() == [
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 1], [1, 0]],
        ]

    def test_if_counterclockwise_square(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        triangles = mkrel.triangulate(coords)
        assert triangles.tolist() == [
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [1, 1], [0, 1]],
        ]


class Test_triangle_areas:
    def test_if_single_triangle(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        area = mkrel.triangle_areas(coords)
        assert area.tolist() == 0.5

    def test_if_multiple_triangles(self):
        coords = np.array([
            [[0, 0], [0, 1], [1, 0]],
            [[0, 0], [0, 2], [1, 0]],
        ])
        area = mkrel.triangle_areas(coords)
        assert area.tolist() == [0.5, 1]


class Test_get_polygon_sample_convex:
    def test_all_inside_when_triangle(self):
        coords = np.array([[5, 3], [5, 1], [6, 3]])
        x, y = mkrel.get_polygon_sample_convex(coords, 100)

        assert np.all(x >= 5)
        assert np.all(y <= 3)
        assert np.all(2 * (y - 1) >= (x - 5))

    def test_all_inside_when_rectangle(self):
        coords = np.array([[1, 10], [2, 10], [2, 12], [1, 12]])
        x, y = mkrel.get_polygon_sample_convex(coords, 100)

        assert np.all(x >= 1)
        assert np.all(x <= 2)
        assert np.all(y >= 10)
        assert np.all(y <= 12)

    def test_does_not_work_when_nonconvex_polygon(self):
        coords = np.array([[0, 0], [10, 0], [10, 10], [9, 1]])
        x, y = mkrel.get_polygon_sample_convex(coords, 100)
        is_inside_forbidden_area = (x < 9) & (y > 1)
        assert np.count_nonzero(is_inside_forbidden_area) > 0


class Test_get_polygon_sample_nonconvex:
    def test_all_inside_when_triangle(self):
        coords = np.array([[5, 3], [5, 1], [6, 3]])
        x, y = mkrel.get_polygon_sample_nonconvex(coords, 100)

        assert np.all(x >= 5)
        assert np.all(y <= 3)
        assert np.all(2 * (y - 1) >= (x - 5))

    def test_all_inside_when_rectangle(self):
        coords = np.array([[1, 10], [2, 10], [2, 12], [1, 12]])
        x, y = mkrel.get_polygon_sample_nonconvex(coords, 100)

        assert np.all(x >= 1)
        assert np.all(x <= 2)
        assert np.all(y >= 10)
        assert np.all(y <= 12)

    def test_works_when_nonconvex_polygon(self):
        coords = np.array([[0, 0], [10, 0], [10, 10], [9, 1]])
        x, y = mkrel.get_polygon_sample_nonconvex(coords, 100)
        is_inside_forbidden_area = (x < 9) & (y > 1)
        assert np.count_nonzero(is_inside_forbidden_area) == 0


class Test_is_convex:
    def test_returns_true_if_clockwise_square(self):
        coords = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert mkrel.is_convex(coords)

    def test_returns_true_if_counterclockwise_square(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert mkrel.is_convex(coords)

    def test_returns_false_if_nonconvex_quadrilateral(self):
        coords = np.array([[0, 0], [1, 0], [.1, .1], [0, 1]])
        assert not mkrel.is_convex(coords)
