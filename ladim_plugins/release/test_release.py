from ladim_plugins.release import make_release
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
