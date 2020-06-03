from ladim_plugins.release import make_release
import pytest


@pytest.fixture()
def conf0():
    return dict(
        date='2000-01-01 01:02:03',
        num=2,
        location=[5, 60]
    )


class Test_make_release:
    def test_returns_dict(self, conf0):
        r = make_release(conf0)
        assert isinstance(r, dict)

    def test_returns_correct_keys(self, conf0):
        r = make_release(conf0)
        assert list(r.keys()) == ['release_time', 'lon', 'lat']

    def test_returns_correct_number(self, conf0):
        r = make_release(conf0)
        assert all(len(v) == conf0['num'] for v in r.values())
