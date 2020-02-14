from ladim_plugins.sedimentation import make_release


class Test_main:
    def test_returns_string_when_empty_config(self):
        config = dict()
        result = make_release.main(**config)
        assert isinstance(result, str)
