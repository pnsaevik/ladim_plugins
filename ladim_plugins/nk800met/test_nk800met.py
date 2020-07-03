import numpy as np
from ladim_plugins.nk800met import gridforce


class Test_OnlineDatabase_request_dset:
    def test_can_get_metadata(self):
        time = np.datetime64('2020-01-01T12:30')
        dbase = gridforce.OnlineDatabase()
        dset = dbase.get_dset(time)
        assert 'depth' in dset.dimensions.keys()
        assert 'w' in dset.variables.keys()
        assert 'Conventions' in dset.ncattrs()

    def test_can_get_metadata_when_async(self):
        time = np.datetime64('2020-01-01T12:30')
        dbase = gridforce.OnlineDatabase()

        def check(dset):
            assert 'depth' in dset.dimensions.keys()
            assert 'w' in dset.variables.keys()
            assert 'Conventions' in dset.ncattrs()

        th = dbase.request_dset(time, check)
        th.join()
