import numpy as np
import netCDF4 as nc
import threading
from pyproj import CRS, Transformer


class Grid:
    def __init__(self, config):
        dbase = OnlineDatabase()
        dset = dbase.request_dset(config['start_time'])

        self._init_proj(dset)
        self._init_gridlimits(dset)

        self.dvars = dict(
            h=dset.variables['h'][:],
        )

    def _init_proj(self, dset):
        nk800_proj4str = dset.variables['projection_stere'].proj4
        nk800 = CRS.from_proj4(nk800_proj4str)
        wgs84 = CRS.from_epsg(4326)
        self.to_wgs84 = Transformer.from_crs(nk800, wgs84, always_xy=True)
        self.from_wgs84 = Transformer.from_crs(wgs84, nk800, always_xy=True)

    def _init_gridlimits(self, dset):
        self.xmin = 0
        self.ymin = 0
        self.xmax = dset.dimensions['X'].size
        self.ymax = dset.dimensions['Y'].size

    def sample_metric(self, x, y):
        return np.ones_like(x)*800, np.ones_like(x)*800

    def sample_depth(self, x, y):
        """Return the depth of grid cells"""
        i = x.round().astype(int)
        j = y.round().astype(int)
        return self.dvars['h'][j, i]

    def ll2xy(self, lon, lat):
        x, y = self.from_wgs84.transform(np.array(lon), np.array(lat))
        return x / 800, y / 800

    def ingrid(self, x, y):
        return np.ones_like(x, dtype=bool)

    def atsea(self, x, y):
        return np.ones_like(x, dtype=bool)


class Forcing:
    def __init__(self, config, grid):
        pass

    def update(self, t):
        pass

    def velocity(self, x, y, z, tstep):
        return np.zeros_like(x), np.zeros_like(x)

    def close(self):
        pass


class OnlineDatabase:
    def __init__(self):
        self._dset_buf = dict()
        self.pattern = "https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.{year:04}{month:02}{day:02}00.nc"

    def _get_pattern(self, time):
        import datetime
        t = np.datetime64(time).astype(datetime.datetime)  # type: datetime.datetime
        return self.pattern.format(year=t.year, month=t.month, day=t.day)

    def get_dset(self, time):
        pat = self._get_pattern(time)
        if pat not in self._dset_buf:
            self._dset_buf[pat] = nc.Dataset(pat)
        return self._dset_buf[pat]

    def request_dset(self, time, when_finished=None):
        if when_finished is None:
            return self.get_dset(time)

        pat = self._get_pattern(time)

        if pat in self._dset_buf:
            def request():
                d = self._dset_buf[pat]
                when_finished(d)

        else:
            def request():
                d = nc.Dataset(pat)
                self._dset_buf[pat] = d
                when_finished(d)

        th = threading.Thread(target=request)
        th.start()
        return th

    def request_var(self, name, time, idx):
        pass

    def close(self):
        for dset in self._dset_buf.values():
            dset.close()
        self._dset_buf = dict()

    def __del__(self):
        self.close()
