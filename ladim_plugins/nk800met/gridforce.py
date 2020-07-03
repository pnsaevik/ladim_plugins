import numpy as np
import netCDF4 as nc
import threading
import datetime
from pyproj import CRS, Transformer


class Grid:
    def __init__(self, config):
        server = config['gridforce'].get('input_file', None)
        self.dbase = OnlineDatabase(server)
        dset = self.dbase.get_dset(config['start_time'])

        self._init_proj(dset)
        self._init_gridlimits(dset)

        self.dvars = dict(
            h=dset.variables['h'][:].filled(0),
            depth=dset.variables['depth'][:].filled(0),
            dx=np.diff(dset.variables['X'][:].filled(0)),
            dy=np.diff(dset.variables['Y'][:].filled(0)),
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
        i = np.clip(self.xmin, x.round().astype(int), self.xmax)
        j = np.clip(self.ymin, y.round().astype(int), self.ymax)
        dx = self.dvars['dx'][i]
        dy = self.dvars['dy'][j]
        return dx, dy

    def sample_depth(self, x, y):
        """Return the depth of grid cells"""
        i = x.round().astype(int)
        j = y.round().astype(int)
        return self.dvars['h'][j, i]

    def z2k(self, k):
        depth = self.dvars['depth']
        return np.interp(k, depth, np.arange(len(depth)))

    def ll2xy(self, lon, lat):
        x, y = self.from_wgs84.transform(np.array(lon), np.array(lat))
        dx, dy = self.sample_metric(np.array(0), np.array(0))
        return x / dx, y / dy

    def ingrid(self, x, y):
        return (
            (self.xmin + 0.5 < x)
            & (x < self.xmax - 0.5)
            & (self.ymin + 0.5 < y)
            & (y < self.ymax - 0.5)
        )

    def atsea(self, x, y):
        return self.sample_depth(x, y) > 5


class Forcing:
    def __init__(self, config, grid):
        self.dbase = grid.dbase
        self.z2k = grid.z2k
        self.timeconfig = dict(start=config['start_time'], step=config['dt'])
        self.current_time = None

    def update(self, t):
        self.current_time = self.timeconfig['start'] + np.timedelta64(
            self.timeconfig['step'] * t, 's')

    def velocity(self, x, y, z, tstep):
        dt = np.timedelta64(int(self.timeconfig['step']), 's')
        time = self.current_time + dt * tstep
        k = self.z2k(z)
        u = interp(self.dbase.get_var('u', time), x, y, k)
        v = interp(self.dbase.get_var('v', time), x, y, k)
        return u, v

    def close(self):
        pass


def interp(arr, i, j, k):
    from scipy.ndimage import map_coordinates
    (arr1, arr2), q = arr
    v1 = map_coordinates(arr1, (k, j, i), order=1, prefilter=False)
    v2 = map_coordinates(arr2, (k, j, i), order=1, prefilter=False)
    return v1 * q + v2 * (1 - q)


class OnlineDatabase:
    default_database = "https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.{year:04}{month:02}{day:02}00.nc"

    def __init__(self, pattern=None):
        self._dset_buf = Buffer()
        self._vars_buf = Buffer()
        self.pattern = pattern or self.default_database

    def get_var(self, name, time):
        val_1 = self._get_var(name, time)
        val_2 = self._get_var(name, time + np.timedelta64(1, 'h'))
        w = (time - time.astype('datetime64[h]')) / np.timedelta64(1, 'h')
        return (val_1, val_2), w

    def _get_var(self, name, time):
        dset = self.get_dset(time)
        tidx = time.astype(datetime.datetime).hour
        tstr = str(time.astype('datetime64[h]'))
        key = (name, tstr)
        if key not in self._vars_buf:
            v = dset[name][tidx, ...].filled(0)
            self._vars_buf.push(key, v, tstr)
        return self._vars_buf[key]

    def get_dset(self, time):
        t = time.astype(datetime.datetime)
        tstr = str(time.astype('datetime64[D]'))
        pat = self.pattern.format(year=t.year, month=t.month, day=t.day)
        if pat not in self._dset_buf:
            self._dset_buf.push(pat, nc.Dataset(pat), tstr)
        return self._dset_buf[pat]

    def request_dset(self, time, when_finished):
        def request():
            return when_finished(self.get_dset(time))
        th = threading.Thread(target=request)
        th.start()
        return th

    def close(self):
        for dset in self._dset_buf.values():
            dset.close()
        self._dset_buf = dict()

    def __del__(self):
        self.close()


class Buffer:
    def __init__(self):
        max_frame_idx = 2
        self.fidx_list = [None] * max_frame_idx
        self.buf = dict()
        self.fidx = dict()

    def prune(self):
        self.buf = {k: v for k, v in self.buf.items() if self.fidx[k] in self.fidx_list}
        self.fidx = {k: v for k, v in self.fidx.items() if v in self.fidx_list}

    def push(self, k, v, frame):
        if frame not in self.fidx_list:
            self.fidx_list = self.fidx_list[1:] + [frame]
            self.prune()

        self.buf[k] = v
        self.fidx[k] = frame

    def __getitem__(self, item):
        return self.buf[item]

    def __contains__(self, item):
        return item in self.buf
