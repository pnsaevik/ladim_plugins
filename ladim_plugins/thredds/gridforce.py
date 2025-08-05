import numpy as np
from netCDF4 import Dataset, num2date
import glob
import datetime
import threading
from ladim.sample import sample2D, bilin_inv
import logging


logger = logging.getLogger(__name__)


class Grid:
    def __init__(self, config):

        logger.info("Initializing ROMS-type grid object")

        # Get url of first input file
        pattern = config['gridforce'].get('input_file', None)
        t = config['start_time'].astype(datetime.datetime)
        grid_file = pattern.format(year=t.year, month=t.month, day=t.day)

        # Open file
        try:
            ncid = Dataset(grid_file)
            ncid.set_auto_mask(False)
        except OSError:
            logger.error("Could not open grid file " + grid_file)
            raise SystemExit(1)

        # Subgrid
        jmax, imax = ncid.variables["h"].shape
        whole_grid = [0, imax, 0, jmax]
        if config["gridforce"].get('subgrid', None):
            limits = list(config["gridforce"]["subgrid"])
        else:
            limits = whole_grid

        # Allow None if no imposed limitation
        for ind, val in enumerate(limits):
            if val is None:
                limits[ind] = whole_grid[ind]

        self.i0, self.i1, self.j0, self.j1 = limits
        self.imax = self.i1 - self.i0
        self.jmax = self.j1 - self.j0

        # Limits for where velocities are defined
        self.xmin = float(self.i0)
        self.xmax = float(self.i1 - 1)
        self.ymin = float(self.j0)
        self.ymax = float(self.j1 - 1)

        # Slices
        #   rho-points
        self.I = slice(self.i0, self.i1)
        self.J = slice(self.j0, self.j1)
        #   U and V-points
        self.Iu = slice(self.i0 - 1, self.i1)
        self.Ju = self.J
        self.Iv = self.I
        self.Jv = slice(self.j0 - 1, self.j1)

        # Vertical grid

        if "Vinfo" in config["gridforce"]:
            Vinfo = config["gridforce"]["Vinfo"]
            self.N = Vinfo["N"]
            self.hc = Vinfo["hc"]
            self.Vstretching = Vinfo.get("Vstretching", 1)
            self.Vtransform = Vinfo.get("Vtransform", 1)
            self.Cs_r = s_stretch(
                self.N,
                Vinfo["theta_s"],
                Vinfo["theta_b"],
                stagger="rho",
                Vstretching=self.Vstretching,
            )
            self.Cs_w = s_stretch(
                self.N,
                Vinfo["theta_s"],
                Vinfo["theta_b"],
                stagger="w",
                Vstretching=self.Vstretching,
            )

        else:
            self.hc = ncid.variables["hc"].getValue()
            self.Cs_r = ncid.variables["Cs_r"][:]
            self.Cs_w = ncid.variables["Cs_w"][:]
            self.N = len(self.Cs_r)
            # Vertical transform
            try:
                self.Vtransform = ncid.variables["Vtransform"].getValue()
            except KeyError:
                self.Vtransform = 1  # Default = old way

        # Read some variables
        self.H = ncid.variables["h"][self.J, self.I]
        self.M = ncid.variables["mask_rho"][self.J, self.I].astype(int)
        # self.Mu = ncid.variables['mask_u'][self.Ju, self.Iu]
        # self.Mv = ncid.variables['mask_v'][self.Jv, self.Iv]
        self.dx = 1.0 / ncid.variables["pm"][self.J, self.I]
        self.dy = 1.0 / ncid.variables["pn"][self.J, self.I]
        self.lon = ncid.variables["lon_rho"][self.J, self.I]
        self.lat = ncid.variables["lat_rho"][self.J, self.I]
        self.angle = ncid.variables["angle"][self.J, self.I]

        self.z_r = sdepth(
            self.H, self.hc, self.Cs_r, stagger="rho", Vtransform=self.Vtransform
        )
        self.z_w = sdepth(
            self.H, self.hc, self.Cs_w, stagger="w", Vtransform=self.Vtransform
        )

        # Land masks at u- and v-points
        M = self.M
        Mu = np.zeros((self.jmax, self.imax + 1), dtype=int)
        Mu[:, 1:-1] = M[:, :-1] * M[:, 1:]
        Mu[:, 0] = M[:, 0]
        Mu[:, -1] = M[:, -1]
        self.Mu = Mu

        Mv = np.zeros((self.jmax + 1, self.imax), dtype=int)
        Mv[1:-1, :] = M[:-1, :] * M[1:, :]
        Mv[0, :] = M[0, :]
        Mv[-1, :] = M[-1, :]
        self.Mv = Mv

        # Close the file(s)
        ncid.close()

    def sample_metric(self, X, Y):
        """Sample the metric coefficients

        Changes slowly, so using nearest neighbour
        """
        I = X.round().astype(int) - self.i0
        J = Y.round().astype(int) - self.j0

        # Constrain to valid indices
        I = np.minimum(np.maximum(I, 0), self.dx.shape[-1] - 2)
        J = np.minimum(np.maximum(J, 0), self.dx.shape[-2] - 2)

        # Metric is conform for PolarStereographic
        A = self.dx[J, I]
        return A, A

    def sample_depth(self, X, Y):
        """Return the depth of grid cells"""
        I = X.round().astype(int) - self.i0
        J = Y.round().astype(int) - self.j0
        I = np.minimum(np.maximum(I, 0), self.H.shape[1] - 1)
        J = np.minimum(np.maximum(J, 0), self.H.shape[0] - 1)
        return self.H[J, I]

    def lonlat(self, X, Y, method="bilinear"):
        """Return the longitude and latitude from grid coordinates"""
        if method == "bilinear":  # More accurate
            return self.xy2ll(X, Y)
        # else: containing grid cell, less accurate
        I = X.round().astype("int") - self.i0
        J = Y.round().astype("int") - self.j0
        I = np.minimum(np.maximum(I, 0), self.lon.shape[1] - 1)
        J = np.minimum(np.maximum(J, 0), self.lon.shape[0] - 1)
        return self.lon[J, I], self.lat[J, I]

    def ingrid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Returns True for points inside the subgrid"""
        return (
            (self.xmin + 0.5 < X)
            & (X < self.xmax - 0.5)
            & (self.ymin + 0.5 < Y)
            & (Y < self.ymax - 0.5)
        )

    def onland(self, X, Y):
        """Returns True for points on land"""
        I = X.round().astype(int) - self.i0
        J = Y.round().astype(int) - self.j0

        # Constrain to valid indices
        I = np.minimum(np.maximum(I, 0), self.M.shape[-1] - 1)
        J = np.minimum(np.maximum(J, 0), self.M.shape[-2] - 1)

        return self.M[J, I] < 1

    # Error if point outside
    def atsea(self, X, Y):
        """Returns True for points at sea"""
        I = X.round().astype(int) - self.i0
        J = Y.round().astype(int) - self.j0

        # Constrain to valid indices
        I = np.minimum(np.maximum(I, 0), self.M.shape[-1] - 1)
        J = np.minimum(np.maximum(J, 0), self.M.shape[-2] - 1)

        return self.M[J, I] > 0

    def xy2ll(self, X, Y):
        return (
            sample2D(self.lon, X - self.i0, Y - self.j0),
            sample2D(self.lat, X - self.i0, Y - self.j0),
        )

    def ll2xy(self, lon, lat):
        Y, X = bilin_inv(lon, lat, self.lon, self.lat)
        return X + self.i0, Y + self.j0


def get_fractional_index(sorted_array, search_value):
    idx_max = len(sorted_array)
    lower_unclipped = np.sum(sorted_array <= search_value) - 1
    lower = np.minimum(idx_max - 2, np.maximum(0, lower_unclipped))
    floor = sorted_array[lower]
    ceil = sorted_array[lower + 1]
    frac_unclipped = (search_value - floor) / (ceil - floor)
    frac = np.minimum(1, np.maximum(0, frac_unclipped))
    return lower + frac


def timestring_formatter(pattern, time):
    t = np.datetime64(time).astype(datetime.datetime)
    return pattern.format(year=t.year, month=t.month, day=t.day)


class ThreddsSource:
    def __init__(self, url_pattern, url_first, blocks):
        self.url_pattern = url_pattern
        self.i_block_offset = blocks['i']['offset']
        self.i_block_size = blocks['i']['size']
        self.j_block_offset = blocks['j']['offset']
        self.j_block_size = blocks['j']['size']

        with Dataset(url_first) as dset:
            first_times = self._load_times(dset)
            if self.i_block_size < 0:
                self.i_block_size = dset.dimensions['xi_rho'].size - self.i_block_offset
            if self.j_block_size < 0:
                self.j_block_size = dset.dimensions['eta_rho'].size - self.j_block_offset
            self.k_block_size = dset.dimensions['s_rho'].size

        self.start_time = first_times[0]
        self.timestep_size = first_times[1] - first_times[0]

    @staticmethod
    def _load_times(dset):
        dset['ocean_time'].set_auto_mask(False)
        return num2date(
            times=dset['ocean_time'][:],
            units=dset['ocean_time'].units,
            calendar=getattr(dset['ocean_time'], 'calendar', 'standard'),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        ).astype('datetime64[us]')

    def url(self, time):
        return timestring_formatter(self.url_pattern, time)

    def load_block(self, i_block, j_block, time_block, varname):
        i0 = self.i_block_offset + self.i_block_size * i_block
        j0 = self.j_block_offset + self.j_block_size * j_block
        i1 = i0 + self.i_block_size
        j1 = j0 + self.j_block_size

        time = self.start_time + self.timestep_size * time_block
        with Dataset(self.url(time)) as dset:
            times = self._load_times(dset)
            l0 = int(get_fractional_index(times, time))
            data = dset[varname][l0, :, j0:j1, i0:i1]
            if hasattr(data, 'filled'):
                data = data.filled(0)
        return data


class FieldCache:
    def __init__(self, src: ThreddsSource, varnames):
        self.source = src
        self.varnames = list(set(varnames))
        self.cache = dict()

    def prepare(self, i, j, k, l):
        # The function loads required blocks into memory
        # At present, the ijk coordinates are disregarded

        # Delete old entries
        for time_block in self.cache.keys():
            if time_block < int(l):
                del self.cache[time_block]

        # Load new entries
        for time_block in [int(l), int(l) + 1]:
            if time_block in self.cache:
                continue

            data = []
            for v in self.varnames:
                data.append(self.source.load_block(0, 0, time_block, v))

            self.cache[time_block] = np.stack(data, dtype='float32')

    def sample(self, i, j, k, l, field):
        i = np.asarray(i)
        j = np.asarray(j)
        k = np.asarray(k)
        l = np.asarray(l)
        v = np.nonzero(np.equal(self.varnames, field))[0][0]

        interp_dims = ''
        if np.issubdtype(i.dtype, np.floating):
            interp_dims += 'x'
        if np.issubdtype(j.dtype, np.floating):
            interp_dims += 'y'
        if np.issubdtype(k.dtype, np.floating):
            interp_dims += 'z'
        if np.issubdtype(l.dtype, np.floating):
            interp_dims += 't'

        if interp_dims == 't':
            func = self._sample_t
        elif interp_dims == 'xt':
            func = self._sample_xt
        elif interp_dims == 'yt':
            func = self._sample_yt
        else:
            raise NotImplementedError()

        return func(v, l, k, j, i)

    def _sample_t(self, v, l, k, j, i):
        l0 = np.int32(l)
        ldiff = np.clip(l - l0, 0, 1)
        data0 = self.cache[l0][v, k, j, i]
        data1 = self.cache[l0 + 1][v, k, j, i]
        return data0 * (1 - ldiff) + data1 * ldiff

    def _sample_xt(self, v, l, k, j, i):
        l0 = np.int32(l)
        _, _, _, i_max = self.cache[l0].shape
        i0 = np.int32(i).clip(0, i_max - 2)

        ldiff = np.clip(l - l0, 0, 1)
        idiff = np.clip(i - i0, 0, 1)

        data00 = self.cache[l0][v, k, j, i0]
        data01 = self.cache[l0][v, k, j, i0 + 1]
        data0_ = data00 * (1 - idiff) + data01 * idiff

        data10 = self.cache[l0 + 1][v, k, j, i0]
        data11 = self.cache[l0 + 1][v, k, j, i0 + 1]
        data1_ = data10 * (1 - idiff) + data11 * idiff

        data = data0_ * (1 - ldiff) + data1_ * ldiff

        return data

    def _sample_yt(self, v, l, k, j, i):
        l0 = np.int32(l)
        _, _, j_max, _ = self.cache[l0].shape
        j0 = np.int32(j).clip(0, j_max - 2)

        ldiff = np.clip(l - l0, 0, 1)
        jdiff = np.clip(j - j0, 0, 1)

        data00 = self.cache[l0][v, k, j0, i]
        data01 = self.cache[l0][v, k, j0 + 1, i]
        data0_ = data00 * (1 - jdiff) + data01 * jdiff

        data10 = self.cache[l0 + 1][v, k, j0, i]
        data11 = self.cache[l0 + 1][v, k, j0 + 1, i]
        data1_ = data10 * (1 - jdiff) + data11 * jdiff

        data = data0_ * (1 - ldiff) + data1_ * ldiff

        return data


class Transformer:
    def __init__(
            self, z_rho, lat_rho, lon_rho, i0, j0, start_time, dt,
            field_start_time, field_dt):
        """
        Store parameters required to compute transforms

        :param z_rho: Depth array (s_rho, eta_rho, xi_rho)
        :param lat_rho: Latitude array (eta_rho, xi_rho)
        :param lon_rho: Longitude array (eta_rho, xi_rho)
        :param i0: xi-direction offset for arrays
        :param j0: eta-direction offset for arrays
        :param start_time: Global time corresponding to model_time = 0
        :param dt: Difference between model_time = 0 and model_time = 1
        :param field_start_time: Global time corresponding to field_time = 0
        :param field_dt: Difference between field_time = 0 and field_time = 1
        """
        self.field_dt = field_dt
        self.z_rho = z_rho
        self.i0 = i0
        self.j0 = j0
        self.lat_rho = lat_rho
        self.lon_rho = lon_rho

        self._t2l_offset = (start_time - field_start_time) / field_dt
        self._t2l_scale = dt / field_dt

    def field_indices(self, x, y, z, t):
        # xyzt denote model coordinates
        # ijkl denote field indices (fractional)
        i = np.asarray(x) - self.i0
        j = np.asarray(y) - self.j0
        k = z2s_frac(self.z_rho, i, j, np.asarray(z))
        l = self._t2l_offset + self._t2l_scale * np.asarray(t)

        return i, j, k, l


class Forcing:
    def __init__(self, config, _):
        logger.info("Initiating forcing")

        # Initialize the Thredds source
        subgrid = config['gridforce'].get('subgrid', [0, None, 0, None])
        url_pattern = config['gridforce']['input_file']
        self.source = ThreddsSource(
            url_pattern=url_pattern,
            url_first=timestring_formatter(url_pattern, config['gridforce']['start_time']),
            blocks=dict(
                i=dict(offset=subgrid[0], size=-1 if subgrid[1] is None else subgrid[1] - subgrid[0]),
                j=dict(offset=subgrid[2], size=-1 if subgrid[3] is None else subgrid[3] - subgrid[2]),
            )
        )

        self.cache = FieldCache(
            src=self.source,
            varnames={'u', 'v'}.union(config['ibm_forcing']),
        )

        grid = Grid(config)
        self.transformer = Transformer(
            z_rho=grid.z_r,
            lat_rho=grid.lat,
            lon_rho=grid.lon,
            i0=grid.i0,
            j0=grid.j0,
            start_time=np.datetime64(config['gridforce']['start_time']),
            dt=np.timedelta64(config['dt'], 's'),
            field_start_time=self.source.start_time,
            field_dt=self.source.timestep_size,
        )

        self._current_time_step = None
        self._grid = grid

    def update(self, t):
        i, j, k, l = self.transformer.field_indices([0], [0], [0], t)
        self.cache.prepare(i, j, k, l)
        self._current_time_step = t
        return

    def close(self):
        pass

    def velocity(self, X, Y, Z, tstep=0):
        to_ijkl = self.transformer.field_indices
        i, j, k, l = to_ijkl(X, Y, Z, self._current_time_step + tstep)
        u = self.cache.sample(
            i - 0.5,
            np.int32(j + 0.5),
            np.int32(k + 0.5),
            l,
            field='u',
        )
        v = self.cache.sample(
            np.int32(i + 0.5),
            j - 0.5,
            np.int32(k + 0.5),
            l,
            field='v',
        )
        return u, v

    def field(self, X, Y, Z, name):
        to_ijkl = self.transformer.field_indices
        i, j, k, l = to_ijkl(X, Y, Z, self._current_time_step)
        return self.cache.sample(
            np.int32(i + 0.5),
            np.int32(j + 0.5),
            np.int32(k + 0.5),
            l,
            field=name,
        )


# ---------------------------------------------
#      Low-level vertical functions
#      more or less from the roppy package
#      https://github.com/bjornaa/roppy
# ----------------------------------------------

def s_stretch(N, theta_s, theta_b, stagger="rho", Vstretching=1):
    """Compute a s-level stretching array

    *N* : Number of vertical levels

    *theta_s* : Surface stretching factor

    *theta_b* : Bottom stretching factor

    *stagger* : "rho"|"w"

    *Vstretching* : 1|2|3|4|5

    """

    # if stagger == "rho":
    #     S = -1.0 + (0.5 + np.arange(N)) / N
    # elif stagger == "w":
    #     S = np.linspace(-1.0, 0.0, N + 1)
    if stagger == "rho":
        K = np.arange(0.5, N)
    elif stagger == "w":
        K = np.arange(N + 1)
    else:
        raise ValueError("stagger must be 'rho' or 'w'")
    S = -1 + K / N

    if Vstretching == 1:
        cff1 = 1.0 / np.sinh(theta_s)
        cff2 = 0.5 / np.tanh(0.5 * theta_s)
        return (1.0 - theta_b) * cff1 * np.sinh(theta_s * S) + theta_b * (
            cff2 * np.tanh(theta_s * (S + 0.5)) - 0.5
        )

    elif Vstretching == 2:
        a, b = 1.0, 1.0
        Csur = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        Cbot = np.sinh(theta_b * (S + 1)) / np.sinh(theta_b) - 1
        mu = (S + 1) ** a * (1 + (a / b) * (1 - (S + 1) ** b))
        return mu * Csur + (1 - mu) * Cbot

    elif Vstretching == 3:
        gamma_ = 3.0
        Csur = -np.log(np.cosh(gamma_ * (-S) ** theta_s)) / np.log(np.cosh(gamma_))
        Cbot = (
            np.log(np.cosh(gamma_ * (S + 1) ** theta_b)) / np.log(np.cosh(gamma_)) - 1
        )
        mu = 0.5 * (1 - np.tanh(gamma_ * (S + 0.5)))
        return mu * Csur + (1 - mu) * Cbot

    elif Vstretching == 4:
        C = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))
        return C

    elif Vstretching == 5:
        S1 = (K * K - 2 * K * N + K + N * N - N) / (N * N - N)
        S2 = (K * K - K * N) / (1 - N)
        S = -S1 - 0.01 * S2

        C = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))
        return C

    else:
        raise

def sdepth(H, Hc, C, stagger="rho", Vtransform=1):
    """Return depth of rho-points in s-levels

    *H* : arraylike
      Bottom depths [meter, positive]

    *Hc* : scalar
       Critical depth

    *cs_r* : 1D array
       s-level stretching curve

    *stagger* : [ 'rho' | 'w' ]

    *Vtransform* : [ 1 | 2 ]
       defines the transform used, defaults 1 = Song-Haidvogel

    Returns an array with ndim = H.ndim + 1 and
    shape = cs_r.shape + H.shape with the depths of the
    mid-points in the s-levels.

    Typical usage::

    >>> fid = Dataset(roms_file)
    >>> H = fid.variables['h'][:, :]
    >>> C = fid.variables['Cs_r'][:]
    >>> Hc = fid.variables['hc'].getValue()
    >>> z_rho = sdepth(H, Hc, C)

    """
    H = np.asarray(H)
    Hshape = H.shape  # Save the shape of H
    H = H.ravel()  # and make H 1D for easy shape maniplation
    C = np.asarray(C)
    N = len(C)
    outshape = (N,) + Hshape  # Shape of output
    if stagger == "rho":
        S = -1.0 + (0.5 + np.arange(N)) / N  # Unstretched coordinates
    elif stagger == "w":
        S = np.linspace(-1.0, 0.0, N)
    else:
        raise ValueError("stagger must be 'rho' or 'w'")

    if Vtransform == 1:  # Default transform by Song and Haidvogel
        A = Hc * (S - C)[:, None]
        B = np.outer(C, H)
        return (A + B).reshape(outshape)

    if Vtransform == 2:  # New transform by Shchepetkin
        N = Hc * S[:, None] + np.outer(C, H)
        D = 1.0 + Hc / H
        return (N / D).reshape(outshape)

    # else:
    raise ValueError("Unknown Vtransform")


# ------------------------
#   Sampling routines
# ------------------------


def z2s_frac(z_rho, i, j, z):

    kmax, jmax, imax = z_rho.shape

    # Constrain to valid indices
    i = np.clip(np.around(i).astype('i4'), 0, imax - 1)
    j = np.clip(np.around(j).astype('i4'), 0, jmax - 1)

    # Vectorized searchsorted
    k_unclipped = np.sum(z_rho[:, j, i] <= -z, axis=0) - 1
    k = np.clip(k_unclipped, 0, kmax - 2)

    frac_unclipped = (-z - z_rho[k, j, i]) / (z_rho[k + 1, j, i] - z_rho[k, j, i])
    frac = np.clip(frac_unclipped, 0, 1)

    return k + frac
