import numpy as np
from scipy.interpolate import RectBivariateSpline


class IBM:
    """Adding a constant horizontal velocity to the particle tracking"""

    def __init__(self, config):
        self.D = config['ibm']['vertical_mixing']
        self.dt = config['dt']

        self.state = None
        self.grid = None
        self.forcing = None

    def update_ibm(self, grid, state, forcing):
        self.state = state
        self.grid = grid
        self.forcing = forcing

        self.vertical_diffuse()

    def vertical_diffuse(self):
        state = self.state

        # Random diffusion velocity
        rand = np.random.normal(size=len(state.Z))
        state.Z += rand * np.sqrt(2 * self.D * self.dt)

        # Keep within vertical limits, reflexive condition
        rmin = 0
        rmax = np.array(self.grid.sample_depth(state.X, state.Y))
        state.Z = reflexive(state.Z, rmin, rmax)


def reflexive(r, rmin=-np.inf, rmax=np.inf):
    r = r.copy()
    shp = np.shape(r)
    rmin = np.broadcast_to(rmin, shp)
    idx = r < rmin
    r[idx] = 2*rmin[idx] - r[idx]

    rmax = np.broadcast_to(rmax, shp)
    idx = r > rmax
    r[idx] = 2*rmax[idx] - r[r > rmax]
    return np.clip(r, rmin, rmax)


def larval_development(temp, stage, active, dt):
    """
    Larval development according to Christensen et al. (2008), doi:10.1139/F08-073

    The function modifies (in-place) the variable `stage`. The length of the larvae is
    assumed to be related to the stage as

       L = L0 * (1 - s) + Lm * s

    where L0 = 7.73mm, Lm = 40mm and s = stage - 1

    :param temp: Ambient temperature, in degrees Celcius
    :param stage: Particle stage (0-1 = egg, 1-2 = larva, 2+ = metamorphosed)
    :param active: 0 if stationary, 1 if the particle follows currents
    :param dt: Time step size, in seconds
    """

    # Christensen's "flexible settlement" scenario is between lengths 37mm and 43mm,
    # which corresponds to a larval stage between 1.907 and 2.093

    idx = (1 <= stage) & (stage < 2)
    s = stage[idx] - 1

    Lm = 40
    L0 = 7.73
    L_inf = 218

    L = L0 + s * (Lm - L0)
    gam = 0.316
    lamb0 = -1.725
    lamb1 = 0.136
    lamb = np.exp(lamb0 + lamb1 * temp[idx])

    dLdt = lamb * np.power(L / L0, gam) * (1 - L / L_inf)
    L_new = L + dLdt * dt / (60 * 60 * 24)
    stage[idx] = 1 + (L_new - L0) / (Lm - L0)

    # Deactivate metamorphosed larvae
    # assert np.all(stage[idx] >= 1)  # stage is only increasing
    active[idx] = stage[idx] < 2


def egg_development(temp, stage, hatch_rate, active, dt):
    """
    Egg development according to Christensen et al. (2008), doi:10.1139/F08-073

    Using initialization model `e` (variable maturation)

    The function modifies (in-place) the variables `stage` and `active`

    :param temp: Ambient temperature
    :param stage: Particle stage (0-1 = egg, 1-2 = larva, 2+ = metamorphosed)
    :param hatch_rate: Number between 0 and 1 indicating hatch rate
    :param active: 0 if stationary, 1 if the particle follows currents
    :param dt: Time step size, in seconds
    """

    # Increase development level of eggs
    idx = stage < 1
    development_time = hatch_time(hatch_rate[idx], temp[idx])
    stage_increase = dt / development_time
    stage[idx] += stage_increase

    # Activate hatched eggs
    active[idx] = stage[idx] >= 1


def get_hatch_time_func():
    """
    Total hatch time based on Smigielski et al. (1984), doi: 10.3354/meps014287

    The underlying data table is taken directly from the paper. Interpolation is
    linear with date and second order in the rate direction.
    """

    days_tab = np.array([
        [61, 51, 39, 25],
        [82, 67, 48, 30],
        [135, 116, 82, 55],
    ])
    temp_tab = np.array([2, 4, 7, 10])
    rate_tab = np.array([0, 0.5, 1])

    spline = RectBivariateSpline(
        x=rate_tab,
        y=temp_tab,
        z=days_tab,
        kx=2,
        ky=1,
    )

    def hatch_time_fn(rate, temp):
        """
        Total hatch time based on Smigielski et al. (1984), doi: 10.3354/meps014287

        The underlying data table is taken directly from the paper. Interpolation is
        linear with date and second order in the rate direction.

        :param rate: 0 = earliest spawners, 0.5 = median spawners, 1 = latest spawners
        :param temp: Ambient temperature, in degrees Celcius
        :return: Total hatch time, in days
        """
        temp = np.minimum(temp_tab[-1], np.maximum(temp_tab[0], temp))
        out = spline(rate.ravel(), temp.ravel(), grid=False)
        return out.reshape(rate.shape)

    return hatch_time_fn


hatch_time = get_hatch_time_func()
