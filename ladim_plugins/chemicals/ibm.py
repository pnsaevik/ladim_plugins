import numpy as np
from ladim_plugins.utils.vertical_mixing import check_no_ibm_vertical_mixing


class IBM:
    def __init__(self, config):
        ibmconf = config.get('ibm', dict())

        # Time before a particle is taken out of the simulation [seconds]
        self.lifespan = ibmconf.get('lifespan', None)

        check_no_ibm_vertical_mixing(config)
        self.dt = config['dt']
        removed = [k for k in ('horzdiff_type', 'horzdiff_max', 'horzdiff_min') if k in ibmconf]
        if removed:
            raise ValueError(f"The options {', '.join('ibm.' + k for k in removed)} are removed. "
                             f"Use numerics.diffusion instead")
        self.vertadv = ibmconf.get('vertical_advection', True)
        self.x = np.array([])
        self.y = np.array([])
        self.pid = np.array([])
        self.land_collision = ibmconf.get('land_collision', 'reposition')
        self.grid = None
        self.state = None
        self.forcing = None

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.state = state
        self.forcing = forcing

        if self.land_collision == "reposition":
            self.reposition()
        elif self.land_collision == "coastal_diffusion":
            self.coastal_diffusion()

        if self.vertadv:
            self.advect()

        if self.land_collision == "reposition":
            self.store_position()

        if self.lifespan is not None:
            self.kill_old()

    def advect(self):
        # Vertical advection
        x = self.state.X
        y = self.state.Y
        z = self.state.Z
        self.state['Z'] -= self.dt * self.forcing.field(x, y, z, 'w')  # w is upwards in ROMS
        self.reflect()

    def kill_old(self):
        state = self.state
        state['age'] += self.dt
        state['alive'] = state.alive & (state.age <= self.lifespan)

    def reflect(self):
        x = self.state.X
        y = self.state.Y
        z = self.state.Z
        H = self.grid.sample_depth(x, y)
        below_seabed = z > H
        z[z < 0] *= -1  # Reflexive boundary at top
        z[below_seabed] = 2 * H[below_seabed] - z[below_seabed]  # Reflexive bottom
        self.state['Z'] = z

    def reposition(self):
        # If particles have not moved: Assume they ended up on land.
        # If that is the case, reposition them within the cell.
        pid, pidx_old, pidx_new = np.intersect1d(self.pid, self.state.pid, return_indices=True)
        onland = ((self.x[pidx_old] == self.state.X[pidx_new]) &
                  (self.y[pidx_old] == self.state.Y[pidx_new]))
        num_onland = np.count_nonzero(onland)
        pidx_new_onland = pidx_new[onland]
        x_new = np.round(self.state.X[pidx_new_onland]) - 0.5 + np.random.rand(num_onland)
        y_new = np.round(self.state.Y[pidx_new_onland]) - 0.5 + np.random.rand(num_onland)
        self.state.X[pidx_new_onland] = x_new
        self.state.Y[pidx_new_onland] = y_new

    def store_position(self):
        self.x = self.state.X
        self.y = self.state.Y
        self.pid = self.state.pid

    def coastal_diffusion(self):
        # If particles are close to coast, reposition them within the cell
        x, y, pid = self.state.X, self.state.Y, self.state.pid
        is_coastal = is_close_to_land(self.grid, x, y)
        num_coastal = np.count_nonzero(is_coastal)
        x_new = np.round(x[is_coastal]) - 0.5 + np.random.rand(num_coastal)
        y_new = np.round(y[is_coastal]) - 0.5 + np.random.rand(num_coastal)
        self.state['X'][is_coastal] = x_new
        self.state['Y'][is_coastal] = y_new


def is_close_to_land(grid, X, Y):
    """True for particles where any of the eight surrounding grid cells is land"""
    i, j = np.round(X), np.round(Y)
    close = np.zeros(np.shape(X), dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                close |= grid.onland(i + di, j + dj)
    return close


def smagorinsky(u_south, u_north, v_west, v_east, dx, atsea=True):
    """Horizontal diffusivity [m2/s] by Smagorinsky (1963)

    This function is not used by the IBM. It is kept for reference, since the
    horizontal diffusion of the IBM was removed in favour of `numerics.diffusion`.

    The diffusivity is proportional to the grid spacing and the horizontal
    shear of the current over one grid cell:

        K = 0.04 * dx * |(u_north - u_south) + (v_east - v_west)|

    :param u_south: x-velocity [m/s] at the southern neighbour grid point
    :param u_north: x-velocity [m/s] at the northern neighbour grid point
    :param v_west: y-velocity [m/s] at the western neighbour grid point
    :param v_east: y-velocity [m/s] at the eastern neighbour grid point
    :param dx: Grid spacing [m]
    :param atsea: False for particles on land, where the diffusivity is zero
    :return: Diffusivity [m2/s]
    """
    K = 0.04 * dx * np.abs((u_north - u_south) + (v_east - v_west))
    return np.where(atsea, K, 0)
