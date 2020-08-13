import numpy as np


class IBM:
    def __init__(self, config):
        # Time before a particle is taken out of the simulation [seconds]
        self.lifespan = config['ibm']['lifespan']

        # Vertical mixing [m*2/s]
        self.diffparams = get_diffparams(config['ibm'].get('vertical_mixing', None))
        self.taucrit_fn = get_taucrit_fn(config['ibm'].get('taucrit', None))

        # Store time step value to calculate age
        self.dt = config['dt']

        # Reference to other modules
        self.grid = None
        self.forcing = None
        self.state = None

        # Parameters for lazy evaluation
        self._ustar = None
        self._ustar_ts = -1

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.forcing = forcing
        self.state = state

        self.resuspend()
        self.diffuse()
        self.sink()
        self.bury()
        self.kill_old()

    def resuspend(self):
        if self.taucrit_fn is None:
            return

        ustar = self.shear_velocity_btm()
        tau = shear_stress_btm(ustar)
        lon, lat = self.grid.lonlat(self.state.X, self.state.Y)
        taucrit = self.taucrit_fn(lon, lat)
        resusp = tau > taucrit
        self.state.active[resusp] = True

    def bury(self):
        grid = self.grid
        a = self.state.active != 0
        X, Y, Z = self.state.X[a], self.state.Y[a], self.state.Z[a]

        # Define which particles have settled to the bottom and which have not
        H = grid.sample_depth(X, Y)  # Water depth
        at_seabed = Z > H
        Z[at_seabed] = H[at_seabed]

        # Store new vertical position
        self.state.Z[a] = Z
        self.state.active[a] = ~at_seabed

    def diffuse(self):
        method = self.diffparams['method']
        if method == 'constant':
            self.diffuse_constant()
        else:
            raise KeyError('Unknown method')

    def diffuse_constant(self):
        # Get parameters
        a = self.state.active != 0
        x, y, z = self.state.X[a], self.state.Y[a], self.state.Z[a]
        h = self.grid.sample_depth(x, y)
        D = self.diffparams['value']

        # Diffusion
        b0 = np.sqrt(2 * D)
        dw = np.random.randn(z.size).reshape(z.shape) * np.sqrt(self.dt)
        z1 = z + b0 * dw

        # Reflexive boundary conditions
        z1[z1 < 0] *= -1  # Surface
        below_seabed = z1 > h
        z1[below_seabed] = 2*h[below_seabed] - z1[below_seabed]

        # Store new vertical position
        self.state.Z[a] = z1

    def sink(self):
        # Get parameters
        a = self.state.active != 0
        z = self.state.Z[a]
        w = self.state.sink_vel[a]  # Sink velocity

        # Euler scheme, no boundary conditions
        self.state.Z[a] = z + self.dt * w

    def kill_old(self):
        state = self.state
        state.age += state.dt
        state.alive = state.alive & (state.age <= self.lifespan)

    def shear_velocity_btm(self):
        # Recompute only if necessary
        if self._ustar_ts < self.state.timestep:
            # Calculate bottom shear velocity from last computational layer
            # velocity
            # returns: Ustar at bottom cell
            x = self.state.X
            y = self.state.Y
            h = self.grid.sample_depth(x, y)

            u_btm, v_btm = self.forcing.velocity(x, y, h, tstep=0)
            U2 = u_btm*u_btm + v_btm*v_btm
            c = 0.003
            ustar = np.sqrt(c * U2)

            self._ustar_ts = self.state.timestep
            self._ustar = ustar

        return self._ustar


def shear_stress_btm(ustar):
    rho = 1000
    return ustar * ustar * rho


def ladis(x0, t0, t1, v, K):
    """
    Lagrangian Advection and DIffusion Solver.

    Solve the diffusion equation in a Lagrangian framework. The equation is

    dc/dt = - grad (vc) + div (K grad c),

    where c is concentration, v is 3-dimensional velocity, K is a diagonal
    tensor (i.e. main axes oriented along coordinate axes) of diffusion.

    This is translated to a stochastic differential equation of the form

    dx = (v_x + d/dx K_xx) * dt + sqrt(2*K_xx) * dw_x,
    dy = (v_y + d/dy K_yy) * dt + sqrt(2*K_yy) * dw_y,
    dz = (v_z + d/dz K_zz) * dt + sqrt(2*K_zz) * dw_z,

    where x, y, z is the spatial position, K_xx, K_yy, K_zz are the diagonal
    elements of K, and dw_x, dw_y, dw_z are Wiener process differential elements
    with zero mean and stdev = sqrt(dt).

    Algorithm:

    Operator splitting: Diffusion first, then advection. Diffusion is solved
    using the gradient-free backwards Itô scheme, according to LaBolle
    (2000, 10.1029/1999WR900224).

    :param x0: An N x M vector of initial values, where N is the number of
               particles and M is the number of coordinates.
    :param t0: The initial time.
    :param t1: The end time.
    :param v:  The velocity. A function (x, t) --> x-like.
    :param K:  The diagonal elements of the diffusion tensor.
               A function (x, t) --> x-like.
    :return:   An x0-like array of the new particle positions.
    """

    dt = t1 - t0

    # --- Diffusion, LaBolle scheme ---

    # First diffusion step (predictor)
    b0 = np.sqrt(2 * K(x0, t0))
    dw = np.random.randn(x0.size).reshape(x0.shape) * np.sqrt(dt)
    x1 = x0 + b0 * dw

    # Second diffusion step (corrector)
    b1 = np.sqrt(2 * K(x1, t0))
    x2 = x0 + b1 * dw

    # --- Advection, forward Euler ---

    a3 = v(x2, t0)
    x3 = x2 + a3 * dt

    return x3


def get_diffparams(subconf):
    if subconf is None:
        subconf = dict(method='constant', value=0)
    elif not isinstance(subconf, dict):
        subconf = dict(method='constant', value=subconf)

    return subconf


def get_turbulence_linear_bounded_fn(max_mixing):
    def linear_bounded_fn(ustar, meters_from_seafloor):
        kappa = 0.41
        dA_dz = kappa * ustar  # Alternative formulation: kappa * ustar * w * exp(-w/w0), where w = z - h
        A = dA_dz * meters_from_seafloor
        cutoff = (A > max_mixing)
        A[cutoff] = max_mixing
        dA_dz[cutoff] = 0
        return A

    return linear_bounded_fn


def get_turbulence_constant_fn(const):
    def constant_fn(ustar, meters_from_seafloor):
        A = np.zeros_like(ustar) + const
        A[meters_from_seafloor < 0] = 0
        return A

    return constant_fn


def get_taucrit_fn(subconf):
    if not isinstance(subconf, dict):
        subconf = dict(method='constant', value=subconf)

    method = subconf['method']
    if method == 'constant':
        value = subconf['value']
        return lambda lon, lat: np.zeros_like(lon) + value
    else:
        raise ValueError(f'Unknown method: {method}')
