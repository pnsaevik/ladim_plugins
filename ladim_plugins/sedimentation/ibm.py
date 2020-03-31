import numpy as np


class IBM:
    def __init__(self, config):
        # Time before a particle is taken out of the simulation [seconds]
        self.lifespan = config['ibm']['lifespan']

        # Vertical mixing [m*2/s]
        self.D = config['ibm']['vertical_mixing']  # 0.001 m2/s -- 0.01 m2/s (?)
        self.taucrit = config['ibm'].get('taucrit', 0.12)
        self.vertical_diffusion = self.D > 0

        # Store time step value to calculate age
        self.dt = config['dt']

        # Reference to other modules
        self.grid = None
        self.forcing = None
        self.state = None

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.forcing = forcing
        self.state = state

        self.resuspend()
        self.sink()
        self.bury()
        self.kill_old()

    def resuspend(self):
        ustar = self.shear_velocity_btm()
        tau = shear_stress_btm(ustar)
        resusp = tau > self.taucrit
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

    def sink(self):
        """Diffuse first (reflective boundaries), then sink (no boundaries)."""

        # Get parameters
        a = self.state.active != 0
        x, y, z = self.state.X[a], self.state.Y[a], self.state.Z[a]

        # Diffusion
        b0 = np.sqrt(2 * self.D)
        dw = np.random.randn(z.size).reshape(z.shape) * np.sqrt(self.dt)
        z1 = z + b0 * dw

        # Reflexive boundary conditions
        z1[z1 < 0] *= -1  # Surface
        h = self.grid.sample_depth(x, y)
        below_seabed = z1 > h
        z1[below_seabed] = 2*h[below_seabed] - z1[below_seabed]

        # Advection
        z2 = z1 + self.dt * self.state.sink_vel[a]  # Sink velocity

        # Store new vertical position
        self.state.Z[a] = z2

    def kill_old(self):
        state = self.state
        state.age += state.dt
        state.alive = state.alive & (state.age <= self.lifespan)

    def shear_velocity_btm(self):
        # Calculates bottom shear velocity from last computational layer
        # velocity
        # returns: Ustar at bottom cell
        x = self.state.X
        y = self.state.Y
        h = self.grid.sample_depth(x, y)

        u_btm, v_btm = self.forcing.velocity(x, y, h, tstep=0)
        U2 = u_btm*u_btm + v_btm*v_btm
        c = 0.003
        return np.sqrt(c * U2)


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
