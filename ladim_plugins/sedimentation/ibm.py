import numpy as np


class IBM:
    def __init__(self, config):
        # Time before a particle is taken out of the simulation [days]
        self.lifespan = config['ibm']['lifespan']

        # Vertical mixing [m*2/s]
        self.D = config['ibm']['vertical_mixing']
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
        pass

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
        state = self.state

        a = state.active != 0
        Z = state.Z[a]

        # Read sinking velocity for settling particles
        W = state.sink_vel[a]

        # Random diffusion velocity
        if self.vertical_diffusion:
            rand = np.random.normal(size=len(W))
            W += rand * (2 * self.D / self.dt) ** 0.5

        # Update vertical position, using reflexive boundary condition at the surface
        Z += W * self.dt
        Z[Z < 0] *= -1

        # Store new vertical position
        state.Z[a] = Z

    def kill_old(self):
        state = self.state
        state.age += state.dt
        state.alive = state.alive & (state.age <= self.lifespan)


def sde_solver(x0, t0, advect_fn, diffuse_fn, dt, method='euler'):
    """
    Solve a stochastic differential equation.

    The equation needs to be an Itö SDE of the form

    dx = a(x, t) * dt + b(x, t) * dw,

    where a(x, t) is the advective term, and the diffusive term b(x, t) is
    related to the physical diffusion coefficient D by b = sqrt(2D). It is
    assumed that the main axes of the diffusion tensor are aligned with the
    coordinate axes.

    Algorithms:

    Euler: The naive Euler–Maruyama method. Works only for homogeneous
    diffusion.

    :param x0: An N x M vector of initial values, where M is the number of coordinates.
    :param t0: The initial time.
    :param advect_fn: A function (x, t) --> x-like, representing the advective term.
    :param diffuse_fn: A function (x, t) --> x-like, representing the diffusive term.
    :param dt: The time step length.
    :param method: Chosen algorithm. Alternatives are 'euler'.
    :return: An x0-like array of the integrated values.
    """
    method_fns = dict(
        euler=_euler_maruyama,
    )

    method_fn = method_fns[method]
    return method_fn(x0, t0, advect_fn, diffuse_fn, dt)


def _euler_maruyama(x0, t0, advect_fn, diffuse_fn, dt):
    a = advect_fn(x0, t0)
    b = diffuse_fn(x0, t0)
    dw = np.random.randn(x0.size).reshape(x0.shape) * np.sqrt(dt)
    return x0 + a * dt + b * dw
