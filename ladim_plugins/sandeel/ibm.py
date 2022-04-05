import numpy as np


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
