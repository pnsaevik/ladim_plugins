import numpy as np


class IBM:
    def __init__(self, config):
        self.D = config["ibm"].get('vertical_mixing', 0)  # Vertical mixing [m*2/s]
        self.dt = config['dt']

    def update_ibm(self, grid, state, forcing):
        # Vertical advection velocity
        W = forcing.forcing.wvel(state.X, state.Y, state.Z)

        # Vertical diffusion velocity
        rand = np.random.normal(size=len(state.X))
        W += rand * (2 * self.D / self.dt) ** 0.5

        # Update vertical position, using reflexive boundary condition at top
        state.Z += W * self.dt
        state.Z[state.Z < 0] *= -1

        # Reflexive boundary condition at bottom
        H = grid.sample_depth(state.X, state.Y)  # Water depth
        below_seabed = state.Z > H
        state.Z[below_seabed] = 2*H[below_seabed] - state.Z[below_seabed]
