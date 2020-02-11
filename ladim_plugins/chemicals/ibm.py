import numpy as np


class IBM:
    def __init__(self, config):
        self.D = config["ibm"].get('vertical_mixing', 0)  # Vertical mixing [m*2/s]
        self.dt = config['dt']
        self.x = np.array([])
        self.y = np.array([])
        self.pid = np.array([])
        self.land_collision = config["ibm"].get('land_collision', 'reposition')

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

        if self.land_collision == "reposition":
            # If particles have not moved: Assume they ended up on land.
            # If that is the case, reposition them within the cell.
            pid, pidx_old, pidx_new = np.intersect1d(self.pid, state.pid, return_indices=True)
            onland = ((self.x[pidx_old] == state.X[pidx_new]) &
                      (self.y[pidx_old] == state.Y[pidx_new]))
            num_onland = np.count_nonzero(onland)
            pidx_new_onland = pidx_new[onland]
            x_new = np.round(state.X[pidx_new_onland]) - 0.5 + np.random.rand(num_onland)
            y_new = np.round(state.Y[pidx_new_onland]) - 0.5 + np.random.rand(num_onland)
            state.X[pidx_new_onland] = x_new
            state.Y[pidx_new_onland] = y_new

            self.x = state.X
            self.y = state.Y
            self.pid = state.pid
