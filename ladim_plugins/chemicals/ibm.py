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
        H = grid.sample_depth(state.X, state.Y)  # Water depth
        MINIMUM_VERTDIFF = 0

        # --- Use Itô backwards scheme (LaBolle et al. 2000) for vertical diffusion ---

        # Vertical advection
        W_adv = forcing.forcing.wvel(state.X, state.Y, state.Z)
        Z0 = state.Z + W_adv * self.dt
        Z0[Z0 < 0] *= -1  # Reflexive boundary at top
        below_seabed = Z0 > H
        Z0[below_seabed] = 2 * H[below_seabed] - Z0[below_seabed]  # Reflexive bottom

        # Stochastic differential
        dW = np.random.normal(size=len(state.X)) * np.sqrt(self.dt)

        # Simple, constant diffusion
        if not isinstance(self.D, str):
            diff_2 = self.D

        # Itô backwards scheme (LaBolle et al. 2000)
        else:
            # Vertical diffusion, intermediate step
            diff_1 = forcing.forcing.field_w(state.X, state.Y, Z0, self.D)
            diff_1 = np.maximum(MINIMUM_VERTDIFF, diff_1)
            Z1 = Z0 + np.sqrt(2 * diff_1) * dW  # Diffusive step
            Z1[Z1 < 0] *= -1                    # Reflexive boundary at top
            below_seabed = Z1 > H
            Z1[below_seabed] = 2*H[below_seabed] - Z1[below_seabed]  # Reflexive bottom

            # Use intermediate step to sample diffusion
            diff_2 = forcing.forcing.field_w(state.X, state.Y, Z1, self.D)
            diff_2 = np.maximum(MINIMUM_VERTDIFF, diff_2)

        # Diffusive step and reflective boundary conditions
        Z2 = Z0 + np.sqrt(2 * diff_2) * dW  # Diffusive step
        Z2[Z2 < 0] *= -1                    # Reflexive boundary at top
        below_seabed = Z2 > H
        Z2[below_seabed] = 2*H[below_seabed] - Z2[below_seabed]  # Reflexive bottom
        state.Z = Z2

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
