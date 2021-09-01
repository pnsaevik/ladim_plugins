import numpy as np


class IBM:
    def __init__(self, config):
        self.D = config["ibm"].get('vertical_mixing', 0)  # Vertical mixing [m*2/s]
        self.dt = config['dt']
        self.vertdiff_dt = config["ibm"].get('vertdiff_dt', self.dt)  # Vertical diffusion timestep [s]
        self.vertdiff_dz = config["ibm"].get('vertdiff_dz', 0)  # Spacing of vertical diffusion sampling [m]
        self.vertdiff_max = config["ibm"].get('vertdiff_max', np.inf)  # Maximal vertical diffusion [m2/s]
        self.x = np.array([])
        self.y = np.array([])
        self.pid = np.array([])
        self.land_collision = config["ibm"].get('land_collision', 'reposition')
        self.grid = None
        self.state = None
        self.forcing = None

        # Issue warning if parameters for vertical diffusion indicates numerical instability
        if self.vertdiff_max < np.inf and self.vertdiff_dz > 0:
            instability = 6 * self.vertdiff_max * self.vertdiff_dt / (self.vertdiff_dz ** 2)
            if instability > 1:
                import logging
                logging.warning('Possible unstable vertical diffusion scheme')
                logging.warning('Reduce time step, increase sampling distance or limit the diffusion coefficient')

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.state = state
        self.forcing = forcing

        self.advect()

        if isinstance(self.D, str):
            self.diffuse_labolle()
        else:
            self.diffuse_const()

        if self.land_collision == "reposition":
            self.reposition()

    def advect(self):
        # Vertical advection
        x = self.state.X
        y = self.state.Y
        z = self.state.Z
        self.state.Z += self.dt * self.forcing.forcing.wvel(x, y, z)
        self.reflect()

    # Itô backwards scheme (LaBolle et al. 2000) for vertical diffusion
    def diffuse_labolle(self):
        x = self.state.X
        y = self.state.Y
        H = self.grid.sample_depth(x, y)

        if self.vertdiff_dz:
            def z_coarse(zz):
                dz = self.vertdiff_dz
                return np.maximum(0.25 * dz, ((zz - 0.5 * dz) // dz) * dz + dz)
        else:
            def z_coarse(zz):
                return zz

        def sample_K(xx, yy, zz):
            kk = self.forcing.forcing.vertdiff(xx, yy, z_coarse(zz), self.D)
            return np.minimum(kk, self.vertdiff_max)

        current_time = 0
        while current_time < self.dt:
            old_time = current_time
            current_time = np.minimum(self.dt, current_time + self.vertdiff_dt)
            ddt = current_time - old_time
            z = self.state.Z

            # Uniform stochastic differential
            dW = (np.random.rand(len(z)) * 2 - 1) * np.sqrt(3 * ddt)

            # Vertical diffusion, intermediate step
            Z1 = z + np.sqrt(2 * sample_K(x, y, z)) * dW  # Diffusive step
            Z1[Z1 < 0] *= -1                    # Reflexive boundary at top
            below_seabed = Z1 > H
            Z1[below_seabed] = 2*H[below_seabed] - Z1[below_seabed]  # Reflexive bottom

            # Diffusive step and reflective boundary conditions
            self.state.Z += np.sqrt(2 * sample_K(x, y, Z1)) * dW  # Diffusive step
            self.reflect()

    def diffuse_const(self):
        # Uniform stochastic differential
        dW = (np.random.rand(len(self.state.Z)) * 2 - 1) * np.sqrt(3 * self.dt)
        self.state.Z += np.sqrt(2 * self.D) * dW
        self.reflect()

    def reflect(self):
        x = self.state.X
        y = self.state.Y
        z = self.state.Z
        H = self.grid.sample_depth(x, y)
        below_seabed = z > H
        z[z < 0] *= -1  # Reflexive boundary at top
        z[below_seabed] = 2 * H[below_seabed] - z[below_seabed]  # Reflexive bottom
        self.state.Z = z

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

        self.x = self.state.X
        self.y = self.state.Y
        self.pid = self.state.pid
