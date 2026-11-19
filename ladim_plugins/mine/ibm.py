import numpy as np
from ladim_plugins.utils.vertical_mixing import check_no_ibm_vertical_mixing


class IBM:
    def __init__(self, config):
        # Time before a particle is taken out of the simulation [seconds]
        self.lifespan = config['ibm']['lifespan']

        check_no_ibm_vertical_mixing(config)
        self.taucrit_fn = get_taucrit_fn(config['ibm'].get('taucrit', 1000))

        # Vertical advection on/off
        self.vadv = config['ibm'].get('vertical_advection', False)

        # Store time step value to calculate age
        self.dt = config['dt']

        # Possible separate output file to record time (and place) of death
        self.output_file = config["ibm"].get('output_file', None)
        self.output_vars = {
            varname: config['nc_attributes'][varname]
            for varname in config['output_instance']
        }
        if self.output_file:
            create_outfile(self.output_file, self.output_vars)

        # Record positions to know if the particles are stuck near land
        self.land_collision = config["ibm"].get('land_collision', 'reposition')
        self.x = np.array([])
        self.y = np.array([])
        self.pid = np.array([])

        # Reference to other modules
        self.grid = None
        self.forcing = None
        self.state = None

        # Variables for lazy evaluation
        self._ustar = None
        self._ustar_tstep = -1

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.forcing = forcing
        self.state = state

        if self.has_active():
            has_been_buried_before = (self.state.active != 1)
        else:
            has_been_buried_before = None

        self.reposition()
        just_resuspended = self.resuspend()
        self.sink(just_resuspended)
        self.bury(just_resuspended)
        self.kill_old()
        self.store()

        if self.has_active():
            is_active = (self.state.active != 0)
            self.state.active[is_active & has_been_buried_before] = 2

    def has_active(self):
        try:
            _ = self.state['active']
            return True
        except KeyError:
            return False

    def active(self):
        if self.has_active():
            return self.state['active']
        else:
            return np.broadcast_to(1, self.state.X.shape)

    def reposition(self):
        if self.land_collision == "reposition":
            state = self.state
            a = self.active()
            X, Y = state['X'], state['Y']

            # If particles have not moved: Assume they ended up on land.
            # If that is the case, reposition them within the cell.
            pid, pidx_old, pidx_new = np.intersect1d(self.pid, state.pid, return_indices=True)
            onland = ((self.x[pidx_old] == X[pidx_new]) &
                      (self.y[pidx_old] == Y[pidx_new]) &
                      np.bool_(a[pidx_new])
                      )
            num_onland = np.count_nonzero(onland)
            pidx_new_onland = pidx_new[onland]
            x_new = np.round(X[pidx_new_onland]) - 0.5 + np.random.rand(num_onland)
            y_new = np.round(Y[pidx_new_onland]) - 0.5 + np.random.rand(num_onland)
            X[pidx_new_onland] = x_new
            Y[pidx_new_onland] = y_new

            state['X'] = X
            state['Y'] = Y
            self.x = np.copy(state.X)
            self.y = np.copy(state.Y)
            self.pid = state.pid

    def resuspend(self):
        """Make settled particles mobile where the bottom shear stress is high enough

        The vertical diffusion is done by ladim before the IBM. Particles which
        are resuspended here stay on the sea floor during this time step (they
        are not sunk or buried), and are lifted by the diffusion in the next one.

        :return: Boolean array, True for the particles which were resuspended
        """
        just_resuspended = np.zeros(np.shape(self.state.X), dtype=bool)
        if self.taucrit_fn is None:
            return just_resuspended

        ustar = self.shear_velocity_btm()
        tau = shear_stress_btm(ustar)
        lon, lat = self.grid.lonlat(self.state.X, self.state.Y)
        taucrit = self.taucrit_fn(lon, lat)
        resusp = tau >= taucrit
        just_resuspended = resusp & (self.state.active == 0)
        self.state.active[resusp] = True
        return just_resuspended

    def bury(self, hold):
        grid = self.grid
        a = (self.active() != 0) & ~hold
        X, Y, Z = self.state.X[a], self.state.Y[a], self.state.Z[a]

        # Define which particles have settled to the bottom and which have not
        H = grid.sample_depth(X, Y)  # Water depth
        at_seabed = Z > H
        Z[at_seabed] = H[at_seabed]

        # Store new vertical position
        self.state.Z[a] = Z
        if self.has_active():
            self.state.active[a] = ~at_seabed

        # Kill buried particles if no resuspension
        if not self.taucrit_fn:
            self.state['alive'][a] &= ~at_seabed

    def sink(self, hold):
        # Get parameters
        a = (self.active() != 0) & ~hold
        z = self.state.Z[a]
        w = self.state.sink_vel[a]  # Sink velocity
        if self.vadv:
            w -= self.forcing.field(self.state.X[a], self.state.Y[a], z, 'w')  # w is upwards in ROMS

        # Euler scheme, no boundary conditions
        self.state.Z[a] = z + self.dt * w

    def kill_old(self):
        state = self.state
        state['age'] += state.dt
        state['alive'] &= state['age'] <= self.lifespan

    def store(self):
        if not self.output_file:
            return

        dead = ~self.state.alive
        new_values = {
            k: self.state[k][dead]
            for k in self.output_vars.keys()
            if k not in ['lon', 'lat']
        }
        if 'lon' in self.output_vars.keys() and 'lat' in self.output_vars.keys():
            new_values['lon'], new_values['lat'] = self.grid.xy2ll(
                new_values['X'], new_values['Y'])

        update_outfile(self.output_file, new_values)

    def shear_velocity_btm(self):
        if self._ustar_tstep < self.state.timestep:
            # Calculate bottom shear velocity from last computational layer
            # velocity
            # returns: Ustar at bottom cell
            x = self.state.X
            y = self.state.Y
            h = self.grid.sample_depth(x, y)

            u_btm, v_btm = self.forcing.velocity(x, y, h, tstep=0)
            U2 = u_btm*u_btm + v_btm*v_btm
            c = 0.003
            self._ustar = np.sqrt(c * U2)
            self._ustar_tstep = self.state.timestep

        return self._ustar


def shear_stress_btm(ustar):
    rho = 1000
    return ustar * ustar * rho


def get_taucrit_fn(value):
    if value >= 1000:  # No resuspension if high value
        return None
    else:
        return lambda lon, lat: np.zeros_like(lon) + value


def create_outfile(fname, variables):
    import netCDF4 as nc
    with nc.Dataset(fname, 'w') as dset:
        dset.createDimension('particle', None)

        for k, v in variables.items():
            var = dset.createVariable(k, v['ncformat'], 'particle')
            for attr_name, attr_val in v.items():
                if attr_name != 'ncformat':
                    var.setncattr(attr_name, attr_val)


def update_outfile(fname, new_values):
    import netCDF4 as nc
    with nc.Dataset(fname, 'a') as dset:
        num_old = dset.dimensions['particle'].size
        num_new = len(next(v for v in new_values.values()))

        for k, v in new_values.items():
            dset.variables[k][num_old:num_old + num_new] = v
