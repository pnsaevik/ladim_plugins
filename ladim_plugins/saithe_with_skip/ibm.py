import numpy as np
from ..utils import light, density, viscosity
from ..larvae.ibm import weight_to_length, sinkvel_egg, growth_cod_larvae


class IBM:

    def __init__(self, config):
        self.k = 0.2  # Light extinction coefficient
        self.vertical_diffusion = True
        self.D = 1e-4  # Vertical mixing [m*2/s]

        self.dt = config['dt']
        self.extra_spreading = config['ibm'].get('extra_spreading', True)

        self.hatch_day = 60      # Hatch day [degree days]
        self.egg_diam = 0.0011   # Egg diameter [m]
        self.swim_speed = 0.2    # Vertical swimming speed [body lengths/second]
        self.max_depth = 60
        self.min_depth = 30
        self.init_larvae_weight = 9.3e-2

        self.grid = None
        self.state = None
        self.forcing = None

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.state = state
        self.forcing = forcing

        # --- Update forcing ---
        state['temp'] = forcing.field(state.X, state.Y, state.Z, 'temp')
        state['salt'] = forcing.field(state.X, state.Y, state.Z, 'salt')

        # --- Ageing ---
        is_egg = state.age <= self.hatch_day
        state['age'] += state.temp * state.dt / 86400

        # --- Horizontal spreading ---
        if self.extra_spreading:
            self.spread()

        # --- Larvae growth ---
        temp_larvae = state.temp[~is_egg]
        weight = np.maximum(state['weight'][~is_egg], self.init_larvae_weight)
        state['weight'][~is_egg] = weight + growth_cod_larvae(temp_larvae, weight, self.dt)

        # --- Egg sinking velocity ---
        W = np.zeros(np.shape(is_egg), dtype=np.float32)
        T, S = state.temp[is_egg], state.salt[is_egg]
        W[is_egg] = sinkvel_egg(
            mu_w=viscosity(T, S),
            dens_w=density(T, S),
            dens_egg=density(T, state.egg_buoy[is_egg]),
            diam_egg=self.egg_diam,
        )

        # --- Larvae swimming velocity ---
        lon, lat = grid.lonlat(state.X[~is_egg], state.Y[~is_egg])
        Eb = light(state.timestamp, lon, lat)
        desired_light = 1
        length = 0.001 * weight_to_length(state.weight[~is_egg])
        W[~is_egg] = self.swim_speed * length * np.sign(Eb - desired_light)

        # --- Vertical turbulent mixing ---
        if self.D:
            W += np.random.normal(size=len(W)) * np.sqrt(2 * self.D / self.dt)

        # --- Execute vertical movement ---
        Z = state.Z + W * self.dt
        Z[~is_egg] = np.clip(Z[~is_egg], self.min_depth, self.max_depth)
        state['Z'] = Z

    def spread(self):
        horizontal_speed = 1  # cm/s
        fraction_directed = 0.93

        # Set direction for new particles
        direction = self.state['direction']
        idx_new_particles = (direction == 0)
        num_new_particles = np.count_nonzero(idx_new_particles)
        new_directions = 2 * np.pi * np.random.rand(num_new_particles)

        # Let some of the new particles be non-directed
        new_directions /= fraction_directed
        new_directions[new_directions > 2 * np.pi] = np.nan
        self.state['direction'][idx_new_particles] = new_directions

        # Compute displacement
        is_directed = ~np.isnan(self.state['direction'])
        is_directed &= (self.state.age > self.hatch_day)  # Exclude eggs
        d = self.state['direction'][is_directed]
        x0 = self.state.X[is_directed]
        y0 = self.state.Y[is_directed]
        dt = self.dt
        om, on = 1 / np.array(self.grid.sample_metric(x0, y0))
        x = x0 + horizontal_speed * 0.01 * om * dt * np.cos(d)
        y = y0 + horizontal_speed * 0.01 * on * dt * np.sin(d)

        # Don't move outside of grid
        outside_grid = ~self.grid.ingrid(x, y)
        x[outside_grid] = x0[outside_grid]
        y[outside_grid] = y0[outside_grid]
        not_alive = np.copy(is_directed)
        not_alive[is_directed] = outside_grid
        self.state['alive'][not_alive] = False

        # Don't move onto land
        on_land = ~self.grid.atsea(x, y)
        x[on_land] = x0[on_land]
        y[on_land] = y0[on_land]

        # Update position
        self.state['X'][is_directed] = x
        self.state['Y'][is_directed] = y
