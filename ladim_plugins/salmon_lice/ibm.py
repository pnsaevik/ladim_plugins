import numpy as np
from ladim.ibms import light
from collections import namedtuple


LegacyModel = namedtuple('Model', ['grid', 'state', 'forcing'])


class IBM:
    def __init__(self, config):

        # Constants
        mortality = 0.17  # [days-1]

        # mm2m = 0.001
        # g = 9.81
        # tempB = 7.0  # set default temperature

        self.k = 0.2  # Light extinction coefficient
        self.swim_vel = 5e-4  # m/s
        self.D = config["ibm"].get('vertical_mixing', 1e-3)  # Vertical mixing [m*2/s]
        self.vertical_diffusion = self.D > 0

        self.dt = config["dt"]
        self.mortality_factor = np.exp(-mortality * self.dt / 86400)

        self.model = {}

    def update_ibm(self, grid, state, forcing):
        self.model = dict(grid=grid, state=state, forcing=forcing)
        self.update()

    def update(self):
        self.initialize()
        self.update_forcing()
        self.development_samsing()
        self.update_infectivity()
        self.ageing()
        self.migration()
        self.diffusion()

    def initialize(self):
        state = self.model['state']
        idx_new_particles = state['salt_preference'][:] == 0

        if np.any(idx_new_particles):
            names = ['salt_preference']
            num = np.count_nonzero(idx_new_particles)
            newdatas = np.random.rand(len(names) * num).reshape((len(names), num))
            for newdata, name in zip(newdatas, names):
                state[name][idx_new_particles] = newdata

    def update_forcing(self):
        state = self.model['state']
        forcing = self.model['forcing']
        x, y, z = state['X'], state['Y'], state['Z']
        state['temp'] = forcing.field(x, y, z, "temp")
        state['salt'] = forcing.field(x, y, z, "salt")

    def development_stien(self):
        """
        Larval development is computed according to Stien et al. (2005), doi: 10.3354/meps290263

        Only the development rate from hatched to copepod stage is used. Stage 0-1 is nauplii,
        stage >= 1 is copepod.
        """
        state = self.model['state']
        number_of_days = self.dt / (60*60*24)
        temp = np.clip(state['temp'], 2, 20)

        beta_1 = 24.79
        beta_2 = 0.525
        rate = ((temp - 10 + beta_1 * beta_2) / beta_1)**2
        state['stage'] += rate * number_of_days

    def development_samsing(self):
        """
        Larval development is computed according to Samsing et al. (2016), doi: 10.1139/cjfas-2016-0050

        Stage 0-1 is nauplii, stage 1-2 is copepod, stage > 2 is dead or attached to host
        """
        state = self.model['state']
        temp = np.clip(state['temp'], 1, 25)
        dt_days = self.dt / (60*60*24)

        beta_0 = 1.4
        beta_1 = -1.48
        tau = np.log(temp / 10)
        nauplius_duration = np.exp(beta_0 + beta_1 * tau)
        nauplius_rate = 1 / nauplius_duration

        beta_0 = 2.6
        beta_1 = -0.26
        beta_2 = -1.03
        cop_duration = np.exp(beta_0 + tau * (beta_1 + tau * beta_2))
        cop_rate = 1 / cop_duration

        rate = np.where(state['stage'] < 1, nauplius_rate, cop_rate)
        state['stage'] += rate * dt_days

    def update_infectivity(self):
        state = self.model['state']
        state['infect'] = infectivity(state['age'], state['temp'], state['super'])

    def ageing(self):
        state = self.model['state']
        dt = self.dt

        # Age in degree-days
        state['age'] += state['temp'] * dt / 86400
        state['days'] += 1.0*(dt/86400)

        # Mortality
        state['super'] *= self.mortality_factor

        # Mark particles older than 200 degree days as dead
        state['alive'] &= (state['age'] < 200)

    def migration(self):
        state = self.model['state']
        x, y, z = state['X'], state['Y'], state['Z']
        timestamp = state.timestamp

        # --- Compute swimming direction ---
        #
        # The algorithm is as follows:
        #
        # 1. If low salinity: Swim downwards
        #
        # 2. Otherwise, check light. If daytime, swim upwards.
        #
        # 3. Default: Don't swim

        # Compute individual freshwater avoidance limit
        salt_pref_min = 23
        salt_pref_max = 31
        salt_limit = salt_pref_min + state['salt_preference'] * (salt_pref_max - salt_pref_min)
        low_salt = state['salt'] < salt_limit

        # Compute darkness indicator
        lon, lat = self.model['grid'].lonlat(x[~low_salt], y[~low_salt])
        light0 = light.surface_light(timestamp, lon, lat)
        Eb = light0 * np.exp(-self.k * state['Z'][~low_salt])
        is_bright = Eb >= 0.01

        velocity = self.swim_vel * np.ones_like(state['Z'])     # Positive = downwards
        velocity[~low_salt] = -velocity[~low_salt] * is_bright  # Upwards if high salt, unless there is darkness

        state['Z'] += velocity * self.dt

    def diffusion(self):
        if not self.vertical_diffusion:
            return

        state = self.model['state']
        num_particles = len(state['Z'])
        x, y = state['X'], state['Y']
        mix_depth = 5
        vert_mix_fn = self.model['forcing'].forcing.vert_mix

        rand = np.random.normal(size=num_particles)
        vert_mix = vert_mix_fn(x, y, np.broadcast_to(mix_depth, x.shape))
        W = rand * (2 * vert_mix / self.dt) ** 0.5

        # Update vertical position, using reflexive boundary condition at the top
        Z = state['Z']
        Z += W * self.dt
        Z[Z < 0] *= -1
        Z[Z >= 20.0] = 19.0
        state['Z'] = Z


# noinspection PyShadowingBuiltins
def infectivity(age, temp, super=1):
    """
    Computes scaled salmon lice infectivity according to Skern-Mauritzen
    et al. (2020), https://doi.org/10.1016/j.jembe.2020.151429.

    :param age: The age of the salmon lice, in degree-days
    :param temp: The ambient temperature
    :param super: The number of lice
    :returns: The infectivity, scaled by number of lice
    """
    coeff = np.array([
        [-3.466e+1, 7.156e-1, -5.354e-3, 1.191e-5],
        [+2.306e+0, -3.577e-2, 2.526e-4, -5.541e-7],
        [-2.585e-2, 0, 0, 0],
    ])

    # Compute infectivity based on Rasmus' formula
    T = np.clip(temp, 5, 15)
    TT = np.array([T ** 0, T ** 1, T ** 2])
    AA = np.array([age ** 0, age ** 1, age ** 2, age ** 3])
    q = ((coeff @ AA) * TT).sum(axis=0)
    ROC_factor = 1.8 / 0.51
    infect = ROC_factor / (1 + np.exp(-q))

    # Define lower limit of infectivity
    b1 = 24.79
    b2 = 0.525
    cop_age_lower = temp * (b1 / (temp - 10 + b1 * b2)) ** 2

    # Set infectivity to zero outside lower and upper limit
    lower_limit = cop_age_lower
    upper_limit = 200
    idx_outside = (age < lower_limit) | (age > upper_limit)
    infect[idx_outside] = 0

    return infect * super
