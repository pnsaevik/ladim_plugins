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
            names = ['salt_preference', 'depth_preference']
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

        # Check larval stage
        is_copepod = state['stage'] >= 1

        # --- Compute individual freshwater avoidance limit ---
        salt_pref_min_cop = 16
        salt_pref_max_cop = 32
        salt_limit_cop = salt_pref_min_cop + state['salt_preference'] * (salt_pref_max_cop - salt_pref_min_cop)

        salt_pref_min_nup = 28
        salt_pref_max_nup = 32
        salt_limit_nup = salt_pref_min_nup + state['salt_preference'] * (salt_pref_max_nup - salt_pref_min_nup)

        salt_limit = np.where(is_copepod, salt_limit_cop, salt_limit_nup)
        is_too_little_salt = state['salt'] < salt_limit

        # --- Compute darkness indicator ---
        light_limit_cop = 0
        light_limit_nup = 0.01
        light_limit = np.where(is_copepod, light_limit_cop, light_limit_nup)

        lon, lat = self.model['grid'].lonlat(x, y)
        light0 = light.surface_light(timestamp, lon, lat)
        Eb = light0 * np.exp(-self.k * state['Z'])
        is_too_dark_to_swim = Eb < light_limit

        # --- Compute depth indicator ---

        # Commented out: Linear depth preference
        # depth_pref_min = 0
        # depth_pref_max = 20
        # depth_limit = depth_pref_min + state['depth_preference'] * (depth_pref_max - depth_pref_min)

        # Exponential depth preference
        depth_pref_scale = 5
        depth_limit = -depth_pref_scale * np.log(1 - state['depth_preference'])

        is_comfortable_depth = state['Z'] < depth_limit

        # --- Determine swim direction ---
        # Interpretation: The conditions are listed by increasing priority
        swim_direction = -np.ones(state['Z'].shape, dtype='i2')  # Negative = upwards
        swim_direction[is_comfortable_depth] = 0
        swim_direction[is_too_dark_to_swim] = 0
        swim_direction[is_too_little_salt] = 1

        state['Z'] += self.swim_vel * swim_direction * self.dt

    def diffusion(self):
        """
        Diffusion scheme by LaBolle et al. (2000), doi: 10.1029/1999WR900224

        Mixing coefficient from ocean model is divided into discrete sections and
        capped from above and below to avoid numerical errors
        """
        if not self.vertical_diffusion:
            return

        state = self.model['state']
        X0, Y0 = state['X'], state['Y']
        depth = self.model['grid'].sample_depth(X0, Y0)
        dt = self.dt

        mix_depth_bins = 1
        max_diff_coeff = 1e-2
        min_diff_coeff = 1e-5
        vertdiff_dt = mix_depth_bins**2 / (6 * max_diff_coeff)

        def vert_mix_fn(xx, yy, zz):
            # Compute coarse-binned mixing coefficient
            dz = mix_depth_bins
            z_coarse = (zz // dz) * dz + 0.5*dz
            vert_mix_value = self.model['forcing'].forcing.vert_mix(xx, yy, z_coarse)
            return np.clip(vert_mix_value, min_diff_coeff, max_diff_coeff)

        current_time = 0
        while current_time < dt:
            old_time = current_time
            current_time = np.minimum(dt, current_time + vertdiff_dt)
            ddt = current_time - old_time
            Z0 = self.model['state']['Z']

            # Uniform stochastic differential
            dW = (np.random.rand(len(Z0)) * 2 - 1) * np.sqrt(3 * ddt)

            # Intermediate diffusion step
            vert_mix_0 = vert_mix_fn(X0, Y0, Z0)
            Z1 = Z0 + np.sqrt(2 * vert_mix_0) * dW  # Diffusive step
            Z1[Z1 < 0] *= -1  # Reflexive boundary at top
            below_seabed = Z1 > depth
            Z1[below_seabed] = 2 * depth[below_seabed] - Z1[below_seabed]  # Reflexive bottom

            # Final diffusion step
            vert_mix_1 = vert_mix_fn(X0, Y0, Z1)
            Z2 = Z0 + np.sqrt(2 * vert_mix_1) * dW  # Diffusive step
            Z2[Z2 < 0] *= -1  # Reflexive boundary at top
            below_seabed = Z2 > depth
            Z2[below_seabed] = 2 * depth[below_seabed] - Z2[below_seabed]  # Reflexive bottom

            state['Z'] = Z2


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
