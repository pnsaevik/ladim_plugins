import numpy as np
from ladim.ibms import light


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

        salinity_model = config["ibm"].get('salinity_model', 'new')
        self.new_salinity_model = (salinity_model == 'new')

    def update_ibm(self, grid, state, forcing):
        # Mortality
        state['super'] *= self.mortality_factor

        # Update forcing
        state['temp'] = forcing.field(state.X, state.Y, state.Z, "temp")
        state['salt'] = forcing.field(state.X, state.Y, state.Z, "salt")

        # Age in degree-days
        state['age'] += state.temp * state.dt / 86400
        state['days'] += 1.0*(state.dt/86400)

        # Light at depth
        lon, lat = grid.lonlat(state.X, state.Y)
        light0 = light.surface_light(state.timestamp, lon, lat)
        Eb = light0 * np.exp(-self.k * state.Z)

        # Swimming velocity
        W = np.zeros_like(state.X)
        # Upwards if light enough (decreasing depth)
        W[Eb >= 0.01] = -self.swim_vel

        if self.new_salinity_model:
            # Mixture of down/up if salinity between 23 and 31
            # Downwards if salinity < 31
            salt_limit = np.random.uniform(23, 31, W.shape)
        else:
            # Downwards if salinity < 20
            salt_limit = 20

        W[state.salt < salt_limit] = self.swim_vel

        # Random diffusion velocity
        if self.vertical_diffusion:
            rand = np.random.normal(size=len(W))
            W += rand * (2 * self.D / self.dt) ** 0.5

        # Update vertical position, using reflexive boundary condition at the top
        Z = state['Z']
        Z += W * self.dt
        Z[Z < 0] *= -1
        Z[Z >= 20.0] = 19.0
        state['Z'] = Z

        # Mark particles older than 200 degree days as dead
        state['alive'] &= (state.age < 200)


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

    # Define function for lower limit of infectivity
    b1 = 24.79
    b2 = 0.525
    cop_age_fn = lambda Temp: Temp * (b1 / (Temp - 10 + b1 * b2)) ** 2

    # Set infectivity to zero outside lower and upper limit
    lower_limit = cop_age_fn(temp)
    upper_limit = 200
    idx_outside = (age < lower_limit) | (age > upper_limit)
    infect[idx_outside] = 0

    return infect * super
