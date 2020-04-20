import numpy as np
from ladim.ibms import light


class IBM:
    def __init__(self, config):
        self.D = config['ibm']['vertical_mixing']  # [m*2/s]
        self.egg_diam = config['ibm']['egg_diam']
        self.hatch_age = config['ibm']['hatch_age']
        self.mortality = config['ibm']['mortality']
        self.light_coeff = config['ibm']['light_coeff']
        self.max_depth = config['ibm']['max_depth']
        self.swim_speed = config['ibm']['swim_speed']
        self.hatch_weight = config['ibm']['hatch_weight']

        self.dt = config['dt']
        self.state = None
        self.grid = None
        self.forcing = None

    def update_ibm(self, grid, state, forcing):
        self.state = state
        self.grid = grid
        self.forcing = forcing

        self.update_forcing()
        self.vertical_migration()
        self.growth_and_ageing()

    def update_forcing(self):
        state = self.state
        state.temp = self.forcing.field(state.X, state.Y, state.Z, 'temp')
        state.salt = self.forcing.field(state.X, state.Y, state.Z, 'salt')

    def vertical_migration(self):
        state = self.state
        grid = self.grid
        egg_diam = self.egg_diam
        is_larvae = (state.age > self.hatch_age)

        # Calculate density and viscosity
        dens_water = calc_density(state.temp, state.salt)
        dens_egg = calc_density(state.temp, state.egg_buoy)
        mu_w = calc_viscosity(state.temp, state.salt)

        # Calculate terminal velocity using Stokes formula
        wvel_egg = egg_velocity(mu_w, dens_water, dens_egg, egg_diam)
        wvel_egg[is_larvae] = 0

        # Random diffusion velocity
        rand = np.random.normal(size=len(state.X))
        wvel_diff = rand * (2 * self.D / self.dt) ** 0.5

        # Light at depth
        lon, lat = grid.lonlat(state.X, state.Y)
        light0 = light.surface_light(state.timestamp, lon, lat)
        Eb = light0 * np.exp(-self.light_coeff * state.Z)

        # Light-induced vertical migration
        # Natt ca 5m, og dag ca 10m Ellertsen 1979
        log_weight = np.log(state.weight)
        mm2m = 0.001
        length = np.exp(2.296 + log_weight * (0.277 + log_weight * -0.005128))
        swim_speed = self.swim_speed * length * mm2m
        swim_up = (Eb < 1)
        swim_speed[swim_up] *= -1
        swim_speed[~is_larvae] = 0

        # Update vertical position.
        W = wvel_egg + swim_speed + wvel_diff
        state.Z += W * self.dt

        # Reflexive boundary condition at the top and at max depth
        state.Z[state.Z < 0] *= -1
        is_deep = state.Z > self.max_depth
        state.Z[is_deep] = 2*self.max_depth - state.Z[is_deep]

    def growth_and_ageing(self):
        state = self.state
        sec2day = 1/86400
        log_weight = np.log(state.weight)
        is_larvae = (state.age > self.hatch_age)

        # Growth rate
        GF = 1.08 + state.temp * (1.79 + log_weight * (-0.074 + log_weight * (
                -0.0965 + log_weight * 0.0112)))
        GR = sec2day * state.dt * np.log(1 + 0.001 * GF)
        GR_gram = (np.exp(GR) - 1) * state.weight
        state.weight[~is_larvae] = self.hatch_weight
        state.weight[is_larvae] += GR_gram[is_larvae]

        # Mortality
        state.super[is_larvae] *= np.exp(-self.mortality * self.dt * sec2day)

        # Age in degree-days
        state.age += state.temp * state.dt * sec2day


def calc_density(temp, salt):
    T68 = temp * 1.00024

    a0 = 999.842594
    a1 = 6.793952e-2
    a2 = -9.095290e-3
    a3 = 1.001685e-4
    a4 = -1.120083e-6
    a5 = 6.536332e-9

    b0 = 8.24493e-1
    b1 = -4.0899e-3
    b2 = 7.6438e-5
    b3 = -8.2467e-7
    b4 = 5.3875e-9

    c0 = -5.72466e-3
    c1 = 1.0227e-4
    c2 = -1.6546e-6

    d0 = 4.8314e-4

    dens_T = a0 + (a1 + (a2 + (a3 + (a4 + a5 * T68) * T68) * T68) * T68) * T68

    dens_ST = dens_T + (
                b0 + (b1 + (b2 + (b3 + b4 * T68) * T68) * T68) * T68) * salt + (
                          c0 + (c1 + c2 * T68) * T68) * salt * (
                          salt ** (1 / 2)) + d0 * salt ** 2

    return dens_ST


def calc_viscosity(temp, salt):
    return 0.001 * (1.7915 + temp * (-0.0538 + temp * 0.0007) + 0.0023 * salt)


def egg_velocity(mu_w, dens_water, dens_egg, egg_diam):
    dmax = (9.0 * mu_w * mu_w / (
            1025.0 * 9.81 * np.abs(dens_water - dens_egg))) ** (1 / 3)
    wvel_small_eggs = (9.81 / 18) * (egg_diam ** 2) * (
                dens_egg - dens_water) / mu_w
    wvel_large_eggs = 0.08825 * (egg_diam - 0.4 * dmax) * (
            dens_egg - dens_water) ** (2 / 3) * mu_w ** (-1 / 3)

    W = wvel_small_eggs
    W[egg_diam > dmax] = wvel_large_eggs[egg_diam > dmax]
    return W
