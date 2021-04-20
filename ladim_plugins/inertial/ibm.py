import numpy as np


class IBM:
    def __init__(self, config):
        self.dt = config['dt']
        self.state = None
        self.grid = None
        self.forcing = None

    def update_ibm(self, grid, state, forcing):
        self.state, self.grid, self.forcing = state, grid, forcing

        # Find sinking velocity
        sinkvel = state['sinkvel']

        # Use explicit Euler to update vertical position
        state['Z'] += self.dt * sinkvel

        # Set as inactive if particle has reached bottom
        X, Y, Z = state.X, state.Y, state.Z
        H = grid.sample_depth(X, Y)  # Water depth
        state['active'][:] = (Z < H)


def terminal_sinkvel(temp, salt, drag, area, density_particle, mass_particle):
    density_water = calc_density(temp, salt)
    g = 9.81
    vv = (
        (mass_particle * g * (density_water/density_particle - 1)) /
        (0.5 * density_water * drag * area)
    )

    return -np.sqrt(np.abs(vv)) * np.sign(vv)


def calc_density(temp, salt):
    # print(temp,salt)
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
