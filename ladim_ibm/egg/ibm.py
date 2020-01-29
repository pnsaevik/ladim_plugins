import numpy as np
from ladim.ibms import light


class IBM:
    def __init__(self, config):
        self.k = 0.2             # Light extinction coefficient
        # self.swim_vel = 5e-4     # m/s
        self.vertical_diffusion = True
        self.D = 5e-5            # Vertical mixing [m*2/s]

        self.dt = config['dt']

    def update_ibm(self, grid, state, forcing):
        egg_diam = 0.0011

        # for n in range(len(state.X)):
        # Update forcing
        state.temp = forcing.field(state.X, state.Y, state.Z, 'temp')
        state.salt = forcing.field(state.X, state.Y, state.Z, 'salt')

        # Calculating density
        dens_water = calc_density(state.temp, state.salt)

        # Calculate dynamic molecular viscosity
        dens_egg = calc_density(state.temp, state.egg_buoy)
        my_w = 0.001*(1.7915 - 0.0538*state.temp + 0.0007*(state.temp**2) + 0.0023*state.salt)

        # Calculate maximum diameter in Stokes formula
        dmax = (9.0*my_w*my_w/(1025.0*9.81*np.abs(dens_water - dens_egg)))**(1/3)

        # Calculate vertical velocity

        W = np.zeros_like(state.X)
        W = np.where(egg_diam <= dmax, (1/18)*(1/my_w)*9.81*(egg_diam**2)*np.abs(dens_water - dens_egg), 0.08825*(egg_diam-0.4*dmax)*(np.abs(dens_water - dens_egg))**(2/3)*my_w**(-1/3))

        W = -W*np.sign(dens_water-dens_egg)

        if self.vertical_diffusion:
            rand = np.random.normal(1)
            W += rand * (2*self.D/self.dt)**0.5

        # Age in degree-days
        state.temp = forcing.field(state.X, state.Y, state.Z, 'temp')
        state.age += state.temp * state.dt / 86400

        # Light at depth
        lon, lat = grid.lonlat(state.X, state.Y)
        light0 = light.surface_light(state.timestamp, lon, lat)
        Eb = light0 * np.exp(-self.k*state.Z)

        # Update vertical position, using reflexive boundary condition
        state.Z += W * self.dt
        # print(state.Z)

        # For z-version, do not go below 100 m
        state.Z[state.Z >= 200.0] = 199.0

        # Mark particles older than 200 degree days as dead
        # state.alive = state.alive & (state.age < 200)


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
