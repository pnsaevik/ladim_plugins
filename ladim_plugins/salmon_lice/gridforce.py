import ladim.gridforce.ROMS
from ladim.gridforce.ROMS import z2s, sample3D
import numpy as np


class Grid(ladim.gridforce.ROMS.Grid):
    def __init__(self, config):
        super().__init__(config)


class Forcing(ladim.gridforce.ROMS.Forcing):
    def __init__(self, config, grid):
        super().__init__(config, grid)

    def vert_mix(self, X, Y, Z):
        MAXIMUM_K = len(self._grid.Cs_w) - 2
        MINIMUM_K = 1
        MINIMUM_D = 1e-7

        I = np.int32(np.round(X)) - self._grid.i0
        J = np.int32(np.round(Y)) - self._grid.j0
        K, A = z2s(self._grid.z_w, I, J, Z)
        K_nearest = np.round(K - A).astype(np.int32)
        K_nearest = np.minimum(MAXIMUM_K, K_nearest)
        K_nearest = np.maximum(MINIMUM_K, K_nearest)
        F = self['ln_AKs']
        return np.maximum(MINIMUM_D, np.exp(F[K_nearest, J, I]))
