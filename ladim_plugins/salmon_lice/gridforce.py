import ladim.gridforce.ROMS
from ladim.gridforce.ROMS import z2s, sample3D


class Grid(ladim.gridforce.ROMS.Grid):
    def __init__(self, config):
        super().__init__(config)


class Forcing(ladim.gridforce.ROMS.Forcing):
    def __init__(self, config, grid):
        super().__init__(config, grid)

    def vert_mix(self, X, Y, Z):
        i0 = self._grid.i0
        j0 = self._grid.j0
        K, A = z2s(self._grid.z_w, X - i0, Y - j0, Z)
        F = self['AKs']
        return sample3D(F, X - i0, Y - j0, K, A, method="nearest")
