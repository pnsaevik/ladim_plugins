import ladim.gridforce.ROMS
import numpy as np


class Grid(ladim.gridforce.ROMS.Grid):
    def __init__(self, config):
        super().__init__(config)


class Forcing(ladim.gridforce.ROMS.Forcing):
    def __init__(self, config, grid):
        super().__init__(config, grid)

        self._fish_u = None
        self._fish_v = None
        self.fish_swim_speed = 0.14  # m/s
        self.use_currents = False
        self.ocean_distance = config['gridforce'].get('ocean_distance', 10)  # km

    def _ocean_dist_cells(self):
        ocean_dist_km = self.ocean_distance
        cell_size_km = self._grid.dx[0, 0] / 1000
        ocean_dist_cells = int(np.round(ocean_dist_km / cell_size_km))
        return ocean_dist_cells

    def _compute_fish_velocity(self):
        from . import ibm
        land = 1 - np.asarray(self._grid.M).astype('int32')
        fjord_idx = ibm.fjord_index(land, self._ocean_dist_cells())
        u, v = ibm.descent(fjord_idx)
        self._fish_u = u * self.fish_swim_speed
        self._fish_v = v * self.fish_swim_speed

    @property
    def fish_u(self):
        if self._fish_u is None:
            self._compute_fish_velocity()
        return self._fish_u

    @property
    def fish_v(self):
        if self._fish_v is None:
            self._compute_fish_velocity()
        return self._fish_v

    def fish_velocity(self, X, Y):
        i0 = self._grid.i0
        j0 = self._grid.j0
        jmax, imax = np.shape(self._grid.M)
        i = np.asarray(X - i0).round().clip(0, imax - 1).astype('int32')
        j = np.asarray(Y - j0).round().clip(0, jmax - 1).astype('int32')
        u = self.fish_u[j, i]
        v = self.fish_v[j, i]
        return u, v

    def velocity(self, X, Y, Z, tstep=0, method="bilinear"):
        fish_u, fish_v = self.fish_velocity(X, Y)
        if self.use_currents:
            u, v = super().velocity(X, Y, Z, tstep, method)
            return u + fish_u, v + fish_v
        else:
            return fish_u, fish_v