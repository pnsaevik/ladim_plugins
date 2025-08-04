import numpy as np


class Grid:
    def __init__(self, config):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

    def sample_metric(self, x, y):
        dx = np.ones_like(x)
        dy = np.ones_like(y)
        return dx, dy

    def sample_depth(self, x, y):
        h = np.ones_like(x)
        return h

    def ll2xy(self, lon, lat):
        x = lon
        y = lat
        return x, y

    def xy2ll(self, x, y):
        return x, y

    def ingrid(self, x, y):
        return np.ones(np.shape(x), dtype=bool)

    def atsea(self, x, y):
        return np.ones(np.shape(x), dtype=bool)


class Forcing:
    def __init__(self, config, grid):
        self._grid = grid

    def update(self, t):
        pass

    def velocity(self, x, y, z, tstep):
        u = np.zeros_like(x)
        v = np.zeros_like(y)
        return u, v

    def close(self):
        pass