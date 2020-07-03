import numpy as np


class Grid:
    def __init__(self, config):
        self.xmin = 0
        self.xmax = 1
        self.ymin = 0
        self.ymax = 1

    def sample_metric(self, x, y):
        return np.ones_like(x), np.ones_like(x)

    def sample_depth(self, x, y):
        return np.ones_like(x) * 100

    def ll2xy(self, lon, lat):
        return lon, lat

    def ingrid(self, x, y):
        return np.ones_like(x, dtype=bool)

    def atsea(self, x, y):
        return np.ones_like(x, dtype=bool)


class Forcing:
    def __init__(self, config, grid):
        pass

    def update(self, t):
        pass

    def velocity(self, x, y, z, tstep):
        return np.zeros_like(x), np.zeros_like(x)

    def close(self):
        pass
