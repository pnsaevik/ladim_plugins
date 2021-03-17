from ladim.gridforce.ROMS import Forcing
from ladim_plugins.chemicals import Grid as LadimGrid


class Grid(LadimGrid):
    def sample_depth(self, X, Y):
        """Return the depth of grid cells"""
        intX = X.astype(int)
        intY = Y.astype(int)
        fracX = X - intX
        fracY = Y - intY

        I = intX - self.i0
        J = intY - self.j0

        # Bilinear interpolation
        return (
            (1 - fracX) * (
                (1 - fracY) * self.H[J, I] +
                fracY * self.H[J + 1, I]
            ) +
            fracX * (
                (1 - fracY) * self.H[J, I + 1] +
                fracY * self.H[J + 1, I + 1]
            )
        )
