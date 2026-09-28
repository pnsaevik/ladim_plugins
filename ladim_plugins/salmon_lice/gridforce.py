import ladim.gridforce.ROMS


class Grid(ladim.gridforce.ROMS.Grid):
    def __init__(self, config):
        super().__init__(config)


class Forcing(ladim.gridforce.ROMS.Forcing):
    def __init__(self, config, grid):
        super().__init__(config, grid)

    def vert_mix(self, X, Y, Z):
        """Vertical diffusivity (AKs) at the particle positions

        Taken from the w level just above the particle (the lowest level if
        the particle is below it) in the nearest water column, without
        interpolation. The forcing files must contain AKs (some, such as
        NorKyst, contain ln_AKs instead).
        """
        return self.field(X, Y, Z, "AKs")
