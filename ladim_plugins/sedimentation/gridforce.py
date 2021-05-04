import ladim.gridforce.ROMS


class Grid(ladim.gridforce.ROMS.Grid):
    def __init__(self, config):
        super().__init__(config)


class Forcing(ladim.gridforce.ROMS.Forcing):
    def __init__(self, config, grid):
        super().__init__(config, grid)
