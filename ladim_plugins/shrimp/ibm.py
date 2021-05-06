class IBM:

    def __init__(self, config):
        self.state = None
        self.grid = None
        self.forcing = None
        self.dt = config['dt']

    def update_ibm(self, grid, state, forcing):
        self.state = state
        self.grid = grid
        self.forcing = forcing
