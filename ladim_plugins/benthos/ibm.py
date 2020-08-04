class IBM:
    def __init__(self, config):
        # Store time step value to calculate age
        self.dt = config['dt']

        # Reference to other modules
        self.grid = None
        self.forcing = None
        self.state = None

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.forcing = forcing
        self.state = state
