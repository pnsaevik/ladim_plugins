class IBM:
    def __init__(self, config):
        self.dt = config['dt']
        self.state = None
        self.grid = None
        self.forcing = None

    def update_ibm(self, grid, state, forcing):
        self.state, self.grid, self.forcing = state, grid, forcing

        # Find sinking velocity
        sinkvel = state['sinkvel']

        # Use explicit Euler to update vertical position
        state['Z'] += self.dt * sinkvel

        # Set as inactive if particle has reached bottom
        X, Y, Z = state.X, state.Y, state.Z
        H = grid.sample_depth(X, Y)  # Water depth
        state['active'][:] = (Z < H)
