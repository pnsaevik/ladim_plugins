import numpy as np


class IBM:
    def __init__(self, config):
        # Time before a particle is taken out of the simulation [days]
        self.lifespan = config['ibm']['lifespan']

        # Vertical mixing [m*2/s]
        self.D = config['ibm']['vertical_mixing']
        self.vertical_diffusion = self.D > 0

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

        self.sink_particles()
        self.bury_particles()
        self.kill_old_particles()

    def bury_particles(self):
        grid = self.grid
        a = self.state.active != 0
        X, Y, Z = self.state.X[a], self.state.Y[a], self.state.Z[a]

        # Define which particles have settled to the bottom and which have not
        H = grid.sample_depth(X, Y)  # Water depth
        at_seabed = Z > H
        Z[at_seabed] = H[at_seabed]

        # Store new vertical position
        self.state.Z[a] = Z
        self.state.active[a] = ~at_seabed

    def sink_particles(self):
        state = self.state

        a = state.active != 0
        Z = state.Z[a]

        # Read sinking velocity for settling particles
        W = state.sink_vel[a]

        # Random diffusion velocity
        if self.vertical_diffusion:
            rand = np.random.normal(size=len(W))
            W += rand * (2 * self.D / self.dt) ** 0.5

        # Update vertical position, using reflexive boundary condition at the surface
        Z += W * self.dt
        Z[Z < 0] *= -1

        # Store new vertical position
        state.Z[a] = Z

    def kill_old_particles(self):
        state = self.state
        state.age += state.dt
        state.alive = state.alive & (state.age <= self.lifespan)
