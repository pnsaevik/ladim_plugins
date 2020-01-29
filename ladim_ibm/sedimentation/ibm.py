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

    def update_ibm(self, grid, state, forcing):
        a = state.active != 0
        X, Y, Z = state.X[a], state.Y[a], state.Z[a]

        # Read sinking velocity for settling particles
        W = state.sink_vel[a]

        # Random diffusion velocity
        if self.vertical_diffusion:
            rand = np.random.normal(size=len(W))
            W += rand * (2 * self.D / self.dt) ** 0.5

        # Update vertical position, using reflexive boundary condition at the surface
        Z += W * self.dt
        Z[Z < 0] *= -1

        # Define which particles have settled to the bottom and which have not
        H = grid.sample_depth(X, Y)  # Water depth
        at_seabed = Z > H
        Z[at_seabed] = H[at_seabed]

        # Store new vertical position
        state.Z[a] = Z
        state.active[a] = ~at_seabed

        # Aging and particle death
        state.age += state.dt
        state.alive = state.alive & (state.age <= self.lifespan)
