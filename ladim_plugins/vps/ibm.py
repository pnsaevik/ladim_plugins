import numpy as np


class IBM:
    def __init__(self, config):
        self.D = config["ibm"].get('vertical_mixing', 1e-4)  # Vertical mixing [m*2/s]
        self.vertical_diffusion = self.D > 0
        self.dt = config["dt"]
        self.fjord_index_file = config["ibm"]["fjord_index_file"]

    def update_ibm(self, grid, state, forcing):
        fjord_index = np.load(self.fjord_index_file)

        state.age += state.dt #/ 86400
        swim_vel = state.size # assume 1bl/sec   

        # Find direction towards the ocean from the fjord index
        # Horizontal swimming:
        delta = [-1, 0, 1]
        ddx = []
        ddy = []

        for n in range(len(state.X)):
            x = int(state.X[n])
            y = int(state.Y[n])
            dx, dy = grid.sample_metric(state.X[n], state.Y[n])
           # swim_vel = float(state.size[n]) # assume 1bl/sek

            xv = [fjord_index[y,x-1],fjord_index[y,x],fjord_index[y,x+1]]
            yv = [fjord_index[y-1,x],fjord_index[y,x],fjord_index[y+1,x]]

            r = np.where(xv==min(xv))[0] # liste x-retn som er nermere havet
            xdir = delta[r[np.random.randint(0,len(r))]] # trekker tilf i lista
            r = np.where(yv==min(yv))[0]
            ydir = delta[r[np.random.randint(0,len(r))]]

            # beregner svommedist i x eller y retn
            if xdir == 0:
                r = 0
            elif ydir == 0:
                r = 1
            else:
                r = np.random.randint(0,2)
            ddx.append(r * swim_vel * xdir * self.dt /dx)
            ddy.append((1-r) * swim_vel * ydir * self.dt /dy)

        # Oppdaterer X, og Y posisjon
        state.X += ddx
        state.Y += ddy
        # Vertical swimming velocity
        W = np.zeros_like(state.X)

        # Random vertical diffusion velocity
        if self.vertical_diffusion:
            rand = np.random.normal(size=len(W))
            W += rand * (2*self.D/self.dt)**0.5

            # Update vertical position
            state.Z += W * self.dt

            # For z-version, reflective boundaries
            state.Z[state.Z < 0.0] = abs(state.Z[state.Z < 0.0])
            state.Z[state.Z >= 2.0] = 2.0 - (state.Z[state.Z >= 2.0] - 2.0)
 
        # Mark particles in the ocean as dead
        state.alive = (state.alive) & (fjord_index[list(map(int,state.Y)),list(map(int,state.X))] > 0) 

  