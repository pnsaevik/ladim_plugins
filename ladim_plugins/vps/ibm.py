import numpy as np
from scipy.ndimage import generic_filter, binary_dilation


class IBM:
    def __init__(self, config):
        self.D = config["ibm"].get('vertical_mixing', 1e-4)  # Vertical mixing [m*2/s]
        self.vertical_diffusion = self.D > 0
        self.dt = config["dt"]
        self.fjord_index_file = config["ibm"]["fjord_index_file"]

        self.grid = None
        self.state = None
        self.forcing = None

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.state = state
        self.forcing = forcing

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
        state.alive = state.alive & (fjord_index[list(map(int,state.Y)),list(map(int,state.X))] > 0)


def _dilate_filter(items):
    up, left, center, right, down = items
    if center != -1:
        return center

    nonnegative_neighbours = [n for n in (up, left, right, down) if n >= 0]
    if not nonnegative_neighbours:
        return center
    else:
        return min(nonnegative_neighbours) + 1


_TAXICAB_FOOTPRINT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def dilate(matrix):
    """
    Compute dilation of a distance matrix

    A "distance matrix" is a numpy array of integers where the value of each
    element represent the taxicab distance to some sources, marked as zeros
    in the matrix. There are also some special values: -1 means that the cell
    has an unknown distance to the nearest source, and -2 means that the cell
    represents an obstacle. A "dilation" means that we assign a positive
    value to all cells on the boundary, i.e., cells with value -1 that are
    also neighbours to a nonnegative cell. The new value is one larger than
    the smallest nonnegative neighbour. Repeated applications of the dilate
    function will build a distance matrix where all positive values represent
    the taxicab distance to the sources.

    :param matrix: The input distance matrix
    :return: Updated distance matrix
    """

    return generic_filter(
      input=matrix,
      function=_dilate_filter,
      footprint=_TAXICAB_FOOTPRINT,
      mode='constant',
      cval=-2,
    )


def distance(matrix, max_dist=None):
    """
    Compute distance matrix

    A "distance matrix" is a numpy array of integers where the value of each
    element represent the taxicab distance to some sources. In the input matrix,
    sources should be marked as zeros, obstacles should be marked as -2, and all
    other elements should be -1. In the output matrix, all elements of -1 are
    replaced with the shortest distance to a nearby source cell, if any route is
    found.

    :param matrix: An initial matrix, containing values of -2 (obstacle), -1
        (unknown distance) and 0 (source cells)
    :param max_dist: If defined, set the maximal number of dilation iterations
        to perform.
    :return: Distance matrix
    """
    matrix = np.asarray(matrix)
    assert len(matrix.shape) == 2

    distmat = matrix
    if max_dist is None:
      max_dist = np.size(matrix)

    for i in range(max_dist):
        old_distmat = distmat
        distmat = dilate(distmat)
        if np.all(distmat == old_distmat):
            break

    return distmat


def fjord_index(land, ocean_dist):
    """
    Compute fjord index matrix

    Computes the distance from any sea cell to the open ocean. An ocean cell is
    defined as any cell that is "far from land", i.e., the distance to land is
    larger than ocean_dist. Distance to land is measured as the number of cells
    in the shortest (taxicab distance) sea-cell route to land.

    :param land: A matrix where 1 denotes land cells and 0 denotes sea cells
    :param ocean_dist: Distance to open ocean (number of cells)
    :return: Fjord index matrix
    """
    land = np.asarray(land).astype(bool).astype('int32')

    # This function gives 0 on ocean and -1 elsewhere
    is_not_ocean = binary_dilation(
        input=land,
        structure=_TAXICAB_FOOTPRINT,
        iterations=ocean_dist - 1,
    )

    # This formula gives 0 on ocean, -2 on land and -1 elsewhere
    input_matrix = -np.asarray(is_not_ocean, dtype='int32') - land

    return distance(input_matrix)