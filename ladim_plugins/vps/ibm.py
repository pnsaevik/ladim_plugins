import numpy as np
from scipy.ndimage import generic_filter, binary_dilation


class IBM:
    def __init__(self, config):
        self.dt = config["dt"]
        self.max_depth = config["ibm"].get('max_depth', 2)  # [m]
        self.max_age = 2**30  # [s]

        self.grid = None
        self.state = None
        self.forcing = None

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.state = state
        self.forcing = forcing

        # Vertical positioning
        state['Z'] = np.random.uniform(0, self.max_depth, size=len(state['Z']))

        # Ageing
        state['age'] = state['age'] + self.dt
        is_not_too_old = state['age'] < self.max_age
        state['alive'] &= is_not_too_old

        # If reached ocean
        x = self.state['X']
        y = self.state['Y']
        u, v = self.forcing.forcing.fish_velocity(x, y)
        has_not_reached_ocean = (u != 0) | (v != 0)
        state['alive'] &= has_not_reached_ocean


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


def _descent_filter_type(items):
    up, left, center, right, down = items
    if center <= 0:
        return 0

    nonnegative_neighbours = [n for n in items if n >= 0]
    smallest_neighbour = min(nonnegative_neighbours)

    idx_smallest = next(
        i for i, n
        in enumerate((center, left, right, down, up))
        if n == smallest_neighbour
    )

    return idx_smallest


def descent(weights):
    """
    Compute directional vector for moving towards smaller weights

    Return two arrays u, v of the same size as the input array weights,
    representing the velocities in the x and y direction, respectively, when
    moving from higher to smaller weights. Negative weights are treated as
    obstacles.

    The function is based on taxicab connectivity, which means that the
    velocity is either left (u = -1, v = 0), right (u = 1, v = 0),
    up (u = 0, v = 1), down (u = 0, v = -1) or zero (u = v = 0).

    :param weights: Input array weights
    :return: Arrays u, v representing velocity in the direction of smaller weights
    """
    if np.size(weights) == 0:
        return (np.zeros(np.shape(weights)), ) * 2

    # Compute direction
    # 0 = zero, 1 = left, 2 = right, 3 = down, 4 = up
    idx_direction = generic_filter(
        input=weights,
        function=_descent_filter_type,
        footprint=_TAXICAB_FOOTPRINT,
        mode='constant',
        cval=-1,
    )

    u_values = np.array([0, -1, 1, 0, 0])
    v_values = np.array([0, 0, 0, -1, 1])

    u = u_values[idx_direction]
    v = v_values[idx_direction]

    return u, v
