from ladim_plugins.vps import ibm
import numpy as np


class Test_dilate:
    def test_accepts_empty_matrix(self):
        result = ibm.dilate([[]])
        assert result.tolist() == [[]]

    def test_accepts_single_element_matrix(self):
        result = ibm.dilate([[0]])
        assert result.tolist() == [[0]]

    def test_no_change_if_center_is_nonnegative(self):
        matrix = np.array([
            [-1, 1, -1],
            [-1, 0, 2],
            [-1, -1, -1],
        ])
        result = ibm.dilate(matrix)
        assert result[1, 1] == 0

    def test_no_change_if_center_is_obstacle(self):
        matrix = np.array([
            [-1, 1, -1],
            [-1, -2, 2],
            [-1, -1, -1],
        ])
        result = ibm.dilate(matrix)
        assert result[1, 1] == -2

    def test_no_change_if_all_neighbours_are_negative(self):
        matrix = np.array([
            [-1, -2, -1],
            [-1, -1, -2],
            [-1, -1, -1],
        ])
        result = ibm.dilate(matrix)
        assert result[1, 1] == -1

    def test_adds_one_to_smallest_nonnegative_neighbour(self):
        matrix = np.array([
            [-1, -2, -1],
            [2, -1, -2],
            [-1, 1, -1],
        ])
        result = ibm.dilate(matrix)
        assert result[1, 1] == 2


class Test_distance:
    def test_can_compute_distance_if_no_obstacles(self):
        matrix = np.array([
            [-1, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, -1, -1],
        ])
        result = ibm.distance(matrix)
        assert result.tolist() == [
            [2, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 2, 3],
        ]

    def test_can_compute_distance_if_obstacles_and_multiple_sources(self):
        matrix = np.array([
            [0, -1, -1, -1],
            [-2, -1, -2, -1],
            [-1, -2, 0, -1],
        ])
        result = ibm.distance(matrix)
        assert result.tolist() == [
            [0, 1, 2, 3],
            [-2, 2, -2, 2],
            [-1, -2, 0, 1],
        ]

    def test_respects_max_dist(self):
        matrix = np.array([
            [-1, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, -1, -1],
        ])
        result = ibm.distance(matrix, max_dist=2)
        assert result.tolist() == [
            [2, 1, 2, -1],
            [1, 0, 1, 2],
            [2, 1, 2, -1],
        ]
