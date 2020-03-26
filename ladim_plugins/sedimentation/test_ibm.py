import numpy as np
from ladim_plugins.sedimentation import ibm


class Test_sde_solver_euler:
    def test_exact_when_trivial(self):
        x0 = np.array([[1, 2, 3], [4, 5, 6]])
        t0 = 0
        dt = 1

        def advect_fn(x, _):
            return np.zeros_like(x)

        def diffuse_fn(x, _):
            return np.zeros_like(x)

        sol = ibm.sde_solver(x0, t0, advect_fn, diffuse_fn, dt, 'euler')

        assert sol.shape == x0.shape
        assert sol.tolist() == x0.tolist()

    def test_exact_when_linear(self):
        x0 = np.array([[1, 2, 3], [4, 5, 6]])
        t0 = 0
        dt = 1

        def advect_fn(x, _):
            return np.ones_like(x) * [1, 2, 3]

        def diffuse_fn(x, _):
            return np.zeros_like(x)

        sol = ibm.sde_solver(x0, t0, advect_fn, diffuse_fn, dt, 'euler')

        assert sol.tolist() == [[2, 4, 6], [5, 7, 9]]
