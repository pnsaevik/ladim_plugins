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

    def test_exact_when_linear_onedim(self):
        x0 = np.array([1, 2, 3])
        t0 = 0
        dt = 1

        def advect_fn(x, _):
            return np.ones_like(x)

        def diffuse_fn(x, _):
            return np.zeros_like(x)

        sol = ibm.sde_solver(x0, t0, advect_fn, diffuse_fn, dt, 'euler')

        assert sol.tolist() == [2, 3, 4]

    def test_not_well_mixed(self):
        np.random.seed(0)
        x0 = np.linspace(0, 1, 100)
        t0 = 0
        dt = 1
        numsteps = 100

        def advect_fn(x, _):
            return np.zeros_like(x)

        def diffuse_fn(x, _):
            return x

        sol = x0
        t = t0
        for i in range(numsteps):
            sol = ibm.sde_solver(sol, t, advect_fn, diffuse_fn, dt, 'euler')
            # Reflective boundaries
            sol[sol < 0] *= -1
            sol[sol > 1] = 2 - sol[sol > 1]

        # Check distribution
        hist_num, hist_edges = np.histogram(sol, bins=[0, .5, 1])
        assert hist_num[0] > len(x0) * .9  # More entries is low-diffusive region
        assert hist_num[1] < len(x0) * .1  # Less entries in high-diffusive region
