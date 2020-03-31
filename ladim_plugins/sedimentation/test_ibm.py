import numpy as np
from ladim_plugins.sedimentation import ibm


class Test_ladis:
    def test_exact_when_trivial(self):
        x0 = np.array([[1, 2, 3], [4, 5, 6]])
        t0 = 0
        dt = 1

        def advect_fn(x, _):
            return np.zeros_like(x)

        def diffuse_fn(x, _):
            return np.zeros_like(x)

        sol = ibm.ladis(x0, t0, t0 + dt, advect_fn, diffuse_fn)

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

        sol = ibm.ladis(x0, t0, t0 + dt, advect_fn, diffuse_fn)

        assert sol.tolist() == [[2, 4, 6], [5, 7, 9]]

    def test_exact_when_linear_onedim(self):
        x0 = np.array([1, 2, 3])
        t0 = 0
        dt = 1

        def advect_fn(x, _):
            return np.ones_like(x)

        def diffuse_fn(x, _):
            return np.zeros_like(x)

        sol = ibm.ladis(x0, t0, t0 + dt, advect_fn, diffuse_fn)

        assert sol.tolist() == [2, 3, 4]

    def test_well_mixed_when_sqrt(self):
        np.random.seed(0)
        x0 = np.linspace(0, 1, 1001)[1:]
        t0 = 0
        dt = .001
        numsteps = 1000

        def advect_fn(x, _):
            return np.zeros_like(x)

        def diffuse_fn(x, _):
            return np.sqrt(2*x)

        sol = x0
        t = t0
        for i in range(numsteps):
            sol = ibm.ladis(x0, t, t + dt, advect_fn, diffuse_fn)
            # Reflective boundaries
            sol[sol > 1] = 2 - sol[sol > 1]
            sol[sol < 0] *= -1

        # Check distribution
        hist_num, hist_edges = np.histogram(sol, bins=np.arange(0, 1.05, .1))

        too_low = hist_num < (len(x0) / len(hist_num)) * .8
        too_high = hist_num > (len(x0) / len(hist_num)) * 1.2

        assert not np.any(too_low)
        assert not np.any(too_high)

    def test_well_mixed_when_linear(self):
        np.random.seed(0)
        x0 = np.linspace(0, 1, 1001)[1:]
        t0 = 0
        dt = .001
        numsteps = 1000

        def advect_fn(x, _):
            return np.zeros_like(x)

        def diffuse_fn(x, _):
            return x

        sol = x0
        t = t0
        for i in range(numsteps):
            sol = ibm.ladis(x0, t, t + dt, advect_fn, diffuse_fn)
            # Reflective boundaries
            sol[sol > 1] = 2 - sol[sol > 1]
            sol[sol < 0] *= -1

        # Check distribution
        hist_num, hist_edges = np.histogram(sol, bins=np.arange(0, 1.05, .1))

        too_low = hist_num < (len(x0) / len(hist_num)) * .8
        too_high = hist_num > (len(x0) / len(hist_num)) * 1.2

        assert not np.any(too_low)
        assert not np.any(too_high)
