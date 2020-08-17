import numpy as np
from ladim_plugins.sedimentation import ibm
import pytest


class Stub:
    def __init__(self, **kwargs):
        self._dic = kwargs

    def __getattr__(self, item):
        return self._dic[item]


class Test_update:
    @staticmethod
    def gsf(num, hvel=0, wvel=0, dt=1):
        zr = np.zeros(num)
        zrl = np.zeros_like
        grid = Stub(
            sample_depth=lambda x, y: zrl(x) + 10,
            lonlat=lambda x, y: (x, y),
        )
        forcing = Stub(velocity=lambda x, y, z, tstep=0: [zrl(x) + hvel] * 2)
        state = Stub(
            X=zr*0, Y=zr*0, Z=zr + 10, active=zr*0, alive=zr == 0, age=zr*0,
            sink_vel=zr + wvel, dt=dt, timestep=0,
        )
        return grid, state, forcing

    def test_does_not_resuspend_when_zero_velocity(self):
        ibmconf = dict(lifespan=100, taucrit=0.12, vertical_mixing=0.01)
        grid, state, forcing = self.gsf(num=5)
        config = dict(dt=state.dt, ibm=ibmconf)
        my_ibm = ibm.IBM(config)

        my_ibm.update_ibm(grid, state, forcing)

        assert np.all(state.Z == 10)

    def test_does_resuspend_when_high_velocity(self):
        ibmconf = dict(lifespan=100, taucrit=0.12, vertical_mixing=0.01)
        grid, state, forcing = self.gsf(num=5, hvel=1)
        config = dict(dt=state.dt, ibm=ibmconf)
        my_ibm = ibm.IBM(config)

        my_ibm.update_ibm(grid, state, forcing)

        assert np.all(state.Z < 10)

    def test_resuspended_particles_have_altered_active_flag(self):
        ibmconf = dict(lifespan=100, taucrit=0.12, vertical_mixing=0.01)
        grid, state, forcing = self.gsf(num=5, hvel=1)
        config = dict(dt=state.dt, ibm=ibmconf)
        my_ibm = ibm.IBM(config)

        my_ibm.update_ibm(grid, state, forcing)

        assert np.all(state.active == 2)

    def test_does_not_resuspend_when_zero_diffusion(self):
        ibmconf = dict(lifespan=100, taucrit=0.12, vertical_mixing=0)
        grid, state, forcing = self.gsf(num=5, hvel=1)
        config = dict(dt=state.dt, ibm=ibmconf)
        my_ibm = ibm.IBM(config)

        my_ibm.update_ibm(grid, state, forcing)

        assert np.all(state.Z == 10)

    def test_does_not_resuspend_when_large_sinkvel(self):
        # On rare occasions, vertical mixing can overcome the sinking velocity
        np.random.seed(0)

        ibmconf = dict(lifespan=100, taucrit=0.12, vertical_mixing=0.01)
        grid, state, forcing = self.gsf(num=5, hvel=1, wvel=1)
        config = dict(dt=state.dt, ibm=ibmconf)
        my_ibm = ibm.IBM(config)

        my_ibm.update_ibm(grid, state, forcing)

        assert np.all(state.Z == 10)


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


def get_grainsize_fixture_fname():
    import ladim_plugins.tests
    import os
    return os.path.join(
        os.path.dirname(ladim_plugins.tests.__file__), 'grainsize.nc')


class Test_taucrit_grain_size_bin:
    @pytest.mark.parametrize("method", ['grain_size_bin', 'grain_size_poly'])
    def test_varying_taucrit_when_regular_grid(self, method):
        grainsize_fname = get_grainsize_fixture_fname()
        ibmconf = dict(
            lifespan=100,
            taucrit=dict(
                method=method,
                source=grainsize_fname,
                varname='grain_size',
            ),
            vertical_mixing=0.01,
        )
        config = dict(dt=1, ibm=ibmconf)
        my_ibm = ibm.IBM(config)

        num = 5
        vel = .12  # Tuned to give tau ~ 0.09
        w = 0

        zr = np.zeros(num)
        rng = np.arange(num)
        zrl = np.zeros_like
        grid = Stub(
            sample_depth=lambda x, y: zrl(x) + 10,
            lonlat=lambda x, y: (
                5.651 + x*0.01,
                59.021 + y*0.01,
            ),
        )
        forcing = Stub(velocity=lambda x, y, z, tstep=0: [zrl(x) + vel] * 2)
        state = Stub(
            X=rng, Y=rng, Z=zr + 10, active=zr, alive=zr == 0, age=zr,
            sink_vel=zr + w, dt=config['dt'], timestep=1,
        )

        my_ibm.update_ibm(grid, state, forcing)

        assert np.any(state.Z < 10)
        assert np.any(state.Z == 10)


class Test_vertical_mixing:
    @staticmethod
    def gfs(num, hvel=0, wvel=0, dt=1):
        zr = np.zeros(num)
        zrl = np.zeros_like
        grid = Stub(
            sample_depth=lambda x, y: zrl(x) + 10,
            lonlat=lambda x, y: (x, y),
        )
        forcing = Stub(velocity=lambda x, y, z, tstep=0: [zrl(x) + hvel] * 2)
        state = Stub(
            X=zrl(zr), Y=zrl(zr), Z=zr + 10, active=zrl(zr), alive=zr == 0,
            age=zrl(zr), sink_vel=zr + wvel, dt=dt, timestep=0,
        )
        return grid, forcing, state

    def test_vertical_movement_when_method_constant(self):
        ibmconf = dict(
            lifespan=100,
            taucrit=0,
            vertical_mixing=dict(
                method='constant',
                value=0.01,
            ),
        )

        grid, forcing, state = self.gfs(num=5)
        config = dict(dt=state.dt, ibm=ibmconf)
        my_ibm = ibm.IBM(config)

        assert np.all(state.Z == 10)
        my_ibm.update_ibm(grid, state, forcing)
        assert np.all(state.Z != 10)

    def test_some_initial_lift_when_method_bounded_linear(self):
        np.random.seed(0)

        ibmconf = dict(
            lifespan=100,
            taucrit=0.12,
            vertical_mixing=dict(
                method='bounded_linear',
                max_diff=0.01,
            ),
        )

        grid, forcing, state = self.gfs(num=5, hvel=1)
        config = dict(dt=state.dt, ibm=ibmconf)
        my_ibm = ibm.IBM(config)

        my_ibm.update_ibm(grid, state, forcing)

        assert np.any(state.Z != 10)
        assert np.any(state.Z == 10)
