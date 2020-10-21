import numpy as np
import pytest
from ladim_plugins.chemicals import gridforce, IBM
from contextlib import contextmanager
from typing import Any


class Test_nearest_unmasked:
    def test_correct_when_all_unmasked(self):
        mask = np.zeros((4, 3))
        i = np.array([1, 1, 2])
        j = np.array([2, 3, 3])
        ii, jj = gridforce.nearest_unmasked(mask, i, j)
        assert ii.tolist() == i.tolist()
        assert jj.tolist() == j.tolist()

    def test_correct_when_south_edge(self):
        mask_south = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
        i = np.array([0, 1, 2.51])
        j = np.array([0, 1, 1.49])
        ii, jj = gridforce.nearest_unmasked(mask_south, i, j)
        assert ii.tolist() == [0, 1, 3]
        assert jj.tolist() == [0, 0, 0]

    def test_correct_when_corner(self):
        mask = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0]])
        i = np.array([0.51, 0.51, 0.99, 1.49, 1.51, 2.00, 3.00])
        j = np.array([0.52, 0.98, 0.52, 1.01, 1.01, 1.01, 1.01])
        ii, jj = gridforce.nearest_unmasked(mask, i, j)
        assert ii.tolist() == [0, 0, 1, 1, 2, 2, 3]
        assert jj.tolist() == [1, 1, 0, 0, 0, 0, 2]


class Test_ibm_land_collision:
    @pytest.fixture()
    def config(self):
        from ladim_plugins.tests.test_examples import get_config
        from ladim.configuration import configure
        return configure(get_config('chemicals'))

    @pytest.fixture()
    def grid(self, config):
        from ladim.gridforce import Grid
        return Grid(config)

    @pytest.fixture()
    def forcing(self, config, grid):
        from ladim.gridforce import Forcing
        return Forcing(config, grid)

    @pytest.fixture()
    def state(self, config, grid):
        from ladim.state import State
        return State(config, grid)

    @pytest.fixture()
    def ibm_chemicals(self, config):
        return IBM(config)

    def test_land_collision(self, ibm_chemicals, grid, state, forcing):
        np.random.seed(0)

        state.X = np.float32([1, 1, 1])
        state.Y = np.float32([1, 1, 1])
        state.Z = np.float32([1, 1, 1])
        state.pid = np.int32([0, 1, 2])
        ibm_chemicals.update_ibm(grid, state, forcing)

        assert state.X.tolist() == [1, 1, 1]
        assert state.Y.tolist() == [1, 1, 1]

        state.X = np.float32([1, 2, 3, 4])
        state.Y = np.float32([1, 1, 1, 1])
        state.Z = np.float32([1, 1, 1, 1])
        state.pid = np.int32([1, 2, 3, 4])
        ibm_chemicals.update_ibm(grid, state, forcing)

        assert state.X.tolist() == [1.4636627435684204, 2, 3, 4]
        assert state.Y.tolist() == [0.8834415078163147, 1, 1, 1]


class Test_compute_w:
    def test_requires_correct_shape(self):
        pn = np.ones((10, 15))
        pm = pn
        u = np.zeros((1, 20, 10, 14))
        v = np.zeros((1, 20, 9, 15))
        z_w = np.zeros((1, 21, 10, 15))
        for i in range(21):
            z_w[:, i, :, :] = i
        z_r = z_w[:, :-1, :, :] + 0.5
        w = gridforce.compute_w(pn, pm, u, v, z_w, z_r)
        assert w.shape == z_w.shape

    def test_zero_when_divergence_free_horizontal_velocity(self):
        t, z, eta, xi = np.meshgrid(
            range(1), range(3), range(4), range(5), indexing='ij')

        eta_u = 0.5 * (eta[:, :, :, :-1] + eta[:, :, :, 1:])
        xi_u = 0.5 * (xi[:, :, :, :-1] + xi[:, :, :, 1:])
        eta_v = 0.5 * (eta[:, :, :-1, :] + eta[:, :, 1:, :])
        # xi_v = 0.5 * (xi[:, :, :-1, :] + xi[:, :, 1:, :])

        z_r = z + 0.5
        z_w = np.concatenate((z, 1 + z[:, -1:, :, :]), axis=1)
        pn = np.ones(xi.shape[-2:])
        pm = pn

        # Define divergence-free field
        u = eta_u * xi_u
        v = - eta_v * eta_v

        w = gridforce.compute_w(pn, pm, u, v, z_w, z_r)
        assert np.max(np.abs(w)) < 1e-7

    def test_computes_positive_velocity_when_downward(self):
        t, z, eta, xi = np.meshgrid(
            range(1), range(3), range(4), range(5), indexing='ij')

        z_r = z + 0.5
        z_w = np.concatenate((z, 1 + z[:, -1:, :, :]), axis=1)

        eta_u = 0.5 * (eta[:, :, :, :-1] + eta[:, :, :, 1:])
        xi_u = 0.5 * (xi[:, :, :, :-1] + xi[:, :, :, 1:])
        z_u = 0.5 * (z_r[:, :, :, :-1] + z_r[:, :, :, 1:])
        eta_v = 0.5 * (eta[:, :, :-1, :] + eta[:, :, 1:, :])
        xi_v = 0.5 * (xi[:, :, :-1, :] + xi[:, :, 1:, :])
        z_v = 0.5 * (z_r[:, :, :-1, :] + z_r[:, :, 1:, :])

        pn = np.ones(xi.shape[-2:])
        pm = pn

        # Define convergence on top, divergence on bottom
        # This gives downward movement
        u = (z_u - 1.5) * (eta_u + xi_u)
        v = (z_v - 1.5) * (eta_v + xi_v)

        w = gridforce.compute_w(pn, pm, u, v, z_w, z_r)

        # Outer edges and top+bottom is zero, so we check internal velocity
        w_internal = w[0, 1:-1, 1:-1, 1:-1]
        assert np.all(w_internal > 0), 'Downward velocity should be positive'


class Test_divergence:
    CONST_DT = 600

    @contextmanager
    def get_forcing(self, u: Any = 0.0, v: Any = 0.0, h: Any = 3.0):
        import netCDF4 as nc
        from uuid import uuid4

        # noinspection PyArgumentList
        dset = nc.Dataset(uuid4(), mode='w', format='NETCDF4', memory=1000)

        dset.createDimension('ocean_time', 2)
        dset.createDimension('xi_rho', 5)
        dset.createDimension('eta_rho', 5)
        dset.createDimension('s_rho', 3)
        dset.createDimension('xi_u', 4)
        dset.createDimension('eta_u', 5)
        dset.createDimension('xi_v', 5)
        dset.createDimension('eta_v', 4)
        dset.createDimension('s_w', 4)

        dset.createVariable('ocean_time', 'd', ('ocean_time', ))
        dset.createVariable('h', 'd', ('eta_rho', 'xi_rho'))
        dset.createVariable('mask_rho', 'd', ('eta_rho', 'xi_rho'))
        dset.createVariable('pn', 'd', ('eta_rho', 'xi_rho'))
        dset.createVariable('pm', 'd', ('eta_rho', 'xi_rho'))
        dset.createVariable('lon_rho', 'd', ('eta_rho', 'xi_rho'))
        dset.createVariable('lat_rho', 'd', ('eta_rho', 'xi_rho'))
        dset.createVariable('angle', 'd', ('eta_rho', 'xi_rho'))
        dset.createVariable('u', 'd', ('ocean_time', 's_rho', 'eta_u', 'xi_u'))
        dset.createVariable('v', 'd', ('ocean_time', 's_rho', 'eta_v', 'xi_v'))
        dset.createVariable('hc', 'd', ())
        dset.createVariable('Vtransform', 'd', ())
        dset.createVariable('Cs_r', 'd', ('s_rho', ))
        dset.createVariable('Cs_w', 'd', ('s_w', ))

        dset.variables['ocean_time'].units = 'seconds since 1970-01-01'
        dset.variables['ocean_time'].calendar = 'proleptic_gregorian'

        dset.variables['ocean_time'][:] = [0, 600]
        dset.variables['mask_rho'][:] = 1
        dset.variables['pm'][:] = 0.00625
        dset.variables['pn'][:] = 0.00625
        dset.variables['hc'][:] = 0
        dset.variables['Vtransform'][:] = 2
        dset.variables['Cs_w'][:] = [-1, -2/3, -1/3, 0]
        dset.variables['Cs_r'][:] = [-5/6, -1/2, -1/6]

        x_rho = np.arange(5)
        y_rho = x_rho
        x_u = x_rho[:-1] + 0.5
        y_u = y_rho
        x_v = x_rho
        y_v = x_u
        z_rho = dset.variables['Cs_r'][:] * h

        if callable(h):
            y, x = np.meshgrid(y_rho, x_rho, indexing='ij')
            h = h(x, y)

        if callable(u):
            z, y, x = np.meshgrid(z_rho, y_u, x_u, indexing='ij')
            u = u(x, y, z)

        if callable(v):
            z, y, x = np.meshgrid(z_rho, y_v, x_v, indexing='ij')
            v = v(x, y, z)

        dset.variables['h'][:] = h
        dset.variables['u'][:] = u
        dset.variables['v'][:] = v

        dset_buf = dset.close()

        config = dict(
            gridforce=dict(
                input_file=[dset_buf],
            ),
            start_time=np.datetime64('1970-01-01T00:00:00'),
            stop_time=np.datetime64('1970-01-01T00:00:10'),
            ibm_forcing=[],
            dt=Test_divergence.CONST_DT,
        )
        grid = gridforce.Grid(config)
        forcing = gridforce.Forcing(config, grid)
        yield forcing
        forcing.close()

    @pytest.fixture()
    def ibm(self):
        config = dict(
            ibm=dict(
                land_collision='freeze',
            ),
            dt=self.CONST_DT,
        )
        return IBM(config)

    @staticmethod
    def get_state(forcing, n):
        class MyState:
            def __init__(self, **kwargs):
                self._dict = dict(**kwargs)

            def __getattr__(self, item):
                return self._dict[item]

            def __getitem__(self, item):
                return self._dict[item]

            def copy(self):
                def deepcopy(item):
                    if isinstance(item, dict):
                        return {k: deepcopy(v) for k, v in item.items()}
                    elif hasattr(item, 'copy'):
                        return item.copy()
                    else:
                        return item

                return MyState(**deepcopy(self._dict))

            def todict(self):
                return {
                    k: v.tolist() if hasattr(v, 'tolist') else v
                    for k, v in self._dict.items()
                }

            @property
            def s_rho(self):
                s_int, s_frac = gridforce.z2s(
                    forcing._grid.z_r,
                    self.X - forcing._grid.i0,
                    self.Y - forcing._grid.j0,
                    self.Z,
                )
                return s_int - s_frac

            @property
            def s_w(self):
                s_int, s_frac = gridforce.z2s(
                    forcing._grid.z_w,
                    self.X - forcing._grid.i0,
                    self.Y - forcing._grid.j0,
                    self.Z,
                )
                return s_int - s_frac

            def in_middle(self):
                idx = np.round(self.X) == 2
                idx &= np.round(self.Y) == 2
                idx &= np.round(self.s_rho) == 1
                return idx

        nn = np.int32(np.round(np.power(n, 1 / 3)))
        n = nn ** 3
        z, y, x = np.meshgrid(
            np.linspace(0, 1, nn),
            np.linspace(1, 3, nn + 2)[1:-1],
            np.linspace(1, 3, nn + 2)[1:-1],
            indexing='ij',
        )

        mystate = MyState(
            X=x.ravel(),
            Y=y.ravel(),
            Z=z.ravel(),
            alive=np.ones(n),
            pid=np.arange(n),
            dt=Test_divergence.CONST_DT,
        )

        h = gridforce.sample2D(
            forcing._grid.H,
            mystate.X - 1, mystate.Y - 1,
        )
        mystate.Z *= h
        return mystate

    @staticmethod
    def one_timestep(state, forcing, ibm=None):
        import ladim.tracker
        config = dict(
            advection='RK4',
            diffusion=0,
            ibm_variables=[],
            dt=Test_divergence.CONST_DT,
        )
        tracker = ladim.tracker.Tracker(config)
        newstate = state.copy()
        tracker.move_particles(forcing._grid, forcing, newstate)
        if ibm is not None:
            class Mock:
                pass
            f = Mock()
            f.forcing = forcing
            ibm.update_ibm(forcing._grid, newstate, f)
        return newstate

    def test_no_divergence_if_velocity_is_zero(self):
        with self.get_forcing() as forcing:
            state = self.get_state(forcing, n=1000)
            newstate = self.one_timestep(state, forcing)
            num_init = np.count_nonzero(state.in_middle())
            num_after = np.count_nonzero(newstate.in_middle())
            assert num_init == num_after

    def test_no_divergence_if_linear_horz_velocity(self):
        with self.get_forcing(.1, .05) as forcing:
            state = self.get_state(forcing, n=1000)
            newstate = self.one_timestep(state, forcing)

            idx_init = state.in_middle()
            idx_after = newstate.in_middle()
            assert np.any(idx_init != idx_after), \
                "Some particles should exit or enter the middle cell"

            num_init = np.count_nonzero(state.in_middle())
            num_after = np.count_nonzero(newstate.in_middle())
            assert num_init == num_after, \
                "Number of particles in middle cell should not change"

    def test_accumulation_if_torus_velocity_and_vertical_adv_off(self):
        u = np.array([[
            [[0] * 4, [0] * 4, [0, -1, 1, 0], [0] * 4, [0] * 4],
            [[0] * 4, [0] * 4, [0, 2, -2, 0], [0] * 4, [0] * 4],
            [[0] * 4, [0] * 4, [0, -1, 1, 0], [0] * 4, [0] * 4],
        ]] * 2) * 0.1

        v = np.swapaxes(u, 2, 3)

        with self.get_forcing(u, v) as forcing:
            n = 9
            state = self.get_state(forcing, n=n**3)
            newstate = self.one_timestep(state, forcing)

            idx_init = state.in_middle()
            idx_after = newstate.in_middle()
            assert np.any(idx_init != idx_after), \
                "Some particles should exit or enter the middle cell"

            num_init = np.count_nonzero(state.in_middle())
            num_after = np.count_nonzero(newstate.in_middle())
            assert num_init * 2 < num_after, \
                "Strong accumulation expected"

    def test_less_accumulation_if_torus_velocity_and_vertical_adv_on(self, ibm):
        u = np.array([[
            [[0] * 4, [0] * 4, [0, -1, 1, 0], [0] * 4, [0] * 4],
            [[0] * 4, [0] * 4, [0, 2, -2, 0], [0] * 4, [0] * 4],
            [[0] * 4, [0] * 4, [0, -1, 1, 0], [0] * 4, [0] * 4],
        ]] * 2) * 0.1

        v = np.swapaxes(u, 2, 3)

        with self.get_forcing(u, v) as forcing:
            n = 9
            state = self.get_state(forcing, n=n**3)
            newstate = self.one_timestep(state, forcing, ibm)

            idx_init = state.in_middle()
            idx_after = newstate.in_middle()
            assert np.any(idx_init != idx_after), \
                "Some particles should exit or enter the middle cell"

            num_init = np.count_nonzero(state.in_middle())
            num_after = np.count_nonzero(newstate.in_middle())
            assert num_init < num_after < 1.2 * num_init, \
                "Very little accumulation expected"
