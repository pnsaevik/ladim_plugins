import numpy as np
from ladim_plugins import inertial


class Stub:
    def __init__(self, **kwargs):
        self._dic = kwargs

    def __getattr__(self, item):
        return self._dic[item]


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
        sinkvel=zr + wvel, dt=dt, timestep=0, density=1100,
    )
    return grid, state, forcing


class Test_IBM_update:
    def test_particles_do_not_sink(self):
        config = dict(dt=1)
        grid, state, forcing = gsf(5, wvel=1, dt=config['dt'])

        depth_before = state.Z.tolist()

        ibm = inertial.IBM(config)
        ibm.update_ibm(grid, state, forcing)

        depth_after = state.Z.tolist()

        assert depth_before == depth_after


def test_snapshot():
    import ladim_plugins.tests.test_examples
    ladim_plugins.tests.test_examples.test_output_matches_snapshot('inertial')
