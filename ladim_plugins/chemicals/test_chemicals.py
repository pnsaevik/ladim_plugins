import numpy as np
from ladim_plugins.chemicals import IBM


class Test_update:
    def test_kills_old_particles(self):
        ibm_conf = {
            'land_collision': 'freeze',
            'vertical_advection': False,
            'lifespan': 20,
        }
        conf = {'dt': 10, 'ibm': ibm_conf}
        state = Stub()
        state['age'] = np.array([0, 10, 20])
        state['alive'] = np.array([False, True, True])
        ibm = IBM(conf)
        ibm.update_ibm(grid=None, state=state, forcing=None)
        assert state['alive'].tolist() == [False, True, False]


class Stub:
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)
