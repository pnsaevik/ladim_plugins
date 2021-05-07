import ladim_plugins.shrimp
import numpy as np


class Dummy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, item):
        return self.kwargs[item]

    def __setitem__(self, key, value):
        self.kwargs[key] = value

    def __getitem__(self, item):
        return self.kwargs[item]


class Test_diel_migration:
    def test_diel_transition(self):
        # Test vertical distribution of particles with time when particles start at the
        # upper layers and diel migration is turned on

        np.random.seed(0)

        num = 1000

        conf = dict(
            ibm=dict(
                stage_duration=[100],
                vertical_mixing=[1e-2],
                vertical_speed=[2e-3],
                mindepth_day=[150],
                maxdepth_day=[200],
                mindepth_night=[0],
                maxdepth_night=[50],
            ),
            dt=3600,
        )

        state = Dummy(
            X=np.zeros(num) + 20,
            Y=np.zeros(num) + 70,
            Z=np.random.uniform(0, 40, size=num),
            stage=np.ones(num),
            age=np.zeros(num),
            timestamp=np.datetime64('2000-03-01T00:00:00'),
        )

        grid = Dummy()
        grid.lonlat = lambda x, y: (x, y)

        ibm = ladim_plugins.shrimp.IBM(conf)

        def single_timestep():
            state.timestamp += np.timedelta64(conf['dt'], 's')
            ibm.update_ibm(grid, state, None)

        # Burn-in 10 days, ending at mid-night
        for i in range(240):
            single_timestep()

        depth_night, _ = np.histogram(state.Z, bins=np.arange(0, 210, 20))
        assert depth_night.tolist() == [35, 173, 392, 190, 121, 66, 17, 6, 0, 0]

        # Proceed to mid-day
        for i in range(12):
            single_timestep()

        depth_day, _ = np.histogram(state.Z, bins=np.arange(0, 210, 20))
        assert depth_day.tolist() == [0, 27, 92, 212, 273, 224, 112, 52, 8, 0]


# def test_snapshot():
#     import ladim_plugins.tests.test_examples
#     import os
#     os.chdir(os.path.dirname(ladim_plugins.tests.test_examples.__file__))
#     ladim_plugins.tests.test_examples.test_output_matches_snapshot('shrimp')
