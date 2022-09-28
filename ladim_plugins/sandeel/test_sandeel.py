from ladim_plugins.sandeel import ibm
import numpy as np

# from ..tests import test_examples


# def test_snapshot():
#    test_examples.test_output_matches_snapshot('sandeel')

class Test_larval_development:
    def test_increases_only_larval_stages(self):
        state = dict(
            temp=np.zeros(4),
            stage=np.array([0, .5, 1, 1.5]),
            dt=1,
        )
        ibm.larval_development(**state)
        assert state['stage'][0] == 0
        assert state['stage'][1] == .5
        assert state['stage'][2] > 1
        assert state['stage'][3] > 1.5

    # def test_looks_nice(self):
    #     L = []
    #     state = dict(
    #         temp=np.array([0, 5, 10]),
    #         stage=np.array([1., 1., 1.]),
    #         dt=60*60*24,
    #     )
    #     days = range(50)
    #     for _ in days:
    #         L.append(np.copy(state['stage']))
    #         ibm.larval_development(**state)
    #
    #     import matplotlib.pyplot as plt
    #     plt.plot(days, np.array(L))
    #     plt.xlabel("days")
    #     plt.ylabel("stage")
    #     plt.show()


class Test_egg_development:
    def test_increases_only_egg_stages(self):
        state = dict(
            temp=np.zeros(4),
            stage=np.array([0, .5, 1, 1.5]),
            active=np.zeros(4),
            hatch_rate=np.zeros(4),
            dt=1,
        )
        ibm.egg_development(**state)
        assert state['stage'][0] > 0
        assert state['stage'][1] > .5
        assert state['stage'][2] == 1
        assert state['stage'][3] == 1.5

    def test_activates_hatched_eggs(self):
        state = dict(
            temp=np.zeros(2),
            stage=np.array([0, .99999999]),
            active=np.zeros(2, dtype=bool),
            hatch_rate=np.zeros(2),
            dt=1,
        )
        ibm.egg_development(**state)

        assert state['stage'][0] < 1
        assert state['active'][0] == 0

        assert state['stage'][1] >= 1
        assert state['active'][1] != 0


class Test_hatch_time:
    def test_can_reproduce_table_from_paper(self):
        rate, temp = np.meshgrid([0, .5, 1], [2, 4, 7, 10], indexing='ij')
        days = ibm.hatch_time(rate, temp)

        assert days.round().astype(int).tolist() == [
            [61, 51, 39, 25],
            [82, 67, 48, 30],
            [135, 116, 82, 55]
        ]

    # def test_looks_nice(self):
        #
        # import matplotlib.pyplot as plt
        #
        # t = np.linspace(0, 10, 100)
        # r = np.array([0, .5, 1])
        # T, R = np.meshgrid(t, r, indexing='ij')
        # D = ibm.hatch_time(R, T)
        # plt.plot(T, D)
        # plt.ylim([0, 200])
        # plt.xlim([2, 10])
        # plt.show()
        #
        # t = np.array([4, 6, 8])
        # r = np.linspace(0, 1, 100)
        # T, R = np.meshgrid(t, r, indexing='ij')
        # D = ibm.hatch_time(R, T)
        # plt.plot(D.T, R.T)
        # plt.xlim([0, 200])
        # plt.show()
        #
        # t = np.linspace(2, 10, 100)
        # r = np.linspace(0, 1, 100)
        # T, R = np.meshgrid(t, r, indexing='ij')
        # D = ibm.hatch_time(R, T)
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(R, T, D, cmap='viridis', edgecolor='none')
        # #plt.xlim([0, 200])
        # plt.show()
        # pass
        #
        # pass
