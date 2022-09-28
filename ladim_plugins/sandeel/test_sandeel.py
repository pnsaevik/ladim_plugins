from ladim_plugins.sandeel import ibm
import numpy as np

# from ..tests import test_examples


# def test_snapshot():
#    test_examples.test_output_matches_snapshot('sandeel')


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
