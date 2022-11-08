from ladim_plugins import salmon_lice
import numpy as np


class Test_infectivity:
    def test_returns_zero_outside_age_range(self):
        age = np.array([0, 1000, -50])
        temp = np.array([10, 10, 10])
        sup = np.array([1, 1, 1])
        infect = salmon_lice.infectivity(age, temp, sup)
        assert infect.tolist() == [0, 0, 0]

    def test_changes_with_age(self):
        age = np.array([50, 100])
        temp = np.array([10, 10])
        sup = np.array([1, 1])
        infect = salmon_lice.infectivity(age, temp, sup)
        assert infect[0] != infect[1]

    def test_extrapolates_temperature_as_constant(self):
        age = np.array([100, 100, 100, 100, 100, 100])
        temp = np.array([5, 3, 1, 15, 16, 17])
        sup = np.array([1, 1, 1, 1, 1, 1])
        infect = salmon_lice.infectivity(age, temp, sup)
        # Below 5 degrees: Returns same as 5 degrees
        assert infect[0] == infect[1]
        assert infect[1] == infect[2]

        # 5 degrees is different from 15 degrees
        assert infect[2] != infect[3]

        # Above 15 degrees: Returns same as 15 degrees
        assert infect[3] == infect[4]
        assert infect[4] == infect[5]

    def test_scales_linearly_with_super(self):
        age = np.array([100, 100, 100])
        temp = np.array([10, 10, 10])
        sup = np.array([1, 2, 4])
        infect = salmon_lice.infectivity(age, temp, sup)
        assert infect[0]*2 == infect[1]
        assert infect[1]*2 == infect[2]
