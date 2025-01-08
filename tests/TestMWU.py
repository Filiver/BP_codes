import pytest
import numpy as np
import numpy.testing as testing
from MWU import multiplicative_weights_update
from DO import lp, solve_sub_game_scipy
from benchmark import generate_gaussian_sum_game_of_size_gen

epsilons = [1e-1, 1e-2, 1e-3]

class TestMWU:
    @pytest.mark.parametrize("epsilon", epsilons)
    def test_matching_pennies(self, epsilon):
        matching_pennies = np.array([[1, -1], [-1, 1]])
        result = multiplicative_weights_update(matching_pennies, epsilon=epsilon)
        expected_row_strategy = np.array([0.5, 0.5])
        expected_column_strategy = np.array([0.5, 0.5])
        expected_value = 0
        testing.assert_allclose(result['row_strategy'], expected_row_strategy)
        testing.assert_allclose(result['column_strategy'], expected_column_strategy)
        assert np.abs(result['value'] - expected_value) < epsilon

    @pytest.mark.parametrize("epsilon", epsilons)
    def test_rocks_paper_scissors(self, epsilon):
        rocks_paper_scissors = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        result = multiplicative_weights_update(rocks_paper_scissors, epsilon=epsilon)
        expected_row_strategy = np.array([1/3, 1/3, 1/3])
        expected_column_strategy = np.array([1/3, 1/3, 1/3])
        expected_value = 0
        testing.assert_allclose(result['row_strategy'], expected_row_strategy)
        testing.assert_allclose(result['column_strategy'], expected_column_strategy)
        assert np.abs(result['value'] - expected_value) < epsilon

    @pytest.mark.parametrize("epsilon", epsilons)
    def test_penalty_kicks(self, epsilon):
        penalty_kicks = np.array([[0.58, 0.95], [0.93, 0.7]])
        result = multiplicative_weights_update(penalty_kicks, epsilon=epsilon)
        expected_row_strategy = np.array([0.38, 0.62])
        expected_column_strategy = np.array([0.42, 0.58])
        expected_value = 0.79584
        # testing.assert_allclose(result['row_strategy'], expected_row_strategy, atol=1e-2)
        # testing.assert_allclose(result['column_strategy'], expected_column_strategy, atol=1e-2)
        assert np.abs(result['value'] - expected_value) < epsilon

    @pytest.mark.parametrize("epsilon", epsilons)
    def test_morra(self, epsilon):
        morra = np.array((
            (0, 2, 2, -3, 0, 0, -4, 0, 0),
            (-2, 0, 0, 0, 3, 3, -4, 0, 0),
            (-2, 0, 0, -3, 0, 0, 0, 4, 4),
            (3, 0, 3, 0, -4, 0, 0, -5, 0),
            (0, -3, 0, 4, 0, 4, 0, -5, 0),
            (0, -3, 0, 0, -4, 0, 5, 0, 5),
            (4, 4, 0, 0, 0, -5, 0, 0, -6),
            (0, 0, -4, 5, 5, 0, 0, 0, -6),
            (0, 0, -4, 0, 0, -5, 6, 6, 0)
        ), dtype=float)
        result = multiplicative_weights_update(morra, epsilon=epsilon)
        expected_row_strategy = np.array([0, 0, 5/12, 0, 4/12, 0, 3/12, 0, 0])
        expected_column_strategy = np.array([0, 0, 5/12, 0, 4/12, 0, 3/12, 0, 0])
        expected_value = 0
        # testing.assert_allclose(result['row_strategy'], expected_row_strategy)
        # testing.assert_allclose(result['column_strategy'], expected_column_strategy)
        assert np.abs(result['value'] - expected_value) < epsilon

    @pytest.mark.parametrize("size", [100, 300])
    @pytest.mark.parametrize("epsilon", epsilons)
    def test_random_same_size_game(self, size, epsilon):
        np.random.seed(0)
        game = generate_gaussian_sum_game_of_size_gen(size)
        result = multiplicative_weights_update(game, epsilon=epsilon)
        result_lp = lp(game, solver=solve_sub_game_scipy)
        assert np.abs(result['value'] - result_lp['value']) < epsilon




