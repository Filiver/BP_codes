import pytest
import numpy as np
import numpy.testing as testing
from DO import double_oracle, solve_sub_game_gurobi

epsilons = [1e-1, 1e-2, 1e-3]

class TestDO:
    @pytest.mark.parametrize("epsilon", epsilons)
    def test_matching_pennies(self, epsilon):
        matching_pennies = np.array([[1, -1], [-1, 1]])
        result = double_oracle(matching_pennies, epsilon=epsilon, solver=solve_sub_game_gurobi)
        expected_value = 0
        assert np.abs(result['value'] - expected_value) < epsilon

    def test_matching_pennies_exact(self):
        matching_pennies = np.array([[1, -1], [-1, 1]])
        result = double_oracle(matching_pennies, solver=solve_sub_game_gurobi)
        expected_row_strategy = np.array([0.5, 0.5])
        expected_column_strategy = np.array([0.5, 0.5])
        expected_value = 0
        testing.assert_allclose(result['row_strategy'], expected_row_strategy)
        testing.assert_allclose(result['column_strategy'], expected_column_strategy)
        assert np.abs(result['value'] - expected_value) < 1e-12

    @pytest.mark.parametrize("epsilon", epsilons)
    def test_rocks_paper_scissors(self, epsilon):
        rocks_paper_scissors = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        result = double_oracle(rocks_paper_scissors, epsilon=epsilon, solver=solve_sub_game_gurobi)
        expected_value = 0
        assert np.abs(result['value'] - expected_value) < epsilon

    def test_rocks_paper_scissors_exact(self):
        rocks_paper_scissors = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        result = double_oracle(rocks_paper_scissors, solver=solve_sub_game_gurobi)
        expected_row_strategy = np.array([1/3, 1/3, 1/3])
        expected_column_strategy = np.array([1/3, 1/3, 1/3])
        expected_value = 0
        testing.assert_allclose(result['row_strategy'], expected_row_strategy)
        testing.assert_allclose(result['column_strategy'], expected_column_strategy)
        assert np.abs(result['value'] - expected_value) < 1e-12

    @pytest.mark.parametrize("epsilon", epsilons)
    def test_penalty_kicks(self, epsilon):
        penalty_kicks = np.array([[0.58, 0.95], [0.93, 0.7]])
        result = double_oracle(penalty_kicks, epsilon=epsilon, solver=solve_sub_game_gurobi)
        expected_value = 0.79584
        assert np.abs(result['value'] - expected_value) < epsilon


    def test_penalty_kicks_exact(self):
        penalty_kicks = np.array([[0.58, 0.95], [0.93, 0.7]])
        result = double_oracle(penalty_kicks, solver=solve_sub_game_gurobi)
        expected_row_strategy = np.array([0.38, 0.62])
        expected_column_strategy = np.array([0.42, 0.58])
        expected_value = 0.79584
        testing.assert_allclose(result['row_strategy'], expected_row_strategy, atol=1e-2)
        testing.assert_allclose(result['column_strategy'], expected_column_strategy, atol=1e-2)
        assert np.abs(result['value'] - expected_value) < 1e-5

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
        result = double_oracle(morra, solver=solve_sub_game_gurobi)
        expected_row_strategy = np.array([0, 0, 5/12, 0, 4/12, 0, 3/12, 0, 0])
        expected_column_strategy = np.array([0, 0, 5/12, 0, 4/12, 0, 3/12, 0, 0])
        expected_value = 0
        # testing.assert_allclose(result['row_strategy'], expected_row_strategy)
        # testing.assert_allclose(result['column_strategy'], expected_column_strategy)
        assert np.abs(result['value'] - expected_value) < epsilon

    @pytest.mark.parametrize("epsilon", epsilons)
    def test_random_game_1(self, epsilon):
        game = np.array([
            [0.9826734, 0.58156181, 0.12181801, 0.92156314, 0.43089033, 0.22006733,
             0.02892908, 0.59451368, 0.60399303, 0.82876719],
            [0.12322226, 0.79551752, 0.92460278, 0.44207347, 0.37506477, 0.26057992,
             0.76810282, 0.64833263, 0.42422216, 0.69629825],
            [0.92334935, 0.32892093, 0.31017389, 0.93825655, 0.26888256, 0.61432575,
             0.32022191, 0.83472426, 0.36875523, 0.51525187],
            [0.20916862, 0.27850357, 0.76924628, 0.79882297, 0.46402023, 0.17374578,
             0.83352433, 0.67562213, 0.7697014, 0.84395946],
            [0.92780047, 0.46265715, 0.32300697, 0.79525371, 0.15980884, 0.81959095,
             0.60644006, 0.53248797, 0.70854751, 0.2455035],
            [0.00991395, 0.47489093, 0.95044781, 0.77690957, 0.15267332, 0.03870271,
             0.64760353, 0.1559749, 0.95872672, 0.25913461],
            [0.36796719, 0.85704295, 0.16695616, 0.07335685, 0.377688, 0.37069128,
             0.8169605, 0.9087131, 0.15503355, 0.66822676],
            [0.39431156, 0.21791312, 0.4265568, 0.91670151, 0.67378034, 0.94598259,
             0.08073087, 0.8784442, 0.99763768, 0.18114017],
            [0.03969656, 0.2775525, 0.46528328, 0.72454631, 0.25458339, 0.2797359,
             0.67864124, 0.12278744, 0.0945402, 0.64170402],
            [0.01300967, 0.03659303, 0.67985665, 0.6934291, 0.1008851, 0.58670546,
             0.52243373, 0.64952729, 0.91732252, 0.07158645]
        ])
        result = double_oracle(game, solver=solve_sub_game_gurobi)
        expected_row_strategy = np.array([0.18309,0.11134,0,0.2399,0.07927,0,0.13071,0.25568,0,0])
        expected_column_strategy = np.array([0.22033,0.09008,0.09543,0,0.40111,0.0384,0.15465,0,0,0])
        expected_value = 0.46628
        # testing.assert_allclose(result['row_strategy'], expected_row_strategy, atol=1e-4)
        # testing.assert_allclose(result['column_strategy'], expected_column_strategy, atol=1e-4)
        assert np.abs(result['value'] - expected_value) < epsilon

