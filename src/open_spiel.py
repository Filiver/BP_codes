import pyspiel
import numpy as np
from DO import *

def solve_openspiel_game(game_name, method=double_oracle, **kwargs):
    print("Loading game:", game)
    matrix = load_openspiel_game_as_matrix(game_name)
    print("Matrix loaded with size:", matrix.shape)
    results = method(matrix, **kwargs)
    return results


def load_openspiel_game_as_matrix(game_name):
    """
    Load an OpenSpiel game and return it as a matrix game

    :param game_name:
    :return: matrix of utilities for the **row** player

    :warning: Do not use if game is not zero-sum
    """
    game = pyspiel.load_game(game_name)
    matrix_game = pyspiel.extensive_to_matrix_game(game)
    matrix = np.stack([matrix_game.row_utilities(), matrix_game.col_utilities()])
    return matrix[0]

if __name__ == "__main__":
    from utils import plt_exploitations
    game = "kuhn_poker"
    # game= "go(board_size=2,komi=6.5)"
    results = solve_openspiel_game(game, solver=solve_sub_game_scipy)
    print(results)
    plt_exploitations(results)