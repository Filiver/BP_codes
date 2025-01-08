import numpy as np
from constants import *


def compute_iterations_limit_freund_schapire(game_size: int, epsilon: float) -> int:
    iterations_limit_theoretical = 4 + (1 + epsilon) ** 2 * np.log(game_size) ** 2 - 4 * np.log(game_size) ** 2 * epsilon
    iterations_limit_theoretical = int(np.ceil((2 * (1 + epsilon) + np.sqrt(iterations_limit_theoretical)) / (2 * epsilon ** 2)))
    return iterations_limit_theoretical


def compute_iterations_limit_bianchi(game_size: int, epsilon: float) -> int:
    return int(np.ceil(np.log(game_size) / (2 * epsilon ** 2)))

def compute_mu(game_size: int, epsilon: float, iterations_limit: int) -> float:
    return np.sqrt(8 * np.log(game_size) / iterations_limit)


def multiplicative_weights_update(game, mu: float = None, epsilon: float = 1e-2, iterations_limit: int = None,
                                  save_strategies: bool = False, calc_exploitations: bool = False) -> dict:
    row_strategy = np.ones(game.shape[0])/game.shape[0]
    column_strategy = np.ones(game.shape[1])/game.shape[1]
    if save_strategies:
        row_strategies = [row_strategy]
        column_strategies = [column_strategy]
    else:
        row_strategies = None
        column_strategies = None

    #game_values = [row_strategy @ game @ column_strategy]
    row_optimal_strategy = row_strategy
    column_optimal_strategy = column_strategy

    # initialize exploitation arrays (for plotting)
    if calc_exploitations:
        exploitations = [[], []]
    else:
        exploitations = None

    iterations_limit_theoretical = compute_iterations_limit_bianchi(max(game.shape), epsilon)
    #print(f"Theoretical iterations limit: {iterations_limit_theoretical}")
    if iterations_limit is None or iterations_limit_theoretical < iterations_limit:
        iterations_limit = iterations_limit_theoretical
    if mu is None:
        mu = compute_mu(max(game.shape), epsilon, iterations_limit)

    mu_game = mu * game
    minus_mu_game = -mu_game
    end_condition = TerminationReasons.NotEnded
    iterations = 0
    while iterations < iterations_limit:
        iterations += 1
        if calc_exploitations:
            row_best_response = np.argmax(game @ column_optimal_strategy)
            column_best_response = np.argmin(row_optimal_strategy @ game)
            row_best_response_value = (game[row_best_response] @ column_optimal_strategy) / iterations
            column_best_response_value = (row_optimal_strategy @ game[:, column_best_response]) / iterations
            exploitations[0].append(row_best_response_value)
            exploitations[1].append(column_best_response_value)

        new_row_strategy = row_strategy * np.exp(mu_game @ column_strategy)
        new_row_strategy /= np.sum(new_row_strategy)
        new_column_strategy = column_strategy * np.exp(row_strategy @ minus_mu_game)
        new_column_strategy /= np.sum(new_column_strategy)

        if save_strategies:
            row_strategies.append(new_row_strategy)
            column_strategies.append(new_column_strategy)

        row_optimal_strategy += new_row_strategy
        column_optimal_strategy += new_column_strategy

        row_strategy = new_row_strategy
        column_strategy = new_column_strategy

    if iterations == iterations_limit_theoretical:
        end_condition = TerminationReasons.Converged
    elif iterations == iterations_limit:
        end_condition = TerminationReasons.IterationsLimit

    row_optimal_strategy /= iterations + 1
    column_optimal_strategy /= iterations + 1

    result = {
        'row_strategy': row_optimal_strategy,
        'column_strategy': column_optimal_strategy,
        'value': row_optimal_strategy @ game @ column_optimal_strategy,
        'iterations': iterations,
        'end_condition': end_condition,
        'row_strategies': row_strategies,
        'column_strategies': column_strategies,
        'exploitations': exploitations
    }
    return result


if __name__ == "__main__":
    matching_pennies = np.array([[1, -1], [-1, 1]])
    penalty_kicks = np.array([[0.58, 0.95], [0.93, 0.7]])
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

    epsilon = 1e-2
    res = multiplicative_weights_update(morra, iterations_limit=None, epsilon=epsilon, save_strategies=True)
    print(f"Value of the game: {res['value']}")
    print(f"Number of iterations: {res['iterations']}")
    print(f"End condition: {res['end_condition']}")

    import matplotlib.pyplot as plt
    from utils import plt_exploitations

    #plt_exploitations(res)

    # x = np.array(res['row_strategies'])[1:, 0]
    # y = np.array(res['column_strategies'])[1:, 0]
    #
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.scatter(x, y, s=np.ones_like(x), c=np.arange(len(x)), cmap='viridis')
    # ax.set_title(f"MWU, penalty kicks, epsilon={epsilon}, value={res['value']}")
    # plt.savefig('mwu_penalty_kicks_spiral.png')
    # plt.show()



