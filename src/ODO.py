import numpy as np
from constants import *
import random
from MWU import compute_iterations_limit_bianchi, compute_eta
from DO import sub_mixed_strategy_to_mixed_strategy

def online_double_oracle(game, epsilon: float = 1e-2, iterations_limit: int = np.inf,
                         verbose: bool = False) -> dict:

    row_strategies = [np.random.randint(0, game.shape[0])]
    column_strategies = [np.random.randint(0, game.shape[1])]
    # row_strategies = [game.shape[0]-1]
    # column_strategies = [game.shape[1]-1]

    row_strategy = np.ones(1)
    column_strategy = np.ones(1)
    average_row_strategy = np.zeros(1)
    average_column_strategy = np.zeros(1)
    k = 1
    theoretical_iterations_limit = 1
    mu = 1

    mu_sub_game = mu * game[row_strategies][:, column_strategies]
    minus_mu_sub_game = -mu_sub_game

    iterations = 0
    window_iterations = 0
    row_iterations_in_window = 0
    column_iterations_in_window = 0
    end_condition = TerminationReasons.NotEnded
    while True:
        window_iterations += 1
        row_iterations_in_window += 1
        column_iterations_in_window += 1
        new_row_strategy = row_strategy * np.exp(mu_sub_game @ column_strategy)
        new_row_strategy /= np.sum(new_row_strategy)
        new_column_strategy = column_strategy * np.exp(row_strategy @ minus_mu_sub_game)
        new_column_strategy /= np.sum(new_column_strategy)

        average_row_strategy = average_row_strategy + (new_row_strategy - average_row_strategy) / (row_iterations_in_window)
        average_column_strategy = average_column_strategy + (new_column_strategy - average_column_strategy) / (column_iterations_in_window)

        row_best_response = np.argmax(game[:, column_strategies] @ average_column_strategy)
        column_best_response = np.argmin(average_row_strategy @ game[row_strategies])

        change = False
        if row_best_response not in row_strategies:
            change = True
            row_strategies.append(row_best_response)
            row_strategies.sort()
            row_strategy = np.ones(len(row_strategies))/len(row_strategies)
            average_row_strategy = np.zeros(len(row_strategies))
            row_iterations_in_window = 0
        else:
            row_strategy = new_row_strategy
        if column_best_response not in column_strategies:
            change = True
            column_strategies.append(column_best_response)
            column_strategies.sort()
            column_strategy = np.ones(len(column_strategies))/len(column_strategies)
            average_column_strategy = np.zeros(len(column_strategies))
            column_iterations_in_window = 0
        else:
            column_strategy = new_column_strategy
        if change:
            k = max(len(row_strategies), len(column_strategies))
            theoretical_iterations_limit = compute_iterations_limit_bianchi(k, epsilon)
            if verbose:
                print("New theoretical iterations limit: ", theoretical_iterations_limit)
            mu = compute_eta(k, theoretical_iterations_limit)
            iterations += window_iterations
            window_iterations = 0
            mu_sub_game = mu * game[row_strategies][:, column_strategies]
            minus_mu_sub_game = -mu_sub_game

        if window_iterations >= theoretical_iterations_limit:
            end_condition = TerminationReasons.Converged
            break
        if iterations >= iterations_limit:
            end_condition = TerminationReasons.IterationsLimit
            break

    iterations += window_iterations
    average_row_strategy_full, average_column_strategy_full = \
        sub_mixed_strategy_to_mixed_strategy(game, row_strategies, column_strategies,
                                             average_row_strategy, average_column_strategy)
    result = {
        'row_strategy': average_row_strategy_full,
        'column_strategy': average_column_strategy_full,
        'value': average_row_strategy_full @ game @ average_column_strategy_full,
        'iterations': iterations,
        'end_condition': end_condition
    }
    # print("Iterations: ", iterations)
    # print('Theoretical iterations limit: ', theoretical_iterations_limit)
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

    b = online_double_oracle(morra)
    print(b)


