import gurobipy
import numpy as np
import random
import scipy.optimize as opt
from constants import *
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib.pyplot as plt

def sub_mixed_strategy_to_mixed_strategy(game, row_strategies, column_strategies, sub_game_sub_row_ne, sub_game_sub_column_ne):
    sub_game_row_ne = np.zeros(game.shape[0])
    sub_game_row_ne[row_strategies] = sub_game_sub_row_ne
    sub_game_column_ne = np.zeros(game.shape[1])
    sub_game_column_ne[column_strategies] = sub_game_sub_column_ne
    return sub_game_row_ne, sub_game_column_ne


def DO_scipy(game: np.array, epsilon: float = 1e-5, verbose: bool = False) -> np.array:
    return double_oracle(game, epsilon, verbose, solve_sub_game_scipy)


def DO_gurobi(game: np.array, epsilon: float = 1e-5, verbose: bool = False) -> np.array:
    return double_oracle(game, epsilon, verbose, solve_sub_game_gurobi)


def solve_sub_game_scipy(subgame: np.array, row_strategies: list, column_strategies) -> tuple:
    """
    Solve the subgame defined by the strategy sets
    :param game: payoff matrix of the game
    :param strategy_sets: strategy sets of the players
    :return: Nash equilibrium of the subgame
    """

    # create subgame matrix
    st = time.perf_counter()
    A_ub = np.hstack((subgame, -np.ones((len(row_strategies), 1))))
    b_ub = np.zeros(len(row_strategies))
    A_eq = np.ones((1, len(column_strategies) + 1))
    A_eq[0, -1] = 0
    b_eq = 1
    c = np.zeros(len(column_strategies) + 1)
    c[-1] = 1

    bounds = [(0, None)] * (len(column_strategies)) + [(None, None)]
    res = opt.linprog(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, c=c, bounds=bounds, method='highs')
    if res.status == 0:
        sub_game_sub_column_ne = res.x[:-1]
        sub_game_sub_row_ne = -res.ineqlin.marginals
        sub_ne_value = res.fun
    else:
        raise Exception(res.message)
    et = time.perf_counter()

    return sub_game_sub_row_ne, sub_game_sub_column_ne, sub_ne_value, et - st


def solve_sub_game_gurobi(subgame: np.array, row_strategies: list, column_strategies) -> tuple:
    """
    Solve the subgame defined by the strategy sets
    :param subgame: payoff matrix of the subgame
    :param row_strategies: row strategy set
    :param column_strategies: column strategy set
    :return: Nash equilibrium of the subgame
    """

    st = time.perf_counter()
    # create subgame matrix
    A_ub = np.hstack((subgame, -np.ones((len(row_strategies), 1))))
    b_ub = np.zeros(len(row_strategies))

    m = gp.Model("subgame")
    m.params.LogToConsole=0
    col_strategy_var = m.addMVar(len(column_strategies) + 1, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="column_strategy")
    col_strategy_var[-1].lb = float("-inf")
    col_strategy_var[-1].ub = float("inf")
    cons = m.addMConstr(A_ub, col_strategy_var, GRB.LESS_EQUAL, b_ub)

    # Set the probabilities to sum to 1
    #m.addConstr(col_strategy_var[:-1].sum() == 1)
    lin_exp = gurobipy.LinExpr()
    lin_exp.addTerms([1.]*len(column_strategies), col_strategy_var[:-1].tolist())
    m.addLConstr(lin_exp, GRB.EQUAL, 1)

    # Set the objective
    m.setObjective(col_strategy_var[-1], GRB.MINIMIZE)
    m.optimize()

    sub_game_sub_column_ne = col_strategy_var[:-1].X
    sub_game_sub_row_ne = -cons.pi
    sub_ne_value = m.ObjVal
    et = time.perf_counter()

    return sub_game_sub_row_ne, sub_game_sub_column_ne, sub_ne_value, et - st


def visualise_game(game: np.array, row_strategies: list, column_strategies: list):
    alphas = np.ones_like(game) / 3
    grid_x, grid_y = np.meshgrid(np.array(row_strategies, dtype=int), np.array(column_strategies, dtype=int))
    alphas[grid_x, grid_y] = 1
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(game, alpha=alphas)
    plt.colorbar(im)
    plt.show()

def double_oracle(game: np.array, epsilon: float = 1e-5, verbose: bool = False, solver=solve_sub_game_scipy,
                  visualise: bool = False) -> np.array:
    """
    Double Oracle algorithm for two-player zero-sum games
    :param game: payoff matrix of the game
    :param epsilon: convergence parameter
    :param verbose: print intermediate results
    :param solver: subgame solver
    :return: Nash equilibrium
    """

    # initialize strategy sets
    row_strategies = [random.randint(0, game.shape[0]-1)]
    column_strategies = [random.randint(0, game.shape[1]-1)]

    # initialize exploitation arrays (for plotting)
    exploitations = [[],[]]

    iterations = 0
    end_condition = TerminationReasons.NotEnded
    t = 0
    while True:
        iterations += 1

        if visualise:
            visualise_game(game, row_strategies, column_strategies)

        # solve subgame and compute the best responses to the strategies
        subgame = game[row_strategies][:, column_strategies]
        sub_game_sub_row_ne, sub_game_sub_column_ne, sub_ne_value, ti = solver(subgame, row_strategies, column_strategies)
        t += ti
        row_best_response = np.argmax(game[:, column_strategies] @ sub_game_sub_column_ne)
        column_best_response = np.argmin(sub_game_sub_row_ne.T @ game[row_strategies])

        # check if the best responses are in the strategy sets
        if row_best_response in row_strategies and column_best_response in column_strategies:
            end_condition = TerminationReasons.NoBetterStrategy
            if verbose:
                print('Best responses already in the strategy sets')
                print('Row best response:', row_best_response)
                print('Row strategy set:', row_strategies)
                print('Column best response:', column_best_response)
                print('Column strategy set:', column_strategies)
            break
        else:
            # check if the subgame value is within epsilon of the best responses
            # print(game[row_best_response][column_strategies])
            # print(sub_game_sub_column_ne)
            row_best_response_value = game[row_best_response][column_strategies] @ sub_game_sub_column_ne
            column_best_response_value = sub_game_sub_row_ne.T @ game[row_strategies][:, column_best_response]
            exploitations[0].append(row_best_response_value)
            exploitations[1].append(column_best_response_value)
            if verbose:
                print(f"{iterations=}, {row_best_response_value=}, {column_best_response_value=}")
            if row_best_response_value - epsilon < sub_ne_value < column_best_response_value + epsilon:
                end_condition = TerminationReasons.Converged
                if verbose:
                    print('Converged')
                break

            # add the best responses to the strategy sets
            if row_best_response not in row_strategies:
                row_strategies.append(row_best_response)
                row_strategies.sort()
            if column_best_response not in column_strategies:
                column_strategies.append(column_best_response)
                column_strategies.sort()

    #print(t)
    row_sub_ne, column_sub_ne = sub_mixed_strategy_to_mixed_strategy(game, row_strategies, column_strategies, sub_game_sub_row_ne, sub_game_sub_column_ne)
    result = {
        'row_strategy': row_sub_ne,
        'column_strategy': column_sub_ne,
        'value': sub_ne_value,
        'iterations': iterations,
        'end_condition': end_condition,
        'exploitations': exploitations
    }
    return result


def lp(game, solver=solve_sub_game_gurobi):
    row_strategies = np.arange(0, game.shape[0]).tolist()
    column_strategies = np.arange(0, game.shape[1]).tolist()
    row_ne, column_ne, value, _ = solver(game, row_strategies, column_strategies)
    result = {
        'row_strategy': row_ne,
        'column_strategy': column_ne,
        'value': value,
        'iterations': 1,
        'end_condition': TerminationReasons.NoBetterStrategy
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
    from benchmark import generate_gaussian_sum_game_of_size_gen, generate_random_game_of_size
    from utils import plt_exploitations

    g = generate_gaussian_sum_game_of_size_gen(100)
    # g = generate_random_game_of_size(4000)

    b = double_oracle(g, solver=solve_sub_game_scipy, verbose=True, epsilon=1e-3, visualise=True)
    # b = lp(g, solver=solve_sub_game_gurobi)
    print(b)
    # plt_exploitations(b)



