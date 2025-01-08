import os

import numpy as np
import scipy
import time
from tqdm import tqdm, trange

from MWU import multiplicative_weights_update
from ODO import online_double_oracle


def benchmark_game_solver(game_solving_functions, game_generation_function, game_solving_args, game_generation_args,
                          iters, iterations_stats: list[bool] = None, verbose: bool = True):
    """
    Benchmark the game solving function
    :param game_solving_functions: function to solve the game
    :param game_generation_function: function to generate the game
    :param game_solving_args: arguments for the game solving function
    :param game_generation_args: arguments for the game generation function
    :return: time taken to solve the game
    """
    np.random.seed(0)
    times_per_iter = np.zeros((len(game_solving_functions), iters))
    iterations_per_iter = np.zeros((len(game_solving_functions), iters))
    solvers_results = [[] for _ in game_solving_functions]
    generating_time = 0
    if iterations_stats is None:
        iterations_stats = [False] * len(game_solving_functions)
    for i in trange(iters, disable=not verbose):
        start_time = time.perf_counter()
        game = game_generation_function(*game_generation_args)
        end_time = time.perf_counter()
        generating_time += end_time - start_time
        for j in range(len(game_solving_functions)):
            start_time = time.perf_counter()
            result = game_solving_functions[j](game, *game_solving_args[j])
            end_time = time.perf_counter()
            solvers_results[j].append(result)
            times_per_iter[j][i] = end_time - start_time
            if iterations_stats[j]:
                iterations_per_iter[j][i] = result['iterations']
    if verbose:
        print()
        print(f"Generating time: {generating_time}")
        for j in range(len(game_solving_functions)):
            print()
            print("------------------------------------")
            print('SOLVER:', game_solving_functions[j].__name__)
            print("------------------------------------")
            print('TIME STATS')
            print("------------------------------------")
            print('Total time:', np.sum(times_per_iter[j]))
            print('Average time:', np.mean(times_per_iter[j]))
            print('Standard deviation:', np.std(times_per_iter[j]))
            print("------------------------------------")
            if iterations_stats[j]:
                print('ITERATIONS STATS')
                print("------------------------------------")
                print('Total iterations:', np.sum(iterations_per_iter[j]))
                print('Average iterations:', np.mean(iterations_per_iter[j]))
                print('Standard deviation:', np.std(iterations_per_iter[j]))
                print("------------------------------------")
    results = {
        "times": times_per_iter,
        "iterations": iterations_per_iter,
        "solvers_results": solvers_results,
        "generating_time": generating_time
    }
    return results


def generate_random_game_of_size(size):
    return np.random.rand(size, size)


def generate_random_gaussian_game_of_size(size):
    return np.random.randn(size, size)


def generate_gaussian_game_of_size(size):
    size = (size // 2) * 2
    mu = np.random.normal(0, size / 4, 2)
    sigmas = np.abs(np.random.randint(size / 4, size * 3 / 4, 2)) * 2
    x = np.arange(-(size - 1) / 2, size / 2)
    y = np.arange(-(size - 1) / 2, size / 2)

    X, Y = np.meshgrid(x, y)

    game = scipy.stats.multivariate_normal.pdf(
        np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1))),
        mu,
        sigmas
    ).reshape(size, size)
    return game

class GaussianSumGameGenerator:
    def __init__(self, gauss_size=500, gauss_number=20, bank=None, bank_size=5000):
        self.GAUSS_SIZE = gauss_size
        self.GAUSS_NUMBER = gauss_number
        self.gauss_factory = np.zeros((self.GAUSS_NUMBER, self.GAUSS_SIZE, self.GAUSS_SIZE))
        for i in range(self.GAUSS_NUMBER):
            sigmas = np.abs(np.random.randint(self.GAUSS_SIZE // 4,  self.GAUSS_SIZE * 3 // 4, 2)) * 10
            x = np.arange(-(self.GAUSS_SIZE - 1) / 2, self.GAUSS_SIZE / 2)
            y = np.arange(-(self.GAUSS_SIZE - 1) / 2, self.GAUSS_SIZE / 2)

            X, Y = np.meshgrid(x, y)

            self.gauss_factory[i] = scipy.stats.multivariate_normal.pdf(
                np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1))),
                np.zeros(2),
                sigmas
            ).reshape(self.GAUSS_SIZE, self.GAUSS_SIZE)
        self.bank_size = bank_size
        if bank is None:
            self.bank = None
        else:
            self.bank = np.zeros((bank, bank_size, bank_size))
            for i in range(bank):
                self.bank[i] = self.new_game(bank_size)

    def new_game(self, size):
        amount = np.random.randint(size//5, size//2)
        game = np.zeros((size + 2 * self.GAUSS_SIZE, size + 2 * self.GAUSS_SIZE))
        for _ in range(amount):
            mu = np.random.randint(self.GAUSS_SIZE, size + self.GAUSS_SIZE, 2)
            i = np.random.randint(0, self.GAUSS_NUMBER - 1)
            game[mu[0] - (self.GAUSS_SIZE) // 2:mu[0] + self.GAUSS_SIZE // 2,
            mu[1] - (self.GAUSS_SIZE) // 2:mu[1] + self.GAUSS_SIZE // 2] += self.gauss_factory[i]
        game = game[self.GAUSS_SIZE:-self.GAUSS_SIZE, self.GAUSS_SIZE:-self.GAUSS_SIZE]
        return game / game.max()

    def generate(self, size):
        if self.bank is None or self.bank_size < size:
            return self.new_game(size)
        else:
            x,y = np.random.randint(0, self.bank_size - size + 1), np.random.randint(0, self.bank_size - size + 1)
            game = self.bank[np.random.randint(0, len(self.bank))][x:x+size, y:y+size]
            #print(game.shape)
            return np.copy(game)

def generate_gaussian_sum_game_of_size_gen(size):
    amount = np.random.randint(size//5, size//2)
    GAUSS_SIZE = 500
    GAUSS_NUMBER = 20
    gauss_factory = np.zeros((20, GAUSS_SIZE, GAUSS_SIZE))
    for i in range(GAUSS_NUMBER):
        sigmas = np.abs(np.random.randint(GAUSS_SIZE // 4,  GAUSS_SIZE * 3 // 4, 2)) * 10
        x = np.arange(-(GAUSS_SIZE - 1) / 2, GAUSS_SIZE / 2)
        y = np.arange(-(GAUSS_SIZE - 1) / 2, GAUSS_SIZE / 2)

        X, Y = np.meshgrid(x, y)

        gauss_factory[i] = scipy.stats.multivariate_normal.pdf(
            np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1))),
            np.zeros(2),
            sigmas
        ).reshape(GAUSS_SIZE, GAUSS_SIZE)

    while True:
        game = np.zeros((size+2*GAUSS_SIZE, size+2*GAUSS_SIZE))
        for _ in range(amount):
            mu = np.random.randint(GAUSS_SIZE, size+GAUSS_SIZE, 2)
            i = np.random.randint(0, GAUSS_NUMBER-1)
            game[mu[0]-(GAUSS_SIZE)//2:mu[0]+GAUSS_SIZE//2, mu[1]-(GAUSS_SIZE)//2:mu[1]+GAUSS_SIZE//2] += gauss_factory[i]
        game = game[GAUSS_SIZE:-GAUSS_SIZE, GAUSS_SIZE:-GAUSS_SIZE]
        return game / game.max()


def generate_gaussian_sum_game_of_size(size):
    amount = np.random.randint(size//2, size)
    game = np.zeros((size, size))
    for _ in range(amount):
        game += generate_gaussian_game_of_size(size)
    return game / game.max()


def load_gaussian_sum_game_of_size(size, presaved=None):
    MAX_SIZE = 2000
    presaved_games_files = os.listdir(os.path.join("", "tmp", f"gaussian_sum_{MAX_SIZE}"))
    if presaved is not None:
        if len(presaved_games_files) > presaved:
            for i in range(presaved - len(presaved_games_files)):
                game = generate_gaussian_sum_game_of_size(MAX_SIZE)
                np.save(os.path.join("", "tmp", f"gaussian_sum_{MAX_SIZE}", f"game_{len(presaved_games_files) + i}"), game)
        else:
            game = np.load(os.path.join("", "tmp", f"gaussian_sum_{MAX_SIZE}", presaved_games_files[presaved]))
        a, b = np.random.randint(0, MAX_SIZE - size), np.random.randint(0, MAX_SIZE - size)
        return game[a:a + size][b:b + size]
    else:
        return generate_gaussian_sum_game_of_size(size)


if __name__ == '__main__':
    np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.9g" % x))
    from DO import *
    # from MWU import *
    # from ODO import *

    game_size = 2000
    iters = 5
    game_generator = GaussianSumGameGenerator(bank=iters, bank_size=game_size)
    results = benchmark_game_solver([multiplicative_weights_update, online_double_oracle],
                                    # game_generator.generate,
                                    generate_gaussian_sum_game_of_size_gen,
                                    # generate_random_game_of_size,
                                    [(), (), ()],
                                    (2000,), iters, [True, True, True])
    # for i in range(iters):
    #     print(results['solvers_results'][0][i]['value'], results['solvers_results'][2][i]['value'])
    #     # assert np.abs(results['solvers_results'][0][i]['value'] - results['solvers_results'][2][i]['value']) < 1e-2

