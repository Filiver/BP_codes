from benchmark import *
from DO import *
from MWU import *

game_solving_functions = [double_oracle, lp]

min_size = 100
max_size = 5000
step = 100

times = np.zeros((len(game_solving_functions), (max_size-min_size)//step + 1))
iters = np.zeros((len(game_solving_functions), (max_size-min_size)//step + 1))


for i, game_size in enumerate(range(min_size, max_size + 1, step)):
    results = benchmark_game_solver(game_solving_functions,
                                    generate_gaussian_sum_game_of_size_gen,
                                    [(), (), ()],
                                    (game_size,), iters, [True, False, True])
    times[i] = results['times'].avg(axis=1)
    iters[i] = results['iterations'].avg(axis=1)


