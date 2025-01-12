import matplotlib.pyplot as plt
import numpy as np
import time
from DO import *
from benchmark import *

SHOW_VALUES = False

#gen = GaussianSumGameGenerator(bank=1, bank_size=5000)
game_size = 100
start_time = time.time()
# game = generate_gaussian_sum_game_of_size_gen(game_size)
game = generate_random_game_of_size(game_size)
# game = gen.generate(game_size)
end_time = time.time()
print(f"Generated: {end_time - start_time}")
result = lp(game)
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(game)
if SHOW_VALUES:
    for i in range(game.shape[0]):
        for j in range(game.shape[1]):
            text = ax.text(j, i, round(game[i, j],2), ha="center", va="center", color="w")
# ax.set_title("Game matrix, solution: " + str(result['value']))
# ax.set_title(f"Example from Gaussian dataset, size: {game_size}")
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('gauss_sum_matrix.png', bbox_inches='tight')
plt.show()
print(result['value'])
print(result['row_strategy'])
print(result['column_strategy'])
print(result['end_condition'])
print(result['iterations'])