import matplotlib.pyplot as plt
import numpy as np
from MWU import compute_iterations_limit_freund_schapire, compute_iterations_limit_bianchi

tested_bounds = [compute_iterations_limit_freund_schapire, compute_iterations_limit_bianchi]

min_size = 100
max_size = 5000
step = 100
epsilon = 1e-2

limits_size = np.zeros((len(tested_bounds), (max_size - min_size) // step + 1))

for j in range(len(tested_bounds)):
    for i, size in enumerate(range(min_size, max_size + 1, step)):
        limits_size[j, i] = tested_bounds[j](size, epsilon)

fig, ax = plt.subplots(figsize=(10, 10))
ax.grid(True)
for j in range(len(tested_bounds)):
    ax.plot(range(min_size, max_size + 1, step), limits_size[j])
ax.set_xlabel('Game size')
ax.set_ylabel('Iterations limit')
ax.set_title(f'Iterations limit of MWU versus game size for epsilon={epsilon}')
plt.savefig('mwu_limits_size.png')
plt.show()


size = 2000
min_epsilon = 1
max_epsilon = 1000000
step = 1
limits_eps = np.zeros((len(tested_bounds), (max_epsilon - min_epsilon) // step + 1))

for j in range(len(tested_bounds)):
    for i, epsilon in enumerate(range(min_epsilon, max_epsilon + 1, step)):
        epsilon = 1 / epsilon
        limits_eps[j, i] = tested_bounds[j](size, epsilon)

fig, ax = plt.subplots(figsize=(10, 10))
ax.grid(True)
for j in range(len(tested_bounds)):
    ax.plot(1/np.arange(min_epsilon, max_epsilon+1), limits_eps[j])
ax.set_xlabel('Epsilon')
ax.set_ylabel('Iterations limit')
ax.loglog()
ax.set_title(f'Iterations limit of MWU versus epsilon for game size={size}')
plt.savefig('mwu_limits_eps.png')
plt.show()
