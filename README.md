# Semester project code

This repository contains the code for the semester project "Analysis of Online Double Oracle Algorithm for Matrix Games" at Czech Technical University in Prague.

## Installation

To install the required packages, run the following command:

```bash
conda create --name ODO --file requirements.txt
```

For any method that uses Gurobi solver, you need to have Gurobi installed on your machine. You can download it from [here](https://www.gurobi.com/downloads/).

Test can be run using the following command:

```bash
python -m pytest --import-mode=importlib tests/
```

## Usage
Following game solving methods are available for the user:
- ```double_oracle``` - runs the double oracle algorithm
    - it accepts any solver that provides solves the subgame and returns the solution as tuple of mixed strategies and the value of the game
    - There are two solvers available:
        - ```solve_sub_game_scipy```
        - ```solve_sub_game_gurobi```
- ```lp``` - runs the linear program for the given game
- ```mulplicative_weights_update``` - runs the multiplicative weights update algorithm
- ```online_double_oracle``` - runs the online double oracle algorithm

There are also some helper functions which generate the game matrices:
- ```generate_random_matrix_game``` - generates a random matrix game
- ```generate_gaussian_sum_game_of_size_gen``` - generates a matrix which is sum of tens to thousands of sampled Gaussian distributions
- etc.