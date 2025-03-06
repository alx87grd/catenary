from catenary.analysis.evaluation import (
    animate_results,
    evaluate,
    plot_results,
    table_add_row,
    table_init,
)
from catenary.analysis.dataset import load_dataset, SimulatedDataset
import numpy as np


def simulated_outliers_analysis(num_outliers_scenarios, plot=False, debug=False):

    datagen_params = {
        "name": "sim_222",
        "n_out": 10,
        "n_frames": 100,
        "n_obs": 10,
        "x_min": -50,
        "x_max": 50,
        "w_l": 0.2,
        "w_o": 50.0,
        "center": [0, 0, 0],
        "partial_obs": False,
        "p_tru": np.array(
            [
                -22.61445006,
                42.86768157,
                14.25202579,
                2.31972922,
                698.6378392,
                5.83313134,
                7.68165757,
                7.28652209,
            ]
        ),
    }

    # Test parameters
    test_params = {
        "name": "Simulated 222",
        "dataset": None,
        "model": "222",
        "p_0": None,  # np.array([-25.0, 40.0, 0.0, 1.0, 700, 6.0, 6.0, 6.0]),
        "Q": 0.01 * np.diag([0.02, 0.02, 0.002, 0.01, 0.0001, 0.02, 0.02, 0.02]),
        "l": 1.0,
        "b": 1000.0,
        "power": 2.0,
        "p_lb": np.array([-100.0, -100.0, 0.0, 1.5, 500.0, 5.0, 6.0, 6.0]),
        "p_ub": np.array([100.0, 100.0, 25.0, 2.5, 1500.0, 7.0, 9.0, 9.0]),
        "n_search": 5,
        "p_var": np.array([50.0, 50.0, 50.0, 5.0, 400.0, 2.0, 2.0, 2.0]),
        "filter_method": "none",  # No filter, as simulated data is already filtered
        "num_randomized_tests": 10,  # Number of tests to execute with randomized initial guess
        "stats_num_frames": 10,  # Number of last frames to use for statistics (experimental results have 100 frames)
        "method": "x",
        "n_sample": 201,
        "x_min": -200,
        "x_max": 200,
        "use_grad": True,
    }

    table = table_init()

    for num_outliers in num_outliers_scenarios:

        datagen_params["n_out"] = num_outliers
        dataset = SimulatedDataset(datagen_params)

        test_params["name"] = f"Sim 222 global obs({num_outliers} outliers)"
        test_params["dataset"] = dataset

        results, stats = evaluate(test_params)

        if plot:
            plot_results(test_params, results, save=True)

        if debug:
            animate_results(test_params, results)

        table_add_row(table, test_params, stats)

    print(table)

    return table


if __name__ == "__main__":

    # Number of outliers to simulate
    num_outliers_scenarios = [1, 10, 20, 50, 100, 200, 500, 1000, 2000]

    table = simulated_outliers_analysis(num_outliers_scenarios, plot=True, debug=False)


# 5 mars
# +-----------------------------------+-------------+-----------------+--------------------------+-------------------------+------------------------------------------------+-------------------------+--------------------+------------------------------------------------------+
# |             Test name             | num. points | Solve time [ms] | on-model point ratio [%] | cost-function ratio [%] |             Translation error [m]              | orientation error [rad] |   sag error [m]    |                  offsets error [m]                   |
# +-----------------------------------+-------------+-----------------+--------------------------+-------------------------+------------------------------------------------+-------------------------+--------------------+------------------------------------------------------+
# |   Sim 222 global obs(1 outliers)  |   61 +/- 0  |  5.43 +/- 0.51  |      100.0% +/- 0.0      |      106.7 +/- 3.79     |    [ 0.16 -0.18 -0.01] +/- [0.69 0.75 0.06]    |        -0. +/- 0.       |   1.06 +/- 19.94   |    [-3.35e-04  4.05e-05 -4.47e-04] +/- [0. 0. 0.]    |
# |  Sim 222 global obs(10 outliers)  |   70 +/- 0  |  5.72 +/- 0.40  |      99.9% +/- 0.4       |      100.8 +/- 0.42     |    [ 0.14 -0.16 -0.02] +/- [0.64 0.7  0.06]    |        -0. +/- 0.       |  -7.13 +/- 22.99   |             [-0. -0. -0.] +/- [0. 0. 0.]             |
# |  Sim 222 global obs(20 outliers)  |   80 +/- 0  |  5.80 +/- 0.44  |      100.0% +/- 0.0      |      100.2 +/- 0.59     |    [ 0.13 -0.14 -0.05] +/- [0.64 0.7  0.08]    |        -0. +/- 0.       |  -58.11 +/- 91.9   |          [-0.   -0.   -0.01] +/- [0. 0. 0.]          |
# |  Sim 222 global obs(50 outliers)  |  110 +/- 0  |  6.24 +/- 0.55  |      100.0% +/- 0.0      |      99.9 +/- 0.52      |    [ 0.1  -0.11 -0.12] +/- [0.58 0.63 0.12]    |        -0. +/- 0.       | -192.75 +/- 199.04 |       [-0.02 -0.01 -0.02] +/- [0.01 0.01 0.01]       |
# |  Sim 222 global obs(100 outliers) |  160 +/- 0  |  7.09 +/- 0.67  |      99.9% +/- 0.6       |      99.9 +/- 0.31      |    [-0.15  0.18 -0.14] +/- [0.44 0.48 0.13]    |     -3.44e-05 +/- 0.    | -312.51 +/- 267.6  |       [-0.03 -0.02 -0.05] +/- [0.01 0.01 0.01]       |
# |  Sim 222 global obs(200 outliers) |  260 +/- 0  |  8.24 +/- 0.84  |      99.1% +/- 2.3       |      100.0 +/- 0.15     |    [-1.28  1.39 -0.15] +/- [1.65 1.77 0.11]    |     5.76e-05 +/- 0.     | -490.76 +/- 281.03 |       [-0.05 -0.06 -0.1 ] +/- [0.02 0.02 0.02]       |
# |  Sim 222 global obs(500 outliers) |  560 +/- 0  |  11.38 +/- 1.70 |      63.0% +/- 41.4      |      99.7 +/- 0.51      | [-33.94  37.74   4.4 ] +/- [37.32 41.57  5.24] |      0.01 +/- 0.01      | -400.41 +/- 216.39 |       [-0.54  0.03 -0.84] +/- [0.49 0.22 0.71]       |
# | Sim 222 global obs(1000 outliers) |  1060 +/- 0 |  15.56 +/- 3.10 |      23.6% +/- 9.5       |      100.9 +/- 0.25     | [-26.59  30.87  10.79] +/- [38.98 44.86  1.79] |      0.02 +/- 0.01      | -395.72 +/- 183.56 | [-1.17  1.03 -1.71] +/- [2.48e-15 1.01e+00 2.63e-15] |
# | Sim 222 global obs(2000 outliers) |  2060 +/- 0 |  21.22 +/- 5.31 |       7.3% +/- 3.5       |      101.9 +/- 0.30     | [-20.69  39.49  13.92] +/- [28.4  41.44  0.96] |      0.19 +/- 0.08      | -526.98 +/- 167.58 | [-1.17 -1.32 -1.71] +/- [2.07e-15 1.36e-14 1.47e-13] |
# +-----------------------------------+-------------+-----------------+--------------------------+-------------------------+------------------------------------------------+-------------------------+--------------------+------------------------------------------------------+
