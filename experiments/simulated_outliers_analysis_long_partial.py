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
        "x_min": -5,
        "x_max": 5,
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

        test_params["name"] = f"Sim222 partial obs ({num_outliers} outliers)"
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

# +------------------------------------+-------------+-----------------+--------------------------+-------------------------+------------------------------------------------+-------------------------+--------------------+------------------------------------------------------+
# |             Test name              | num. points | Solve time [ms] | on-model point ratio [%] | cost-function ratio [%] |             Translation error [m]              | orientation error [rad] |   sag error [m]    |                  offsets error [m]                   |
# +------------------------------------+-------------+-----------------+--------------------------+-------------------------+------------------------------------------------+-------------------------+--------------------+------------------------------------------------------+
# |  Sim222 partial obs (1 outliers)   |   61 +/- 0  |  6.40 +/- 0.78  |      100.0% +/- 0.0      |      99.4 +/- 9.61      | [-21.67  23.02   0.73] +/- [23.2  24.66  0.85] |      -0.01 +/- 0.01     | -429.93 +/- 350.33 |             [-0.  0. -0.] +/- [0. 0. 0.]             |
# |  Sim222 partial obs (10 outliers)  |   70 +/- 0  |  6.53 +/- 0.89  |      100.1% +/- 0.5      |      100.7 +/- 2.05     | [-22.85  24.07   0.8 ] +/- [24.14 25.42  1.04] |      -0.01 +/- 0.02     | -370.18 +/- 350.44 |             [-0. -0. -0.] +/- [0. 0. 0.]             |
# |  Sim222 partial obs (20 outliers)  |   80 +/- 0  |  6.57 +/- 0.80  |      100.5% +/- 0.9      |      100.4 +/- 1.40     | [-25.22  26.82   0.93] +/- [24.44 25.93  1.09] |       -0. +/- 0.02      | -392.39 +/- 329.11 |       [-0.01 -0.   -0.01] +/- [0.   0.   0.01]       |
# |  Sim222 partial obs (50 outliers)  |  110 +/- 0  |  7.28 +/- 0.95  |      100.3% +/- 0.7      |      100.2 +/- 0.79     | [-28.61  31.65   1.18] +/- [24.78 27.29  1.31] |      0.02 +/- 0.03      | -448.36 +/- 285.82 |       [-0.01 -0.01 -0.02] +/- [0.01 0.01 0.01]       |
# | Sim222 partial obs (100 outliers)  |  160 +/- 0  |  7.77 +/- 0.94  |      100.1% +/- 1.1      |      100.3 +/- 0.43     | [-33.25  35.97   1.9 ] +/- [32.56 34.88  1.82] |       0. +/- 0.03       | -460.72 +/- 237.5  |       [-0.03 -0.02 -0.04] +/- [0.01 0.02 0.01]       |
# | Sim222 partial obs (200 outliers)  |  260 +/- 0  |  8.93 +/- 1.13  |      100.3% +/- 2.1      |      100.7 +/- 0.37     | [-35.23  42.95   2.62] +/- [33.71 41.03  2.57] |      0.05 +/- 0.04      | -442.68 +/- 216.86 |       [-0.05 -0.04 -0.11] +/- [0.01 0.02 0.02]       |
# | Sim222 partial obs (500 outliers)  |  560 +/- 0  |  11.63 +/- 1.76 |      84.1% +/- 14.7      |      100.9 +/- 0.38     | [-34.02  46.62   4.31] +/- [35.6  48.66  4.21] |      0.13 +/- 0.04      | -377.38 +/- 239.86 |       [-0.22 -0.08 -0.36] +/- [0.2  0.36 0.28]       |
# | Sim222 partial obs (1000 outliers) |  1060 +/- 0 |  16.36 +/- 2.90 |      28.1% +/- 5.3       |      101.3 +/- 0.45     | [-22.81  38.83  11.14] +/- [27.48 46.75  2.49] |       0.2 +/- 0.07      | -417.34 +/- 231.27 | [-1.17  1.38 -1.71] +/- [1.08e-03 7.14e-01 3.00e-14] |
# | Sim222 partial obs (2000 outliers) |  2060 +/- 0 |  21.16 +/- 4.79 |      22.9% +/- 8.4       |      101.9 +/- 0.26     | [-17.36  33.57  12.31] +/- [27.31 45.02  1.66] |       0.18 +/- 0.1      | -454.65 +/- 261.73 | [-1.17 -1.32 -1.71] +/- [2.96e-15 1.26e-15 4.02e-15] |
# +------------------------------------+-------------+-----------------+--------------------------+-------------------------+------------------------------------------------+-------------------------+--------------------+------------------------------------------------------+
