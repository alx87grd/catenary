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
        "n_search": 2,
        "p_var": np.array([50.0, 50.0, 50.0, 5.0, 400.0, 2.0, 2.0, 2.0]),
        "filter_method": "none",  # No filter, as simulated data is already filtered
        "num_randomized_tests": 4,  # Number of tests to execute with randomized initial guess
        "stats_num_frames": 50,  # Number of last frames to use for statistics (experimental results have 100 frames)
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

        test_params["name"] = f"Simulated 222 ({num_outliers} outliers)"
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
    # num_outliers_scenarios = [1, 10, 100, 500, 1000]
    num_outliers_scenarios = [1, 3, 5]

    table = simulated_outliers_analysis(num_outliers_scenarios, plot=True, debug=False)
