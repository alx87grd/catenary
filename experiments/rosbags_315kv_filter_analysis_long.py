from catenary.analysis.evaluation import (
    evaluate,
    animate_results,
    plot_results,
    table_add_row,
    table_init,
)
from catenary.analysis.dataset import load_dataset, SimulatedDataset
import numpy as np


def run_315kv_filter_analysis(filter_methods, plot=False, debug=False):
    exp_315kv_test1_params = {
        "name": "315kv_test1_quick",
        "dataset": None,
        "model": "222",
        "p_0": None,  # np.array([-25.0, 40.0, 0.0, 1.0, 700, 6.0, 6.0, 6.0]),
        "Q": 0.01
        * np.diag(
            [
                1.0 / 50,
                1.0 / 50,
                1.0 / 50,
                1.0 / 2.0,
                1.0 / 2000,
                1.0 / 2,
                1.0 / 2,
                1.0 / 2,
            ]
        ),
        "l": 1.0,
        "b": 100.0,
        "power": 2.0,
        "p_lb": np.array([-100.0, -100.0, 12.0, 1.5, 500.0, 5.0, 6.0, 6.0]),
        "p_ub": np.array([100.0, 100.0, 25.0, 2.5, 1500.0, 7.0, 9.0, 9.0]),
        "n_search": 5,
        "p_var": np.array([5.0, 5.0, 5.0, 1.0, 400.0, 2.0, 2.0, 2.0]),
        "filter_method": "corridor",  # No filter, as simulated data is already filtered
        # "filter_method": "clustering",  # No filter, as simulated data is already filtered
        "num_randomized_tests": 100,  # Number of tests to execute with randomized initial guess
        "stats_num_frames": 10,  # Number of last frames to use for statistics (experimental results have 100 frames)
        "method": "x",
        "n_sample": 201,
        "x_min": -200,
        "x_max": 200,
        "use_grad": True,
    }

    table = table_init()
    for filter_method in filter_methods:
        params = exp_315kv_test1_params.copy()
        params["dataset"] = load_dataset("ligne315kv_test1")
        params["filter_method"] = filter_method
        params["name"] = f"315kV with filter = {filter_method})"
        results, stats = evaluate(params)
        table_add_row(table, params, stats)

        if plot:
            plot_results(params, results, save=True)

        if debug:
            animate_results(params, results)

    print(table)
    return table


if __name__ == "__main__":

    # Filtering methods to tests
    filter_methods = ["corridor", "ground_filter", "clustering", "none"]

    table = run_315kv_filter_analysis(filter_methods, plot=True, debug=False)
