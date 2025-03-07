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
        "name": "315kV Test 1",
        "dataset": load_dataset("ligne315kv_test1"),
        "model": "222",
        "p_0": None,  # np.array([-25.0, 40.0, 0.0, 1.0, 700, 6.0, 6.0, 6.0]),
        "Q": 0.001 * np.diag([0.02, 0.02, 0.002, 0.01, 0.0001, 0.02, 0.02, 0.02]),
        "l": 1.0,
        "b": 1000.0,
        "power": 2.0,
        "p_lb": np.array([-100.0, -100.0, 0.0, 1.5, 500.0, 5.0, 6.0, 6.0]),
        "p_ub": np.array([100.0, 100.0, 25.0, 2.5, 1500.0, 7.0, 9.0, 9.0]),
        "n_search": 5,
        "p_var": np.array([50.0, 50.0, 50.0, 5.0, 200.0, 2.0, 2.0, 2.0]),
        "filter_method": "",
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
        params["filter_method"] = filter_method
        params["name"] = f"315kV Test 1 ({filter_method})"
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
    # Available methods: none, ground_filter, clustering and corridor
    # filter_methods = ["ground_filter", "clustering", "corridor"]
    filter_methods = ["corridor", "ground_filter", "clustering"]
    # filter_methods = ["corridor"]

    table = run_315kv_filter_analysis(filter_methods, plot=True, debug=False)
