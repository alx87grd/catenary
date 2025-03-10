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
    filter_methods = ["corridor", "ground_filter", "clustering"]
    # filter_methods = ["none"]

    table = run_315kv_filter_analysis(filter_methods, plot=True, debug=False)


# +---------------------------------------+------------+-----------------+----------------+-------------------------+--------------------+
# |               Test name               |   n pts    | Solve time [ms] |  accuracy [%]  | orientation error [rad] |   sag error [m]    |
# +---------------------------------------+------------+-----------------+----------------+-------------------------+--------------------+
# |    V2 315kV with filter = corridor)   | 172 +/- 7  |  9.33 +/- 2.87  | 100.0% +/- 0.0 |       0. +/- 0.01       | -346.07 +/- 271.88 |
# | V2 315kV with filter = ground_filter) | 613 +/- 52 | 17.31 +/- 11.73 | 11.7% +/- 3.2  |    0.82 +/- 3.95e-09    | -397.45 +/- 258.11 |
# |   V2 315kV with filter = clustering)  | 185 +/- 15 |  19.16 +/- 9.84 | 100.0% +/- 0.0 |      0.01 +/- 0.02      | -349.35 +/- 271.23 |
# +---------------------------------------+------------+-----------------+----------------+-------------------------+--------------------+

# +---------------------------------------+------------+-----------------+----------------+-------------------------+--------------------+
# |               Test name               |   n pts    | Solve time [ms] |  accuracy [%]  | orientation error [rad] |   sag error [m]    |
# +---------------------------------------+------------+-----------------+----------------+-------------------------+--------------------+
# |    V3 315kV with filter = corridor)   | 172 +/- 7  | 21.87 +/- 24.08 | 100.0% +/- 0.0 |       0. +/- 0.01       | -346.07 +/- 271.88 |
# | V3 315kV with filter = ground_filter) | 614 +/- 50 | 15.62 +/- 11.07 | 11.5% +/- 3.2  |    0.82 +/- 9.45e-09    | -397.57 +/- 260.36 |
# |   V3 315kV with filter = clustering)  | 185 +/- 15 |  8.54 +/- 1.35  | 100.0% +/- 0.0 |      0.01 +/- 0.02      | -349.17 +/- 273.03 |
# +-----

# +---------------------------+--------------+-------------------+--------------+-------------------------+--------------------+
# |         Test name         |    n pts     |  Solve time [ms]  | accuracy [%] | orientation error [rad] |   sag error [m]    |
# +---------------------------+--------------+-------------------+--------------+-------------------------+--------------------+
# | 315kV with filter = none) | 13467 +/- 16 | 197.98 +/- 181.59 | 1.3% +/- 1.0 |    0.82 +/- 3.64e-12    | -321.71 +/- 258.26 |
# +---------------------------+--------------+-------------------+--------------+-------------------------+--------------------+
