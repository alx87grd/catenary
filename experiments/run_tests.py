from experiments.test_runner import run_test, plot_results
from experiments.dataset import load_dataset, SimulatedDataset
import numpy as np
from prettytable import PrettyTable


def table_init():
    table = PrettyTable()
    # table.field_names = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg"]
    table.field_names = [
        "Test name",
        "num. points",
        "Solve time [ms]",
        "on-model point ratio [%]",
        "cost-function ratio [%]",
        "Translation error [m]",
        "orientation error [rad]",
        "sag error [m]",
        "offsets error [m]",
    ]
    return table


def table_add_row(table, params, stats):
    table.add_row(
        [
            params["name"],
            f'{stats["num_points_mean_after_filter"]:.0f} +/- {stats["num_points_std_after_filter"]:.0f}',
            f'{stats["solve_time_per_seach_mean"]*1000:.2f} +/- {stats["solve_time_per_seach_std"]*1000:.2f}',
            f'{stats["n_in_ratio_mean"]*100:.1f}% +/- {stats["n_in_ratio_std"]*100:.1f}',
            f'{stats["J_ratio_mean"]*100:.1f} +/- {stats["J_ratio_std"]*100:.2f}',
            f'{np.array2string(stats["p_err_mean"][0:3], precision=2)} +/- {np.array2string(stats["p_err_std"][0:3], precision=2)}',
            f'{np.array2string(stats["p_err_mean"][3], precision=2)} +/- {np.array2string(stats["p_err_std"][3], precision=2)}',
            f'{np.array2string(stats["p_err_mean"][4], precision=2)} +/- {np.array2string(stats["p_err_std"][4], precision=2)}',
            f'{np.array2string(stats["p_err_mean"][5:], precision=2)} +/- {np.array2string(stats["p_err_std"][5:], precision=2)}',
            #     f'({stats["p_err_mean"][0]:.2f}, {stats["p_err_mean"][1]:.2f}, {stats["p_err_mean"][2]:.2f}) +/- '
            #     + f'({stats["p_err_std"][0]:.2f}, {stats["p_err_std"][1]:.2f}, {stats["p_err_std"][2]:.2f})',
        ]
    )


def run_simulated_tests(table, plot=False):
    sim222_dataset = SimulatedDataset("sim_222")

    # Test parameters
    sim222_params = {
        "name": "Simulated 222",
        "dataset": None,
        "model": "222",
        "p_0": np.array([-25.0, 40.0, 0.0, 1.0, 700, 6.0, 6.0, 6.0]),
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
        "stats_num_frames": 50,  # Number of last frames to use for statistics (experimental results have 100 frames)
        "num_outliers": 0,  # Number of outliers to simulate
    }

    # Number of outliers to simulate
    num_outliers_scenarios = [10, 100, 500, 1000]

    for num_outliers in num_outliers_scenarios:
        params = sim222_params.copy()
        params["name"] = f"Simulated 222 ({num_outliers} outliers)"
        params["num_outliers"] = num_outliers
        params["dataset"] = SimulatedDataset("sim_222", num_outliers)
        results, stats = run_test(params)
        if plot:
            plot_results(params, results)
        table_add_row(table, params, stats)


def run_experimental_tests(table, plot=False):
    exp_315kv_test1_params = {
        "name": "315kV Test 1",
        "dataset": load_dataset("ligne315kv_test1"),
        "model": "222",
        "p_0": np.array([-25.0, 40.0, 0.0, 1.0, 700, 6.0, 6.0, 6.0]),
        "Q": 0.01 * np.diag([0.02, 0.02, 0.002, 0.01, 0.0001, 0.02, 0.02, 0.02]),
        "l": 1.0,
        "b": 1000.0,
        "power": 2.0,
        "p_lb": np.array([-100.0, -100.0, 0.0, 1.5, 500.0, 5.0, 6.0, 6.0]),
        "p_ub": np.array([100.0, 100.0, 25.0, 2.5, 1500.0, 7.0, 9.0, 9.0]),
        "n_search": 5,
        "p_var": np.array([50.0, 50.0, 50.0, 5.0, 200.0, 2.0, 2.0, 2.0]),
        "filter_method": "",
        "num_randomized_tests": 10,  # Number of tests to execute with randomized initial guess
        "stats_num_frames": 50,  # Number of last frames to use for statistics (experimental results have 100 frames)
    }

    # Filtering methods to tests
    # Available methods: none, ground_filter, clustering and corridor
    # filter_methods = ["ground_filter", "clustering", "corridor"]

    filter_methods = ["corridor", "ground_filter", "clustering", "None"]

    for filter_method in filter_methods:
        params = exp_315kv_test1_params.copy()
        params["filter_method"] = filter_method
        params["name"] = f"315kV Test 1 ({filter_method})"
        results, stats = run_test(params)
        if plot:
            plot_results(params, results)
        table_add_row(table, params, stats)


if __name__ == "__main__":
    table = table_init()
    plot = True  # plotting result will take more much time
    run_simulated_tests(table, plot=False)
    # run_experimental_tests(table, plot=False)

    print(table)
