import numpy as np
from catenary.analysis.dataset import SimulatedDataset, load_dataset
from catenary.analysis.evaluation import evaluate, plot_results, animate_results


def simtest():

    datagen_params = {
        "name": "sim_32",
        "n_out": 10,
        "n_frames": 100,
        "n_obs": 10,
        "x_min": -50,
        "x_max": 50,
        "w_l": 0.5,
        "w_o": 50.0,
        "center": [0, 0, 0],
        "partial_obs": True,
        "p_tru": np.array([10, 10, 10, 1.0, 300, 20.0, 10.0, 20.0]),
    }

    test_params = {
        "name": "GlobalConvergenceTest",
        "dataset": None,
        "model": "32",
        "p_0": None,  # np.array([-25.0, 40.0, 0.0, 1.0, 700, 6.0, 6.0, 6.0]),
        "Q": 0.0001 * np.diag([20.0, 20.0, 20.0, 1000.0, 0.0001, 800.0, 200.0, 200.0]),
        "l": 1.0,
        "b": 1000.0,
        "power": 2.0,
        "p_lb": np.array([-50, -50, 0, 0.0, 200, 19.0, 9.0, 19.0]),
        "p_ub": np.array([50, 50, 50, 2.0, 900, 21.0, 11.0, 21.0]),
        "n_search": 3,
        "p_var": np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0]),
        "filter_method": "corridor",  # No filter, as simulated data is already filtered
        "num_randomized_tests": 1,  # Number of tests to execute with randomized initial guess
        "stats_num_frames": 50,  # Number of last frames to use for statistics (experimental results have 100 frames)
        "method": "x",
        "n_sample": 100,
        "x_min": -200,
        "x_max": 200,
        "use_grad": True,
    }

    dataset = SimulatedDataset(datagen_params)

    test_params["dataset"] = dataset
    results, stats = evaluate(test_params)

    plot_results(test_params, results, save=False, n_run_plot=5)
    animate_results(test_params, results)


def rosbagtest():

    test_params = {
        "name": "RosbagTest",
        "dataset": None,
        "model": "222",
        "p_0": None,  # np.array([-25.0, 40.0, 0.0, 1.0, 700, 6.0, 6.0, 6.0]),
        "Q": 0.01 * np.diag([0.02, 0.02, 0.002, 0.01, 0.00001, 0.02, 0.02, 0.02]),
        "l": 1.0,
        "b": 100.0,
        "power": 2.0,
        "p_lb": np.array([-100.0, -100.0, 0.0, 1.5, 500.0, 5.0, 6.0, 6.0]),
        "p_ub": np.array([100.0, 100.0, 25.0, 2.5, 1500.0, 7.0, 9.0, 9.0]),
        "n_search": 2,
        "p_var": np.array([50.0, 50.0, 50.0, 5.0, 200.0, 2.0, 2.0, 2.0]),
        "filter_method": "corridor",  # No filter, as simulated data is already filtered
        "num_randomized_tests": 2,  # Number of tests to execute with randomized initial guess
        "stats_num_frames": 50,  # Number of last frames to use for statistics (experimental results have 100 frames)
        "method": "x",
        "n_sample": 201,
        "x_min": -200,
        "x_max": 200,
        "use_grad": True,
    }

    dataset = load_dataset("ligne315kv_test1")

    test_params["dataset"] = dataset
    results, stats = evaluate(test_params)

    plot_results(test_params, results, save=True, n_run_plot=5)
    animate_results(test_params, results)


if __name__ == "__main__":
    simtest()
    rosbagtest()
