from catenary import powerline
from catenary.filter import remove_ground_plane, filter_cable_points
from experiments.dataset import load_dataset


import matplotlib.pyplot as plt
import numpy as np
import time


def print_progress_bar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    print_end="\r",
):
    """
    Call in a loop to create terminal progress bar.

    Ref: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters

    Parameters
    ----------
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def run_test(params: dict):
    """
    Run a test case.

    Parameters
    ----------
    params: dict
        Dictionary of test parameters.

    Return
    ------
    results: dict
        Dictionary of results.
    """
    test_name = params["name"]

    # Load dataset
    dataset = params["dataset"]

    # Initialize model
    model = powerline.create_array_model(params["model"])

    # Number of tests with randomized initial guess
    num_randomized_tests = params["num_randomized_tests"]

    # Number of last frames to use for statistics, allowing for convergence
    # before computing statistics
    stats_num_frames = params["stats_num_frames"]

    # Filter method
    filter_method = params["filter_method"]

    # Array of results (one element for each randomized test)
    results = []

    # Initialize random number generator with seed to have reproductible results
    rng = np.random.default_rng(seed=1)

    total_frames = dataset.frame_count() * num_randomized_tests
    current_frame = 0

    for i in range(num_randomized_tests):

        # Result dictionary for this test
        result = {}

        # Randomize initial guess within [p_lb, p_ub] bounds
        p_0 = rng.uniform(params["p_lb"], params["p_ub"])

        # Use this to disable randomization of initial guess
        # p_0 = params['p_0']

        # Initialize estimator
        estimator = powerline.ArrayEstimator(model, p_0)

        # Regulation weight matrix
        estimator.Q = params["Q"]

        # Parameters of lorentzian cost shaping function
        estimator.l = params["l"]
        estimator.power = params["power"]
        estimator.b = params["b"]

        # Parameters upper and lower bounds
        estimator.p_ub = params["p_ub"]
        estimator.p_lb = params["p_lb"]

        # Search parameters

        # Number of search
        estimator.n_search = params["n_search"]

        # Parameters std deviation for searching
        estimator.p_var = params["p_var"]

        # Initialize p_hat key with an numpy array of dataset.frame_count() elements
        result["p_hat"] = np.zeros((dataset.frame_count(), p_0.shape[0]))
        result["p_err"] = np.zeros((dataset.frame_count(), p_0.shape[0]))
        result["p_err_mean"] = np.zeros((p_0.shape[0]))
        result["p_err_std"] = np.zeros((p_0.shape[0]))
        result["num_points_before_filter"] = np.zeros((dataset.frame_count()))
        result["num_points_after_filter"] = np.zeros((dataset.frame_count()))
        result["num_points_close_model_tru"] = np.zeros((dataset.frame_count()))
        result["num_points_close_model_hat"] = np.zeros((dataset.frame_count()))
        result["J_tru"] = np.zeros((dataset.frame_count()))
        result["J_hat"] = np.zeros((dataset.frame_count()))
        result["n_in_ratio"] = np.zeros((dataset.frame_count()))
        result["J_ratio"] = np.zeros((dataset.frame_count()))
        result["solve_time_per_seach"] = np.zeros((dataset.frame_count()))
        result["num_points_mean_before_filter"] = 0
        result["num_points_mean_after_filter"] = 0
        result["num_points_std_before_filter"] = 0
        result["num_points_std_after_filter"] = 0

        result["points"] = []

        for pt_id in range(dataset.frame_count()):

            current_frame = current_frame + 1

            # Print progress
            print_progress_bar(
                current_frame,
                total_frames,
                prefix=f"TEST: {test_name}",
                suffix="Complete",
                length=50,
            )

            # Number of points in the lidar frame before filtering
            n_points_before_filter = dataset.lidar_points(pt_id).shape[1]

            # Filter lidar points
            if filter_method == "none":
                points = dataset.lidar_points(pt_id)
            elif filter_method == "ground_filter":
                points = remove_ground_plane(
                    dataset.lidar_points(pt_id).T, distance_threshold=5.0, ransac_n=5
                ).T
            elif filter_method == "clustering":
                points = filter_cable_points(dataset.lidar_points(pt_id).T).T
            elif filter_method == "corridor":
                points = dataset.filtered_lidar_points(pt_id)
            else:
                raise ValueError(f"Filter method {filter_method} not recognized.")

            # Store filtered points used for optimization
            result["points"].append(points)

            # Number of points in the lidar frame after filter
            n_points_after_filter = points.shape[1]

            # Execute estimator
            if pt_id == 0:
                p_hat = p_0
            ########################################
            start_time = time.time()
            p_hat = estimator.solve_with_search(points, p_hat)
            solve_time = time.time() - start_time
            solve_time_per_seach = solve_time / estimator.n_search
            ########################################

            # Compute actual cost (no regulation)
            J_param = estimator.get_cost_parameters(m=n_points_after_filter)
            J_hat = powerline.J(p_hat, points, p_hat, J_param)

            # Compute ground truth cost (with points in current frame)
            p_tru = dataset.ground_thruth_params(pt_id)
            J_tru = powerline.J(p_tru, points, p_tru, J_param)

            # Compute number of points close to the power line model
            pts_in_hat = estimator.get_array_group(p_hat, points)
            n_in_hat = pts_in_hat.shape[1]

            pts_in_tru = estimator.get_array_group(p_tru, points)
            n_in_tru = pts_in_tru.shape[1]

            # Ratio of performance vs. ground truth
            n_in_ratio = n_in_hat / n_in_tru
            J_ratio = J_tru / J_hat

            # Store result
            result["solve_time_per_seach"][pt_id] = solve_time_per_seach
            result["num_points_close_model_tru"][pt_id] = n_in_tru
            result["num_points_close_model_hat"][pt_id] = n_in_hat
            result["n_in_ratio"][pt_id] = n_in_ratio
            result["J_ratio"][pt_id] = J_ratio
            result["J_tru"][pt_id] = J_tru
            result["J_hat"][pt_id] = J_hat
            result["p_hat"][pt_id] = p_hat
            result["p_err"][pt_id] = dataset.ground_thruth_params(pt_id) - p_hat
            result["num_points_before_filter"][pt_id] = n_points_before_filter
            result["num_points_after_filter"][pt_id] = n_points_after_filter

        results.append(result)

        stats = {}

        # Combine results for each randomized by taking only the last stats_num_frames frames
        p_err = np.vstack([res["p_err"][-stats_num_frames:] for res in results])
        num_points_before_filter = np.vstack(
            [res["num_points_before_filter"][-stats_num_frames:] for res in results]
        )
        num_points_after_filter = np.vstack(
            [res["num_points_after_filter"][-stats_num_frames:] for res in results]
        )

        num_solve_time_per_seach = np.vstack(
            [res["solve_time_per_seach"][-stats_num_frames:] for res in results]
        )

        num_n_in_ratio = np.vstack(
            [res["n_in_ratio"][-stats_num_frames:] for res in results]
        )

        num_J_ratio = np.vstack([res["J_ratio"][-stats_num_frames:] for res in results])

        # Compute statistics on combined results

        stats["p_err_mean"] = np.mean(p_err, axis=0)
        stats["p_err_std"] = np.std(p_err, axis=0)

        stats["num_points_mean_before_filter"] = np.mean(num_points_before_filter)
        stats["num_points_std_before_filter"] = np.std(num_points_before_filter)

        stats["num_points_mean_after_filter"] = np.mean(num_points_after_filter)
        stats["num_points_std_after_filter"] = np.std(num_points_after_filter)

        stats["solve_time_per_seach_mean"] = np.mean(num_solve_time_per_seach)
        stats["solve_time_per_seach_std"] = np.std(num_solve_time_per_seach)

        stats["n_in_ratio_mean"] = np.mean(num_n_in_ratio)
        stats["n_in_ratio_std"] = np.std(num_n_in_ratio)

        stats["J_ratio_mean"] = np.mean(num_J_ratio)
        stats["J_ratio_std"] = np.std(num_J_ratio)

    return results, stats


def animate_results(params, results):
    """
    Animate test results.

    Parameters
    ----------
    params: dict
        Parameters dictionary.
    results: array
        Array of results dictionary.
    """

    # Figure 1 : Plot estimated power line and ground thruth as animation
    fig1 = plt.figure(1, figsize=(14, 10))
    ax1 = plt.axes(projection="3d")

    dataset = params["dataset"]
    model = powerline.create_array_model(params["model"])

    for result_idx, result in enumerate(results):
        for idx in range(dataset.frame_count()):
            ax1.clear()

            p_hat = result["p_hat"][idx]

            p_ground_thruth = dataset.ground_thruth_params(idx)

            # Compute projected power line points using estimated model
            pts_hat = model.p2r_w(p_hat, x_min=-100, x_max=100, n=200)[1]

            # Compute ground thruth line points
            pts_ground_thruth = model.p2r_w(
                p_ground_thruth, x_min=-100, x_max=100, n=200
            )[1]

            # Plot raw lidar points
            ax1.scatter(
                dataset.lidar_points(idx)[0],
                dataset.lidar_points(idx)[1],
                dataset.lidar_points(idx)[2],
                color="red",
                alpha=0.5,
                s=1,
            )

            # # Plot filtered lidar points
            ax1.scatter(
                result["points"][idx][0],
                result["points"][idx][1],
                result["points"][idx][2],
                color="blue",
                alpha=1,
                s=5,
            )

            for i in range(pts_hat.shape[2]):
                ax1.plot3D(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], "-k")

            for i in range(pts_ground_thruth.shape[2]):
                ax1.plot3D(
                    pts_ground_thruth[0, :, i],
                    pts_ground_thruth[1, :, i],
                    pts_ground_thruth[2, :, i],
                    "-g",
                )

            # Set fixed scale
            ax1.set_xlim([-50, 50])
            ax1.set_ylim([-50, 50])
            ax1.set_zlim([0, 50])

            n_in_tru = result["num_points_close_model_tru"][idx]
            J_tru = result["J_tru"][idx]
            n_in_hat = result["num_points_close_model_hat"][idx]
            J_hat = result["J_hat"][idx]
            dt = result["solve_time_per_seach"][idx]

            n_in_ratio = result["n_in_ratio"][idx]
            J_ratio = result["J_ratio"][idx]

            # Display test name with run number on graph
            ax1.text2D(
                0.05,
                0.95,
                f"Test: {params['name']}, run {result_idx+1}/{len(results)}, frame {idx+1}/{dataset.frame_count()}, solve time per search [ms]: {dt*1000:.2f}, n_in ratio: {n_in_ratio:.2f}, J ratio: {J_ratio:.2f}",
                transform=ax1.transAxes,
            )

            ax1.text2D(
                0.05,
                0.90,
                f"TRU n_in: {n_in_tru}, J: {J_tru}, p: {np.array2string(p_ground_thruth, precision=2)}",
                transform=ax1.transAxes,
            )

            ax1.text2D(
                0.05,
                0.85,
                f"HAT n_in: {n_in_hat}, J: {J_hat}, p: {np.array2string(p_hat, precision=2)}",
                transform=ax1.transAxes,
            )

            plt.pause(0.001)


def plot_results(params, results):
    """
    Animate test results.

    Parameters
    ----------
    params: dict
        Parameters dictionary.
    results: array
        Array of results dictionary.
    """

    # Figure 1 : Plot estimated power line and ground thruth as animation
    fig1 = plt.figure(1, figsize=(14, 10))
    ax1 = plt.axes(projection="3d")

    dataset = params["dataset"]
    model = powerline.create_array_model(params["model"])

    for result_idx, result in enumerate(results):
        for idx in range(dataset.frame_count()):
            ax1.clear()

            p_hat = result["p_hat"][idx]

            p_ground_thruth = dataset.ground_thruth_params(idx)

            # Compute projected power line points using estimated model
            pts_hat = model.p2r_w(p_hat, x_min=-100, x_max=100, n=200)[1]

            # Compute ground thruth line points
            pts_ground_thruth = model.p2r_w(
                p_ground_thruth, x_min=-100, x_max=100, n=200
            )[1]

            # Plot raw lidar points
            ax1.scatter(
                dataset.lidar_points(idx)[0],
                dataset.lidar_points(idx)[1],
                dataset.lidar_points(idx)[2],
                color="red",
                alpha=0.5,
                s=1,
            )

            # # Plot filtered lidar points
            ax1.scatter(
                result["points"][idx][0],
                result["points"][idx][1],
                result["points"][idx][2],
                color="blue",
                alpha=1,
                s=5,
            )

            for i in range(pts_hat.shape[2]):
                ax1.plot3D(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], "-k")

            for i in range(pts_ground_thruth.shape[2]):
                ax1.plot3D(
                    pts_ground_thruth[0, :, i],
                    pts_ground_thruth[1, :, i],
                    pts_ground_thruth[2, :, i],
                    "-g",
                )

            # Set fixed scale
            ax1.set_xlim([-50, 50])
            ax1.set_ylim([-50, 50])
            ax1.set_zlim([0, 50])

            n_in_tru = result["num_points_close_model_tru"][idx]
            J_tru = result["J_tru"][idx]
            n_in_hat = result["num_points_close_model_hat"][idx]
            J_hat = result["J_hat"][idx]
            dt = result["solve_time_per_seach"][idx]

            n_in_ratio = result["n_in_ratio"][idx]
            J_ratio = result["J_ratio"][idx]

            # Display test name with run number on graph
            ax1.text2D(
                0.05,
                0.95,
                f"Test: {params['name']}, run {result_idx+1}/{len(results)}, frame {idx+1}/{dataset.frame_count()}, solve time per search [ms]: {dt*1000:.2f}, n_in ratio: {n_in_ratio:.2f}, J ratio: {J_ratio:.2f}",
                transform=ax1.transAxes,
            )

            ax1.text2D(
                0.05,
                0.90,
                f"TRU n_in: {n_in_tru}, J: {J_tru}, p: {np.array2string(p_ground_thruth, precision=2)}",
                transform=ax1.transAxes,
            )

            ax1.text2D(
                0.05,
                0.85,
                f"HAT n_in: {n_in_hat}, J: {J_hat}, p: {np.array2string(p_hat, precision=2)}",
                transform=ax1.transAxes,
            )

            plt.pause(0.001)


if __name__ == "__main__":

    # from experiments.test_runner import run_test, plot_results
    from experiments.dataset import load_dataset, SimulatedDataset

    # self._ground_truth = np.array(
    #             [
    #                 -22.61445006,
    #                 42.86768157,
    #                 14.25202579,
    #                 2.31972922,
    #                 698.6378392,
    #                 5.83313134,
    #                 7.68165757,
    #                 7.28652209,
    #             ]
    #         )

    # Test parameters
    params = {
        "name": "test",
        "dataset": None,
        "model": "222",
        "p_0": np.array([-25.0, 40.0, 0.0, 1.0, 700, 6.0, 6.0, 6.0]),
        "Q": 0.1 * np.diag([0.02, 0.02, 0.002, 0.01, 0.0001, 0.02, 0.02, 0.02]),
        "l": 1.0,
        "b": 100.0,
        "power": 2.0,
        "p_lb": np.array([-100.0, -100.0, 0.0, 1.5, 500.0, 5.0, 6.0, 6.0]),
        "p_ub": np.array([100.0, 100.0, 25.0, 2.5, 1500.0, 7.0, 9.0, 9.0]),
        "n_search": 2,
        "p_var": np.array([50.0, 50.0, 50.0, 5.0, 200.0, 2.0, 2.0, 2.0]),
        "filter_method": "corridor",  # No filter, as simulated data is already filtered
        "num_randomized_tests": 5,  # Number of tests to execute with randomized initial guess
        "stats_num_frames": 50,  # Number of last frames to use for statistics (experimental results have 100 frames)
        "num_outliers": 10,  # Number of outliers to simulate
    }

    params["dataset"] = SimulatedDataset("sim_222", params["num_outliers"])
    # params["dataset"] = load_dataset("ligne315kv_test1")
    results, stats = run_test(params)
    animate_results(params, results)
