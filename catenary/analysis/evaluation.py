import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import numpy as np
import time

from catenary.kinematic import powerline

from catenary.estimation.filter import filter_cable_points, remove_ground_plane
from catenary.estimation.estimator import ArrayEstimator
from catenary.estimation import costfunction
from catenary.tools import print_progress_bar
from catenary.analysis.dataset import Dataset, SimulatedDataset


def evaluate(params: dict):
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

    # Load dataset vs simulated dataset
    if isinstance(params["dataset"], Dataset):
        datagen = False
        dataset = params["dataset"]
    else:
        datagen = True
        datagen_params = params["dataset"]
        datagen_seed = datagen_params["seed"]
        dataset = SimulatedDataset(datagen_params)

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
        if params["p_0"] is None:
            p_0 = rng.uniform(params["p_lb"], params["p_ub"])
            # print(f"Randomized initial guess: {p_0}")
        else:
            p_0 = params["p_0"]
            print(f"Fixed initial guess: {p_0}")

        # Randomize noise in dataset
        if datagen:
            datagen_params["seed"] = datagen_seed + i * 100
            dataset = SimulatedDataset(datagen_params)

        # Initialize estimator
        estimator = ArrayEstimator(model, p_0)

        # Regulation weight matrix
        estimator.Q = params["Q"]

        # Parameters of lorentzian cost shaping function
        estimator.l = params["l"]
        estimator.power = params["power"]
        estimator.b = params["b"]

        # Parameters upper and lower bounds
        estimator.p_ub = params["p_ub"]
        estimator.p_lb = params["p_lb"]

        # default cost function parameters
        estimator.method = params["method"]  # "x"
        estimator.n_sample = params["n_sample"]
        estimator.x_min = params["x_min"]
        estimator.x_max = params["x_max"]

        estimator.use_grad = params["use_grad"]

        # Search parameters
        # Number of search
        estimator.n_search = params["n_search"]
        # Parameters std deviation for searching
        estimator.p_var = params["p_var"]

        # Initialize p_hat key with an numpy array of dataset.frame_count() elements
        result["p_0"] = p_0
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
        result["n_in_ratio_tru"] = np.zeros((dataset.frame_count()))
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
            J_hat = costfunction.J(p_hat, points, p_hat, J_param)

            # Compute ground truth cost (with points in current frame)
            p_tru = dataset.ground_thruth_params(pt_id)
            J_tru = costfunction.J(p_tru, points, p_tru, J_param)

            # Compute number of points close to the power line model
            pts_in_hat = estimator.get_array_group(p_hat, points)
            n_in_hat = pts_in_hat.shape[1]

            pts_in_tru = estimator.get_array_group(p_tru, points)
            n_in_tru = pts_in_tru.shape[1]

            # Ratio of performance vs. ground truth
            n_in_ratio = n_in_hat / n_in_tru
            J_ratio = J_tru / J_hat

            if dataset.outliers_count() > 0:
                n_tot_line_tru = points.shape[1] - dataset.outliers_count()
                n_in_ratio_tru = n_in_hat / n_tot_line_tru
            else:
                n_in_ratio_tru = -1.0  # Undefined for real datasets

            # Store result
            result["solve_time_per_seach"][pt_id] = solve_time_per_seach
            result["num_points_close_model_tru"][pt_id] = n_in_tru
            result["num_points_close_model_hat"][pt_id] = n_in_hat
            result["n_in_ratio"][pt_id] = n_in_ratio
            result["n_in_ratio_tru"][pt_id] = n_in_ratio_tru
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

        num_n_in_ratio_tru = np.vstack(
            [res["n_in_ratio_tru"][-stats_num_frames:] for res in results]
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

        stats["n_in_ratio_tru_mean"] = np.mean(num_n_in_ratio_tru)
        stats["n_in_ratio_tru_std"] = np.std(num_n_in_ratio_tru)

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

    # Load dataset vs simulated dataset
    if isinstance(params["dataset"], Dataset):
        datagen = False
        dataset = params["dataset"]
    else:
        datagen = True
        datagen_params = params["dataset"]
        datagen_seed = datagen_params["seed"]
        dataset = SimulatedDataset(datagen_params)

    model = powerline.create_array_model(params["model"])

    for result_idx, result in enumerate(results):

        p_hat = result["p_0"]
        p_tru = dataset.ground_thruth_params(0)

        plot_3d = powerline.EstimationPlot(
            p_tru, p_hat, None, model.p2r_w, xmin=-200, xmax=200
        )

        # Display test name with run number on graph
        text1 = plot_3d.ax.text2D(
            0.0,
            1.0,
            f"Test: ",
            transform=plot_3d.ax.transAxes,
            fontsize=4,
        )

        text2 = plot_3d.ax.text2D(
            0.0,
            0.95,
            f"TRU ",
            transform=plot_3d.ax.transAxes,
            fontsize=4,
        )

        text3 = plot_3d.ax.text2D(
            0.0,
            0.9,
            f"HAT ",
            transform=plot_3d.ax.transAxes,
            fontsize=4,
        )

        for idx in range(dataset.frame_count()):

            p_hat = result["p_hat"][idx]
            p_ground_thruth = dataset.ground_thruth_params(idx)

            plot_3d.update_estimation(p_hat)
            plot_3d.update_pts(result["points"][idx])

            n_in_tru = result["num_points_close_model_tru"][idx]
            J_tru = result["J_tru"][idx]
            n_in_hat = result["num_points_close_model_hat"][idx]
            J_hat = result["J_hat"][idx]
            dt = result["solve_time_per_seach"][idx]

            n_in_ratio = result["n_in_ratio"][idx]
            n_in_ratio_tru = result["n_in_ratio_tru"][idx]
            J_ratio = result["J_ratio"][idx]

            text1.set_text(
                f"Test: {params['name']}, run {result_idx+1}/{len(results)}, frame {idx+1}/{dataset.frame_count()}, solve time per search [ms]: {dt*1000:.2f}, n_in ratio: {n_in_ratio:.2f}, n_in tru: {n_in_ratio_tru:.2f}, J ratio: {J_ratio:.2f}"
            )
            text2.set_text(
                f"TRU n_in: {n_in_tru}, J: {J_tru}, p: {np.array2string(p_ground_thruth, precision=2)}"
            )
            text3.set_text(
                f"HAT n_in: {n_in_hat}, J: {J_hat}, p: {np.array2string(p_hat, precision=2)}"
            )


def animate_results2(params, results):
    """
    Animate test results.

    Parameters
    ----------
    params: dict
        Parameters dictionary.
    results: array
        Array of results dictionary.
    """

    fig = plt.figure(figsize=(4, 3), dpi=300, frameon=True)
    ax = fig.add_subplot(projection="3d")

    # Load dataset vs simulated dataset
    if isinstance(params["dataset"], Dataset):
        datagen = False
        dataset = params["dataset"]
    else:
        datagen = True
        datagen_params = params["dataset"]
        datagen_seed = datagen_params["seed"]
        dataset = SimulatedDataset(datagen_params)

    model = powerline.create_array_model(params["model"])

    for result_idx, result in enumerate(results):

        for idx in range(dataset.frame_count()):
            ax.clear()

            p_hat = result["p_hat"][idx]

            p_ground_thruth = dataset.ground_thruth_params(idx)

            # Compute projected power line points using estimated model
            pts_hat = model.p2r_w(p_hat, x_min=-100, x_max=100, n=200)[1]

            # Compute ground thruth line points
            pts_ground_thruth = model.p2r_w(
                p_ground_thruth, x_min=-100, x_max=100, n=200
            )[1]

            for i in range(pts_ground_thruth.shape[2]):

                if i == 0:
                    ax.plot(
                        pts_ground_thruth[0, :, i],
                        pts_ground_thruth[1, :, i],
                        pts_ground_thruth[2, :, i],
                        "-k",
                        label="True",
                    )
                else:
                    ax.plot(
                        pts_ground_thruth[0, :, i],
                        pts_ground_thruth[1, :, i],
                        pts_ground_thruth[2, :, i],
                        "-k",
                    )

            for i in range(pts_hat.shape[2]):
                if i == 0:
                    ax.plot(
                        pts_hat[0, :, i],
                        pts_hat[1, :, i],
                        pts_hat[2, :, i],
                        "--",
                        label="Est.",
                    )
                else:
                    ax.plot(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], "--")

            # # Plot filtered lidar points
            ax.scatter(
                result["points"][idx][0],
                result["points"][idx][1],
                result["points"][idx][2],
                color="blue",
                alpha=0.5,
                s=3,
                label="Pts",
            )

            # # Plot raw lidar points
            # ax.scatter(
            #     dataset.lidar_points(idx)[0],
            #     dataset.lidar_points(idx)[1],
            #     dataset.lidar_points(idx)[2],
            #     color="red",
            #     alpha=0.2,
            #     s=1,
            # )

            # Set fixed scale
            ax.set_xlim([-50, 50])
            ax.set_ylim([-50, 50])
            ax.set_zlim([-25, 75])

            n_in_tru = result["num_points_close_model_tru"][idx]
            J_tru = result["J_tru"][idx]
            n_in_hat = result["num_points_close_model_hat"][idx]
            J_hat = result["J_hat"][idx]
            dt = result["solve_time_per_seach"][idx]

            n_in_ratio = result["n_in_ratio"][idx]
            J_ratio = result["J_ratio"][idx]

            # Display test name with run number on graph
            ax.text2D(
                0.05,
                0.95,
                f"Test: {params['name']}, run {result_idx+1}/{len(results)}, frame {idx+1}/{dataset.frame_count()}, solve time per search [ms]: {dt*1000:.2f}, n_in ratio: {n_in_ratio:.2f}, J ratio: {J_ratio:.2f}",
                transform=ax.transAxes,
                fontsize=4,
            )

            ax.text2D(
                0.05,
                0.90,
                f"TRU n_in: {n_in_tru}, J: {J_tru}, p: {np.array2string(p_ground_thruth, precision=2)}",
                transform=ax.transAxes,
                fontsize=4,
            )

            ax.text2D(
                0.05,
                0.85,
                f"HAT n_in: {n_in_hat}, J: {J_hat}, p: {np.array2string(p_hat, precision=2)}",
                transform=ax.transAxes,
                fontsize=4,
            )

            plt.pause(0.001)
            ax.legend(loc="upper right")

    fig.show()


def plot_results(params, results, save=False, n_run_plot=5, fs=8):
    """
    Plot performance figure

    Parameters
    ----------
    params: dict
        Parameters dictionary.
    results: array
        Array of results dictionary.
    """

    name = params["name"]

    n_run = len(results)

    # Load dataset vs simulated dataset
    if isinstance(params["dataset"], Dataset):
        datagen = False
        dataset = params["dataset"]
    else:
        datagen = True
        datagen_params = params["dataset"]
        datagen_seed = datagen_params["seed"]
        dataset = SimulatedDataset(datagen_params)

    model = powerline.create_array_model(params["model"])

    p_tru = dataset.ground_thruth_params(0)

    n_p = p_tru.shape[0]
    n_frame = dataset.frame_count()

    if n_run_plot > n_run:
        n_run_plot = n_run

    # Parameter errors
    PE = np.zeros((n_p, n_frame + 1, n_run))

    # Time to solve
    t_solve = np.zeros((n_frame, n_run))

    # Number of points close to model
    pt_in = np.zeros((n_frame, n_run))

    # Number of points close to model vs. ground truth
    pt_ratio = np.zeros((n_frame, n_run))
    pt_ratio_tru = np.zeros((n_frame, n_run))

    # Cost function
    cost = np.zeros((n_frame, n_run))

    # Cost function vs. ground truth
    cost_ratio = np.zeros((n_frame, n_run))

    # For all runs
    for run_id, result in enumerate(results):

        PE[:, 0, run_id] = p_tru - result["p_0"]  # Initial guess error

        p_init = result["p_0"]

        if datagen:
            datagen_params["seed"] = datagen_seed + run_id * 100
            dataset = SimulatedDataset(datagen_params)

        # For all frames in the run
        for frame_id in range(dataset.frame_count()):

            p_hat = result["p_hat"][frame_id]
            p_tru = dataset.ground_thruth_params(frame_id)

            PE[:, frame_id + 1, run_id] = p_tru - p_hat

            t_solve[frame_id, run_id] = result["solve_time_per_seach"][frame_id]
            pt_in[frame_id, run_id] = result["num_points_close_model_hat"][frame_id]
            pt_ratio[frame_id, run_id] = result["n_in_ratio"][frame_id]
            pt_ratio_tru[frame_id, run_id] = result["n_in_ratio_tru"][frame_id]
            cost[frame_id, run_id] = result["J_hat"][frame_id]
            cost_ratio[frame_id, run_id] = result["J_ratio"][frame_id]

    PE_mean = np.mean(PE, axis=2)
    PE_std = np.std(PE, axis=2)
    t_solve_mean = np.mean(t_solve, axis=1)
    t_solve_std = np.std(t_solve, axis=1)
    pt_in_mean = np.mean(pt_in, axis=1)
    pt_in_std = np.std(pt_in, axis=1)
    pt_ratio_mean = np.mean(pt_ratio, axis=1)
    pt_ratio_std = np.std(pt_ratio, axis=1)
    pt_ratio_tru_mean = np.mean(pt_ratio_tru, axis=1)
    pt_ratio_tru_std = np.std(pt_ratio_tru, axis=1)
    cost_mean = np.mean(cost, axis=1)
    cost_std = np.std(cost, axis=1)
    cost_ratio_mean = np.mean(cost_ratio, axis=1)
    cost_ratio_std = np.std(cost_ratio, axis=1)

    frame = np.linspace(0, n_frame, n_frame + 1)

    # ###################################################################
    # fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    # # Plot n runs sample
    # for j in range(n_run_plot):
    #     ax.plot(
    #         frame,
    #         PE[0, :, j],
    #         "--k",
    #         linewidth=0.25,
    #     )

    # # Plot mean
    # ax.plot(frame, PE_mean[0, :], "-r")

    # # Plot std
    # ax.fill_between(
    #     frame,
    #     PE_mean[0, :] - PE_std[0, :],
    #     PE_mean[0, :] + PE_std[0, :],
    #     color="#DDDDDD",
    # )

    # # ax.legend( loc = 'upper right' , fontsize = fs)
    # ax.set_xlabel("frames", fontsize=fs)
    # ax.set_ylabel("$x_o$[m]", fontsize=fs)
    # ax.grid(True)
    # fig.tight_layout()
    # fig.show()

    # if save:
    #     fig.savefig(name + "_x_error.pdf")

    # ###################################################################

    # fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    # # Plot n runs sample
    # for j in range(n_run_plot):
    #     ax.plot(
    #         frame,
    #         PE[1, :, j],
    #         "--k",
    #         linewidth=0.25,
    #     )

    # # Plot mean
    # ax.plot(frame, PE_mean[1, :], "-r")

    # # Plot std
    # ax.fill_between(
    #     frame,
    #     PE_mean[1, :] - PE_std[1, :],
    #     PE_mean[1, :] + PE_std[1, :],
    #     color="#DDDDDD",
    # )

    # # ax.legend( loc = 'upper right' , fontsize = fs)
    # ax.set_xlabel("frames", fontsize=fs)
    # ax.set_ylabel("$y_o$[m]", fontsize=fs)
    # ax.grid(True)
    # fig.tight_layout()
    # fig.show()

    # if save:
    #     fig.savefig(name + "_y_error.pdf")

    # ###################################################################

    # fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    # # Plot n runs sample
    # for j in range(n_run_plot):
    #     ax.plot(
    #         frame,
    #         PE[2, :, j],
    #         "--k",
    #         linewidth=0.25,
    #     )

    # # Plot mean
    # ax.plot(frame, PE_mean[2, :], "-r")

    # # Plot std
    # ax.fill_between(
    #     frame,
    #     PE_mean[2, :] - PE_std[2, :],
    #     PE_mean[2, :] + PE_std[2, :],
    #     color="#DDDDDD",
    # )

    # # ax.legend( loc = 'upper right' , fontsize = fs)
    # ax.set_xlabel("frames", fontsize=fs)
    # ax.set_ylabel("$z_o$[m]", fontsize=fs)
    # ax.grid(True)
    # fig.tight_layout()
    # fig.show()

    # if save:
    #     fig.savefig(name + "_z_error.pdf")

    # ###########################################################

    # fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    # for j in range(n_run_plot):
    #     ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
    # ax.plot(frame, PE_mean[3, :], "-r")
    # ax.fill_between(
    #     frame,
    #     PE_mean[3, :] - PE_std[3, :],
    #     PE_mean[3, :] + PE_std[3, :],
    #     color="#DDDDDD",
    # )

    # # ax.legend( loc = 'upper right' , fontsize = fs)
    # ax.set_xlabel("frames", fontsize=fs)
    # ax.set_ylabel("$\psi$ [rad]", fontsize=fs)
    # ax.grid(True)
    # fig.tight_layout()
    # fig.show()

    # if save:
    #     fig.savefig(name + "_orientation_error.pdf")

    # ###########################################################

    # fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    # for j in range(n_run_plot):
    #     ax.plot(frame, PE[4, :, j], "--k", linewidth=0.25)
    # ax.plot(frame, PE_mean[4, :], "-r")
    # ax.fill_between(
    #     frame,
    #     PE_mean[4, :] - PE_std[4, :],
    #     PE_mean[4, :] + PE_std[4, :],
    #     color="#DDDDDD",
    # )

    # # ax.legend( loc = 'upper right' , fontsize = fs)
    # ax.set_xlabel("frames", fontsize=fs)
    # ax.set_ylabel("$a$[m]", fontsize=fs)
    # ax.grid(True)
    # fig.tight_layout()
    # fig.show()
    # if save:
    #     fig.savefig(name + "_sag_error.pdf")

    # ###########################################################

    # fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    # l_n = PE.shape[0] - 5

    # for i in range(l_n):
    #     k = 5 + i
    #     for j in range(n_run_plot):
    #         ax.plot(frame, PE[k, :, j], "--k", linewidth=0.25)
    #     ax.plot(frame, PE_mean[k, :], "-r")
    #     ax.fill_between(
    #         frame,
    #         PE_mean[k, :] - PE_std[k, :],
    #         PE_mean[k, :] + PE_std[k, :],
    #         color="#DDDDDD",
    #     )

    # # ax.legend( loc = 'upper right' , fontsize = fs)
    # ax.set_xlabel("frames", fontsize=fs)
    # ax.set_ylabel("$\Delta$[m]", fontsize=fs)
    # ax.grid(True)
    # fig.tight_layout()
    # fig.show()
    # if save:
    #     fig.savefig(name + "_internaloffsets_error.pdf")

    ###########################################################

    fig, axes = plt.subplots(8, figsize=(4, 8), dpi=300, frameon=True)

    # Subplot 1 : Accuracy
    ax = axes[0]

    for j in range(n_run_plot):
        if j == 0:
            ax.plot(
                frame[1:], pt_ratio[:, j], "--k", label="Sample run", linewidth=0.25
            )
        else:
            ax.plot(frame[1:], pt_ratio[:, j], "--k", linewidth=0.25)
    ax.plot(frame[1:], pt_ratio_mean[:], "-r", label="Average")
    ax.fill_between(
        frame[1:],
        pt_ratio_mean[:] - pt_ratio_std[:],
        pt_ratio_mean[:] + pt_ratio_std[:],
        color="#DDDDDD",
        label="Std. dev.",
    )

    ax.set_ylabel("Accuracy [-]", fontsize=fs)
    ax.legend(loc="lower right", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 2 : x
    ax = axes[1]

    # Plot n runs sample
    for j in range(n_run_plot):
        ax.plot(
            frame,
            PE[0, :, j],
            "--k",
            linewidth=0.25,
        )

    # Plot mean
    ax.plot(frame, PE_mean[0, :], "-r")

    # Plot std
    ax.fill_between(
        frame,
        PE_mean[0, :] - PE_std[0, :],
        PE_mean[0, :] + PE_std[0, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$x$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 3 : y
    ax = axes[2]

    # Plot n runs sample
    for j in range(n_run_plot):
        ax.plot(
            frame,
            PE[1, :, j],
            "--k",
            linewidth=0.25,
        )

    # Plot mean
    ax.plot(frame, PE_mean[1, :], "-r")

    # Plot std
    ax.fill_between(
        frame,
        PE_mean[1, :] - PE_std[1, :],
        PE_mean[1, :] + PE_std[1, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$y$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 4 : z
    ax = axes[3]

    # Plot n runs sample
    for j in range(n_run_plot):
        ax.plot(
            frame,
            PE[2, :, j],
            "--k",
            linewidth=0.25,
        )

    # Plot mean
    ax.plot(frame, PE_mean[2, :], "-r")

    # Plot std
    ax.fill_between(
        frame,
        PE_mean[2, :] - PE_std[2, :],
        PE_mean[2, :] + PE_std[2, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$z$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 5 : psi
    ax = axes[4]

    for j in range(n_run_plot):
        if j == 0:
            ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
        else:
            ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
    ax.plot(frame, PE_mean[3, :], "-r")
    ax.fill_between(
        frame,
        PE_mean[3, :] - PE_std[3, :],
        PE_mean[3, :] + PE_std[3, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$\psi$ [rad]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 6 : sag
    ax = axes[5]

    for j in range(n_run_plot):
        ax.plot(frame, PE[4, :, j], "--k", linewidth=0.25)
    ax.plot(frame, PE_mean[4, :], "-r")
    ax.fill_between(
        frame,
        PE_mean[4, :] - PE_std[4, :],
        PE_mean[4, :] + PE_std[4, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$a$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 7 : offsets
    ax = axes[6]

    l_n = PE.shape[0] - 5

    for i in range(l_n):
        k = 5 + i
        for j in range(n_run_plot):
            ax.plot(frame, PE[k, :, j], "--k", linewidth=0.25)
        ax.plot(frame, PE_mean[k, :], "-r")
        ax.fill_between(
            frame,
            PE_mean[k, :] - PE_std[k, :],
            PE_mean[k, :] + PE_std[k, :],
            color="#DDDDDD",
        )

    ax.set_ylabel("$\Delta$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 8 : solver time
    ax = axes[7]

    for j in range(n_run_plot):
        ax.plot(frame[1:], t_solve[:, j], "--k", linewidth=0.25)
    ax.plot(frame[1:], t_solve_mean, "-r")
    ax.fill_between(
        frame[1:],
        t_solve_mean - t_solve_std,
        t_solve_mean + t_solve_std,
        color="#DDDDDD",
    )
    ax.set_ylabel("solver time [sec]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Finalizing

    ax.set_xlabel("frames", fontsize=fs)
    fig.tight_layout()
    fig.show()

    if save:
        fig.savefig(name + "_solver_time.pdf")

    ###########################################################

    # fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    # for j in range(n_run_plot):
    #     ax.plot(frame[1:], pt_ratio[:, j], "--k", linewidth=0.25)
    # ax.plot(frame[1:], pt_ratio_mean[:], "-r")
    # ax.fill_between(
    #     frame[1:],
    #     pt_ratio_mean[:] - pt_ratio_std[:],
    #     pt_ratio_mean[:] + pt_ratio_std[:],
    #     color="#DDDDDD",
    # )

    # ax.set_xlabel("frames", fontsize=fs)
    # ax.set_ylabel("accuracy (inliers)", fontsize=fs)
    # ax.grid(True)
    # fig.tight_layout()
    # fig.show()
    # if save:
    #     fig.savefig(name + "_pt_ratio.pdf")

    ###########################################################

    # fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    # for j in range(n_run_plot):
    #     ax.plot(frame[1:], pt_ratio_tru[:, j], "--k", linewidth=0.25)
    # ax.plot(frame[1:], pt_ratio_mean[:], "-r")
    # ax.fill_between(
    #     frame[1:],
    #     pt_ratio_tru_mean[:] - pt_ratio_tru_std[:],
    #     pt_ratio_tru_mean[:] + pt_ratio_tru_std[:],
    #     color="#DDDDDD",
    # )

    # ax.set_xlabel("frames", fontsize=fs)
    # ax.set_ylabel("accuracy (inliers)", fontsize=fs)
    # ax.grid(True)
    # fig.tight_layout()
    # fig.show()
    # if save:
    #     fig.savefig(name + "_pt_ratio_tru.pdf")

    ###########################################################

    # fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    # for j in range(n_run_plot):
    #     ax.plot(frame[1:], cost_ratio[:, j], "--k", linewidth=0.25)
    # ax.plot(frame[1:], cost_ratio_mean[:], "-r")
    # ax.fill_between(
    #     frame[1:],
    #     cost_ratio_mean[:] - cost_ratio_std[:],
    #     cost_ratio_mean[:] + cost_ratio_std[:],
    #     color="#DDDDDD",
    # )

    # ax.set_xlabel("frames", fontsize=fs)
    # ax.set_ylabel("accuracy (cost)", fontsize=fs)
    # ax.grid(True)
    # fig.tight_layout()
    # fig.show()
    # if save:
    #     fig.savefig(name + "_cost_ratio.pdf")

    ###########################################################

    fig = plt.figure(figsize=(4, 3), dpi=300, frameon=True)
    ax = fig.add_subplot(projection="3d")

    # Compute projected power line points using estimated model
    pts_hat = model.p2r_w(p_hat, x_min=-100, x_max=100, n=200)[1]

    for i in range(pts_hat.shape[2]):
        if i == 0:
            ax.plot(
                pts_hat[0, :, i],
                pts_hat[1, :, i],
                pts_hat[2, :, i],
                "-k",
                label="Estimation",
            )
        else:
            ax.plot(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], "-k")

    # # Plot filtered lidar points
    ax.scatter(
        result["points"][-1][0],
        result["points"][-1][1],
        result["points"][-1][2],
        color="red",
        alpha=0.9,
        s=4,
        label="LiDAR points",
    )

    # Set fixed scale
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-25, 75])
    ax.tick_params(labelsize=fs)

    ax.legend(loc="upper right", fontsize=fs)

    ax.view_init(elev=12, azim=-71)

    fig.tight_layout()
    fig.show()

    if save:
        fig.savefig(name + "_3D.pdf")
        fig.savefig(name + "_3D.png")

    ###########################################################

    fig = plt.figure(figsize=(4, 3), dpi=300, frameon=True)
    ax = plt.axes(projection="3d")

    # Compute projected power line points using estimated model
    pts_hat = model.p2r_w(p_hat, x_min=-100, x_max=100, n=200)[1]

    # # Plot filtered lidar points
    ax.scatter(
        result["points"][-1][0],
        result["points"][-1][1],
        result["points"][-1][2],
        color="blue",
        alpha=0.5,
        label="Filtered points",
        s=3,
    )

    for i in range(pts_hat.shape[2]):
        if i == 0:
            ax.plot(
                pts_hat[0, :, i],
                pts_hat[1, :, i],
                pts_hat[2, :, i],
                "-k",
                label="Estimation",
            )
        else:
            ax.plot(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], "-k")

    # Plot raw lidar points
    ax.scatter(
        dataset.lidar_points(99)[0],
        dataset.lidar_points(99)[1],
        dataset.lidar_points(99)[2],
        label="LiDAR points",
        color="red",
        alpha=0.2,
        s=0.1,
    )

    # Set fixed scale
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-25, 75])
    ax.tick_params(labelsize=fs)

    ax.legend(loc="upper right", fontsize=fs)

    ax.view_init(elev=12, azim=-71)

    fig.tight_layout()
    fig.show()

    if save:
        fig.savefig(name + "_3D_with_lidar.pdf")
        fig.savefig(name + "_3D_with_lidar.png")

    ###########################################################
    # Figure for acc, psi, sag
    ###########################################################

    fig, axes = plt.subplots(3, figsize=(4, 4), dpi=300, frameon=True)

    # Subplot 1
    ax = axes[0]
    for j in range(n_run_plot):
        if j == 0:
            ax.plot(
                frame[1:], pt_ratio[:, j], "--k", label="Sample run", linewidth=0.25
            )
        else:
            ax.plot(frame[1:], pt_ratio[:, j], "--k", linewidth=0.25)
    ax.plot(frame[1:], pt_ratio_mean[:], "-r", label="Average")
    ax.fill_between(
        frame[1:],
        pt_ratio_mean[:] - pt_ratio_std[:],
        pt_ratio_mean[:] + pt_ratio_std[:],
        color="#DDDDDD",
        label="Std. dev.",
    )

    ax.set_ylabel("Accuracy [-]", fontsize=fs)
    ax.legend(loc="lower right", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 2
    ax = axes[1]
    for j in range(n_run_plot):
        if j == 0:
            ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
        else:
            ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
    ax.plot(frame, PE_mean[3, :], "-r")
    ax.fill_between(
        frame,
        PE_mean[3, :] - PE_std[3, :],
        PE_mean[3, :] + PE_std[3, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$\psi$ [rad]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 1
    ax = axes[2]

    for j in range(n_run_plot):
        ax.plot(frame, PE[4, :, j], "--k", linewidth=0.25)
    ax.plot(frame, PE_mean[4, :], "-r")
    ax.fill_between(
        frame,
        PE_mean[4, :] - PE_std[4, :],
        PE_mean[4, :] + PE_std[4, :],
        color="#DDDDDD",
    )

    # ax.legend( loc = 'upper right' , fontsize = fs)
    ax.set_xlabel("frames", fontsize=fs)
    ax.set_ylabel("$a$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    ax.set_xlabel("frames", fontsize=fs)
    fig.tight_layout()
    fig.show()
    if save:
        fig.savefig(name + "_acc_psi_sag.pdf")
        fig.savefig(name + "_acc_psi_sag.png")

    ###########################################################
    # Figure for all plots
    ###########################################################

    fig, axes = plt.subplots(7, figsize=(4, 8), dpi=300, frameon=True)

    # Subplot 1 : Accuracy
    ax = axes[0]

    for j in range(n_run_plot):
        if j == 0:
            ax.plot(
                frame[1:],
                pt_ratio[:, j] * 100,
                "--k",
                label="Sample run",
                linewidth=0.25,
            )
        else:
            ax.plot(frame[1:], pt_ratio[:, j] * 100, "--k", linewidth=0.25)
    ax.plot(frame[1:], pt_ratio_mean[:] * 100, "-r", label="Average")
    ax.fill_between(
        frame[1:],
        pt_ratio_mean[:] * 100 - pt_ratio_std[:] * 100,
        pt_ratio_mean[:] * 100 + pt_ratio_std[:] * 100,
        color="#DDDDDD",
        label="Std. dev.",
    )

    ax.set_ylabel("Accuracy [%]", fontsize=fs)
    ax.legend(loc="lower right", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 2 : x
    ax = axes[1]

    # Plot n runs sample
    for j in range(n_run_plot):
        ax.plot(
            frame,
            PE[0, :, j],
            "--k",
            linewidth=0.25,
        )

    # Plot mean
    ax.plot(frame, PE_mean[0, :], "-r")

    # Plot std
    ax.fill_between(
        frame,
        PE_mean[0, :] - PE_std[0, :],
        PE_mean[0, :] + PE_std[0, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$x$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 3 : y
    ax = axes[2]

    # Plot n runs sample
    for j in range(n_run_plot):
        ax.plot(
            frame,
            PE[1, :, j],
            "--k",
            linewidth=0.25,
        )

    # Plot mean
    ax.plot(frame, PE_mean[1, :], "-r")

    # Plot std
    ax.fill_between(
        frame,
        PE_mean[1, :] - PE_std[1, :],
        PE_mean[1, :] + PE_std[1, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$y$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 4 : z
    ax = axes[3]

    # Plot n runs sample
    for j in range(n_run_plot):
        ax.plot(
            frame,
            PE[2, :, j],
            "--k",
            linewidth=0.25,
        )

    # Plot mean
    ax.plot(frame, PE_mean[2, :], "-r")

    # Plot std
    ax.fill_between(
        frame,
        PE_mean[2, :] - PE_std[2, :],
        PE_mean[2, :] + PE_std[2, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$z$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 5 : psi
    ax = axes[4]

    for j in range(n_run_plot):
        if j == 0:
            ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
        else:
            ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
    ax.plot(frame, PE_mean[3, :], "-r")
    ax.fill_between(
        frame,
        PE_mean[3, :] - PE_std[3, :],
        PE_mean[3, :] + PE_std[3, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$\psi$ [rad]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 6 : sag
    ax = axes[5]

    for j in range(n_run_plot):
        ax.plot(frame, PE[4, :, j], "--k", linewidth=0.25)
    ax.plot(frame, PE_mean[4, :], "-r")
    ax.fill_between(
        frame,
        PE_mean[4, :] - PE_std[4, :],
        PE_mean[4, :] + PE_std[4, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$a$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 7 : offsets
    ax = axes[6]

    l_n = PE.shape[0] - 5

    for i in range(l_n):
        k = 5 + i
        for j in range(n_run_plot):
            ax.plot(frame, PE[k, :, j], "--k", linewidth=0.25)
        ax.plot(frame, PE_mean[k, :], "-r")
        ax.fill_between(
            frame,
            PE_mean[k, :] - PE_std[k, :],
            PE_mean[k, :] + PE_std[k, :],
            color="#DDDDDD",
        )

    ax.set_ylabel("$\Delta$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Finalizing

    ax.set_xlabel("frames", fontsize=fs)
    fig.tight_layout()
    fig.show()
    if save:
        fig.savefig(name + "_all.pdf")
        fig.savefig(name + "_all.png")

    ###########################################################

    ###########################################################
    # Figure for all plots
    ###########################################################

    fig, axes = plt.subplots(7, figsize=(4, 8), dpi=300, frameon=True)

    # Subplot 1 : Accuracy
    ax = axes[0]

    for j in range(n_run_plot):
        if j == 0:
            ax.plot(
                frame[1:],
                pt_ratio[:, j] * 100,
                "--k",
                label="Sample run",
                linewidth=0.25,
            )
        else:
            ax.plot(frame[1:], pt_ratio[:, j] * 100, "--k", linewidth=0.25)
    ax.plot(frame[1:], pt_ratio_mean[:] * 100, "-r", label="Average")
    ax.fill_between(
        frame[1:],
        pt_ratio_mean[:] * 100 - pt_ratio_std[:] * 100,
        pt_ratio_mean[:] * 100 + pt_ratio_std[:] * 100,
        color="#DDDDDD",
        label="Std. dev.",
    )

    ax.set_ylabel("Accuracy [%]", fontsize=fs)
    ax.legend(loc="lower right", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 2 : x
    ax = axes[1]

    # Plot n runs sample
    for j in range(n_run_plot):
        ax.plot(
            frame,
            PE[0, :, j],
            "--k",
            linewidth=0.25,
        )

    # Plot mean
    ax.plot(frame, PE_mean[0, :], "-r")

    # Plot std
    ax.fill_between(
        frame,
        PE_mean[0, :] - PE_std[0, :],
        PE_mean[0, :] + PE_std[0, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$x_e$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 3 : y
    ax = axes[2]

    # Plot n runs sample
    for j in range(n_run_plot):
        ax.plot(
            frame,
            PE[1, :, j],
            "--k",
            linewidth=0.25,
        )

    # Plot mean
    ax.plot(frame, PE_mean[1, :], "-r")

    # Plot std
    ax.fill_between(
        frame,
        PE_mean[1, :] - PE_std[1, :],
        PE_mean[1, :] + PE_std[1, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$y_e$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 4 : z
    ax = axes[3]

    # Plot n runs sample
    for j in range(n_run_plot):
        ax.plot(
            frame,
            PE[2, :, j],
            "--k",
            linewidth=0.25,
        )

    # Plot mean
    ax.plot(frame, PE_mean[2, :], "-r")

    # Plot std
    ax.fill_between(
        frame,
        PE_mean[2, :] - PE_std[2, :],
        PE_mean[2, :] + PE_std[2, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$z_e$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 5 : psi
    ax = axes[4]

    for j in range(n_run_plot):
        if j == 0:
            ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
        else:
            ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
    ax.plot(frame, PE_mean[3, :], "-r")
    ax.fill_between(
        frame,
        PE_mean[3, :] - PE_std[3, :],
        PE_mean[3, :] + PE_std[3, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$\psi_e$ [rad]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 6 : sag
    ax = axes[5]

    for j in range(n_run_plot):
        ax.plot(frame, PE[4, :, j], "--k", linewidth=0.25)
    ax.plot(frame, PE_mean[4, :], "-r")
    ax.fill_between(
        frame,
        PE_mean[4, :] - PE_std[4, :],
        PE_mean[4, :] + PE_std[4, :],
        color="#DDDDDD",
    )
    ax.set_ylabel("$a_e$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Subplot 7 : offsets
    ax = axes[6]

    l_n = PE.shape[0] - 5

    for i in range(l_n):
        k = 5 + i
        for j in range(n_run_plot):
            ax.plot(frame, PE[k, :, j], "--k", linewidth=0.25)
        ax.plot(frame, PE_mean[k, :], "-r")
        ax.fill_between(
            frame,
            PE_mean[k, :] - PE_std[k, :],
            PE_mean[k, :] + PE_std[k, :],
            color="#DDDDDD",
        )

    ax.set_ylabel("$\Delta_e$ [m]", fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs)

    # Finalizing

    ax.set_xlabel("frames", fontsize=fs)
    fig.tight_layout()
    fig.show()
    if save:
        fig.savefig(name + "_all_v3.pdf")
        fig.savefig(name + "_all_v3.png")


def table_init():
    from prettytable import PrettyTable

    table = PrettyTable()
    # table.field_names = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg"]
    table.field_names = [
        "name",
        "n pts",
        "time [ms]",
        "acc. [%]",
        # "on-model point ratio tru [%]",
        # "cost-function ratio [%]",
        # "Translation error [m]",
        "psi [rad]",
        "a [m]",
        # "offsets error [m]",
    ]
    return table


def table_add_row(table, params, stats):
    table.add_row(
        [
            params["name"],
            f'{stats["num_points_mean_after_filter"]:.0f} +/- {stats["num_points_std_after_filter"]:.0f}',
            f'{stats["solve_time_per_seach_mean"]*1000:.0f} +/- {stats["solve_time_per_seach_std"]*1000:.0f}',
            f'{stats["n_in_ratio_mean"]*100:.0f} +/- {stats["n_in_ratio_std"]*100:.0f}',
            # f'{stats["n_in_ratio_tru_mean"]*100:.1f}% +/- {stats["n_in_ratio_tru_std"]*100:.1f}',
            # f'{stats["J_ratio_mean"]*100:.1f} +/- {stats["J_ratio_std"]*100:.2f}',
            # f'{np.array2string(stats["p_err_mean"][0:3], precision=2)} +/- {np.array2string(stats["p_err_std"][0:3], precision=2)}',
            f'{np.array2string(stats["p_err_mean"][3], precision=1)} +/- {np.array2string(stats["p_err_std"][3], precision=1)}',
            f'{np.array2string(stats["p_err_mean"][4], precision=0)} +/- {np.array2string(stats["p_err_std"][4], precision=0)}',
            # f'{np.array2string(stats["p_err_mean"][5:], precision=2)} +/- {np.array2string(stats["p_err_std"][5:], precision=2)}',
            #     f'({stats["p_err_mean"][0]:.2f}, {stats["p_err_mean"][1]:.2f}, {stats["p_err_mean"][2]:.2f}) +/- '
            #     + f'({stats["p_err_std"][0]:.2f}, {stats["p_err_std"][1]:.2f}, {stats["p_err_std"][2]:.2f})',
        ]
    )


###############################################################################
### Real-time plotting
###############################################################################
class ErrorPlot:
    """ """

    ############################
    def __init__(self, p_true, p_hat, n_steps, n_run=1):

        self.n_p = p_true.shape[0]
        self.n_steps = n_steps
        self.n_run = n_run

        self.p_true = p_true

        self.PE = np.zeros((self.n_p, n_steps + 1, n_run))

        self.t = np.zeros((n_steps, n_run))
        self.n_in = np.zeros((n_steps, n_run))

        self.step = 0
        self.run = 0

        self.PE[:, self.step, self.run] = p_true - p_hat

    ############################
    def init_new_run(self, p_true, p_hat):

        self.p_true = p_true

        self.step = 0
        self.run = self.run + 1

        self.PE[:, self.step, self.run] = self.p_true - p_hat

    ############################
    def save_new_estimation(self, p_hat, t_solve, n_in=0):

        self.t[self.step, self.run] = t_solve
        self.n_in[self.step, self.run] = n_in

        self.step = self.step + 1

        self.PE[:, self.step, self.run] = self.p_true - p_hat

    ############################
    def plot_error_mean_std(self, fs=10, save=False, name="test", n_run_plot=10):

        PE = self.PE
        t = self.t
        n_in = self.n_in

        if n_run_plot > self.n_run:
            n_run_plot = self.n_run

        PE_mean = np.mean(PE, axis=2)
        PE_std = np.std(PE, axis=2)
        t_mean = np.mean(t, axis=1)
        t_std = np.std(t, axis=1)
        n_mean = np.mean(n_in, axis=1)
        n_std = np.std(n_in, axis=1)

        frame = np.linspace(0, self.n_steps, self.n_steps + 1)

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(n_run_plot):
            ax.plot(
                frame,
                (PE[0, :, j] + PE[1, :, j] + PE[3, :, j]) / 3,
                "--k",
                linewidth=0.25,
            )

        ax.plot(frame, (PE_mean[0, :] + PE_mean[1, :] + PE_mean[1, :]) / 3, "-r")
        ax.fill_between(
            frame,
            PE_mean[0, :] - PE_std[0, :],
            PE_mean[0, :] + PE_std[0, :],
            color="#DDDDDD",
        )
        ax.fill_between(
            frame,
            PE_mean[1, :] - PE_std[1, :],
            PE_mean[1, :] + PE_std[1, :],
            color="#DDDDDD",
        )
        ax.fill_between(
            frame,
            PE_mean[2, :] - PE_std[2, :],
            PE_mean[2, :] + PE_std[2, :],
            color="#DDDDDD",
        )

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$(x_o,y_o,z_o)$[m]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_translation_error.pdf")

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(n_run_plot):
            ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
        ax.plot(frame, PE_mean[3, :], "-r")
        ax.fill_between(
            frame,
            PE_mean[3, :] - PE_std[3, :],
            PE_mean[3, :] + PE_std[3, :],
            color="#DDDDDD",
        )

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$\psi$ [rad]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_orientation_error.pdf")

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(n_run_plot):
            ax.plot(frame, PE[4, :, j], "--k", linewidth=0.25)
        ax.plot(frame, PE_mean[4, :], "-r")
        ax.fill_between(
            frame,
            PE_mean[4, :] - PE_std[4, :],
            PE_mean[4, :] + PE_std[4, :],
            color="#DDDDDD",
        )

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$a$[m]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_sag_error.pdf")

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        l_n = PE.shape[0] - 5

        for i in range(l_n):
            k = 5 + i
            for j in range(n_run_plot):
                ax.plot(frame, PE[k, :, j], "--k", linewidth=0.25)
            ax.plot(frame, PE_mean[k, :], "-r")
            ax.fill_between(
                frame,
                PE_mean[k, :] - PE_std[k, :],
                PE_mean[k, :] + PE_std[k, :],
                color="#DDDDDD",
            )

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$\Delta$[m]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_internaloffsets_error.pdf")

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(n_run_plot):
            ax.plot(frame[1:], t[:, j], "--k", linewidth=0.25)
        ax.plot(frame[1:], t_mean, "-r")
        ax.fill_between(frame[1:], t_mean - t_std, t_mean + t_std, color="#DDDDDD")

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$\Delta t$[sec]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_solver_time.pdf")

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(n_run_plot):
            ax.plot(frame[1:], n_in[:, j], "--k", linewidth=0.25)
        ax.plot(frame[1:], n_mean[:], "-r")
        ax.fill_between(
            frame[1:], n_mean[:] - n_std[:], n_mean[:] + n_std[:], color="#DDDDDD"
        )

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$n_{in}[\%]$", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_nin.pdf")


###############################################################################
def ArrayModelEstimatorTest(
    save=True,
    plot=True,
    name="test",
    n_run=1,
    n_steps=10,
    # Model
    model=powerline.ArrayModel32(),
    p_hat=np.array([50, 50, 50, 1.0, 600, 50.0, 30.0, 50.0]),
    p_ub=np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0]),
    p_lb=np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0]),
    # Fake data Distribution param
    n_obs=20,
    n_out=0,
    x_min=-200,
    x_max=200,
    w_l=0.5,
    w_o=100,
    center=[0, 0, 0],
    partial_obs=False,
    # Solver param
    n_sea=2,
    var=np.array([10, 10, 10, 1.0, 200, 1.0, 1.0, 1.0]),
    Q=0.0 * np.diag([0.0002, 0.0002, 0.0002, 0.001, 0.0001, 0.002, 0.002, 0.002]),
    l=1.0,
    power=2.0,
    b=1000.0,
    method="x",
    n_s=100,
    x_min_s=-200,
    x_max_s=+200,
    use_grad=True,
):

    estimator = ArrayEstimator(model, p_hat)

    estimator.p_var = var
    estimator.n_search = n_sea
    estimator.use_grad = use_grad
    estimator.Q = Q
    estimator.p_lb = p_lb
    estimator.p_ub = p_ub
    estimator.method = method
    estimator.Q = Q
    estimator.b = b
    estimator.l = l
    estimator.power = power
    estimator.x_min = x_min_s
    estimator.x_max = x_max_s
    estimator.n_sample = n_s
    estimator.d_th = w_l * 5.0

    for j in range(n_run):

        print("Run no", j)

        # Random true line position
        p_true = np.random.uniform(p_lb, p_ub)

        if plot:
            plot_3d = powerline.EstimationPlot(
                p_true, p_hat, None, model.p2r_w, xmin=-200, xmax=200
            )

        if j == 0:
            e_plot = ErrorPlot(p_true, p_hat, n_steps, n_run)
        else:
            e_plot.init_new_run(p_true, p_hat)

        # Alway plot the 3d graph for the last run
        if j == (n_run - 1):
            plot = True
            plot_3d = powerline.EstimationPlot(p_true, p_hat, None, model.p2r_w)
            plot_3d.show = False

        for i in range(n_steps):

            # Generate fake noisy data
            pts = model.generate_test_data(
                p_true, n_obs, x_min, x_max, w_l, n_out, center, w_o, partial_obs
            )

            if plot:
                plot_3d.update_pts(pts)

            start_time = time.time()
            ##################################################################
            p_hat = estimator.solve_with_search(pts, p_hat)
            ##################################################################
            solve_time = time.time() - start_time

            if plot:
                plot_3d.update_estimation(p_hat)

            ##################################################################
            n_tot = pts.shape[1] - n_out
            pts_in = estimator.get_array_group(p_hat, pts)
            n_in = pts_in.shape[1] / n_tot * 100
            ##################################################################

            # print(pts.shape,pts_in.shape)

            e_plot.save_new_estimation(p_hat, solve_time, n_in)

        # Plot pts_in
        if plot:
            plot_3d.add_pts(pts_in)

    # Finalize figures
    if save:
        plot_3d.save(name=name)
    # e_plot.plot_error_all_run( save = save , name = ( name + 'All') )
    e_plot.plot_error_mean_std(save=save, name=name)

    return e_plot


###############################################################

if __name__ == "__main__":

    from catenary.analysis.dataset import load_dataset, SimulatedDataset

    datagen_params = {
        "name": "sim_222",
        "n_out": 10,
        "n_frames": 100,
        "n_obs": 10,
        "x_min": -10,
        "x_max": 10,
        "w_l": 0.2,
        "w_o": 10.0,
        "center": [0, 0, -25],
        "partial_obs": True,
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
        "seed": 0,
    }

    # Test parameters
    test_params = {
        "name": "test",
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
        "n_search": 1,
        "p_var": np.array([5.0, 5.0, 5.0, 1.0, 400.0, 2.0, 2.0, 2.0]),
        "filter_method": "corridor",  # No filter, as simulated data is already filtered
        # "filter_method": "clustering",  # No filter, as simulated data is already filtered
        "num_randomized_tests": 10,  # Number of tests to execute with randomized initial guess
        "stats_num_frames": 10,  # Number of last frames to use for statistics (experimental results have 100 frames)
        "method": "x",
        "n_sample": 201,
        "x_min": -200,
        "x_max": 200,
        "use_grad": True,
    }

    # Real data
    # dataset = load_dataset("ligne315kv_test1")
    # test_params["dataset"] = dataset

    # Simulated data
    test_params["dataset"] = datagen_params

    results, stats = evaluate(test_params)
    plot_results(test_params, results, save=True, n_run_plot=5)
    # animate_results(test_params, results)
    # animate_results2(test_params, results)
