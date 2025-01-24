from catenary import powerline
from catenary.filter import remove_ground_plane, filter_cable_points
from experiments.dataset import load_dataset
import matplotlib.pyplot as plt
import numpy as np

def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', print_end = "\r"):
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
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = print_end)
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
    test_name = params['name']

    # Load dataset
    dataset = load_dataset(params['dataset'])

    # Initialize model
    model = powerline.create_array_model(params['model'])

    # Initialize initial guess
    p_0 = params['p_0']

    # Initialize estimator
    estimator = powerline.ArrayEstimator(model, p_0)

    # Regulation weight matrix
    estimator.Q = params['Q']

    # Parameters of lorentzian cost shaping function
    estimator.l = params['l']
    estimator.power = params['power']
    estimator.b = params['b']

    # Parameters upper and lower bounds
    estimator.p_ub = params['p_ub']
    estimator.p_lb = params['p_lb']

    # Search parameters

    # Number of search
    estimator.n_search = params['n_search']

    # Parameters std deviation for searching
    estimator.p_var = params['p_var']

    # Filter method
    filter_method = params['filter_method']

    # Results dictionary
    results = {}

    # Initialize p_hat key with an numpy array of dataset.frame_count() elements
    results['p_hat'] = np.zeros((dataset.frame_count(), p_0.shape[0]))
    results['p_err'] = np.zeros((dataset.frame_count(), p_0.shape[0]))
    results['p_err_mean'] = np.zeros((p_0.shape[0]))
    results['p_err_std'] = np.zeros((p_0.shape[0]))
    results['num_points_before_filter'] = np.zeros((dataset.frame_count()))
    results['num_points_after_filter'] = np.zeros((dataset.frame_count()))
    results['num_points_mean_before_filter'] = 0
    results['num_points_mean_after_filter'] = 0
    results['num_points_std_before_filter'] = 0
    results['num_points_std_after_filter'] = 0

    for pt_id in range(dataset.frame_count()):

        # Print progress
        print_progress_bar(pt_id, dataset.frame_count()-1, prefix = f'TEST: {test_name}', suffix = 'Complete', length = 50)

        # Number of points in the lidar frame before filtering
        n_points_before_filter = dataset.lidar_points(pt_id).shape[1]

        # Filter lidar points
        if filter_method == 'none':
            points = dataset.lidar_points(pt_id)
        elif filter_method == 'ground_filter':
            points = remove_ground_plane(dataset.lidar_points(pt_id).T, distance_threshold=5.0, ransac_n=5).T
        elif filter_method == 'clustering':
            points = filter_cable_points(dataset.lidar_points(pt_id).T).T
        elif filter_method == 'corridor':
            points = dataset.filtered_lidar_points(pt_id)
        else:
            raise ValueError(f"Filter method {filter_method} not recognized.")

        # Number of points in the lidar frame after filter
        n_points_after_filter = points.shape[1]

        # Execute estimator
        if pt_id == 0:
            p_hat = p_0

        p_hat = estimator.solve_with_search(points, p_hat)

        # Store result
        results['p_hat'][pt_id] = p_hat
        results['p_err'][pt_id] = dataset.ground_thruth_params(pt_id) - p_hat
        results['num_points_before_filter'][pt_id] = n_points_before_filter
        results['num_points_after_filter'][pt_id] = n_points_after_filter

    # Compute stats on results
    results['p_err_mean'] = np.mean(results['p_err'], axis=0)
    results['p_err_std'] = np.std(results['p_err'], axis=0)
    results['num_points_mean_before_filter'] = np.mean(results['num_points_before_filter'])
    results['num_points_mean_after_filter'] = np.mean(results['num_points_after_filter'])
    results['num_points_std_before_filter'] = np.std(results['num_points_before_filter'])
    results['num_points_std_after_filter'] = np.std(results['num_points_after_filter'])

    return results

def plot_results(params, results):
    """
    Plot test results.

    Parameters
    ----------
    params: dict
        Parameters dictionary.
    results: dict
        Results dictionary.
    """

    # Figure 1 : Plot estimated power line and ground thruth as animation
    fig1 = plt.figure(1, figsize=(14, 10))
    ax1 = plt.axes(projection="3d")

    dataset = load_dataset(params['dataset'])
    model = powerline.create_array_model(params['model'])

    for idx in range(dataset.frame_count()):
        ax1.clear()

        p_hat = results['p_hat'][idx]

        p_ground_thruth = dataset.ground_thruth_params(idx)

        # Compute projected power line points using estimated model
        pts_hat = model.p2r_w(p_hat, x_min=-100, x_max=100, n=200)[1]

        # Compute ground thruth line points
        pts_ground_thruth = model.p2r_w(p_ground_thruth, x_min=-100, x_max=100, n=200)[1]

        ax1.scatter3D(
            dataset.lidar_points(idx)[0],
            dataset.lidar_points(idx)[1],
            dataset.lidar_points(idx)[2],
            color="red",
            alpha=1,
            s=1,
        )

        for i in range(pts_hat.shape[2]):
            ax1.plot3D(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], "-k")

        for i in range(pts_ground_thruth.shape[2]):
            ax1.plot3D(pts_ground_thruth[0, :, i], pts_ground_thruth[1, :, i], pts_ground_thruth[2, :, i], "-g")

        # Set fixed scale
        ax1.set_xlim([-50, 50])
        ax1.set_ylim([-50, 50])
        ax1.set_zlim([0, 50])

        plt.pause(0.001)
