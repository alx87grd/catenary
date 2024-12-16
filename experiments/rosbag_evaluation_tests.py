import numpy as np
import matplotlib.pyplot as plt

from catenary import powerline


def rosbagEvaluation(
    bagname, param_powerline, estimator, points=(-100, 100, 200), filtered=True
):
    """Evaluation of the rosbag data"""

    line_pts = np.load(("rosbag/" + bagname + "/filtered_cloud_points.npy"))
    drone_pos = np.load(("rosbag/" + bagname + "/drone_pose.npy"))
    velodyne_pts = np.load(("rosbag/" + bagname + "/velodyne_points.npy"))
    timestamps = np.load(("rosbag/" + bagname + "/timestamps.npy"), allow_pickle=True)

    images = []

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection="3d")

    for pt_id in range(len(line_pts)):
        pt = line_pts[pt_id]
        pt = pt[:, ~np.isnan(pt[0])]
        param_powerline = estimator.solve_with_search(pt, param_powerline)
        pts_hat = model.p2r_w(param_powerline, points[0], points[1], points[2])[1]
        ax.clear()
        if filtered:
            ax.scatter3D(pt[0], pt[1], pt[2], color="green", alpha=0.5)
        ax.scatter3D(
            drone_pos[pt_id][0],
            drone_pos[pt_id][1],
            drone_pos[pt_id][2],
            color="blue",
            s=50,
            marker="*",
        )
        for i in range(pts_hat.shape[2]):
            ax.plot3D(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], "-k")

        ax.scatter3D(
            velodyne_pts[pt_id][0],
            velodyne_pts[pt_id][1],
            velodyne_pts[pt_id][2],
            color="red",
            alpha=1,
            s=1,
        )

        # Set fixed scale
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_zlim([0, 50])

        # add timestamp to the plot
        ax.text2D(0.05, 0.95, str(timestamps[pt_id]), transform=ax.transAxes)

        plt.pause(0.05)

        save_gif = False
        if save_gif:
            if pt_id % 3 == 0:
                # Create a bytes buffer to save the plot
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)

                # Open the PNG image from the buffer and convert it to a NumPy array
                image = Image.open(buf)
                buf.seek(0)
                images.append(image)

    if save_gif:
        # save as a gif
        images[0].save(
            "figures/" + bagname + ".gif",
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=500,
            loop=0,
        )
        # Close the buffer
        buf.close()


if __name__ == "__main__":
    """MAIN TEST"""
    #################################################
    ############### ligne315kv_test1 ################
    #################################################
    bagname = "ligne120kv_test1"

    param_powerline = np.array([-1.0, 64.0, 11.0, 1.7, 500, 5.5, 6.0, 7.0])

    model = ArrayModel222()
    estimator = ArrayEstimator(model, param_powerline)

    estimator.Q = 1 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([200.0, 200.0, 25.0, 3.14, 500.0, 7.0, 9.0, 9.0])
    estimator.p_lb = np.array([-200.0, -200.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0])

    estimator.p_var[0:3] = 5.0
    estimator.p_var[3] = 5.0
    estimator.p_var[4] = 500.0
    estimator.p_var[5:] = 5.5

    # rosbagEvaluation(bagname, param_powerline, estimator)

    #################################################
    ############### ligne120kv_test2 ################
    #################################################
    bagname = "ligne120kv_test2"

    param_powerline = np.array([-1.0, 64.0, 11.0, 1.7, 500, 5.5, 6.0, 7.0])

    model = ArrayModel222()
    estimator = ArrayEstimator(model, param_powerline)

    estimator.Q = 1 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([200.0, 200.0, 25.0, 3.14, 500.0, 7.0, 9.0, 9.0])
    estimator.p_lb = np.array([-200.0, -200.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0])

    estimator.p_var[0:3] = 5.0
    estimator.p_var[3] = 5.0
    estimator.p_var[4] = 500.0
    estimator.p_var[5:] = 5.5

    # rosbagEvaluation(bagname, param_powerline, estimator)

    #################################################
    ############### ligne315kv_test1 ################
    #################################################
    bagname = "ligne315kv_test1"

    param_powerline = np.array([-30.0, 50.0, 11.0, 2.3, 500, 6.0, 7.8, 7.5])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 1 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([200.0, 200.0, 25.0, 3.14, 500.0, 7.0, 9.0, 9.0])
    estimator.p_lb = np.array([-200.0, -200.0, 0.0, 0.0, 5.0, 5.0, 6.0, 6.0])

    estimator.p_var[0:3] = 5.0
    estimator.p_var[3] = 5.0
    estimator.p_var[4] = 500.0
    estimator.p_var[5:] = 5.5

    rosbagEvaluation(bagname, param_powerline, estimator)

    #################################################
    ############### ligne315kv_test2 ################
    #################################################
    bagname = "ligne315kv_test2"

    param_powerline = np.array([-30.0, 50.0, 11.0, 2.3, 500, 6.0, 7.8, 7.5])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 1 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([200.0, 200.0, 25.0, 3.14, 500.0, 7.0, 9.0, 9.0])
    estimator.p_lb = np.array([-200.0, -200.0, 0.0, 0.0, 5.0, 5.0, 6.0, 6.0])

    estimator.p_var[0:3] = 5.0
    estimator.p_var[3] = 5.0
    estimator.p_var[4] = 500.0
    estimator.p_var[5:] = 5.5

    # rosbagEvaluation(bagname, param_powerline, estimator)

    #################################################
    ############### contournement_pylone ############
    #################################################
    bagname = "contournement_pylone"

    param_powerline = np.array([-0.0, 50.0, 15.0, 1.8, 500, 6.0, 7.8, 7.5])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.0001 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02])
    estimator.l = 1.0
    estimator.b = 8.0
    estimator.power = 2.0
    estimator.n_search = 3
    estimator.p_ub = np.array([200.0, 200.0, 15.0, 1.9, 600.0, 7.0, 9.0, 9.0])
    estimator.p_lb = np.array([-200.0, -200.0, 10.0, 1.5, 5.0, 5.0, 5.0, 5.0])

    estimator.p_var[0:3] = 10.0
    estimator.p_var[3] = 2.0
    estimator.p_var[4] = 500.0
    estimator.p_var[5:] = 5.5

    # rosbagEvaluation(bagname, param_powerline, estimator, points=(50, -50, 100))


def test4():

    #################################################
    ####### scenario_failsafe_long_approach #########
    #################################################
    bagname = "scenario_failsafe_long_approach"

    # param_powerline = np.array([  -28.,  60., 18., 2.3, 500, 6., 7.8, 7.5 ])

    param_powerline = np.array([-25.0, 50.0, 18.0, 2.3, 500, 6.0, 7.8, 7.5])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.001 * np.diag([0.02, 0.02, 0.002, 0.01, 0.00000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.b = 5.0
    estimator.power = 2.0
    estimator.n_search = 5
    estimator.p_ub = np.array([200.0, 200.0, 25.0, 2.8, 1000.0, 7.0, 9.0, 9.0])
    estimator.p_lb = np.array([-200.0, -200.0, 10.0, 1.8, 5.0, 5.0, 6.0, 6.0])

    estimator.p_var[0:3] = 50.0
    estimator.p_var[3] = 5.0
    estimator.p_var[4] = 200.0
    estimator.p_var[5:] = 2.0

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=False)


def test5():

    # Test 1
    ############### ligne315kv_test1 ################
    bagname = "ligne315kv_test1"

    # param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6., 7.8, 7.5 ])

    param_powerline = np.array([-25.0, 40.0, 0.0, 1.0, 700, 6.0, 6.0, 6.0])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 1 * np.diag([0.02, 0.02, 0.002, 0.01, 0.0001, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([200.0, 200.0, 25.0, 3.14, 1000.0, 7.0, 9.0, 9.0])
    estimator.p_lb = np.array([-200.0, -200.0, 0.0, 0.0, 5.0, 5.0, 6.0, 6.0])

    estimator.p_var[0:3] = 50.0
    estimator.p_var[3] = 5.0
    estimator.p_var[4] = 200.0
    estimator.p_var[5:] = 2.0

    rosbagEvaluation(bagname, param_powerline, estimator, points=(100, -100, 100))
