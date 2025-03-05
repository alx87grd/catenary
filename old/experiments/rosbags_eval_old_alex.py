import numpy as np
import matplotlib.pyplot as plt

from catenary import powerline
from catenary import filter


def rosbagEvaluation(
    bagname,
    param_powerline,
    estimator,
    points=(-100, 100, 200),
    filtered=True,
    path="/Users/agirard/data/catenary/",
):
    """Evaluation of the rosbag data"""

    bagfolder = path + bagname

    line_pts = np.load((bagfolder + "/filtered_cloud_points.npy"))
    drone_pos = np.load((bagfolder + "/drone_pose.npy"))
    velodyne_pts = np.load((bagfolder + "/velodyne_points.npy"))
    timestamps = np.load((bagfolder + "/timestamps.npy"), allow_pickle=True)

    images = []

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection="3d")

    # Initial estimation
    pt_id = 0
    if filtered:
        # pt = filter.filter_cable_points(velodyne_pts[pt_id].T).T
        pt = line_pts[pt_id]
    else:
        pt = velodyne_pts[pt_id]
    pts_hat = estimator.model.p2r_w(param_powerline, points[0], points[1], points[2])[1]
    print(param_powerline)
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

    # Projected powerline points
    for i in range(pts_hat.shape[2]):
        ax.plot3D(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], "-k")

    # Raw velodyne points
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
    plt.pause(2.001)

    for pt_id in range(len(line_pts)):

        if filtered:
            # pt = filter.filter_cable_points(velodyne_pts[pt_id].T).T
            pt = line_pts[pt_id]
        else:
            pt = velodyne_pts[pt_id]

        pt = pt[:, ~np.isnan(pt[0])]

        param_powerline = estimator.solve_with_search(pt, param_powerline)
        pts_hat = estimator.model.p2r_w(
            param_powerline, points[0], points[1], points[2]
        )[1]
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

        plt.pause(0.001)

        print(param_powerline)


def rosbagView(
    bagname,
    path="/Users/agirard/data/catenary/",
    start=0,
    # end=100,
    step=10,
):
    """Evaluation of the rosbag data"""

    bagfolder = path + bagname

    line_pts = np.load((bagfolder + "/filtered_cloud_points.npy"))
    drone_pos = np.load((bagfolder + "/drone_pose.npy"))
    velodyne_pts = np.load((bagfolder + "/velodyne_points.npy"))
    timestamps = np.load((bagfolder + "/timestamps.npy"), allow_pickle=True)

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection="3d")

    for pt_id in range(start, len(line_pts), step):

        print(pt_id)
        print(len(line_pts))

        pt = filter.filter_cable_points(velodyne_pts[pt_id].T).T
        ax.clear()
        ax.scatter3D(pt[0], pt[1], pt[2], color="green", alpha=0.5)

        ax.scatter3D(
            drone_pos[pt_id][0],
            drone_pos[pt_id][1],
            drone_pos[pt_id][2],
            color="blue",
            s=50,
            marker="*",
        )

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

        plt.pause(0.001)


####################################################################################
# TESTS
####################################################################################


def test_baseline():

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

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=True)


def no_regulation():

    # Test 1
    ############### ligne315kv_test1 ################
    bagname = "ligne315kv_test1"

    # param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6., 7.8, 7.5 ])

    param_powerline = np.array([-0.0, 0.0, 0.0, 2.0, 700, 6.0, 6.0, 6.0])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.0 * np.diag([0.02, 0.02, 0.002, 0.01, 0.0001, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 4
    estimator.p_ub = np.array([200.0, 200.0, 25.0, 3.14, 1000.0, 7.0, 9.0, 9.0])
    estimator.p_lb = np.array([-200.0, -200.0, 0.0, 0.0, 5.0, 5.0, 6.0, 6.0])

    estimator.p_var[0:3] = 100.0
    estimator.p_var[3] = 2.0
    estimator.p_var[4] = 500.0
    estimator.p_var[5:] = 4.0

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=True)


def test2_baseline():

    # Test 2
    ############### ligne315kv_test2 ################
    bagname = "ligne120kv_test1"

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


def test2_lidar_bad():

    # Test 2
    ############### ligne315kv_test2 ################
    bagname = "ligne120kv_test1"

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

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=False)


def test2_lidar_good():

    # Test 2
    ############### ligne315kv_test2 ################
    bagname = "ligne120kv_test1"

    param_powerline = np.array([-0.0, 50.0, 15.0, 1.8, 500, 6.0, 7.8, 7.5])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.1 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
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

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=False)


def test2():

    # Test 3
    ############### contournement_pylone ################

    bagname = "ligne120kv_test2"

    param_powerline = np.array([-0.0, 50.0, 15.0, 1.8, 500, 6.0, 7.8, 7.5])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.1 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
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

    rosbagEvaluation(bagname, param_powerline, estimator)


def test3():

    # Test 3
    ############### contournement_pylone ################

    bagname = "ligne120kv_test3"

    param_powerline = np.array([-0.0, 50.0, 15.0, 1.8, 500, 6.0, 7.8, 7.5])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.1 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
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

    rosbagEvaluation(bagname, param_powerline, estimator)


def test4():

    # Test 4
    ############### contournement_pylone ################

    bagname = "ligne120kv_test4"

    param_powerline = np.array([-0.0, 50.0, 15.0, 1.8, 500, 6.0, 7.8, 7.5])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.1 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
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

    rosbagEvaluation(bagname, param_powerline, estimator)


def approach():

    # Test 4
    ############### contournement_pylone ################

    bagname = "approach"

    param_powerline = np.array([0.0, 0.0, 15.0, 2.5, 2000, 6.0, 4, 6])

    model = powerline.ArrayModel32()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    # 10,7,8,12,2.3,6000,6,3.5,6

    estimator.Q = 1.5 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.b = 8.0
    estimator.power = 2.0
    estimator.n_search = 3
    estimator.p_ub = np.array([200.0, 200.0, 20.0, 2.5, 10000.0, 8.0, 5.0, 8.0])
    estimator.p_lb = np.array([-200.0, -200.0, 10.0, 2.1, 500.0, 4.0, 2.0, 4.0])

    estimator.p_var[0:3] = 10.0
    estimator.p_var[3] = 2.0
    estimator.p_var[4] = 500.0
    estimator.p_var[5:] = 5.5

    rosbagEvaluation(bagname, param_powerline, estimator)


def pylon():

    # Test 4
    ############### contournement_pylone ################

    bagname = "pylon"

    param_powerline = np.array([0.0, 0.0, 15.0, 2.5, 2000, 6.0, 4, 6])

    model = powerline.ArrayModel32()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    # 10,7,8,12,2.3,6000,6,3.5,6

    estimator.Q = 1.5 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.b = 8.0
    estimator.power = 2.0
    estimator.n_search = 3
    estimator.p_ub = np.array([200.0, 200.0, 20.0, 2.5, 10000.0, 8.0, 5.0, 8.0])
    estimator.p_lb = np.array([-200.0, -200.0, 10.0, 2.1, 500.0, 4.0, 2.0, 4.0])

    estimator.p_var[0:3] = 10.0
    estimator.p_var[3] = 2.0
    estimator.p_var[4] = 500.0
    estimator.p_var[5:] = 5.5

    rosbagEvaluation(bagname, param_powerline, estimator)


def manual_315():

    # Test 4
    ############### contournement_pylone ################

    bagname = "315kV_manual_2"

    param_powerline = np.array([0.0, 0.0, 15.0, 2.5, 2000, 6.0, 4, 6])

    model = powerline.ArrayModel32()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    # 10,7,8,12,2.3,6000,6,3.5,6

    estimator.Q = 1.5 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.b = 8.0
    estimator.power = 2.0
    estimator.n_search = 3
    estimator.p_ub = np.array([200.0, 200.0, 20.0, 2.5, 10000.0, 8.0, 5.0, 8.0])
    estimator.p_lb = np.array([-200.0, -200.0, 10.0, 2.1, 500.0, 4.0, 2.0, 4.0])

    estimator.p_var[0:3] = 10.0
    estimator.p_var[3] = 2.0
    estimator.p_var[4] = 500.0
    estimator.p_var[5:] = 5.5

    rosbagEvaluation(bagname, param_powerline, estimator)


def full_mission_rtl_bug():

    # Test 4
    ############### contournement_pylone ################

    bagname = "full_mission_rtl_bug"

    param_powerline = np.array([20.0, 20.0, 15.0, 2.2, 2000, 6.0])

    model = powerline.ArrayModel2()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    # 10,7,8,12,2.3,6000,6,3.5,6

    estimator.Q = 5.5 * np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02])
    estimator.l = 1.0
    estimator.b = 8.0
    estimator.power = 2.0
    estimator.n_search = 3
    estimator.p_ub = np.array([200.0, 200.0, 15.0, 2.5, 10000.0, 8.0])
    estimator.p_lb = np.array([-200.0, -200.0, 10.0, 2.1, 2000.0, 4.0])

    estimator.p_var[0:3] = 10.0
    estimator.p_var[3] = 2.0
    estimator.p_var[4] = 500.0
    estimator.p_var[5:] = 5.5

    rosbagEvaluation(bagname, param_powerline, estimator)


if __name__ == "__main__":
    """MAIN TEST"""

    # test_baseline()

    # no_regulation()

    # test2_baseline()
    # test2_lidar_bad()
    # test2_lidar_good()

    # test2()
    # test3()
    # test4()

    # approach()

    # pylon()

    # manual_315()
    full_mission_rtl_bug()
