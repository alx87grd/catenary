import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


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

        ax.clear()

        print(pt_id)
        print(len(line_pts))

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


if __name__ == "__main__":
    """MAIN TEST"""

    ## Bag viewing and description

    # Landing on model222, always aligned with the cable, energized lines
    # rosbagView("315kV_manual_1")

    # Landing on cable 1 of model222, always aligned with the cable, energized lines
    # rosbagView("315kV_manual_2")

    # Muliplt lines crossing near a pylon ???
    # rosbagView("approach")

    #  nice 180 close to lines nd pylon, but multipl close lines confusing???
    # rosbagView("approach2")

    # many close line again..
    # rosbagView("failsafe_long_approach")

    # Nice mission with rotation, lots of lines however, config = dual cable??
    # rosbagView("full_mission_rtl_bug")
    # rosbagView(
    #     "full_mission_rtl_bug", start=500, step=1
    # )  # To test the estimator !!!!!!!!!
    rosbagView(
        "full_mission_rtl_bug", start=250, step=1
    )  # To test the estimator !!!!!!!!!

    # again similar to full_mission_rtl_bug
    # rosbagView("landing_on_cable")

    # Landing on cable 1 of model2221, always aligned with the cable
    # rosbagView("ligne120kv_test1")

    # Landing on cable 1 of model2221, always aligned with the cable
    # rosbagView("ligne120kv_test2")

    # Landing on cable 3 of model2221, always aligned with the cable
    # rosbagView("ligne120kv_test3")

    # Landing on cable 4 of model2221, always aligned with the cable
    # rosbagView("ligne120kv_test4")

    # Landing on cable 1 of model222, always aligned with the cable
    # rosbagView("ligne315kv_test1")

    # again similar to full_mission_rtl_bug
    # rosbagView("mission_base_contournement")

    # Close to a pylon and rotating 180 degrees but not sure about how many lines?????
    # rosbagView("pylon")

    # Landing on cable 2 of model222, always aligned with the cable
    # rosbagView("test_autonome1")

    # On ground test
    # rosbagView("test_de_poussiere_ecole_des_monteurs")
