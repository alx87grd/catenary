import os
import numpy as np
from datetime import datetime

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from tqdm import tqdm
from pathlib import Path


def rosbag_to_array(bag_file, topic):

    pts = []
    drone_pos = []
    timestamps = []

    velodyne_rot = np.array([0.0, 0.0, 0.0, 1.0])
    velodyne_translation = np.array([0.0, 0.0, 0.0])
    base_link_rot = np.array([0.0, 0.0, 0.0, 1.0])
    base_link_translation = np.array([0.0, 0.0, 0.0])
    offset_base_link_rot = np.array([0.0, 0.0, 0.0, 1.0])
    offset_base_link_translation = np.array([0.0, 0.0, 0.0])

    bag_path = os.path.join(
        os.path.dirname(__file__), "..", "rosbag", bag_file + ".bag"
    )

    with Reader(bag_file) as reader:
        for connection, timestamp, rawdata in tqdm(reader.messages()):
            if connection.topic == "/tf":
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                for transform in msg.transforms:
                    # print(transform.child_frame_id)
                    # if transform.child_frame_id not in child_frames:
                    #     print(transform.header)
                    #     print(transform.child_frame_id)
                    #     child_frames.append(transform.child_frame_id)
                    pass
                    if transform.child_frame_id == "base_link":
                        base_link_rot = quaternion_rotation_matrix(
                            np.array(
                                [
                                    transform.transform.rotation.w,
                                    transform.transform.rotation.x,
                                    transform.transform.rotation.y,
                                    transform.transform.rotation.z,
                                ]
                            )
                        )
                        base_link_translation = np.array(
                            [
                                transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z,
                            ]
                        )

                    if transform.child_frame_id == "offset_base_link":
                        drone_pos.append(
                            [
                                transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z,
                            ]
                        )
                        offset_base_link_rot = quaternion_rotation_matrix(
                            np.array(
                                [
                                    transform.transform.rotation.w,
                                    transform.transform.rotation.x,
                                    transform.transform.rotation.y,
                                    transform.transform.rotation.z,
                                ]
                            )
                        )
                        offset_base_link_translation = np.array(
                            [
                                transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z,
                            ]
                        )

                    if transform.child_frame_id == "velodyne":
                        velodyne_rot = quaternion_rotation_matrix(
                            np.array(
                                [
                                    transform.transform.rotation.w,
                                    transform.transform.rotation.x,
                                    transform.transform.rotation.y,
                                    transform.transform.rotation.z,
                                ]
                            )
                        )
                        velodyne_translation = np.array(
                            [
                                transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z,
                            ]
                        )

            if connection.topic == topic:
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                # print(msg.header.frame_id)

                # reshape the data to a 2D array of shape (point_step, width)
                data = msg.data.reshape(msg.width, msg.point_step)[:, 0:12]

                # Ensure the array is C-contiguous
                data = np.ascontiguousarray(data)

                # reshape the data from 12 bytes to 3 floats
                data = data.view(np.float32).reshape(msg.width, 3)

                if msg.header.frame_id == "velodyne":
                    data = data @ velodyne_rot.T + velodyne_translation
                    data = data @ base_link_rot.T + base_link_translation
                    data = data @ offset_base_link_rot.T + offset_base_link_translation

                pts.append(data.T)

                # convert timestamp to date format
                timestamp = datetime.utcfromtimestamp(timestamp / 1e9)
                timestamps.append(timestamp)

    return pts, np.array(drone_pos), timestamps


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]  # w
    q1 = Q[1]  # #
    q2 = Q[2]  # y
    q3 = Q[3]  # z

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return rot_matrix


def list_to_np_array(lst):
    length = max([len(lst[i][0]) for i in range(len(lst))])

    for i in range(len(lst)):

        lst_x = np.array(
            [np.pad(lst[i][0], (0, length - len(lst[i][0])), constant_values=np.nan)]
        )
        lst_y = np.array(
            [np.pad(lst[i][1], (0, length - len(lst[i][0])), constant_values=np.nan)]
        )
        lst_z = np.array(
            [np.pad(lst[i][2], (0, length - len(lst[i][0])), constant_values=np.nan)]
        )

        lst[i] = np.array([lst_x, lst_y, lst_z]).squeeze()

    return np.array(lst)


########################################################################################
######################################## main ##########################################
########################################################################################

if __name__ == "__main__":

    path = "/Users/agirard/data/catenary/"
    # iterate over all the rosbag files in the rosbag folder

    # Create a typestore and get the string class.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # tf2_msg = Path('rosbag/TFMessage.msg').read_text()
    tf2_msg = Path(os.path.join(path, "TFMessage.msg")).read_text()
    typetf2 = get_types_from_msg(tf2_msg, "tf2_msgs/msg/TFMessage")

    typestore.register(typetf2)

    for bagname in os.listdir(path):
        if bagname.endswith(".bag"):
            bagname = bagname[:-4]
            bag_file = path + bagname + ".bag"
            if not os.path.exists("rosbag/" + bagname):
                print(bagname)

                line_pts, drone_pos, timestamps = rosbag_to_array(
                    bag_file, "/filtered_cloud_points"
                )
                velodyne_pts, _, _ = rosbag_to_array(bag_file, "/velodyne_points")

                # remove the first 500 points each cloud
                # line_pts = line_pts[500:]
                # velodyne_pts = velodyne_pts[500:]
                # drone_pos = drone_pos[500:]
                # timestamps = timestamps[500:]

                line_pts = list_to_np_array(line_pts)
                velodyne_pts = list_to_np_array(velodyne_pts)
                drone_pos = np.array(drone_pos)
                timestamps = np.array(timestamps)

                print(line_pts.shape)
                print(velodyne_pts.shape)
                print(drone_pos.shape)
                print(timestamps.shape)

                os.mkdir(path + bagname)

                np.save((path + bagname + "/filtered_cloud_points"), line_pts)
                np.save((path + bagname + "/drone_pose"), drone_pos)
                np.save((path + bagname + "/velodyne_points"), velodyne_pts)
                np.save((path + bagname + "/timestamps"), timestamps)
