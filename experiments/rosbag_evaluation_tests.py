import numpy as np
import matplotlib.pyplot as plt

from catenary.powerline import *

def rosbagEvaluation(bagname, param_powerline, estimator, save=False, points=(0, -100, 100)):
    """ Evaluation of the rosbag data """


    line_pts = np.load(('rosbag/' + bagname + '/filtered_cloud_points.npy'))
    drone_pos = np.load(('rosbag/' + bagname + '/drone_pose.npy'))
    velodyne_pts = np.load(('rosbag/' + bagname + '/velodyne_points.npy'))
    timestamps = np.load(('rosbag/' + bagname + '/timestamps.npy'), allow_pickle=True)


    images = []

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection='3d')

    for pt_id in range(len(line_pts)):
        pt = line_pts[pt_id]
        pt = pt[:, ~np.isnan(pt[0])]
        param_powerline = estimator.solve_with_search(pt, param_powerline)
        pts_hat = model.p2r_w(param_powerline, points[0], points[1], points[2])[1]
        ax.clear()
        ax.scatter3D(velodyne_pts[pt_id][0], velodyne_pts[pt_id][1], velodyne_pts[pt_id][2], color='red', alpha=1, s=1)
        ax.scatter3D(pt[0], pt[1], pt[2], color='green', alpha=0.5)
        ax.scatter3D(drone_pos[pt_id][0], drone_pos[pt_id][1], drone_pos[pt_id][2], color='blue', s=50, marker='*')
        for i in range(pts_hat.shape[2]):
            ax.plot3D(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], '-k')
        
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
                plt.savefig(buf, format='png')
                buf.seek(0)

                # Open the PNG image from the buffer and convert it to a NumPy array
                image = Image.open(buf)
                buf.seek(0)
                images.append(image)

    if save_gif:
        # save as a gif   
        images[0].save('figures/' + bagname + '.gif', save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)
        # Close the buffer
        buf.close()
        



if __name__ == "__main__":     
    """ MAIN TEST """
    #################################################
    ############### ligne315kv_test1 ################
    #################################################
    bagname = 'ligne120kv_test1'

    param_powerline = np.array([  -1.,  64., 11., 1.7, 500, 5.5, 6., 7. ])

    model = ArrayModel222()
    estimator = ArrayEstimator(model, param_powerline)

    estimator.Q = 1*np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([ 200.,  200., 25.,  3.14, 500., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  0., 0.0,  5., 5., 5., 5.])

    estimator.p_var[0:3] = 5.0
    estimator.p_var[3]   = 5.0
    estimator.p_var[4]   = 500.0
    estimator.p_var[5:]  = 5.5

    # rosbagEvaluation(bagname, param_powerline, estimator)

    #################################################
    ############### ligne120kv_test2 ################
    #################################################
    bagname = 'ligne120kv_test2'

    param_powerline = np.array([  -1.,  64., 11., 1.7, 500, 5.5, 6., 7. ])

    model = ArrayModel222()
    estimator = ArrayEstimator(model, param_powerline)

    estimator.Q = 1*np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([ 200.,  200., 25.,  3.14, 500., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  0., 0.0,  5., 5., 5., 5.])

    estimator.p_var[0:3] = 5.0
    estimator.p_var[3]   = 5.0
    estimator.p_var[4]   = 500.0
    estimator.p_var[5:]  = 5.5

    # rosbagEvaluation(bagname, param_powerline, estimator)

    #################################################
    ############### ligne315kv_test1 ################
    #################################################
    bagname = 'ligne315kv_test1'

    param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6., 7.8, 7.5 ])

    model = ArrayModel222()
    estimator = ArrayEstimator(model, param_powerline)

    estimator.Q = 1*np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([ 200.,  200., 25.,  3.14, 500., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  0., 0.0,  5., 5., 6., 6.])

    estimator.p_var[0:3] = 5.0
    estimator.p_var[3]   = 5.0
    estimator.p_var[4]   = 500.0
    estimator.p_var[5:]  = 5.5

    # rosbagEvaluation(bagname, param_powerline, estimator)


    #################################################
    ############### ligne315kv_test2 ################
    #################################################
    bagname = 'ligne315kv_test2'

    param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6., 7.8, 7.5 ])

    model = ArrayModel222()
    estimator = ArrayEstimator(model, param_powerline)

    estimator.Q = 1*np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([ 200.,  200., 25.,  3.14, 500., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  0., 0.0,  5., 5., 6., 6.])

    estimator.p_var[0:3] = 5.0
    estimator.p_var[3]   = 5.0
    estimator.p_var[4]   = 500.0
    estimator.p_var[5:]  = 5.5

    # rosbagEvaluation(bagname, param_powerline, estimator)


    #################################################
    ############### contournement_pylone ############
    #################################################
    bagname = 'contournement_pylone'

    param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6. ])

    model = ArrayModel()
    estimator = ArrayEstimator(model, param_powerline)

    estimator.Q = 0.0001*np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([ 200.,  200., 25.,  3.14, 500., 7.])
    estimator.p_lb = np.array([-200., -200.,  0., 0.0,  5., 4.])

    estimator.p_var[0:3] = 5.0
    estimator.p_var[3]   = 5.0
    estimator.p_var[4]   = 500.0
    estimator.p_var[5:]  = 5.5

    # rosbagEvaluation(bagname, param_powerline, estimator, points=(50, -50, 100))


    #################################################
    ####### scenario_failsafe_long_approach #########
    #################################################
    bagname = 'scenario_failsafe_long_approach'

    param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6. ])

    model = ArrayModel()
    estimator = ArrayEstimator(model, param_powerline)

    estimator.Q = 0.1*np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([ 200.,  200., 25.,  3.14, 500., 7.])
    estimator.p_lb = np.array([-200., -200.,  0., 0.0,  5., 4.])

    estimator.p_var[0:3] = 5.0
    estimator.p_var[3]   = 5.0
    estimator.p_var[4]   = 500.0
    estimator.p_var[5:]  = 5.5

    rosbagEvaluation(bagname, param_powerline, estimator, points=(100, -100, 100))








