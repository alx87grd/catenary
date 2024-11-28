import numpy as np
import matplotlib.pyplot as plt

from catenary import powerline

def rosbagEvaluation(bagname, param_powerline, estimator, points=(-100, 100, 200), filtered = True):
    """ Evaluation of the rosbag data """

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection='3d')
    line_pts, drone_pos = powerline.rosbag_to_array(bagname, '/filtered_cloud_points')
    velodyne_pts, _ = powerline.rosbag_to_array(bagname, '/velodyne_points')

    # Initial estimation
    pt_id = 0
    if filtered:
        pt = line_pts[pt_id]
    else:
        pt = velodyne_pts[pt_id]
    pts_hat = estimator.model.p2r_w(param_powerline, points[0], points[1], points[2])[1]
    print(param_powerline)
    if filtered:
        ax.scatter3D(pt[0], pt[1], pt[2], color='green', alpha=0.5)
    ax.scatter3D(drone_pos[pt_id][0], drone_pos[pt_id][1], drone_pos[pt_id][2], color='blue', s=50, marker='*')
    for i in range(pts_hat.shape[2]):
        ax.plot3D(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], '-k')

    ax.scatter3D(velodyne_pts[pt_id][0], velodyne_pts[pt_id][1], velodyne_pts[pt_id][2], color='red', alpha=1, s=1)
    # Set fixed scale
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([0, 50])
    plt.pause(2.001)
    
    # Loop over the frames
    for pt_id in range(len(line_pts)):

        if filtered:
            pt = line_pts[pt_id]
        else:
            pt = velodyne_pts[pt_id]

        param_powerline = estimator.solve_with_search(pt, param_powerline)
        pts_hat = estimator.model.p2r_w(param_powerline, points[0], points[1], points[2])[1]
        # pts_hat = model.p2r_w(param_powerline, 50, -50, 100)[1]
        print(param_powerline)
        ax.clear()
        if filtered:
            ax.scatter3D(pt[0], pt[1], pt[2], color='green', alpha=0.5)
        ax.scatter3D(drone_pos[pt_id][0], drone_pos[pt_id][1], drone_pos[pt_id][2], color='blue', s=50, marker='*')
        for i in range(pts_hat.shape[2]):
            ax.plot3D(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], '-k')

        ax.scatter3D(velodyne_pts[pt_id][0], velodyne_pts[pt_id][1], velodyne_pts[pt_id][2], color='red', alpha=1, s=1)

        # Set fixed scale
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_zlim([0, 50])
        
        plt.pause(0.001)


def test_baseline():

     # Test 1
    ############### ligne315kv_test1 ################
    bagname = 'ligne315kv_test1'

    # param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6., 7.8, 7.5 ])

    param_powerline = np.array([  -25.0,  40.0, 0.0, 1.0, 700, 6., 6.0, 6.0 ])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 1*np.diag([0.02, 0.02, 0.002, 0.01, 0.0001, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([ 200.,  200., 25.,  3.14, 1000., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  0., 0.0,  5., 5., 6., 6.])

    estimator.p_var[0:3] = 50.0
    estimator.p_var[3]   = 5.0
    estimator.p_var[4]   = 200.0
    estimator.p_var[5:]  = 2.0

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=True)

def no_regulation():

     # Test 1
    ############### ligne315kv_test1 ################
    bagname = 'ligne315kv_test1'

    # param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6., 7.8, 7.5 ])

    param_powerline = np.array([  -0.0,  0.0, 0.0, 2.0, 700, 6., 6.0, 6.0 ])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.0*np.diag([0.02, 0.02, 0.002, 0.01, 0.0001, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 4
    estimator.p_ub = np.array([ 200.,  200., 25.,  3.14, 1000., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  0., 0.0,  5., 5., 6., 6.])

    estimator.p_var[0:3] = 100.0
    estimator.p_var[3]   = 2.0
    estimator.p_var[4]   = 500.0
    estimator.p_var[5:]  = 4.0

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=True)


def test2_baseline():

    # Test 2
    ############### ligne315kv_test2 ################
    bagname = 'ligne120kv_test1'

    param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6., 7.8, 7.5 ])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

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

    rosbagEvaluation(bagname, param_powerline, estimator)


def test2_lidar_bad():

    # Test 2
    ############### ligne315kv_test2 ################
    bagname = 'ligne120kv_test1'

    param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6., 7.8, 7.5 ])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

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

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=False)

def test2_lidar_good():

    # Test 2
    ############### ligne315kv_test2 ################
    bagname = 'ligne120kv_test1'

    param_powerline = np.array([  -0.,  50., 15., 1.8, 500, 6., 7.8, 7.5 ])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.1*np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.b = 8.0
    estimator.power = 2.0
    estimator.n_search = 3
    estimator.p_ub = np.array([ 200.,  200., 15.,  1.9, 600., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  10., 1.5,  5., 5., 5., 5.])

    estimator.p_var[0:3] = 10.0
    estimator.p_var[3]   = 2.0
    estimator.p_var[4]   = 500.0
    estimator.p_var[5:]  = 5.5

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=False)


def test3():

    # Test 3
    ############### contournement_pylone ################

    bagname = 'ligne120kv_test2'

    param_powerline = np.array([  -0.,  50., 15., 1.8, 500, 6., 7.8, 7.5 ])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.1*np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.b = 8.0
    estimator.power = 2.0
    estimator.n_search = 3
    estimator.p_ub = np.array([ 200.,  200., 15.,  1.9, 600., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  10., 1.5,  5., 5., 5., 5.])

    estimator.p_var[0:3] = 10.0
    estimator.p_var[3]   = 2.0
    estimator.p_var[4]   = 500.0
    estimator.p_var[5:]  = 5.5

    rosbagEvaluation(bagname, param_powerline, estimator)


def test4():

    # Test 4
    ############### contournement_pylone ################

    bagname = 'ligne120kv_test4'

    param_powerline = np.array([  -0.,  50., 15., 1.8, 500, 6., 7.8, 7.5 ])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.1*np.diag([0.002, 0.002, 0.002, 0.01, 0.000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.b = 8.0
    estimator.power = 2.0
    estimator.n_search = 3
    estimator.p_ub = np.array([ 200.,  200., 15.,  1.9, 600., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  10., 1.5,  5., 5., 5., 5.])

    estimator.p_var[0:3] = 10.0
    estimator.p_var[3]   = 2.0
    estimator.p_var[4]   = 500.0
    estimator.p_var[5:]  = 5.5

    rosbagEvaluation(bagname, param_powerline, estimator)

        
def test_lidar_pts():

    #############
    ## Alex
    ###############

    # Direct velodyne points???
    ############### ligne315kv_test1 ################
    bagname = 'ligne315kv_test1'

    # param_powerline = np.array([  -28.,  60., 18., 2.3, 500, 6., 7.8, 7.5 ])

    param_powerline = np.array([  -25.,  50., 18., 2.3, 500, 6., 7.8, 7.5 ])


    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 0.001*np.diag([0.02, 0.02, 0.002, 0.01, 0.00000, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.b = 5.0
    estimator.power = 2.0
    estimator.n_search = 5
    estimator.p_ub = np.array([ 200.,  200., 25.,  2.8, 1000., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  10., 1.8,  5., 5., 6., 6.])

    estimator.p_var[0:3] = 50.0
    estimator.p_var[3]   = 5.0
    estimator.p_var[4]   = 200.0
    estimator.p_var[5:]  = 2.0

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=False)


def test5():

     # Test 1
    ############### ligne315kv_test1 ################
    bagname = 'ligne315kv_test1'

    # param_powerline = np.array([  -30.,  50., 11., 2.3, 500, 6., 7.8, 7.5 ])

    param_powerline = np.array([  -25.0,  40.0, 0.0, 1.0, 700, 6., 6.0, 6.0 ])

    model = powerline.ArrayModel222()
    estimator = powerline.ArrayEstimator(model, param_powerline)

    estimator.Q = 1*np.diag([0.02, 0.02, 0.002, 0.01, 0.0001, 0.02, 0.02, 0.02])
    estimator.l = 1.0
    estimator.power = 2.0
    estimator.n_search = 2
    estimator.p_ub = np.array([ 200.,  200., 25.,  3.14, 1000., 7., 9., 9.])
    estimator.p_lb = np.array([-200., -200.,  0., 0.0,  5., 5., 6., 6.])

    estimator.p_var[0:3] = 50.0
    estimator.p_var[3]   = 5.0
    estimator.p_var[4]   = 200.0
    estimator.p_var[5:]  = 2.0

    rosbagEvaluation(bagname, param_powerline, estimator, filtered=True)


if __name__ == "__main__":     
    """ MAIN TEST """

    test_baseline()
    # test_lidar_pts()
    # no_regulation()

    # test2_baseline()
    # test2_lidar_bad()
    # test2_lidar_good()


    # test3()
    # test4()
    # test5()

    
   








    










