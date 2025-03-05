#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:15:46 2023

@author: alex
"""


import numpy as np

from catenary import powerline


###############################################################################
def GlobalConvergenceTest(n_run=5, plot=False, save=True):

    # Baseline:
    save = save
    plot = plot
    name = "GlobalConvergence"
    n_run = n_run
    n_steps = 100
    model = powerline.ArrayModel32()
    p_hat = np.array([50, 50, 50, 1.0, 300, 50.0, 30.0, 50.0])
    p_ub = np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0])
    p_lb = np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0])
    # Fake data Distribution param
    n_obs = 10
    n_out = 20
    x_min = -200
    x_max = 200
    w_l = 0.5
    w_o = 50.0
    center = [0, 0, -200]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 5 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
    )


###############################################################################
def GlobalConvergenceBagTest(n_run=5, plot=False, save=True):

    # Baseline:
    save = save
    plot = plot
    rosbag = True
    rosfile = "ligne315kv_test2"
    name = "GlobalConvergence"
    n_run = n_run
    n_steps = 50
    model = powerline.ArrayModel222()
    p_hat = np.array([-30.0, 50.0, 11.0, 2.3, 500, 6.0, 7.8, 7.5])
    p_ub = np.array([100.0, 100.0, 15.0, 3.14, 1000.0, 7.0, 9.0, 9.0])
    p_lb = np.array([-100.0, -100.0, 5.0, 1.5, 20.0, 5.0, 6.0, 6.0])

    # Fake data Distribution param
    n_obs = 10
    n_out = 20
    x_min = -200
    x_max = 200
    w_l = 0.5
    w_o = 50.0
    center = [0, 0, -200]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 50 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
        rosbag,
        rosfile,
    )


###############################################################################
def PartialObsTest(n_run=5, plot=False, save=True):

    # Baseline:
    save = save
    plot = plot
    name = "PartialObs"
    n_run = n_run
    n_steps = 200
    model = powerline.ArrayModel32()
    p_hat = np.array([50, 50, 50, 1.0, 300, 50.0, 30.0, 50.0])
    p_ub = np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0])
    p_lb = np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0])
    # Fake data Distribution param
    n_obs = 16
    n_out = 5
    x_min = -100
    x_max = -70
    w_l = 0.5
    w_o = 10.0
    center = [-50, -50, -50]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 5 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
    )


###############################################################################
def RegulationVsNoRegulation(n_run=5, plot=False, save=True):

    # Baseline:
    save = save
    plot = plot
    name = "PartialObs"
    n_run = n_run
    n_steps = 100
    model = powerline.ArrayModel32()
    p_hat = np.array([50, 50, 50, 1.0, 300, 50.0, 30.0, 50.0])
    p_ub = np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0])
    p_lb = np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0])
    # Fake data Distribution param
    n_obs = 16
    n_out = 5
    x_min = -100
    x_max = -70
    w_l = 0.5
    w_o = 10.0
    center = [-50, -50, -50]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 5 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    name = "WithoutRegulation"
    Q = 0.00 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
    )


###############################################################################
def MethodTests(n_run=5, plot=False, save=True):

    # Baseline:
    save = save
    plot = plot
    name = "GlobalConvergence"
    n_run = n_run
    n_steps = 100
    model = powerline.ArrayModel32()
    p_hat = np.array([50, 50, 50, 1.0, 300, 50.0, 30.0, 50.0])
    p_ub = np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0])
    p_lb = np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0])
    # Fake data Distribution param
    n_obs = 10
    n_out = 20
    x_min = -200
    x_max = 200
    w_l = 0.5
    w_o = 50.0
    center = [0, 0, -200]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 5 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    # Baseline:
    name = "MethodSample"
    method = "sample"

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
    )


###############################################################################
def GradientTests(n_run=5, plot=False, save=True):

    # Baseline:
    save = save
    plot = plot
    name = "GlobalConvergence"
    n_run = n_run
    n_steps = 100
    model = powerline.ArrayModel32()
    p_hat = np.array([50, 50, 50, 1.0, 300, 50.0, 30.0, 50.0])
    p_ub = np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0])
    p_lb = np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0])
    # Fake data Distribution param
    n_obs = 10
    n_out = 20
    x_min = -200
    x_max = 200
    w_l = 0.5
    w_o = 50.0
    center = [0, 0, -200]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 5 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    # Baseline:
    name = "GradOff"
    use_grad = False

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
    )


###############################################################################
def SearchTests(n_run=5, plot=False, save=True):

    # Baseline:
    save = save
    plot = plot
    name = "PartialObs"
    n_run = n_run
    n_steps = 100
    model = powerline.ArrayModel32()
    p_hat = np.array([50, 50, 50, 1.0, 300, 50.0, 30.0, 50.0])
    p_ub = np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0])
    p_lb = np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0])
    # Fake data Distribution param
    n_obs = 16
    n_out = 5
    x_min = -100
    x_max = -70
    w_l = 0.5
    w_o = 10.0
    center = [-50, -50, -50]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 5 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    # Baseline:
    name = "no_search"
    n_sea = 1

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
    )


###############################################################################
def QuadLossTest(n_run=5, plot=False, save=True):

    # Baseline:
    save = save
    plot = plot
    name = "GlobalConvergence"
    n_run = n_run
    n_steps = 100
    model = powerline.ArrayModel32()
    p_hat = np.array([50, 50, 50, 1.0, 300, 50.0, 30.0, 50.0])
    p_ub = np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0])
    p_lb = np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0])
    # Fake data Distribution param
    n_obs = 10
    n_out = 20
    x_min = -200
    x_max = 200
    w_l = 0.5
    w_o = 50.0
    center = [0, 0, -200]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 5 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    # Baseline:
    name = "QuadLoss"
    l = 0

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
    )


###############################################################################
def CrazyOutliers(n_run=1, plot=True, save=False):

    # Baseline:
    save = save
    plot = plot
    name = "GlobalConvergence"
    n_run = n_run
    n_steps = 100
    model = powerline.ArrayModel32()
    p_hat = np.array([50, 50, 50, 1.0, 300, 50.0, 30.0, 50.0])
    p_ub = np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0])
    p_lb = np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0])
    # Fake data Distribution param
    n_obs = 10
    n_out = 1000
    x_min = -200
    x_max = 200
    w_l = 0.5
    w_o = 50.0
    center = [0, 0, -200]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 5 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 100.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
    )


###############################################################################
def Snowing(n_run=1, plot=True, save=False):

    # Baseline:
    save = save
    plot = plot
    name = "GlobalConvergence"
    n_run = n_run
    n_steps = 100
    model = powerline.ArrayModel32()
    p_hat = np.array([50, 50, 50, 1.0, 300, 50.0, 30.0, 50.0])
    p_ub = np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0])
    p_lb = np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0])
    # Fake data Distribution param
    n_obs = 10
    n_out = 100
    x_min = -200
    x_max = 200
    w_l = 0.5
    w_o = 150.0
    center = [0, 0, 0]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 5 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
    )


###############################################################################
def OutliersTest(n_out=10, n_run=5, plot=False, save=True):

    # Baseline:
    save = save
    plot = plot
    name = "Out" + str(n_out)
    n_run = n_run
    n_steps = 100
    model = powerline.ArrayModel32()
    p_hat = np.array([50, 50, 50, 1.0, 300, 50.0, 30.0, 50.0])
    p_ub = np.array([150, 150, 150, 2.0, 900, 51.0, 31.0, 51.0])
    p_lb = np.array([0, 0, 0, 0.0, 300, 49.0, 29.0, 49.0])
    # Fake data Distribution param
    n_obs = 10
    n_out = n_out
    x_min = -200
    x_max = 200
    w_l = 0.5
    w_o = 50.0
    center = [0, 0, 0]
    partial_obs = True
    # Solver param
    n_sea = 3
    var = np.array([50, 50, 50, 1.0, 200, 1.0, 1.0, 1.0])
    Q = 5 * 1e-6 * np.diag([20.0, 20.0, 20.0, 1000.0, 1.0, 800.0, 200.0, 200.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n_s = 100
    x_min_s = -200
    x_max_s = +200
    use_grad = True

    powerline.ArrayModelEstimatorTest(
        save,
        plot,
        name,
        n_run,
        n_steps,
        model,
        p_hat,
        p_ub,
        p_lb,
        n_obs,
        n_out,
        x_min,
        x_max,
        w_l,
        w_o,
        center,
        partial_obs,
        n_sea,
        var,
        Q,
        l,
        power,
        b,
        method,
        n_s,
        x_min_s,
        x_max_s,
        use_grad,
    )


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    # Demos
    ###############################
    # GlobalConvergenceTest(2, True, True)
    # GlobalConvergenceBagTest( 2 , True , False )
    # PartialObsTest( 2 , True , False )

    # CrazyOutliers(2 , True , False )
    Snowing(2, True, False)

    # SearchTests(  2 , True , False )
    # QuadLossTest( 2 , True , False )
    #
    # Short Tests
    ###############################
    # GlobalConvergenceTest( 12 , False , True )
    # PartialObsTest(12, True, True)
    # RegulationVsNoRegulation( 12 , False, True)
    # SearchTests( 12 , False , True )
    # GradientTests( 12 , False , True )
    # MethodTests( 12 , False , True )
    # QuadLossTest( 12 , False , True )

    # Outliers Tests
    ###############################
    # OutliersTest( 1 , 5 )
    # OutliersTest( 10 , 5 )
    # OutliersTest( 20 , 5 )
    # OutliersTest( 50 , 5 )
    # OutliersTest( 100 , 5 )
    # OutliersTest( 200 , 5 )
    # OutliersTest( 500 , 5 )

    # Long Run
    ###############################
    # # Baseline Plot
    # GlobalConvergenceTest( 1000 , False , True )
    # PartialObsTest( 1000 , False , True )
    # # Regulation
    # RegulationVsNoRegulation( 1000 , False, True)
    # # X vs sample
    # MethodTests( 1000 , False , True )
    # # Gradient
    # GradientTests( 1000 , False , True )
    # # Search
    # SearchTests( 1000 , False , True )
    # QuadLoss
    # QuadLossTest( 100 , False , True )
