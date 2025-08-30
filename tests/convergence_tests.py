#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:46:33 2023

@author: alex
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
from scipy.optimize import minimize
import time


from catenary.kinematic import singleline as catenary
from catenary.kinematic import powerline
from catenary.estimation import costfunction as cf


def basic_array3_convergence_test():

    p_1 = np.array([28.0, 30.0, 77.0, 0.0, 53.0])
    p_2 = np.array([28.0, 50.0, 77.0, 0.0, 53.0])
    p_3 = np.array([28.0, 70.0, 77.0, 0.0, 53.0])
    # p_4  =  np.array([ 28.0, 35.0, 97.0, 0.0, 53.0])
    # p_5  =  np.array([ 28.0, 65.0, 97.0, 0.0, 53.0])

    pts1 = catenary.generate_test_data(
        p_1, n_obs=10, n_out=5, center=[50, 50, 50], x_min=0, x_max=10
    )
    pts2 = catenary.generate_test_data(
        p_2, n_obs=10, n_out=5, center=[50, 50, 50], x_min=-20, x_max=30
    )
    pts3 = catenary.generate_test_data(
        p_3, n_obs=10, n_out=5, center=[50, 50, 50], x_min=-10, x_max=15
    )

    pts = np.hstack((pts1, pts2, pts3))

    p = np.array([28.0, 50.0, 77.0, 0.0, 53, 20.0])

    p_init = np.array([10.0, 10.0, 10.0, 1.0, 80, 16.0])

    bounds = [(0, 200), (0, 200), (0, 200), (0, 0.3), (10, 200), (15, 30)]

    model = powerline.ArrayModel()

    R = np.ones(pts.shape[1]) / pts.shape[1]
    Q = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n = 100
    x_min = -50
    x_max = +50

    params = [model, R, Q, l, power, b, method, n, x_min, x_max]

    start_time = time.time()
    plot = powerline.EstimationPlot(p, p_init, pts, model.p2r_w, 25, -50, 50)

    func = lambda p: cf.J(p, pts, p_init, params)
    grad = lambda p: cf.dJ_dp(p, pts, p_init, params)

    res = minimize(
        func,
        p_init,
        method="SLSQP",
        bounds=bounds,
        jac=grad,
        # constraints=constraints,
        callback=plot.update_estimation,
        options={"disp": True, "maxiter": 500},
    )

    p_hat = res.x

    print(
        f" Optimzation completed in : { time.time() - start_time } sec \n"
        + f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n"
        + f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n"
        + f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
    )


############################
def basic_array32_convergence_test():

    model = powerline.ArrayModel32()

    p_1 = np.array([28.0, 30.0, 77.0, 0.0, 53.0])
    p_2 = np.array([28.0, 50.0, 77.0, 0.0, 53.0])
    p_3 = np.array([28.0, 70.0, 77.0, 0.0, 53.0])
    p_4 = np.array([28.0, 35.0, 97.0, 0.0, 53.0])
    p_5 = np.array([28.0, 65.0, 97.0, 0.0, 53.0])

    pts1 = catenary.generate_test_data(
        p_1, n_obs=10, n_out=5, center=[50, 50, 50], x_min=0, x_max=10
    )
    pts2 = catenary.generate_test_data(
        p_2, n_obs=10, n_out=5, center=[50, 50, 50], x_min=-20, x_max=30
    )
    pts3 = catenary.generate_test_data(
        p_3, n_obs=10, n_out=5, center=[50, 50, 50], x_min=-10, x_max=15
    )
    pts4 = catenary.generate_test_data(
        p_4, n_obs=10, n_out=5, center=[50, 50, 50], x_min=-30, x_max=30
    )
    pts5 = catenary.generate_test_data(
        p_5, n_obs=10, n_out=5, center=[50, 50, 50], x_min=-20, x_max=20
    )

    pts = np.hstack((pts1, pts2, pts3, pts4, pts5))

    p = np.array([28.0, 50.0, 77.0, 0.0, 53, 20.0, 15.0, 20.0])

    p_init = np.array([10.0, 10.0, 10.0, 1.0, 80, 16.0, 15.0, 16.0])

    bounds = [
        (0, 200),
        (0, 200),
        (0, 200),
        (0, 0.3),
        (10, 200),
        (15, 30),
        (15, 15),
        (15, 30),
    ]

    R = np.ones(pts.shape[1]) / pts.shape[1]
    Q = np.diag(np.ones(p.shape[0])) * 0.0
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n = 100
    x_min = -50
    x_max = +50

    params = [model, R, Q, l, power, b, method, n, x_min, x_max]

    start_time = time.time()
    plot = powerline.EstimationPlot(p, p_init, pts, model.p2r_w, 25, -50, 50)

    func = lambda p: cf.J(p, pts, p_init, params)

    res = minimize(
        func,
        p_init,
        method="SLSQP",
        bounds=bounds,
        # constraints=constraints,
        callback=plot.update_estimation,
        options={"disp": True, "maxiter": 500},
    )

    p_hat = res.x

    print(
        f" Optimzation completed in : { time.time() - start_time } sec \n"
        + f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n"
        + f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n"
        + f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
    )


############################
def basic_array32_tracking_test():

    model = powerline.ArrayModel32()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 30.0, 50.0])
    p_hat = np.array([100, 100, 100, 1.0, 300, 40.0, 25.0, 25])

    pts = model.generate_test_data(p, partial_obs=True)

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

    bounds = [
        (0, 200),
        (0, 200),
        (0, 200),
        (0, 3.14),
        (100, 2000),
        (15, 60),
        (15, 50),
        (15, 50),
    ]

    R = np.ones(pts.shape[1]) / pts.shape[1]
    Q = 10 * np.diag([0.0002, 0.0002, 0.0002, 0.001, 0.0001, 0.002, 0.002, 0.002])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n = 501
    x_min = -200
    x_max = +200

    params = [model, R, Q, l, power, b, method, n, x_min, x_max]

    for i in range(200):

        pts = model.generate_test_data(p, partial_obs=True)

        plot.update_pts(pts)

        start_time = time.time()

        params[1] = np.ones(pts.shape[1]) / pts.shape[1]

        func = lambda p: cf.J(p, pts, p_hat, params)

        res = minimize(
            func,
            p_hat,
            method="SLSQP",
            bounds=bounds,
            # constraints=constraints,
            # callback=plot.update_estimation,
            options={"disp": True, "maxiter": 500},
        )

        p_hat = res.x

        plot.update_estimation(p_hat)

        print(
            f" Optimzation completed in : { time.time() - start_time } sec \n"
            f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n"
            + f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )


############################
def hard_array32_tracking_test():

    model = powerline.ArrayModel32()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 25.0, 50.0])
    p_hat = np.array([0, 0, 0, 1.2, 500, 40.0, 25.0, 25])

    pts = model.generate_test_data(
        p,
        partial_obs=True,
        n_obs=16,
        x_min=-100,
        x_max=-50,
        n_out=10,
        center=[0, 0, 0],
        w_o=20,
    )

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

    bounds = [
        (0, 200),
        (0, 200),
        (0, 200),
        (0.5, 1.5),
        (100, 2000),
        (30, 60),
        (25, 25),
        (25, 60),
    ]

    R = np.ones(pts.shape[1]) / pts.shape[1]
    Q = 1 * np.diag([0.0002, 0.0002, 0.0002, 0.001, 0.0001, 0.002, 0.002, 0.002])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n = 501
    x_min = -200
    x_max = +200

    params = [model, R, Q, l, power, b, method, n, x_min, x_max]

    for i in range(500):

        pts = model.generate_test_data(
            p,
            partial_obs=True,
            n_obs=16,
            x_min=-100,
            x_max=-70,
            n_out=10,
            center=[-50, -50, -50],
            w_o=10,
        )

        plot.update_pts(pts)

        start_time = time.time()

        params[1] = np.ones(pts.shape[1]) / pts.shape[1]

        func = lambda p: cf.J(p, pts, p_hat, params)

        res = minimize(
            func,
            p_hat,
            method="SLSQP",
            bounds=bounds,
            # constraints=constraints,
            # callback=plot.update_estimation,
            options={"disp": True, "maxiter": 500},
        )

        p_hat = res.x

        plot.update_estimation(p_hat)

        print(
            f" Optimzation completed in : { time.time() - start_time } sec \n"
            f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n"
            + f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )


############################
def basic_array2221_tracking_test():

    model = powerline.ArrayModel2221()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 70.0, 50.0, 30, 30, 30])
    p_hat = np.array([100, 100, 100, 1.0, 300, 40.0, 25.0, 25, 25, 25, 25])

    pts = model.generate_test_data(p, partial_obs=True)

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

    bounds = [
        (0, 200),
        (0, 200),
        (0, 200),
        (0, 3.14),
        (100, 2000),
        (40, 60),
        (40, 80),
        (40, 60),
        (20, 40),
        (20, 40),
        (20, 40),
    ]

    R = np.ones(pts.shape[1]) / pts.shape[1]
    Q = 2 * np.diag(
        [
            0.0002,
            0.0002,
            0.0002,
            0.001,
            0.0001,
            0.002,
            0.002,
            0.002,
            0.002,
            0.002,
            0.002,
        ]
    )
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n = 501
    x_min = -200
    x_max = +200

    params = [model, R, Q, l, power, b, method, n, x_min, x_max]

    for i in range(50):

        pts = model.generate_test_data(p, partial_obs=True)

        plot.update_pts(pts)

        start_time = time.time()

        params[1] = np.ones(pts.shape[1]) / pts.shape[1]

        func = lambda p: cf.J(p, pts, p_hat, params)

        res = minimize(
            func,
            p_hat,
            method="SLSQP",
            bounds=bounds,
            # constraints=constraints,
            # callback=plot.update_estimation,
            options={"disp": True, "maxiter": 500},
        )

        p_hat = res.x

        plot.update_estimation(p_hat)

        print(
            f" Optimzation completed in : { time.time() - start_time } sec \n"
            f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n"
            + f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )


############################
def hard_array2221_tracking_test():

    model = powerline.ArrayModel2221()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 70.0, 50.0, 30, 30, 30])
    p_hat = np.array([100, 100, 100, 1.0, 300, 40.0, 25.0, 25, 25, 25, 25])

    pts = model.generate_test_data(
        p,
        partial_obs=True,
        n_obs=16,
        x_min=-100,
        x_max=-50,
        n_out=10,
        center=[0, 0, 0],
        w_o=20,
    )

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

    bounds = [
        (0, 200),
        (0, 200),
        (0, 200),
        (0, 3.14),
        (100, 2000),
        (40, 60),
        (40, 80),
        (40, 60),
        (20, 40),
        (20, 40),
        (20, 40),
    ]

    R = np.ones(pts.shape[1]) / pts.shape[1]
    Q = 1 * np.diag(
        [
            0.0002,
            0.0002,
            0.0002,
            0.001,
            0.0001,
            0.002,
            0.002,
            0.002,
            0.002,
            0.002,
            0.002,
        ]
    )
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n = 501
    x_min = -200
    x_max = +200

    params = [model, R, Q, l, power, b, method, n, x_min, x_max]

    for i in range(100):

        pts = model.generate_test_data(
            p,
            partial_obs=True,
            n_obs=16,
            x_min=-100,
            x_max=-70,
            n_out=10,
            center=[-50, -50, -50],
            w_o=10,
        )

        plot.update_pts(pts)

        start_time = time.time()

        params[1] = np.ones(pts.shape[1]) / pts.shape[1]

        func = lambda p: cf.J(p, pts, p_hat, params)

        res = minimize(
            func,
            p_hat,
            method="SLSQP",
            bounds=bounds,
            # constraints=constraints,
            # callback=plot.update_estimation,
            options={"disp": True, "maxiter": 500},
        )

        p_hat = res.x

        plot.update_estimation(p_hat)

        print(
            f" Optimzation completed in : { time.time() - start_time } sec \n"
            f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n"
            + f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )


############################
def hard_arrayconstant2221_tracking_test():

    model = powerline.ArrayModelConstant2221()

    p = np.array([1, -3, 14, 0.2, 500, 4.0, 4, 9])
    p_hat = np.array([0, 0, 0, 0.0, 1000, 5.0, 5.0, 10])

    pts = model.generate_test_data(
        p,
        partial_obs=True,
        n_obs=16,
        x_min=-100,
        x_max=-50,
        n_out=10,
        center=[0, 0, 0],
        w_o=20,
    )

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

    bounds = [
        (-5, 5),
        (-5, 5),
        (-15, 15),
        (-0.3, 0.3),
        (100, 2000),
        (3, 6),
        (3, 6),
        (5, 15),
    ]

    R = np.ones(pts.shape[1]) / pts.shape[1]
    Q = 10 * np.diag([0.0002, 0.0002, 0.0002, 0.001, 0.0001, 0.002, 0.002, 0.002])
    l = 1.0
    power = 2.0
    b = 1000.0
    method = "x"
    n = 501
    x_min = -200
    x_max = +200

    params = [model, R, Q, l, power, b, method, n, x_min, x_max]

    for i in range(500):

        pts = model.generate_test_data(
            p,
            partial_obs=True,
            n_obs=16,
            x_min=-10,
            x_max=+20,
            n_out=10,
            center=[-50, -50, -50],
            w_o=10,
        )

        plot.update_pts(pts)

        start_time = time.time()

        params[1] = np.ones(pts.shape[1]) / pts.shape[1]

        func = lambda p: cf.J(p, pts, p_hat, params)

        res = minimize(
            func,
            p_hat,
            method="SLSQP",
            bounds=bounds,
            # constraints=constraints,
            # callback=plot.update_estimation,
            options={"disp": True, "maxiter": 500},
        )

        p_hat = res.x

        plot.update_estimation(p_hat)

        print(
            f" Optimzation completed in : { time.time() - start_time } sec \n"
            f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n"
            + f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )


###########################
# Speed tests
###########################


def speed_test(plot=False):

    model = powerline.ArrayModel32()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 30.0, 50.0])
    p_hat = np.array([55, 66, 32, 1.2, 400, 49.0, 29.0, 58.0])

    bounds = [
        (0, 200),
        (0, 200),
        (0, 200),
        (0.5, 1.5),
        (100, 800),
        (30, 60),
        (15, 50),
        (15, 50),
    ]

    pts = model.generate_test_data(
        p,
        n_obs=10,
        x_min=-50,
        x_max=50,
        w_l=0.5,
        n_out=3,
        center=[0, 0, 0],
        w_o=100,
        partial_obs=False,
    )
    m = pts.shape[1]

    # plot   = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )

    R = np.ones((m)) * 1 / m
    Q = 0 * np.diag([0.0002, 0.0002, 0.0002, 0.001, 0.0001, 0.002, 0.002, 0.002])
    b = 1000.0
    l = 1.0
    power = 2.0
    method = "sample"
    n = 101
    x_min = -50
    x_max = +50

    params = [model, R, Q, l, power, b, method, n, x_min, x_max]

    ###########################
    # 1
    ###########################

    func = lambda p: cf.J(p, pts, p_hat, params)

    start_time = time.time()

    res = minimize(
        func,
        p_hat,
        method="SLSQP",
        bounds=bounds,
        # jac = jac,
        # constraints=constraints,
        # callback=plot.update_estimation,
        options={"disp": False, "maxiter": 500},
    )

    t1 = time.time() - start_time
    p1 = res.x

    ###########################
    # 2
    ###########################

    jac = lambda p: cf.dJ_dp(p, pts, p_hat, params)

    start_time = time.time()

    res = minimize(
        func,
        p_hat,
        method="SLSQP",
        bounds=bounds,
        jac=jac,
        # constraints=constraints,
        # callback=plot.update_estimation,
        options={"disp": False, "maxiter": 500},
    )

    t2 = time.time() - start_time
    p2 = res.x

    ###########################
    # 3
    ###########################

    params[6] = "x"

    func = lambda p: cf.J(p, pts, p_hat, params)

    start_time = time.time()

    res = minimize(
        func,
        p_hat,
        method="SLSQP",
        bounds=bounds,
        # jac = jac,
        # constraints=constraints,
        # callback=plot.update_estimation,
        options={"disp": False, "maxiter": 500},
    )

    t3 = time.time() - start_time
    p3 = res.x

    ###########################
    # 4
    ###########################

    jac = lambda p: cf.dJ_dp(p, pts, p_hat, params)

    start_time = time.time()

    res = minimize(
        func,
        p_hat,
        method="SLSQP",
        bounds=bounds,
        jac=jac,
        # constraints=constraints,
        # callback=plot.update_estimation,
        options={"disp": False, "maxiter": 500},
    )

    t4 = time.time() - start_time
    p4 = res.x

    print(
        f" Init: {np.array2string(p_hat, precision=2, floatmode='fixed')} \n"
        + f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n"
        + f" p1: {np.array2string(p1, precision=2, floatmode='fixed')} \n"
        + f" p2: {np.array2string(p2, precision=2, floatmode='fixed')} \n"
        + f" p3: {np.array2string(p3, precision=2, floatmode='fixed')} \n"
        + f" p4: {np.array2string(p4, precision=2, floatmode='fixed')} \n"
    )

    print("Sample no grad   t=", t1)
    print("Sample with grad t=", t2)
    print("x no grad t=", t3)
    print("x with grad t=", t4)

    if plot:

        plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

        plot = powerline.EstimationPlot(p, p1, pts, model.p2r_w)

        plot = powerline.EstimationPlot(p, p2, pts, model.p2r_w)

        plot = powerline.EstimationPlot(p, p3, pts, model.p2r_w)

        plot = powerline.EstimationPlot(p, p4, pts, model.p2r_w)


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    # basic_array3_convergence_test()

    # basic_array32_convergence_test()

    # basic_array32_tracking_test()

    # hard_array32_tracking_test()

    # basic_array2221_tracking_test()

    # hard_array2221_tracking_test()

    # hard_arrayconstant2221_tracking_test()

    speed_test(False)
