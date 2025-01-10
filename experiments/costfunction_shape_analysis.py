#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:39:56 2023

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time


from catenary import singleline as catenary
from catenary import powerline


############################
def array32_cost_shape_analysis(model=powerline.ArrayModel32()):

    p_hat = np.array([0, 0, 0, 0.0, 500, 5.0, 3, 5])
    p = np.array([0, 0, 0, 0.0, 500, 5.0, 3.0, 5])

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

    R = np.ones(pts.shape[1]) / pts.shape[1]
    Q = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    l = 1.0
    power = 2.0
    b = 1.0
    method = "x"
    n = 25
    x_min = -20
    x_max = +20

    params = [model, R, Q, l, power, b, method, n, x_min, x_max]

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

    n = 100
    zs = np.linspace(-25, 25, n)
    cs = np.zeros(n)

    for i in range(n):

        p_hat[2] = zs[i]

        cs[i] = powerline.J(p_hat, pts, p_hat, params)

        plot.update_estimation(p_hat)

    fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    ax = [ax]
    ax[0].plot(zs, cs)
    ax[0].set_xlabel("z_hat", fontsize=5)
    ax[0].set_ylabel("J(p)", fontsize=5)
    ax[0].grid(True)
    ax[0].legend()
    ax[0].tick_params(labelsize=5)


############################
def hard_array32_tracking_local_minima_analysis(n_run=10):

    model = powerline.ArrayModel32()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 25.0, 50.0])
    p_hat = np.array([3.56, 26.8, 25.82, 1.05, 499.95, 44.12, 25.00, 28.1])

    Q = 0 * np.diag([0.0002, 0.0002, 0.0002, 0.001, 0.0001, 0.002, 0.002, 0.002])
    l = 1.0
    power = 2.0
    b = 1.0
    method = "x"
    n = 501
    x_min = -200
    x_max = +200

    n = 200
    zs = np.linspace(-100, 100, n)
    cs = np.zeros((n, n_run))

    for j in range(n_run):

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

        R = np.ones(pts.shape[1]) / pts.shape[1]
        params = [model, R, Q, l, power, b, method, n, x_min, x_max]

        if j == 0:
            plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

        for i in range(n):

            p_hat[2] = p[2] + zs[i]

            cs[i, j] = powerline.J(p_hat, pts, p_hat, params)

            if j == 0:
                plot.update_estimation(p_hat)

    fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    ax = [ax]
    for j in range(n_run):
        ax[0].plot(zs, cs[:, j])
    ax[0].set_xlabel("$\hat{z}_o$", fontsize=10)
    ax[0].set_ylabel("$J(p)$", fontsize=10)
    ax[0].grid(True)
    # ax[0].legend()
    ax[0].tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig("costshape32.pdf")


############################
def arrayconstant2221_cost_shape_analysis(n_run=10):

    model = powerline.ArrayModelConstant2221()

    p_hat = np.array([0, 0, 0, 0.0, 500, 5.0, 5, 10])
    p = np.array([0, 0, 0, 0.0, 500, 5.0, 5.0, 10])

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

    Q = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    l = 1.0
    power = 2.0
    b = 1.0
    method = "x"
    n = 501
    x_min = -200
    x_max = +200

    n = 200
    zs = np.linspace(-20, 20, n)
    cs = np.zeros((n, n_run))

    for j in range(n_run):

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

        R = np.ones(pts.shape[1]) / pts.shape[1]
        params = [model, R, Q, l, power, b, method, n, x_min, x_max]

        if j == 0:
            plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

        for i in range(n):

            p_hat[2] = p[2] + zs[i]

            cs[i, j] = powerline.J(p_hat, pts, p_hat, params)

            if j == 0:
                plot.update_estimation(p_hat)

    fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

    ax = [ax]
    for j in range(n_run):
        ax[0].plot(zs, cs[:, j])
    ax[0].set_xlabel("$\hat{z}_o$", fontsize=10)
    ax[0].set_ylabel("$J(p)$", fontsize=10)
    ax[0].grid(True)
    # ax[0].legend()
    ax[0].tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig("costshape2221.pdf")


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    # array32_cost_shape_analysis()

    hard_array32_tracking_local_minima_analysis()

    arrayconstant2221_cost_shape_analysis()
