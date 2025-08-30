#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:52:04 2023

@author: alex
"""

from catenary import tools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


###########################
# Catenary function
###########################


############################
def cat(x, a=1.0):
    """
    inputs
    --------
    x : position along cable in local frame ( x = 0 @ z_min )
    a : sag parameter


    output
    -------
    z : elevation in local frame ( z = 0 is z_min )

    Note on local cable frame
    --------------------------
    origin (0,0,0) is lowest elevation point on the cable
    x-axis is tangential to the cable at the origin
    y-axis is the normal of the plane formed by the cable
    z-axis is positive in the curvature direction at the origin

    """

    z = a * (np.cosh(x / a) - 1.0)

    return z


############################
def w_T_c(phi, x, y, z):
    """
    Tranformation matrix from catenary local frame to world frame

    inputs
    ----------
    phi : z rotation of catenary local basis with respect to world basis
    x   : x translation of catenary local frame orign in world frame
    y   : y translation of catenary local frame orign in world frame
    z   : z translation of catenary local frame orign in world frame

    outputs
    ----------
    world_T_catenary  : 4x4 Transformation Matrix

    """

    s = np.sin(phi)
    c = np.cos(phi)

    T = np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1.0, z], [0, 0, 0, 1.0]])

    return T


############################
def p2r_w(
    p,
    x_min=-200,
    x_max=200,
    n=200,
):
    """
    Compute n pts coord in world frame based on a parameter vector

    inputs
    --------
    p      : vector of parameters

        x_0 : x translation of local frame orign in world frame
        y_0 : y translation of local frame orign in world frame
        z_0 : z translation of local frame orign in world frame
        phi : z rotation of local frame basis in in world frame
        a   : sag parameter

    x_min  : start of points in cable local frame
    x_max  : end   of points in cable local frame
    n      : number of pts

    outputs
    ----------
    r_w[0,:] : x positions in world frame
    r_w[1,:] : y positions in world frame
    r_w[2,:] : z positions in world frame
    x_c      : vector of x coord in catenary frame

    """

    x_c = np.linspace(x_min, x_max, n)

    # params
    x_0 = p[0]
    y_0 = p[1]
    z_0 = p[2]
    phi = p[3]
    a = p[4]

    # catenary frame z
    z_c = cat(x_c, a)

    r_c = np.zeros((4, n))
    r_c[0, :] = x_c
    r_c[2, :] = z_c
    r_c[3, :] = np.ones((n))

    r_w = w_T_c(phi, x_0, y_0, z_0) @ r_c

    return (r_w[0:3, :], x_c)


# ###########################
# Plotting
# ###########################


###############################################################################
class CatenaryEstimationPlot:
    """ """

    ############################
    def __init__(self, p_true, p_hat, pts, n=100, xmin=-200, xmax=200):

        fig = plt.figure(figsize=(4, 3), dpi=300, frameon=True)
        ax = fig.add_subplot(projection="3d")

        pts_true = p2r_w(p_true, xmin, xmax, n)[0]
        pts_hat = p2r_w(p_hat, xmin, xmax, n)[0]
        pts_noisy = pts

        line_true = ax.plot(
            pts_true[0, :], pts_true[1, :], pts_true[2, :], label="True equation"
        )
        line_hat = ax.plot(
            pts_hat[0, :],
            pts_hat[1, :],
            pts_hat[2, :],
            "--",
            label="Estimated equation",
        )
        line_noisy = ax.plot(
            pts_noisy[0, :], pts_noisy[1, :], pts_noisy[2, :], "x", label="Measurements"
        )

        ax.axis("equal")
        ax.legend(loc="upper right", fontsize=5)
        ax.set_xlabel("x", fontsize=5)
        ax.grid(True)

        self.fig = fig
        self.ax = ax

        self.n = n
        self.xmin = xmin
        self.xmax = xmax

        self.line_true = line_true
        self.line_hat = line_hat
        self.line_noisy = line_noisy

    ############################
    def update_estimation(self, p_hat):

        pts_hat = p2r_w(p_hat, self.xmin, self.xmax, self.n)[0]

        self.line_hat[0].set_data(pts_hat[0, :], pts_hat[1, :])
        self.line_hat[0].set_3d_properties(pts_hat[2, :])

        plt.pause(0.001)

    ############################
    def update_pts(self, pts_noisy):

        self.line_noisy[0].set_data(pts_noisy[0, :], pts_noisy[1, :])
        self.line_noisy[0].set_3d_properties(pts_noisy[2, :])

        plt.pause(0.001)

    ############################
    def update_true(self, p_true):

        pts_true = p2r_w(p_true, self.xmin, self.xmax, self.n)[0]

        self.line_true[0].set_data(pts_true[0, :], pts_true[1, :])
        self.line_true[0].set_3d_properties(pts_true[2, :])

        plt.pause(0.001)


###############################
# Data generation for testing
##############################


############################
def noisy_p2r_w(p, x_min=-200, x_max=200, n=400, w=0.5, seed=None):
    """
    p2r_w but with added gaussian noise of standard deviation w

    """

    # true points on the curve
    r_line = p2r_w(p, x_min, x_max, n)[0]

    # adding measurements noise
    rng = np.random.default_rng(seed=seed)
    r_noisy = r_line + w * rng.standard_normal((3, n))

    return r_noisy


############################
def outliers(n=10, center=[0, 0, 0], w=100, seed=None):
    """
    Create random 3D world pts

    """

    # Outliers randoms points (all pts are initialized as random outliers)
    rng = np.random.default_rng(seed=seed)
    noise = w * rng.standard_normal((3, n))
    pts = np.zeros((3, n))

    pts[0, :] = center[0]
    pts[1, :] = center[1]
    pts[2, :] = center[2]

    pts = pts + noise  # noise randoms points

    return pts


############################
def generate_test_data(
    p,
    partial_obs=False,
    n_obs=20,
    x_min=-100,
    x_max=100,
    w_l=0.5,
    n_out=10,
    center=[0, 0, 0],
    w_o=100,
    seed=None,
):
    """
    generate pts for a line and outliers

    """

    if partial_obs:

        n_obs = np.random.randint(1, n_obs)
        x_min = np.random.randint(x_min, x_max)
        x_max = np.random.randint(x_min, x_max)

    r_line = noisy_p2r_w(p, x_min, x_max, n_obs, w_l, seed)

    r_out = outliers(n_out, center, w_o, seed)

    pts = np.append(r_line, r_out, axis=1)

    return pts


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    p = np.array([10.0, 10.0, 10.0, 0.5, 100.0])

    p2 = np.array([10.0, 10.0, 10.0, 1.5, 600.0])

    plot = CatenaryEstimationPlot(p, p2, generate_test_data(p))

    plot.update_estimation(p2)
