#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:52:04 2023

@author: alex
"""

from catenary import tools

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


###########################
# Catenary function
###########################

from catenary.kinematic import singleline as catenary


############################
def lorentzian(d, l=1.0, power=2, b=50.0):
    """

    Cost shaping function that smooth out the cost of large distance to
    minimize the effect of outliers.

    c = np.log10( 1 + ( b * x ) ** power / l )

    inputs
    --------
    d  : input vector of distance

    l     : shaping parameter or if set to zero c = d
    power : shaping parameter
    b     : shaping parameter

    outputs
    ----------
    c  : output vector of cost for each distances

    """

    # Saturation
    d_sat = np.clip(d, -b, b)

    # Cost shaping
    c = np.log10(1 + d_sat**power / l)

    return c


############################
def lorentzian_grad(d, l=1.0, power=2, b=50.0):
    """
    grad of previous fonction

    """

    # Cost shaping
    dc_dd = power * d ** (power - 1) / (np.log(10) * (l + d**power))

    # Saturation
    dc_dd[d > +b] = 0.0
    dc_dd[d < -b] = 0.0

    return dc_dd


###############################################
def compute_d_min(r_measurements, r_model):
    """

    Compute the distance between a list of 3D measurements and the closest
    point in a list of model points

    Inputs
    ----------
    r_measurements : (3 x m) array
    r_model        : (3 x n) array

    Ouputs
    -------
    d_min : (1 x m) array

    """

    # Vectors between measurements and all model pts
    e = r_measurements[:, :, np.newaxis] - r_model[:, np.newaxis, :]

    # Distances between measurements and model sample pts
    d = np.linalg.norm(e, axis=0)

    # Minimum distances to model for all measurements
    d_min = d.min(axis=1)

    return d_min


############################

default_cost_param = [
    "sample",
    np.diag([0.0, 0.0, 0.0, 0.0, 0.0]),
    1000.0,
    1.0,
    2,
    1000,
    -200,
    200,
]


############################
def J_single(p, pts, p_nom, param=default_cost_param):
    """
    Cost function for curve fitting a catenary model on a point cloud

    J = average_cost_per_measurement + regulation_term

    see attached notes.

    inputs
    --------
    p     : 5x1 parameter vector
    pts   : 3xm cloud point
    p_nom : 5x1 expected parameter vector ( optionnal for regulation )
    param : list of cost function parameter and options

    default_param = [ method = 'sample' ,
                     Q = np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                     b = 1.0 ,
                     l = 1.0 ,
                     power = 2 ,
                     n = 1000 ,
                     x_min = -200 ,
                     x_max = 200    ]

    method = 'sample' : sample-based brute force scheme
    method = 'x'      : data association is based on local x in cat frame

    Q      : 5x5 regulation weight matrix
    b      : scalar parameter in lorentzian function
    l      : scalar parameter in lorentzian function
    power  : scalar parameter in lorentzian function
    n      : number of sample (only used with method = 'sample' )
    x_min  : x start point of model catenary (only used with method = 'sample')
    x_max  : x end point of model catenary (only used with method = 'sample' )

    outputs
    ----------
    J : cost scalar

    """

    m = pts.shape[1]  # number of measurements

    method = param[0]
    Q = param[1]
    b = param[2]
    l = param[3]
    power = param[4]
    n = param[5]
    x_min = param[6]
    x_max = param[7]

    ###################################################
    if method == "sample":
        """data association is sampled-based"""

        # generate a list of sample point on the model curve
        r_model = p2r_w(p, x_min, x_max, n)[0]

        # Minimum distances to model for all measurements
        d_min = compute_d_min(pts, r_model)

    ###################################################
    elif method == "x":
        """data association is based on local x-coord in cat frame"""

        x0 = p[0]
        y0 = p[1]
        z0 = p[2]
        psi = p[3]
        a = p[4]

        c_T_w = np.linalg.inv(w_T_c(psi, x0, y0, z0))

        r_w = np.vstack([pts, np.ones(m)])

        # Compute measurements points positions in local cable frame
        r_c = c_T_w @ r_w

        # Compute expected z position based on x coord and catanery model
        z_hat = cat(r_c[0, :], a)

        # Compute delta vector between measurements and expected position
        e = np.zeros((3, m))
        e[1, :] = r_c[1, :]
        e[2, :] = r_c[2, :] - z_hat

        d_min = np.linalg.norm(e, axis=0)

    ###################################################

    # Cost shaping function
    c = lorentzian(d_min, l, power, b)

    # Average costs per measurement plus regulation
    pts_cost = c.sum() / m

    # Regulation
    p_e = p_nom - p

    # Total cost with regulation
    cost = pts_cost + p_e.T @ Q @ p_e

    return cost


############################
def dJ_dp_single(p, pts, p_nom, param=default_cost_param, num=False):
    """

    Gradient of J with respect to parameters p

    inputs
    --------
    see J function doc

    outputs
    ----------
    dJ_dp : 1 x 5 gradient evaluated at p

    """

    if num:

        dJ_dp = np.zeros(p.shape[0])
        dp = np.array([0.001, 0.001, 0.001, 0.001, 0.001])

        for i in range(5):

            pp = p.copy()
            pm = p.copy()
            pp[i] = p[i] + dp[i]
            pm[i] = p[i] - dp[i]
            cp = J_single(pp, pts, p_nom, param)
            cm = J_single(pm, pts, p_nom, param)

            dJ_dp[i] = (cp - cm) / (2.0 * dp[i])

    else:

        #########################
        # Analytical gratient
        #########################

        m = pts.shape[1]  # number of measurements

        method = param[0]
        Q = param[1]
        b = param[2]
        l = param[3]
        power = param[4]
        n = param[5]
        x_min = param[6]
        x_max = param[7]

        x0 = p[0]
        y0 = p[1]
        z0 = p[2]
        phi = p[3]
        a = p[4]

        if method == "sample":

            # generate a list of sample point on the model curve
            r_hat, x_l = p2r_w(p, x_min, x_max, n)

            # Vectors between measurements and model sample pts
            e = pts[:, :, np.newaxis] - r_hat[:, np.newaxis, :]

            # Distances between measurements and model sample pts
            d = np.linalg.norm(e, axis=0)

            # Minimum distances to model for all measurements
            d_min = d.min(axis=1)

            # Errors vector to closet model point
            j_min = d.argmin(axis=1)  # index of closet model point
            i = np.arange(j_min.shape[0])
            e_min = e[:, i, j_min]
            x = x_l[j_min]  # local x of closest pts on the model

            # Error Grad for each pts
            eT_de_dp = np.zeros((5, pts.shape[1]))

            eT_de_dp[0, :] = -e_min[0, :]
            eT_de_dp[1, :] = -e_min[1, :]
            eT_de_dp[2, :] = -e_min[2, :]
            eT_de_dp[3, :] = -(
                e_min[0, :] * (-np.sin(phi) * x) + e_min[1, :] * (+np.cos(phi) * x)
            )
            eT_de_dp[4, :] = -e_min[2, :] * (
                np.cosh(x / a) - x / a * np.sinh(x / a) - 1
            )

            # Norm grad
            dd_dp = eT_de_dp / d_min

        elif method == "x":

            # Pts in world frame
            x_w = pts[0, :]
            y_w = pts[1, :]
            z_w = pts[2, :]

            # Compute measurements points positions in local cable frame
            r_w = np.vstack([pts, np.ones(m)])
            c_T_w = np.linalg.inv(w_T_c(phi, x0, y0, z0))
            r_c = c_T_w @ r_w

            x_c = r_c[0, :]
            y_c = r_c[1, :]
            z_c = r_c[2, :]

            # Model z
            z_hat = cat(x_c, a)

            # Error in local frame
            e = np.zeros((3, m))
            e[1, :] = y_c
            e[2, :] = z_c - z_hat

            d_min = np.linalg.norm(e, axis=0)

            ey = e[1, :]
            ez = e[2, :]

            # print( x_c )
            # print( np.sinh( x_c ) )

            de_dx0 = np.sinh(x_c / a) * np.cos(phi)
            de_dy0 = np.sinh(x_c / a) * np.sin(phi)
            de_dphi = -np.sinh(x_c / a) * (
                -np.sin(phi) * (x_w - x0) + np.cos(phi) * (y_w - y0)
            )

            c = np.cos(phi)
            s = np.sin(phi)

            # Error Grad for each pts
            eT_de_dp = np.zeros((5, pts.shape[1]))

            eT_de_dp[0, :] = ey * s + ez * de_dx0
            eT_de_dp[1, :] = ey * -c + ez * de_dy0
            eT_de_dp[2, :] = -ez
            eT_de_dp[3, :] = ey * (c * (x0 - x_w) + s * (y0 - y_w)) + ez * de_dphi
            eT_de_dp[4, :] = -ez * (np.cosh(x_c / a) - x_c / a * np.sinh(x_c / a) - 1)

            # Norm grad
            dd_dp = eT_de_dp / d_min

        ################################

        # Smoothing grad
        dc_dd = lorentzian_grad(d_min, l, power, b)

        dc_dp = dc_dd * dd_dp

        # Average grad per point
        dc_cp_average = dc_dp.sum(axis=1) / m

        # Regulation
        p_e = p_nom - p

        # Total cost with regulation
        dJ_dp = dc_cp_average - 2 * p_e.T @ Q

    return dJ_dp


# ###########################
# Optimization Powerline
# ###########################


#######################################
def find_closest_distance(p, pts, p2r_w, x_min=-200, x_max=+200, n=100):

    r_model_flat = p2r_w(p, x_min, x_max, n)[0]

    # Vectors between measurements and all model pts
    e_flat = pts[:, :, np.newaxis] - r_model_flat[:, np.newaxis, :]

    # Distances between measurements and model sample pts
    d_flat = np.linalg.norm(e_flat, axis=0)

    # Minimum distances to model for all measurements
    d_min_flat = d_flat.min(axis=1)

    return d_min_flat


#######################################
def find_closest_distance_cable_point(p, pts, p2r_w, x_min=-200, x_max=+200, n=100):

    # number of measurements
    m = pts.shape[1]
    ind = np.arange(0, m)

    r_model = p2r_w(p, x_min, x_max, n)[1]

    # Vectors between measurements and all model pts of all cables
    e = pts[:, :, np.newaxis, np.newaxis] - r_model[:, np.newaxis, :, :]

    # Distances between measurements and model sample pts
    d = np.linalg.norm(e, axis=0)

    # Minimum distances to all cable and closet model points index j
    d_min = d.min(axis=1)
    j_min = d.argmin(axis=1)

    # Closest cable
    k = d_min.argmin(axis=1)  # closest cable index
    j = j_min[ind, k]  # closest point index on closest cable

    # d_min_min = d_min.min( axis = 1 )
    d_min_min = d[ind, j, k]  # Alternative computation

    return (d_min_min, j, k)


###############################################################################
# Cost function of match between pts cloud and an ArrayModel parameter vector
###############################################################################


def J(p, pts, p_nom, param):
    """
    cost function for curve fitting a catenary model on a point cloud

    J = sum_of_cost_per_measurement + regulation_term

    see attached notes.

    inputs
    --------
    p     : dim (n_p) parameter vector
    pts   : dim (3,m) list of m 3d measurement points
    p_nom : dim (n_p) expected parameter vector ( optionnal for regulation )

    param : list of cost function parameter and options

    [ model , R , Q , l , power , b , method , n , x_min , x_max ]

    model  : instance of ArrayModel

    R      : dim (m) weight for each measurement points
    Q      : dim (n_p,n_p) regulation weight matrix

    b      : distance saturation
    l      : scalar parameter in lorentzian function
    power  : scalar parameter in lorentzian function

    method = 'sample' : sample-based brute force scheme
    method = 'x'      : data association is based on local x in cat frame

    n      : number of sample (only used with method = 'sample' )
    x_min  : x start point of model catenary (only used with method = 'sample')
    x_max  : x end point of model catenary (only used with method = 'sample' )

    outputs
    ----------
    J : cost scalar

    """

    m = pts.shape[1]  # number of measurements
    n_p = p.shape[0]  # number of model parameters
    model = param[0]  # array model
    R = param[1]  # vector of pts weight
    Q = param[2]  # regulation matrix
    l = param[3]  # cost function shaping param
    power = param[4]  # cost function shaping param
    b = param[5]  # distance saturation
    method = param[6]  # data association method
    n = param[7]  # number of model pts (for sample method)
    x_min = param[8]  # start of sample points (for sample method)
    x_max = param[9]  # end of sample points (for sample method)

    ###################################################
    if method == "sample":
        """data association is sampled-based"""

        p2r_w = model.p2r_w

        # Minimum distances to model for all measurements
        d_min = find_closest_distance(p, pts, p2r_w, x_min, x_max, n)

    ###################################################
    elif method == "x":
        """data association is based on local x-coord in cat frame"""

        x0 = p[0]
        y0 = p[1]
        z0 = p[2]
        psi = p[3]
        a = p[4]

        # Array offsets
        r_k = model.p2deltas(p)

        # Transformation Matrix
        c_T_w = np.linalg.inv(catenary.w_T_c(psi, x0, y0, z0))

        # Compute measurements points positions in local cable frame
        r_w = np.vstack([pts, np.ones(m)])
        r_i = (c_T_w @ r_w)[0:3, :]

        # Catenary elevation at ref point
        x_j = r_i[0, :, np.newaxis] - r_k[0, np.newaxis, :]
        z_j = catenary.cat(x_j, a)

        # Reference model points
        r_j = np.zeros((3, m, model.q))

        r_j[0, :, :] = r_i[0, :, np.newaxis]
        r_j[1, :, :] = r_k[1, np.newaxis, :]
        r_j[2, :, :] = z_j + r_k[2, np.newaxis, :]

        # All error vectors
        E = r_i[:, :, np.newaxis] - r_j

        # Distances between measurements and model ref
        D = np.linalg.norm(E, axis=0)

        # Minimum distances to model for all measurements
        d_min = D.min(axis=1)
        # k_min = D.argmin( axis = 1 )

    ###################################################

    # From distance to individual pts cost
    if not (l == 0):
        # lorentzian cost shaping function
        c = lorentzian(d_min, l, power, b)

    else:
        # No cost shaping
        c = d_min

    # Regulation error
    p_e = p_nom - p

    # Total cost with regulation
    cost = R.T @ c + p_e.T @ Q @ p_e

    return cost


######################################################
def dJ_dp(p, pts, p_nom, param, num=False):
    """

    Gradient of J with respect to parameters p

    inputs
    --------
    see J function doc

    outputs
    ----------
    dJ_dp : 1 x n_p gradient evaluated at p

    """

    m = pts.shape[1]  # number of measurements
    n_p = p.shape[0]  # number of model parameters

    if num:

        dJ_dp = np.zeros(n_p)
        dp = 0.0001 * np.ones(n_p)

        for i in range(n_p):

            pp = p.copy()
            pm = p.copy()
            pp[i] = p[i] + dp[i]
            pm[i] = p[i] - dp[i]
            cp = J(pp, pts, p_nom, param)
            cm = J(pm, pts, p_nom, param)

            dJ_dp[i] = (cp - cm) / (2.0 * dp[i])

    else:

        #########################
        # Analytical gratient
        #########################

        model = param[0]  # array model
        R = param[1]  # vector of pts weight
        Q = param[2]  # regulation matrix
        l = param[3]  # cost function shaping param
        power = param[4]  # cost function shaping param
        b = param[5]  # cost function shaping param
        method = param[6]  # data association method
        n = param[7]  # number of model pts (for sample method)
        x_min = param[8]  # start of sample points (for sample method)
        x_max = param[9]  # end of sample points (for sample method)

        x0 = p[0]
        y0 = p[1]
        z0 = p[2]
        phi = p[3]
        a = p[4]

        if method == "sample":

            # number of measurements
            ind = np.arange(0, m)

            # generate a list of sample point on the model curve
            r_flat, r, xs = model.p2r_w(p, x_min, x_max, n)

            # Vectors between measurements and all model pts of all cables
            E = pts[:, :, np.newaxis, np.newaxis] - r[:, np.newaxis, :, :]

            # Distances between measurements and model sample pts
            D = np.linalg.norm(E, axis=0)

            # Minimum distances to all cable and closet model points index j
            D_min = D.min(axis=1)
            j_min = D.argmin(axis=1)

            # Closest cable
            k = D_min.argmin(axis=1)  # closest cable index
            j = j_min[ind, k]  # closest point index on closest cable
            xj = xs[j]  # local x of closest pts on the model
            d = D[ind, j, k]  # Closest distnace
            e = E[:, ind, j, k]  # Closest error vector

            # Array offsets
            deltas = model.p2deltas(p)
            deltas_grad = model.deltas_grad()

            xk = deltas[0, k]
            yk = deltas[1, k]
            zk = deltas[2, k]

            # pre-computation
            s = np.sin(phi)
            c = np.cos(phi)
            sh = np.sinh(xj / a)
            ch = np.cosh(xj / a)
            ex = e[0, :]
            ey = e[1, :]
            ez = e[2, :]

            # Error Grad for each pts
            eT_de_dp = np.zeros((n_p, pts.shape[1]))

            eT_de_dp[0, :] = -ex
            eT_de_dp[1, :] = -ey
            eT_de_dp[2, :] = -ez
            eT_de_dp[3, :] = ex * ((xj + xk) * s + yk * c) + ey * (
                -(xj + xk) * c + yk * s
            )
            eT_de_dp[4, :] = ez * (1 + (xj / a) * sh - ch)

            # for all offset parameters
            for i_p in range(5, n_p):

                dxk_dp = deltas_grad[0, k, i_p - 5]
                dyk_dp = deltas_grad[1, k, i_p - 5]
                dzk_dp = deltas_grad[2, k, i_p - 5]

                eT_de_dp[i_p, :] = (
                    ex * (-c * dxk_dp + s * dyk_dp)
                    + ey * (-s * dxk_dp - c * dyk_dp)
                    + ez * (-dzk_dp)
                )

            # Norm grad
            dd_dp = eT_de_dp / d

        elif method == "x":

            ind = np.arange(0, m)

            # Array offsets
            r_k = model.p2deltas(p)

            # Transformation Matrix
            c_T_w = np.linalg.inv(catenary.w_T_c(phi, x0, y0, z0))

            # Compute measurements points positions in local cable frame
            r_w = np.vstack([pts, np.ones(m)])
            r_i = (c_T_w @ r_w)[0:3, :]

            # Catenary elevation at ref point
            x_j = r_i[0, :, np.newaxis] - r_k[0, np.newaxis, :]
            z_j = catenary.cat(x_j, a)

            # Reference model points
            r_j = np.zeros((3, m, model.q))

            r_j[0, :, :] = r_i[0, :, np.newaxis]
            r_j[1, :, :] = r_k[1, np.newaxis, :]
            r_j[2, :, :] = z_j + r_k[2, np.newaxis, :]

            # All error vectors
            E = r_i[:, :, np.newaxis] - r_j

            # Distances between measurements and model ref
            D = np.linalg.norm(E, axis=0)

            # Minimum distances to model for all measurements
            d = D.min(axis=1)
            k = D.argmin(axis=1)
            e = E[:, ind, k]

            # Array offsets
            deltas = model.p2deltas(p)
            deltas_grad = model.deltas_grad()

            xk = deltas[0, k]
            yk = deltas[1, k]
            zk = deltas[2, k]
            xj = x_j[ind, k]

            # pre-computation
            s = np.sin(phi)
            c = np.cos(phi)
            sh = np.sinh(xj / a)
            ch = np.cosh(xj / a)
            ex = e[0, :]
            ey = e[1, :]
            ez = e[2, :]

            xi = pts[0, :]
            yi = pts[1, :]

            dz_da = ch - xj / a * sh - 1
            dz_dxj = sh
            dxj_dx0 = -c
            dxj_dy0 = -s
            dxj_dpsi = -s * (xi - x0) + c * (yi - y0)
            dxj_dxk = -1

            # Error Grad for each pts
            eT_de_dp = np.zeros((n_p, pts.shape[1]))

            eT_de_dp[0, :] = ey * +s - ez * dz_dxj * dxj_dx0
            eT_de_dp[1, :] = ey * -c - ez * dz_dxj * dxj_dy0
            eT_de_dp[2, :] = ez * -1
            eT_de_dp[3, :] = (
                ey * (c * (x0 - xi) + s * (y0 - yi)) - ez * dz_dxj * dxj_dpsi
            )
            eT_de_dp[4, :] = ez * -dz_da

            # for all offset parameters
            for i_p in range(5, n_p):

                dxk_dp = deltas_grad[0, k, i_p - 5]
                dyk_dp = deltas_grad[1, k, i_p - 5]
                dzk_dp = deltas_grad[2, k, i_p - 5]

                eT_de_dp[i_p, :] = (
                    ex * (0.0 * dxk_dp + 0.0 * dyk_dp + 0.0 * dzk_dp)
                    + ey * (0.0 * dxk_dp - 1.0 * dyk_dp + 0.0 * dzk_dp)
                    + ez * (sh * dxk_dp - 0.0 * dyk_dp - 1.0 * dzk_dp)
                )

            # Norm grad
            dd_dp = eT_de_dp / d

        ################################

        # From distance to individual pts cost
        if not (l == 0):
            # Cost shaping function
            # dc_dd = power * d ** ( power - 1 ) / ( np.log( 10 ) * ( l + d ** power ) )
            dc_dd = lorentzian_grad(d, l, power, b)

        else:
            # No cost shaping
            dc_dd = 1.0

        # Smoothing grad
        dc_dp = dc_dd * dd_dp

        # Regulation
        p_e = p_nom - p

        # Total cost with regulation
        dJ_dp = R.T @ dc_dp.T - 2 * p_e.T @ Q

    return dJ_dp


# ###########################
# # Segmentation
# ###########################


############################
def get_catanery_group(p, pts, d_th=1.0, n_sample=1000, x_min=-200, x_max=200):

    # generate a list of sample point on the model curve
    r_model = p2r_w(p, x_min, x_max, n_sample)[0]

    # Minimum distances to model for all measurements
    d_min = compute_d_min(pts, r_model)

    # Group based on threshlod
    pts_in = pts[:, d_min < d_th]

    return pts_in


# ###########################
# Plots
# ###########################


############################
def plot_lorentzian(l=10, power=2, b=50.0, ax=None):
    """
    x : input
    a : flattening parameter
    """

    if ax is None:
        fig, ax = plt.subplots(2, figsize=(4, 3), dpi=300, frameon=True)

    x = np.linspace(-100.0, 100.0, 1000)
    y = lorentzian(x, l, power, b)

    ax[0].plot(x, y, label=r"$l =$ %0.1f  $p =$ %0.1f  $b =$ %0.1f" % (l, power, b))
    ax[0].set_xlabel("$d_i$", fontsize=8)
    ax[0].grid(True)
    # ax[0].legend()
    ax[0].tick_params(labelsize=8)
    ax[0].set_ylabel("$c_i$", fontsize=8)

    x = np.linspace(-1.0, 1.0, 1000)
    y = lorentzian(x, l, power, b)

    ax[1].plot(x, y, label=r"$l =$ %0.1f  $p =$ %0.1f  $b =$ %0.1f" % (l, power, b))
    ax[1].set_xlabel("$d_i$", fontsize=8)
    ax[1].grid(True)
    # ax[1].legend()
    ax[1].tick_params(labelsize=8)
    ax[1].set_ylabel("$c_i$", fontsize=8)

    try:
        fig.tight_layout()
    except:
        pass

    return ax


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    ax = plot_lorentzian(l=1.0, power=2.0, b=25.0)

    plot_lorentzian(l=1.0, power=2, b=2.0, ax=ax)

    plot_lorentzian(l=1.0, power=2, b=2.0, ax=ax)
