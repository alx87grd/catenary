#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:46:33 2023

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time


from catenary.kinematic import singleline as catenary
from catenary.kinematic import powerline
from catenary.estimation import costfunction as cf
from catenary.estimation.estimator import ArrayEstimator


###########################


###############################################################################
class ErrorPlot:
    """ """

    ############################
    def __init__(self, p_true, p_hat, n_steps, n_run=1):

        self.n_p = p_true.shape[0]
        self.n_steps = n_steps
        self.n_run = n_run

        self.p_true = p_true

        self.PE = np.zeros((self.n_p, n_steps + 1, n_run))

        self.t = np.zeros((n_steps, n_run))
        self.n_in = np.zeros((n_steps, n_run))

        self.step = 0
        self.run = 0

        self.PE[:, self.step, self.run] = p_true - p_hat

    ############################
    def init_new_run(self, p_true, p_hat):

        self.p_true = p_true

        self.step = 0
        self.run = self.run + 1

        self.PE[:, self.step, self.run] = self.p_true - p_hat

    ############################
    def save_new_estimation(self, p_hat, t_solve, n_in=0):

        self.t[self.step, self.run] = t_solve
        self.n_in[self.step, self.run] = n_in

        self.step = self.step + 1

        self.PE[:, self.step, self.run] = self.p_true - p_hat

    ############################
    def plot_error_all_run(self, fs=10, save=False, name="test"):

        PE = self.PE
        t = self.t

        frame = np.linspace(0, self.n_steps, self.n_steps + 1)

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(self.n_run):
            ax.plot(frame, PE[0, :, j], "--k")
            ax.plot(frame, PE[1, :, j], "--r")
            ax.plot(frame, PE[2, :, j], "--b")

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("[m]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_translation_error.pdf")
            # fig.savefig( name + '_translation_error.png')
            # fig.savefig( name + '_translation_error.jpg')

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(self.n_run):
            ax.plot(frame, PE[3, :, j])

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$\psi$ [rad]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_orientation_error.pdf")
            # fig.savefig( name + '_orientation_error.png')
            # fig.savefig( name + '_orientation_error.jpg')

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(self.n_run):
            ax.plot(frame, PE[4, :, j], "--b")

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("[m]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_sag_error.pdf")
            # fig.savefig( name + '_sag_error.png')
            # fig.savefig( name + '_sag_error.jpg')

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        l_n = PE.shape[0] - 5

        for i in range(l_n):
            for j in range(self.n_run):
                ax.plot(frame, PE[5 + i, :, j])

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("[m]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_internaloffsets_error.pdf")
            # fig.savefig( name + '_internaloffsets_error.png')
            # fig.savefig( name + '_internaloffsets_error.jpg')

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(self.n_run):
            ax.plot(frame[1:], t[:, j])

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("[sec]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_solver_time.pdf")
            # fig.savefig( name + '_solver_time.png')
            # fig.savefig( name + '_solver_time.jpg')

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(self.n_run):
            ax.plot(frame[1:], self.n_in[:, j])

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$n_{in}[\%]$", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_nin.pdf")
            # fig.savefig( name + '_nin.png')
            # fig.savefig( name + '_nin.jpg')

    ############################
    def plot_error_mean_std(self, fs=10, save=False, name="test", n_run_plot=10):

        PE = self.PE
        t = self.t
        n_in = self.n_in

        if n_run_plot > self.n_run:
            n_run_plot = self.n_run

        PE_mean = np.mean(PE, axis=2)
        PE_std = np.std(PE, axis=2)
        t_mean = np.mean(t, axis=1)
        t_std = np.std(t, axis=1)
        n_mean = np.mean(n_in, axis=1)
        n_std = np.std(n_in, axis=1)

        frame = np.linspace(0, self.n_steps, self.n_steps + 1)

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(n_run_plot):
            ax.plot(
                frame,
                (PE[0, :, j] + PE[1, :, j] + PE[3, :, j]) / 3,
                "--k",
                linewidth=0.25,
            )

        ax.plot(frame, (PE_mean[0, :] + PE_mean[1, :] + PE_mean[1, :]) / 3, "-r")
        ax.fill_between(
            frame,
            PE_mean[0, :] - PE_std[0, :],
            PE_mean[0, :] + PE_std[0, :],
            color="#DDDDDD",
        )
        ax.fill_between(
            frame,
            PE_mean[1, :] - PE_std[1, :],
            PE_mean[1, :] + PE_std[1, :],
            color="#DDDDDD",
        )
        ax.fill_between(
            frame,
            PE_mean[2, :] - PE_std[2, :],
            PE_mean[2, :] + PE_std[2, :],
            color="#DDDDDD",
        )

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$(x_o,y_o,z_o)$[m]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_translation_error.pdf")

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(n_run_plot):
            ax.plot(frame, PE[3, :, j], "--k", linewidth=0.25)
        ax.plot(frame, PE_mean[3, :], "-r")
        ax.fill_between(
            frame,
            PE_mean[3, :] - PE_std[3, :],
            PE_mean[3, :] + PE_std[3, :],
            color="#DDDDDD",
        )

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$\psi$ [rad]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_orientation_error.pdf")

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(n_run_plot):
            ax.plot(frame, PE[4, :, j], "--k", linewidth=0.25)
        ax.plot(frame, PE_mean[4, :], "-r")
        ax.fill_between(
            frame,
            PE_mean[4, :] - PE_std[4, :],
            PE_mean[4, :] + PE_std[4, :],
            color="#DDDDDD",
        )

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$a$[m]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_sag_error.pdf")

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        l_n = PE.shape[0] - 5

        for i in range(l_n):
            k = 5 + i
            for j in range(n_run_plot):
                ax.plot(frame, PE[k, :, j], "--k", linewidth=0.25)
            ax.plot(frame, PE_mean[k, :], "-r")
            ax.fill_between(
                frame,
                PE_mean[k, :] - PE_std[k, :],
                PE_mean[k, :] + PE_std[k, :],
                color="#DDDDDD",
            )

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$\Delta$[m]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_internaloffsets_error.pdf")

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(n_run_plot):
            ax.plot(frame[1:], t[:, j], "--k", linewidth=0.25)
        ax.plot(frame[1:], t_mean, "-r")
        ax.fill_between(frame[1:], t_mean - t_std, t_mean + t_std, color="#DDDDDD")

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$\Delta t$[sec]", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_solver_time.pdf")

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=300, frameon=True)

        for j in range(n_run_plot):
            ax.plot(frame[1:], n_in[:, j], "--k", linewidth=0.25)
        ax.plot(frame[1:], n_mean[:], "-r")
        ax.fill_between(
            frame[1:], n_mean[:] - n_std[:], n_mean[:] + n_std[:], color="#DDDDDD"
        )

        # ax.legend( loc = 'upper right' , fontsize = fs)
        ax.set_xlabel("steps", fontsize=fs)
        ax.set_ylabel("$n_{in}[\%]$", fontsize=fs)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(name + "_nin.pdf")


###########################
# Powerline Model
###########################


############################
def basic_array32_estimator_test(n_steps=50):

    model = powerline.ArrayModel32()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 30.0, 50.0])
    p_hat = np.array([100, 100, 100, 1.0, 300, 49.0, 29.0, 49])

    pts = model.generate_test_data(p, partial_obs=True)

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)
    plot2 = ErrorPlot(p, p_hat, n_steps)

    estimator = ArrayEstimator(model, p_hat)

    estimator.Q = 10 * np.diag(
        [0.0002, 0.0002, 0.0002, 0.001, 0.0001, 0.002, 0.002, 0.002]
    )

    for i in range(n_steps):

        pts = model.generate_test_data(p, partial_obs=True)

        plot.update_pts(pts)

        start_time = time.time()
        p_hat = estimator.solve(pts, p_hat)
        solve_time = time.time() - start_time

        target = estimator.is_target_aquired(p_hat, pts)

        plot.update_estimation(p_hat)
        plot2.save_new_estimation(p_hat, solve_time)

        print(
            " Solve time : "
            + str(solve_time)
            + "\n"
            + " Target acquired: "
            + str(target)
            + "\n"
            + f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n"
            + f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )

    plot2.plot_error_mean_std()

    return estimator


############################
def basic_array_constant2221_estimator_test(n=5, var=5.0):

    model = powerline.ArrayModelConstant2221()

    p = np.array([50, 50, 50, 1.3, 600, 4.0, 5.0, 6.0])
    p_hat = np.array([0, 0, 0, 1.0, 800, 3.0, 4.0, 7.0])

    pts = model.generate_test_data(p, partial_obs=False)

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

    estimator = ArrayEstimator(model, p_hat)

    estimator.Q = 10 * np.diag(
        [0.0002, 0.0002, 0.0002, 0.001, 0.000001, 0.002, 0.002, 0.002]
    )
    estimator.n_search = n
    estimator.p_var = np.array([var, var, var, 0, 0, 0, 0, 0])

    for i in range(100):

        pts = model.generate_test_data(p, partial_obs=True)

        plot.update_pts(pts)

        start_time = time.time()
        p_hat = estimator.solve_with_search(pts, p_hat)
        solve_time = time.time() - start_time

        target = estimator.is_target_aquired(p_hat, pts)

        plot.update_estimation(p_hat)

        print(
            " Solve time : "
            + str(solve_time)
            + "\n"
            + " Target acquired: "
            + str(target)
            + "\n"
            + f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n"
            + f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )

    return estimator


############################
def hard_array_constant2221_estimator_test(n=2, var=5.0):

    model = powerline.ArrayModelConstant2221()

    p = np.array([15, -20, 50, 0.3, 600, 5.0, 5.0, 6.0])
    p_hat = np.array([0, 0, 0, 0.0, 800, 4.0, 4.0, 7.0])

    pts = model.generate_test_data(
        p,
        n_obs=10,
        x_min=-50,
        x_max=-40,
        w_l=0.5,
        n_out=3,
        center=[0, 0, 0],
        w_o=10,
        partial_obs=False,
    )

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

    estimator = ArrayEstimator(model, p_hat)

    estimator.Q = 10 * np.diag(
        [0.0002, 0.0002, 0.0002, 0.001, 0.000001, 0.002, 0.002, 0.002]
    )
    estimator.n_search = n
    estimator.p_var = np.array([var, var, var, 0, 0, 0, 0, 0])

    for i in range(100):

        pts = model.generate_test_data(
            p,
            n_obs=6,
            x_min=-50,
            x_max=-40,
            w_l=0.5,
            n_out=3,
            center=[0, 0, 0],
            w_o=10,
            partial_obs=False,
        )

        plot.update_pts(pts)

        start_time = time.time()
        p_hat = estimator.solve_with_search(pts, p_hat)
        solve_time = time.time() - start_time

        target = estimator.is_target_aquired(p_hat, pts)

        plot.update_estimation(p_hat)

        print(
            " Solve time : "
            + str(solve_time)
            + "\n"
            + " Target acquired: "
            + str(target)
            + "\n"
            + f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n"
            + f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )

    return estimator


############################
def translation_search_test(search=True, n=3, var=10.0):

    model = powerline.ArrayModel32()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 25.0, 50.0])
    p_hat = np.array([0, 0, 150, 1.2, 500, 51.0, 25.0, 49])

    pts = model.generate_test_data(
        p,
        partial_obs=True,
        n_obs=16,
        x_min=-100,
        x_max=-50,
        n_out=5,
        center=[0, 0, 0],
        w_o=20,
    )

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

    estimator = ArrayEstimator(model, p_hat)

    # estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.000002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    estimator.Q = 10 * np.diag(
        [0.0002, 0.0002, 0.0, 0.001, 0.0001, 0.002, 0.002, 0.002]
    )
    estimator.n_search = n
    estimator.p_var = np.array([var, var, var, 0, 0, 0, 0, 0])

    for i in range(50):

        pts = model.generate_test_data(
            p,
            partial_obs=True,
            n_obs=16,
            x_min=-100,
            x_max=-70,
            n_out=5,
            center=[-50, -50, -50],
            w_o=10,
        )

        plot.update_pts(pts)

        if search:
            p_hat = estimator.solve_with_search(pts, p_hat)

        else:
            p_hat = estimator.solve(pts, p_hat)

        target = estimator.is_target_aquired(p_hat, pts)

        plot.update_estimation(p_hat)

        print(
            " Target acquired: "
            + str(target)
            + "\n"
            + f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n"
            + f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )

    return estimator


############################
def hard_test(search=True, method="x", n=2, var=10, n_steps=50):

    model = powerline.ArrayModel32()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 25.0, 50.0])
    p_hat = np.array([0, 0, 150, 1.2, 500, 51.0, 26.0, 49])

    pts = model.generate_test_data(
        p,
        partial_obs=True,
        n_obs=16,
        w_l=0.2,
        x_min=-100,
        x_max=-50,
        n_out=3,
        center=[0, 0, 0],
        w_o=20,
    )

    pts = pts[:, :30]  # remover one cable

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)
    plot2 = ErrorPlot(p, p_hat, n_steps)
    plot.plot_model(p_hat)

    estimator = ArrayEstimator(model, p_hat)

    # estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.000002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    estimator.Q = 0 * np.diag([0.0002, 0.0002, 0.0, 0.001, 0.0001, 0.002, 0.002, 0.002])

    estimator.n_search = n
    estimator.p_var = np.array([var, var, var, 0, 0, 0, 0, 0])

    estimator.d_th = 3.0
    estimator.succes_ratio = 0.7
    estimator.method = method

    for i in range(n_steps):

        pts = model.generate_test_data(
            p,
            partial_obs=True,
            n_obs=16,
            x_min=-100,
            x_max=-70,
            n_out=5,
            center=[-50, -50, -50],
            w_o=10,
        )

        pts = pts[:, :30]  # remover one cable

        plot.update_pts(pts)

        start_time = time.time()
        if search:
            p_hat = estimator.solve_with_search(pts, p_hat)

        else:
            p_hat = estimator.solve(pts, p_hat)

        solve_time = time.time() - start_time

        target = estimator.is_target_aquired(p_hat, pts)

        plot.update_estimation(p_hat)
        plot2.save_new_estimation(p_hat, solve_time)

        print(
            " Solve time : "
            + str(solve_time)
            + "\n"
            + " Target acquired: "
            + str(target)
            + "\n"
            + f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n"
            + f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )

    plot2.plot_error_mean_std()

    return estimator


############################
def very_hard_test(search=True, method="x", n=5, var=100):

    model = powerline.ArrayModel32()

    p = np.array([50, 50, 50, 0.0, 200, 50.0, 25.0, 150.0])
    p_hat = np.array([0, 0, 150, 0.2, 300, 51.0, 26.0, 149])

    ps = model.p2ps(p)

    ps[4, 3] = 2000
    ps[4, 4] = 2000

    pts0 = catenary.generate_test_data(ps[:, 0], n_obs=10, n_out=2, x_min=-50, x_max=50)
    pts1 = catenary.generate_test_data(ps[:, 1], n_obs=10, n_out=2, x_min=-50, x_max=50)
    pts2 = catenary.generate_test_data(ps[:, 2], n_obs=10, n_out=2, x_min=-50, x_max=50)
    pts3 = catenary.generate_test_data(ps[:, 3], n_obs=10, n_out=2, x_min=-50, x_max=50)
    pts4 = catenary.generate_test_data(ps[:, 4], n_obs=10, n_out=2, x_min=-50, x_max=50)

    # pts = np.hstack( ( pts0 , pts1 , pts2 , pts3 , pts4 ))

    pts = np.hstack((pts1, pts3, pts4))

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)

    estimator = ArrayEstimator(model, p_hat)

    # estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.000002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    estimator.Q = 10 * np.diag(
        [0.0002, 0.0002, 0.0, 0.001, 0.0001, 0.002, 0.002, 0.002]
    )

    estimator.n_search = n
    estimator.p_var = np.array([var, var, var, 0, 0, 0, 0, 0])

    estimator.d_th = 5.0
    estimator.succes_ratio = 0.5
    estimator.method = method

    for i in range(50):

        pts0 = catenary.generate_test_data(
            ps[:, 0], n_obs=10, n_out=2, x_min=-50, x_max=50
        )
        pts1 = catenary.generate_test_data(
            ps[:, 1], n_obs=7, n_out=2, x_min=-50, x_max=50
        )
        pts2 = catenary.generate_test_data(
            ps[:, 2], n_obs=10, n_out=2, x_min=-50, x_max=50
        )
        pts3 = catenary.generate_test_data(
            ps[:, 3], n_obs=6, n_out=2, x_min=-50, x_max=50
        )
        pts4 = catenary.generate_test_data(
            ps[:, 4], n_obs=3, n_out=2, x_min=-30, x_max=20
        )

        # pts = np.hstack( ( pts0 , pts1 , pts2 , pts3 , pts4 ))

        pts = np.hstack((pts1, pts3, pts4))

        plot.update_pts(pts)

        start_time = time.time()

        if search:
            p_hat = estimator.solve_with_search(pts, p_hat)

        else:
            p_hat = estimator.solve(pts, p_hat)

        solve_time = time.time() - start_time

        target = estimator.is_target_aquired(p_hat, pts)

        plot.update_estimation(p_hat)

        print(
            " Solve time : "
            + str(solve_time)
            + "\n"
            + " Target acquired: "
            + str(target)
            + "\n"
            + f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n"
            + f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )

    return estimator


############################
def quad_test(search=True, method="x", n=5, var=10):

    # 3 x quad powerlines
    quad = powerline.Quad()

    p4_1 = np.array([50, 40, 50, 0.0, 500, 0.2, 0.4])
    p4_2 = np.array([50, 50, 50, 0.0, 500, 0.2, 0.4])
    p4_3 = np.array([50, 60, 50, 0.0, 500, 0.2, 0.4])

    pts4_1 = quad.generate_test_data(
        p4_1, n_obs=16, w_l=0.05, n_out=2, x_min=+80, x_max=100
    )
    pts4_2 = quad.generate_test_data(
        p4_2, n_obs=16, w_l=0.05, n_out=2, x_min=+80, x_max=100
    )
    pts4_3 = quad.generate_test_data(
        p4_3, n_obs=16, w_l=0.05, n_out=2, x_min=+80, x_max=100
    )

    # 2x guard cables
    pg1 = np.array([50, 45, 60, 0.0, 800])
    pg2 = np.array([50, 55, 60, 0.0, 800])

    pts_g1 = catenary.generate_test_data(pg1, n_obs=10, n_out=2, x_min=+80, x_max=100)
    pts_g2 = catenary.generate_test_data(pg2, n_obs=10, n_out=2, x_min=+80, x_max=100)

    pts = np.hstack((pts4_1, pts4_2, pts4_3, pts_g1, pts_g2))

    #  Estimation Model
    model = powerline.ArrayModel32()

    p_true = np.array([50, 50, 50, 0.0, 500, 10.0, 5.0, 10.0])
    p_hat = np.array([0, 0, 0, 0.3, 800, 9.0, 4.0, 9.0])

    plot = powerline.EstimationPlot(p_true, p_hat, pts, model.p2r_w)

    callback = None  # plot.update_estimation

    estimator = ArrayEstimator(model, p_hat)

    estimator.Q = 10 * np.diag(
        [0.0002, 0.0002, 0.0, 0.001, 0.0001, 0.002, 0.002, 0.002]
    )

    estimator.n_search = n
    estimator.p_var = np.array([var, var, var, 0, 0, 0, 0, 0])

    estimator.method = method

    for i in range(50):

        # 3 x quad powerlines
        quad = powerline.Quad()

        p4_1 = np.array([50, 40, 50, 0.0, 500, 0.2, 0.4])
        p4_2 = np.array([50, 50, 50, 0.0, 500, 0.2, 0.4])
        p4_3 = np.array([50, 60, 50, 0.0, 500, 0.2, 0.4])

        pts4_1 = quad.generate_test_data(
            p4_1, n_obs=16, w_l=0.05, n_out=2, x_min=+80, x_max=100
        )
        pts4_2 = quad.generate_test_data(
            p4_2, n_obs=16, w_l=0.05, n_out=2, x_min=+80, x_max=100
        )
        pts4_3 = quad.generate_test_data(
            p4_3, n_obs=16, w_l=0.05, n_out=2, x_min=+80, x_max=100
        )

        # 2x guard cables
        pg1 = np.array([50, 45, 60, 0.0, 800])
        pg2 = np.array([50, 55, 60, 0.0, 800])

        pts_g1 = catenary.generate_test_data(
            pg1, n_obs=10, n_out=2, x_min=+80, x_max=100
        )
        pts_g2 = catenary.generate_test_data(
            pg2, n_obs=10, n_out=2, x_min=+80, x_max=100
        )

        pts = np.hstack((pts4_1, pts4_2, pts4_3, pts_g1, pts_g2))

        plot.update_pts(pts)

        start_time = time.time()

        if search:
            p_hat = estimator.solve_with_search(pts, p_hat, callback)

        else:
            p_hat = estimator.solve(pts, p_hat, callback)

        solve_time = time.time() - start_time

        target = estimator.is_target_aquired(p_hat, pts)

        plot.update_estimation(p_hat)

        print(
            " Solve time : "
            + str(solve_time)
            + "\n"
            + " Target acquired: "
            + str(target)
            + "\n"
            + f" p_true : {np.array2string(p_true, precision=2, floatmode='fixed')}  \n"
            + f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"
        )


############################
def global_convergence_test(n_steps=100):

    xm = -200
    xp = 200

    n_out = 500

    model = powerline.ArrayModel32()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 30.0, 50.0])
    p_hat = np.array([100, 100, 100, 0.6, 300, 49.0, 29.0, 49])

    pts = model.generate_test_data(p, partial_obs=True)

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)
    plot2 = ErrorPlot(p, p_hat, n_steps)

    estimator = ArrayEstimator(model, p_hat)

    estimator.Q = 1.0 * np.diag(
        [0.0002, 0.0002, 0.0002, 0.01, 0.00001, 0.002, 0.002, 0.002]
    )

    estimator.n_search = 3
    estimator.p_var = np.array([50.0, 50.0, 50.0, 0, 0, 0, 0, 0])

    estimator.b = 200

    plot.plot_model(p_hat)

    for i in range(n_steps):

        pts = model.generate_test_data(
            p,
            partial_obs=True,
            x_min=xm,
            x_max=xp,
            n_out=n_out,
            w_o=50.0,
            center=[0, 0, -200],
        )

        plot.update_pts(pts)

        start_time = time.time()
        p_hat = estimator.solve_with_search(pts, p_hat)
        solve_time = time.time() - start_time

        plot.update_estimation(p_hat)

        n_tot = pts.shape[1] - n_out
        pts_in = estimator.get_array_group(p_hat, pts)
        n_in = pts_in.shape[1] / n_tot * 100.0

        plot2.save_new_estimation(p_hat, solve_time, n_in)

    plot2.plot_error_mean_std()


############################
def ransac_test(n_steps=100):

    xm = -200
    xp = 200

    n_obs = 20
    n_out = 500

    model = powerline.ArrayModel32()

    p = np.array([50, 50, 50, 1.0, 600, 50.0, 30.0, 50.0])
    p_hat = np.array([100, 100, 25, 0.6, 300, 49.0, 29.0, 49])

    pts = model.generate_test_data(p, partial_obs=True)

    plot = powerline.EstimationPlot(p, p_hat, pts, model.p2r_w)
    plot2 = ErrorPlot(p, p_hat, n_steps)

    estimator = ArrayEstimator(model, p_hat)

    estimator.Q = 1.0 * np.diag(
        [0.0002, 0.0002, 0.0002, 0.01, 0.00001, 0.002, 0.002, 0.002]
    )

    estimator.n_search = 3
    estimator.p_var = np.array([50.0, 50.0, 50.0, 0, 0, 0, 0, 0])

    estimator.b = 200

    plot.plot_model(p_hat)

    for i in range(n_steps):

        pts = model.generate_test_data(
            p,
            partial_obs=True,
            x_min=xm,
            x_max=xp,
            n_obs=n_obs,
            n_out=n_out,
            w_o=200.0,
            center=[0, 0, -200],
        )

        plot.update_pts(pts)

        start_time = time.time()
        p_hat = estimator.solve_with_ransac_search(pts, p_hat, n_iter=10, n_pts=50)
        # p_hat = estimator.solve_with_search(pts, p_hat)
        solve_time = time.time() - start_time

        plot.update_estimation(p_hat)

        n_tot = pts.shape[1] - n_out
        pts_in = estimator.get_array_group(p_hat, pts)
        n_in = pts_in.shape[1] / n_tot * 100.0

        plot2.save_new_estimation(p_hat, solve_time, n_in)

    plot2.plot_error_mean_std()


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    # basic_array32_estimator_test(100)

    # basic_array_constant2221_estimator_test()
    # hard_array_constant2221_estimator_test()

    translation_search_test(False)
    translation_search_test(True)

    # hard_test( method = 'sample' )
    # hard_test( method = 'x' , n_steps = 100 )

    # very_hard_test()

    # quad_test()

    global_convergence_test()

    # ransac_test()
