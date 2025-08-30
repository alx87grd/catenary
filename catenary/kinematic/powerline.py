#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:03:21 2023

@author: alex
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


###########################
# catenary function
###########################

from catenary import tools
from catenary.kinematic import singleline as catenary

# from mpl_toolkits.mplot3d import Axes3D


###########################
# Powerline Model
###########################


###############################################################################
class ArrayModel:
    """
    A class wraping up usefull functions related to model of array of catenary
    line where each line as the same sag parameter and the same orientation

    """

    #################################################
    def __init__(self, l=6, q=3):
        """
        l : int
            The number of model parameters
        q : int
            The number of discret lines in the array

        """

        self.q = q  # number of lines
        self.l = l  # number of model parameters

    ############################
    def p2deltas(self, p):
        """
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame

        this is a model for 3 catenary with  equal horizontal d1 offset

        ----------------------------------------------------------


                    0            1             2

                                      d1
                                |------------->

        ----------------------------------------------------------
        """

        d1 = p[5]

        delta = np.array([[0.0, 0.0, 0.0], [-d1, 0.0, +d1], [0.0, 0.0, 0.0]])

        return delta

    ############################
    def deltas_grad(self):
        """
        Compute the gradient of deltas with respect to offset parameters
        """

        grad = np.zeros((3, self.q, (self.l - 5)))

        grad[:, :, 0] = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, +1.0], [0.0, 0.0, 0.0]])

        return grad

    ############################
    def p2r_w(
        self,
        p,
        x_min=-200,
        x_max=200,
        n=400,
    ):
        """
        Compute n pts coord in world frame based on a parameter vector p



        inputs
        --------
        p      : vector of parameters

            x_0 : x translation of local frame orign in world frame
            y_0 : y translation of local frame orign in world frame
            z_0 : z translation of local frame orign in world frame
            phi : z rotation of local frame basis in in world frame
            a   : sag parameter
            d1  : horizontal distance between power lines

        x_min  : start of points in cable local frame
        x_max  : end   of points in cable local frame
        n      : number of pts

        outputs
        ----------
        r_w_flat : dim (3,n * q)  all world pts
        r_w      : dim (3,n,q)    all world pts splitted by line id
        x_c      : dim (n)        array of x coord in catenary frame

        """

        x_c = np.linspace(x_min, x_max, n)

        # params
        x_0 = p[0]
        y_0 = p[1]
        z_0 = p[2]
        phi = p[3]
        a = p[4]

        # catenary frame z
        z_c = catenary.cat(x_c, a)

        # Offset in local catenary frame
        delta = self.p2deltas(p)

        r_c = np.zeros((4, n, self.q))
        r_w = np.zeros((4, n, self.q))

        # Foward kinematic for all lines in the array
        for i in range(self.q):

            r_c[0, :, i] = x_c + delta[0, i]
            r_c[1, :, i] = 0.0 + delta[1, i]
            r_c[2, :, i] = z_c + delta[2, i]
            r_c[3, :, i] = np.ones((n))

            r_w[:, :, i] = catenary.w_T_c(phi, x_0, y_0, z_0) @ r_c[:, :, i]

        r_w_flat = r_w.reshape((4, n * self.q), order="F")

        return (r_w_flat[0:3, :], r_w[0:3, :, :], x_c)

    ############################
    def flat2line(self, r):
        """split a list of pts by line"""

        return r.reshape((3, -1, self.q), order="F")

    ############################
    def line2flat(self, r):
        """flatten the list of pts by line"""

        return r.reshape((3, -1), order="F")

    ############################
    def p2ps(self, p):
        """
        Input: model parameter vector  ( l x 1 array )
        Ouput: list of q catenary parameter vector ( 5 x q array )

        """

        # params
        x_0 = p[0]
        y_0 = p[1]
        z_0 = p[2]
        phi = p[3]
        a = p[4]

        # Offset in local catenary frame
        delta = self.p2deltas(p)

        ps = np.zeros((5, self.q))

        for i in range(self.q):

            r0_c = np.hstack((delta[:, i], np.array([1.0])))

            r0_w = catenary.w_T_c(phi, x_0, y_0, z_0) @ r0_c

            ps[0, i] = r0_w[0]
            ps[1, i] = r0_w[1]
            ps[2, i] = r0_w[2]
            ps[3, i] = phi
            ps[4, i] = a

        return ps

    ############################
    def generate_test_data(
        self,
        p,
        n_obs=20,
        x_min=-200,
        x_max=200,
        w_l=0.5,
        n_out=10,
        center=[0, 0, 0],
        w_o=100,
        partial_obs=False,
        seed=None,
    ):
        """
        generate pts for a line and outliers

        """

        # outliers
        pts = catenary.outliers(n_out, center, w_o)

        # Individual catenary parameters
        ps = self.p2ps(p)

        for i in range(self.q):

            p_line = ps[:, i]  # parameter vector of ith line

            if partial_obs:

                xn = np.random.randint(1, n_obs)
                xm = np.random.randint(x_min, x_max)
                xp = np.random.randint(x_min, x_max)

                r_line = catenary.noisy_p2r_w(p_line, xm, xp, xn, w_l, seed)

            else:

                r_line = catenary.noisy_p2r_w(p_line, x_min, x_max, n_obs, w_l, seed)

            pts = np.append(pts, r_line, axis=1)

        return pts


###############################################################################
class ArrayModel32(ArrayModel):
    """
    ArrayModel 32 is a model for 5 catenary with 3 offsets variables

    ----------------------------------------------------------

                                d2
                             |---->
     ^                 3          4
     |
     h
     |
     _          0            1             2

                                  d1
                            |------------->

    ----------------------------------------------------------

    p      :  8 x 1 array of parameters

        x_0 : x translation of local frame orign in world frame
        y_0 : y translation of local frame orign in world frame
        z_0 : z translation of local frame orign in world frame
        phi : z rotation of local frame basis in in world frame
        a   : sag parameter
        d1  : horizontal distance between power lines
        d2  : horizontal distance between guard cable
        h   : vertical distance between power lines and guard cables

    """

    #################################################
    def __init__(self):

        ArrayModel.__init__(self, l=8, q=5)

    ############################
    def p2deltas(self, p):
        """
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame
        """

        delta = np.zeros((3, self.q))

        d1 = p[5]
        d2 = p[6]
        h = p[7]

        # Offset in local catenary frame

        delta[1, 0] = -d1  # y offset of cable 0
        delta[1, 2] = +d1  # y offset of cable 2
        delta[1, 3] = -d2  # y offset of cable 3
        delta[1, 4] = +d2  # y offset of cable 4

        delta[2, 3] = +h  # z offset of cable 3
        delta[2, 4] = +h  # z offset of cable 4

        return delta

    ############################
    def deltas_grad(self):
        """
        Compute the gradient of deltas with respect to offset parameters
        """

        grad = np.zeros((3, self.q, (self.l - 5)))

        grad[:, :, 0] = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, +1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        grad[:, :, 1] = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, +1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        grad[:, :, 2] = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, +1.0, +1.0],
            ]
        )

        return grad


###############################################################################


###############################################################################
class ArrayModel2(ArrayModel):
    """
    ArrayModel 2 is a model for 2 catenary with 1 offsets variables

    ----------------------------------------------------------


                           0             1

                                  d1
                            |------------->

    ----------------------------------------------------------

    p      :  6 x 1 array of parameters

        x_0 : x translation of local frame orign in world frame
        y_0 : y translation of local frame orign in world frame
        z_0 : z translation of local frame orign in world frame
        phi : z rotation of local frame basis in in world frame
        a   : sag parameter
        d1  : horizontal distance between power lines

    """

    #################################################
    def __init__(self):

        ArrayModel.__init__(self, l=6, q=2)

    ############################
    def p2deltas(self, p):
        """
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame
        """

        delta = np.zeros((3, self.q))

        d1 = p[5]

        # Offset in local catenary frame

        delta[1, 1] = +d1  # y offset of cable 2

        return delta

    ############################
    def deltas_grad(self):
        """
        Compute the gradient of deltas with respect to offset parameters
        """

        grad = np.zeros((3, self.q, (self.l - 5)))

        grad[:, :, 0] = np.array(
            [
                [0.0, 0.0],
                [0.0, +1.0],
                [0.0, 0.0],
            ]
        )

        return grad


###############################################################################
class ArrayModel2221(ArrayModel):
    """
    ArrayModel 32 is a model for 7 catenary with 6 offsets variables

    inputs
    --------
    p      : vector of parameters

        x_0 : x translation of local frame orign in world frame
        y_0 : y translation of local frame orign in world frame
        z_0 : z translation of local frame orign in world frame
        phi : z rotation of local frame basis in in world frame
        a   : sag parameter
        d1  : horizontal distance between power lines
        d2  : horizontal distance between power lines
        d3  : horizontal distance between power lines
        h1  : vertical distance between power lines
        h2  : vertical distance between power lines
        h3  : vertical distance between power lines

    ----------------------------------------------------------

     ^                       6
     |
     h3
     |
     _          4                 d3     5
     ^                       |----------->
     |
     h2
     |
     _     2                                     3
     ^                                 d2
     |                       |------------------->
     h1
     |
     _         0                           1

                                  d1
                            |------------->


    ----------------------------------------------------------

    """

    #################################################
    def __init__(self):

        ArrayModel.__init__(self, l=11, q=7)

    ############################
    def p2deltas(self, p):
        """
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame
        """

        delta = np.zeros((3, self.q))

        d1 = p[5]
        d2 = p[6]
        d3 = p[7]
        h1 = p[8]
        h2 = p[9]
        h3 = p[10]

        # Offset in local catenary frame
        delta[1, 0] = -d1  # y offset of cable 0
        delta[1, 1] = +d1  # y offset of cable 1
        delta[1, 2] = -d2  # y offset of cable 2
        delta[1, 3] = +d2  # y offset of cable 3
        delta[1, 4] = -d3  # y offset of cable 4
        delta[1, 5] = +d3  # y offset of cable 5

        delta[2, 2] = +h1  # z offset of cable 2
        delta[2, 3] = +h1  # z offset of cable 3
        delta[2, 4] = +h1 + h2  # z offset of cable 4
        delta[2, 5] = +h1 + h2  # z offset of cable 5
        delta[2, 6] = +h1 + h2 + h3  # z offset of cable 2

        return delta

    ############################
    def deltas_grad(self):
        """
        Compute the gradient of deltas with respect to offset parameters
        """

        grad = np.zeros((3, self.q, (self.l - 5)))

        grad[:, :, 0] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        grad[:, :, 1] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, +1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        grad[:, :, 2] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, +1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        grad[:, :, 3] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, +1.0, +1.0, +1.0, +1.0, +1.0],
            ]
        )

        grad[:, :, 4] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, +1.0, +1.0, +1.0],
            ]
        )

        grad[:, :, 5] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +1.0],
            ]
        )

        return grad


###############################################################################
class ArrayModelConstant2221(ArrayModel):
    """
    ArrayModel 32 is a model for 7 catenary with 6 offsets variables

    ----------------------------------------------------------

     ^                       6
     |
     h3
     |
     _          4                 d1       5
     ^                       |------------->
     |
     h1
     |
     _          2                          3
     ^                                 d1
     |                       |------------>
     h1
     |
     _         0                           1

                                  d1
                            |------------->


    ----------------------------------------------------------

    inputs
    --------
    p      : vector of parameters

        x_0 : x translation of local frame orign in world frame
        y_0 : y translation of local frame orign in world frame
        z_0 : z translation of local frame orign in world frame
        phi : z rotation of local frame basis in in world frame
        a   : sag parameter
        d1  : horizontal distance between power lines
        h1  : vertical distance between power lines
        h3  : vertical distance between power lines


    ----------------------------------------------------------

    """

    #################################################
    def __init__(self):

        ArrayModel.__init__(self, l=8, q=7)

    ############################
    def p2deltas(self, p):
        """
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame
        """

        delta = np.zeros((3, self.q))

        d1 = p[5]
        h1 = p[6]
        h3 = p[7]

        # Offset in local catenary frame
        delta[1, 0] = -d1  # y offset of cable 0
        delta[1, 1] = +d1  # y offset of cable 1
        delta[1, 2] = -d1  # y offset of cable 2
        delta[1, 3] = +d1  # y offset of cable 3
        delta[1, 4] = -d1  # y offset of cable 4
        delta[1, 5] = +d1  # y offset of cable 5

        delta[2, 2] = +h1  # z offset of cable 2
        delta[2, 3] = +h1  # z offset of cable 3
        delta[2, 4] = +h1 + h1  # z offset of cable 4
        delta[2, 5] = +h1 + h1  # z offset of cable 5
        delta[2, 6] = +h1 + h1 + h3  # z offset of cable 2

        return delta

    ############################
    def deltas_grad(self):
        """
        Compute the gradient of deltas with respect to offset parameters
        """

        grad = np.zeros((3, self.q, (self.l - 5)))

        grad[:, :, 0] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, +1.0, -1.0, +1.0, -1.0, +1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        grad[:, :, 1] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, +1.0, +1.0, +2.0, +2.0, +2.0],
            ]
        )

        grad[:, :, 2] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +1.0],
            ]
        )

        return grad


##############################################################################
class Quad(ArrayModel):
    """
    ArrayModel 32 is a model for 7 catenary with 6 offsets variables

    ----------------------------------------------------------

     _          2                          3
     ^                                 d1
     |                       |------------>
     h1
     |
     _         0                           1

                                  d1
                            |------------->


    ----------------------------------------------------------

    inputs
    --------
    p      : vector of parameters

        x_0 : x translation of local frame orign in world frame
        y_0 : y translation of local frame orign in world frame
        z_0 : z translation of local frame orign in world frame
        phi : z rotation of local frame basis in in world frame
        a   : sag parameter
        d1  : horizontal distance between power lines
        h1  : vertical distance between power lines


    ----------------------------------------------------------

    """

    #################################################
    def __init__(self):

        ArrayModel.__init__(self, l=7, q=4)

    ############################
    def p2deltas(self, p):
        """
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame
        """

        delta = np.zeros((3, self.q))

        d1 = p[5]
        h1 = p[6]

        # Offset in local catenary frame
        delta[1, 0] = -d1  # y offset of cable 0
        delta[1, 1] = +d1  # y offset of cable 1
        delta[1, 2] = -d1  # y offset of cable 2
        delta[1, 3] = +d1  # y offset of cable 3

        delta[2, 2] = +h1  # z offset of cable 2
        delta[2, 3] = +h1  # z offset of cable 3

        return delta

    ############################
    def deltas_grad(self):
        """
        Compute the gradient of deltas with respect to offset parameters
        """

        grad = np.zeros((3, self.q, (self.l - 5)))

        grad[:, :, 0] = np.array(
            [[0.0, 0.0, 0, 0.0], [-1.0, +1.0, -1.0, +1.0], [0.0, 0.0, 0.0, 0.0]]
        )

        grad[:, :, 1] = np.array(
            [[0.0, 0.0, 0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, +1.0, +1.0]]
        )

        return grad


class ArrayModel222(ArrayModel):
    """
    ArrayModel 222 is a model for 6 catenary with 3 offsets variables

    inputs
    --------
    p      : vector of parameters

        x_0 : x translation of local frame orign in world frame
        y_0 : y translation of local frame orign in world frame
        z_0 : z translation of local frame orign in world frame
        phi : z rotation of local frame basis in in world frame
        a   : sag parameter
        d1  : horizontal distance between power lines
        d2  : horizontal distance between power lines
        h1  : vertical distance between power lines

    ----------------------------------------------------------

     _         4                  d1       5
     ^                       |----------->
     |
     h1
     |
     _     2                                     3
     ^                                 d2
     |                       |------------------->
     h1
     |
     _         0                           1

                                  d1
                            |------------->


    ----------------------------------------------------------

    """

    #################################################
    def __init__(self):

        ArrayModel.__init__(self, l=8, q=6)

    ############################
    def p2deltas(self, p):
        """
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame
        """

        delta = np.zeros((3, self.q))

        d1 = p[5]
        d2 = p[6]
        h1 = p[7]

        # Offset in local catenary frame
        delta[1, 0] = -d1  # y offset of cable 0
        delta[1, 1] = +d1  # y offset of cable 1
        delta[1, 2] = -d2  # y offset of cable 2
        delta[1, 3] = +d2  # y offset of cable 3
        delta[1, 4] = -d1  # y offset of cable 4
        delta[1, 5] = +d1  # y offset of cable 5

        delta[2, 2] = +h1  # z offset of cable 2
        delta[2, 3] = +h1  # z offset of cable 3
        delta[2, 4] = +2 * h1  # z offset of cable 4
        delta[2, 5] = +2 * h1  # z offset of cable 5

        return delta

    ############################
    def deltas_grad(self):
        """
        Compute the gradient of deltas with respect to offset parameters
        """

        grad = np.zeros((3, self.q, (self.l - 5)))

        grad[:, :, 0] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0],
                [-1.0, +1.0, 0.0, 0.0, -1.0, +1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        grad[:, :, 1] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, +1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        grad[:, :, 2] = np.array(
            [
                [0.0, 0.0, 0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, +1.0, +1.0, +2.0, +2.0],
            ]
        )

        return grad


# Model factory
def create_array_model(model_name: str) -> ArrayModel:
    if model_name == "222":
        return ArrayModel222()
    elif model_name == "32":
        return ArrayModel32()
    elif model_name == "2221":
        return ArrayModel2221()
    elif model_name == "constant2221":
        return ArrayModelConstant2221()
    elif model_name == "quad":
        return Quad()
    elif model_name == "2":
        return ArrayModel2()
    else:
        raise ValueError(f"Model {model_name} not recognized.")


# ###########################
# Plotting
# ###########################


###############################################################################
class EstimationPlot:
    """ """

    ############################
    def __init__(self, p_true, p_hat, pts, p2r_w, n=100, xmin=-200, xmax=200):

        fig = plt.figure(figsize=(4, 3), dpi=300, frameon=True)
        ax = fig.add_subplot(projection="3d")

        pts_true = p2r_w(p_true, xmin, xmax, n)[1]
        pts_hat = p2r_w(p_hat, xmin, xmax, n)[1]
        pts_noisy = pts

        self.fig = fig
        self.ax = ax
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.p2r_w = p2r_w
        self.n_line = pts_true.shape[2]
        self.show = True

        # Plot init position (saved)
        self.plot_model(p_hat)

        lines_true = []
        lines_hat = []

        # Plot true line position
        lines_true.append(
            ax.plot(
                pts_true[0, :, 0],
                pts_true[1, :, 0],
                pts_true[2, :, 0],
                "-k",
                label="True",
            )
        )
        for i in range(self.n_line - 1):
            i = i + 1
            lines_true.append(
                ax.plot(pts_true[0, :, i], pts_true[1, :, i], pts_true[2, :, i], "-k")
            )  # , label= 'True line %d ' %i ) )

        # Plot measurements
        if pts is not None:
            line_noisy = ax.plot(
                pts_noisy[0, :],
                pts_noisy[1, :],
                pts_noisy[2, :],
                "x",
                color="k",
                label="Pts",
            )
            self.line_noisy = line_noisy

        # Plot estimation
        lines_hat.append(
            ax.plot(
                pts_hat[0, :, 0], pts_hat[1, :, 0], pts_hat[2, :, 0], "--", label="Est."
            )
        )
        for i in range(self.n_line - 1):
            i = i + 1
            # lines_true.append( ax.plot( pts_true[0,:,i]  , pts_true[1,:,i]  , pts_true[2,:,i] , '-k' ) ) #, label= 'True line %d ' %i ) )
            lines_hat.append(
                ax.plot(pts_hat[0, :, i], pts_hat[1, :, i], pts_hat[2, :, i], "--")
            )

        self.lines_true = lines_true
        self.lines_hat = lines_hat

        # ax.axis('equal')
        # ax.set_xlabel( 'x', fontsize = 5)
        ax.grid(True)

    ############################
    def plot_model(self, p_hat, style="-."):

        pts_hat = self.p2r_w(p_hat, self.xmin, self.xmax, self.n)[1]

        self.lines_model = []

        self.lines_model.append(
            self.ax.plot(
                pts_hat[0, :, 0],
                pts_hat[1, :, 0],
                pts_hat[2, :, 0],
                style,
                color="grey",
                label="Init",
            )
        )
        for i in range(self.n_line - 1):
            i = i + 1
            self.lines_model.append(
                self.ax.plot(
                    pts_hat[0, :, i],
                    pts_hat[1, :, i],
                    pts_hat[2, :, i],
                    style,
                    color="grey",
                )
            )

    ############################
    def update_estimation(self, p_hat):

        pts_hat = self.p2r_w(p_hat, self.xmin, self.xmax, self.n)[1]

        for i in range(self.n_line):
            self.lines_hat[i][0].set_data(pts_hat[0, :, i], pts_hat[1, :, i])
            self.lines_hat[i][0].set_3d_properties(pts_hat[2, :, i])

        if self.show:
            plt.pause(0.001)

    ############################
    def update_pts(self, pts_noisy):

        try:

            self.line_noisy[0].set_data(pts_noisy[0, :], pts_noisy[1, :])
            self.line_noisy[0].set_3d_properties(pts_noisy[2, :])

        except:

            line_noisy = self.ax.plot(
                pts_noisy[0, :],
                pts_noisy[1, :],
                pts_noisy[2, :],
                "x",
                color="k",
                label="Pts",
            )
            self.line_noisy = line_noisy
            self.ax.legend(loc="upper right", fontsize=8)

        if self.show:
            plt.pause(0.001)

    ############################
    def update_true(self, p_true):

        pts_true = self.p2r_w(p_true, self.xmin, self.xmax, self.n)[1]

        for i in range(self.n_line):
            self.lines_true[i][0].set_data(pts_true[0, :, i], pts_true[1, :, i])
            self.lines_true[i][0].set_3d_properties(pts_true[2, :, i])

        if self.show:
            plt.pause(0.001)

    ############################
    def add_pts(self, pts, label="$n_{in}$ group"):

        self.ax.plot(pts[0, :], pts[1, :], pts[2, :], "xk", label=label)

        if self.show:
            plt.pause(0.001)

    ############################
    def save(self, name="test"):

        self.fig.savefig(name + "_3d_estimation.pdf")


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    p = np.array([-10.0, -10.0, -10.0, 0.0, 500, 15.0, 5.0, 10.0])
    p_nom = np.array([-11.0, -11.0, -10.1, 0.5, 600, 16.0, 10.0, 10.0])

    model = ArrayModel32()

    pts = model.generate_test_data(p, partial_obs=True)

    EstimationPlot(p, p_nom, pts, model.p2r_w)
