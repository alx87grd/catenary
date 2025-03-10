import numpy as np
from scipy.optimize import minimize
from catenary.kinematic import singleline as catenary
from catenary.estimation import costfunction as cf


############################
# Estimation Class
############################


###############################################################################
class ArrayEstimator:
    """ """

    #################################################
    def __init__(self, model, p_0):

        self.model = model
        # self.p2r_w  = model.p2r_w        # forward kinematic
        self.n_p = model.l  # number of parameters
        self.n_line = model.q  # number of lines in the array

        # default parameter range
        self.p_ub = np.zeros(self.n_p)
        self.p_lb = np.zeros(self.n_p)
        # translation
        self.p_ub[0:2] = p_0[0:2] + 50.0
        self.p_lb[0:2] = p_0[0:1] - 50.0
        self.p_ub[2] = p_0[2] + 100.0
        self.p_lb[2] = p_0[2] - 100.0
        # rotation
        self.p_ub[3] = p_0[3] + 0.5
        self.p_lb[3] = p_0[3] - 0.5
        # sag
        self.p_ub[4] = p_0[4] + 500
        self.p_lb[4] = 100
        # intercablw distance
        self.p_ub[5:] = p_0[5:] + 2.0
        self.p_lb[5:] = p_0[5:] - 2.0

        # default sampling parameters
        self.x_min = -200
        self.x_max = +200
        self.n_sample = 201

        # default cost function parameters
        self.method = "x"
        self.Q = np.diag(np.zeros(self.n_p))
        self.b = 1000.0
        self.l = 1.0
        self.power = 2

        self.use_grad = True

        # search param
        self.n_search = 3
        self.p_var = np.zeros(self.n_p)
        self.p_var[0:3] = 10.0
        self.p_var[3] = 1.0
        self.p_var[4] = 200.0
        self.p_var[5:] = 0.5

        # initialize random number generator with seed to have reproductible results
        self.rng = np.random.default_rng(seed=1)

        # grouping param
        self.d_th = 1.0
        self.succes_ratio = 0.8

    #####################################################
    def get_bounds(self):

        bounds = []

        for i in range(self.n_p):

            bounds.append((self.p_lb[i], self.p_ub[i]))

        return bounds

    #####################################################
    def get_cost_parameters(self, m):

        R = np.ones((m)) * 1 / m

        param = [
            self.model,
            R,
            self.Q,
            self.l,
            self.power,
            self.b,
            self.method,
            self.n_sample,
            self.x_min,
            self.x_max,
        ]

        return param

    #####################################################
    def solve(self, pts, p_init, callback=None):

        bounds = self.get_bounds()
        param = self.get_cost_parameters(m=pts.shape[1])
        func = lambda p: cf.J(p, pts, p_init, param)

        if self.use_grad:
            grad = lambda p: cf.dJ_dp(p, pts, p_init, param, num=False)
        else:
            grad = None

        res = minimize(
            func,
            p_init,
            method="SLSQP",
            bounds=bounds,
            jac=grad,
            # constraints=constraints,
            callback=callback,
            options={"disp": True, "maxiter": 500},
        )

        p_hat = res.x
        # j_hat = res.fun

        return p_hat

    #####################################################
    def solve_with_search(self, pts, p_init, callback=None):

        bounds = self.get_bounds()
        param = self.get_cost_parameters(m=pts.shape[1])
        func = lambda p: cf.J(p, pts, p_init, param)

        if self.use_grad:
            grad = lambda p: cf.dJ_dp(p, pts, p_init, param, num=False)
        else:
            grad = None

        # variation to params
        deltas = self.p_var[:, np.newaxis] * self.rng.standard_normal(
            (self.n_p, self.n_search)
        )

        # keep original solution
        deltas[:, 0] = np.zeros((self.n_p))

        # solutions
        ps = np.zeros((self.n_p, self.n_search))
        js = np.zeros(self.n_search)

        for i in range(self.n_search):

            p = p_init + deltas[:, i]

            res = minimize(
                func,
                p,
                method="SLSQP",
                bounds=bounds,
                jac=grad,
                # constraints=constraints,
                callback=callback,
                options={"disp": False, "maxiter": 500},
            )

            ps[:, i] = res.x
            js[i] = res.fun

        i_star = js.argmin()
        p_hat = ps[:, i_star]

        return p_hat

    #####################################################
    def solve_with_ransac_search(
        self, pts, p_init, callback=None, n_iter=10, n_pts=200
    ):

        # solutions
        ps = np.zeros((self.n_p, n_iter))
        ns = np.zeros(n_iter)

        for i in range(n_iter):

            pts_test = pts[:, np.random.randint(pts.shape[1], size=n_pts)]

            p_hat = self.solve_with_search(pts_test, p_init, callback)
            pts_in = self.get_array_group(p_hat, pts_test)

            ps[:, i] = p_hat
            ns[i] = pts_in.shape[1]

        # Debug
        print(ns)

        return ps[:, ns.argmax()]

    #####################################################
    def get_array_group(self, p, pts, xm=-200, xp=+200, n_s=5000):

        # generate a list of sample point on the model curve
        r_model = self.model.p2r_w(p, xm, xp, n_s)[0]

        # Minimum distances to model for all measurements
        d_min = cf.compute_d_min(pts, r_model)

        # Group based on threshlod
        pts_in = pts[:, d_min < self.d_th]

        return pts_in

    #####################################################
    def is_target_aquired(self, p, pts):

        # Group based on threshlod
        pts_in = self.get_array_group(p, pts)

        # Ratio of point in range of the model
        ratio = pts_in.shape[1] / pts.shape[1]

        succes = ratio > self.succes_ratio

        return succes
