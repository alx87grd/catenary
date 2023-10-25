#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:03:21 2023

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

###########################
# catenary function
###########################

import catenary

from powerline32 import p2r_w

# ###########################
# Optimization
# ###########################

default_cost_param = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                        1.0 , 1.0 , 2 , 1000 , -200 , 200, p2r_w ] 

############################
def J( p , pts , p_nom , param = default_cost_param ):
    """ 
    Cost function for curve fitting a catenary model on a point cloud
    
    J = average_cost_per_measurement + regulation_term
    
    see attached notes.
    
    inputs
    --------
    p     : 8x1 parameter vector
    pts   : 3xm cloud point
    p_nom : 8x1 expected parameter vector ( optionnal for regulation )
    param : list of cost function parameter and options
    
    default_param = [ method = 'sample' , 
                     Q = np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                     b = 1.0 , 
                     l = 1.0 , 
                     power = 2 , 
                     n = 1000 , 
                     x_min = -200 , 
                     x_max = 200 ,
                     r2w   = fonction used to generate samples] 
    
    method = 'sample' : sample-based brute force scheme
    method = 'x'      : data association is based on local x in cat frame
    
    Q      : 8x8 regulation weight matrix
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
    
    m      = pts.shape[1]  # number of measurements
    
    method = param[0]
    Q      = param[1]
    b      = param[2]
    l      = param[3]
    power  = param[4]
    n      = param[5]
    x_min  = param[6]
    x_max  = param[7]
    p2r_w  = param[8]
    
    ###################################################
    if method == 'sample':
        """ data association is sampled-based """
    
        # generate a list of sample point on the model curve
        r_model  = p2r_w( p, n , x_min , x_max )[0]
        
        # Minimum distances to model for all measurements
        d_min = catenary.compute_d_min( pts , r_model )
    
    ###################################################
    elif method == 'x':
        """ data association is based on local x-coord in cat frame """
        
        raise NotImplementedError
        
    ###################################################
        
    # Cost shaping function
    c = catenary.lorentzian( d_min , l , power , b )
    
    # Average costs per measurement plus regulation
    pts_cost = c.sum() / m 
    
    # Regulation
    p_e = p_nom - p
    
    # Total cost with regulation
    cost = pts_cost + p_e.T @ Q @ p_e
    
    return cost


############################
# Estimation Class
############################



###############################################################################
class ArrayEstimator:
    """ 
    """
    
    #################################################
    def __init__(self, p2r_w , p_0 ):
        
        
        self.n_p    = p_0.shape[0]                 # number of parameters
        
        self.p2r_w  = p2r_w                        # forward kinematic
        self.n_line = p2r_w( p_0 )[1].shape[2]     # number of lines in the array
        
        
        # default parameter range
        self.p_ub      = np.zeros( self.n_p )
        self.p_lb      = np.zeros( self.n_p )
        #translation
        self.p_ub[0:3] = p_0[0:3] + 50.0
        self.p_lb[0:3] = p_0[0:3] - 50.0
        # rotation
        self.p_ub[3] = p_0[3] + 0.5
        self.p_lb[3] = p_0[3] - 0.5
        # sag
        self.p_ub[4] = p_0[4] + 500
        self.p_lb[4] = 100
        # intercablw distance
        self.p_ub[4:] = p_0[4:] + 2.0
        self.p_lb[4:] = p_0[4:] - 2.0
        
        # default sampling parameters
        self.x_min     = -200
        self.x_max     = +200
        self.n_sample  =  501
        
        # default cost function parameters
        self.method = 'sample'
        self.Q      = np.diag( np.zeros( self.n_p ) )
        self.b      = 1.0
        self.l      = 1.0
        self.power  = 2
        
        # grouping param
        self.d_th         = 1.0
        self.succes_ratio = 0.9
        
        
    #####################################################
    def compute_average_dmin( self, p , pts ):
        
        # number of measurements
        m      = pts.shape[1]
        
        # generate a list of sample point on the model curve
        r_model  = self.p2r_w( p, self.n_sample , self.x_min , self.x_max )[0]
        
        # Minimum distances to model for all measurements
        d_min = catenary.compute_d_min( pts , r_model )
        
        # Average
        d_min_average = d_min.sum() / m 
        
        return d_min_average

            
    #####################################################
    def get_array_group( self, p , pts ):
        
        # generate a list of sample point on the model curve
        r_model  = self.p2r_w( p, self.n_sample , self.x_min , self.x_max )[0]
        
        # Minimum distances to model for all measurements
        d_min = catenary.compute_d_min( pts , r_model )
        
        # Group based on threshlod
        pts_in = pts[ : , d_min < self.d_th ]
        
        return pts_in
    
    
    #####################################################
    def get_bounds( self):
        
        bounds = []
        
        for i in range(self.n_p):
            
            bounds.append( ( self.p_lb[i] , self.p_ub[i] ) )
            
        return bounds
    
    
    #####################################################
    def get_cost_parameters(self):
        
        param = [self.method,
                 self.Q,
                 self.b,
                 self.l,
                 self.power,
                 self.n_sample,
                 self.x_min,
                 self.x_max,
                 self.p2r_w   ]
            
        return param
            
    
    
    #####################################################
    def solve( self, pts , p_init , callback = None ):
        
        bounds = self.get_bounds()
        param  = self.get_cost_parameters()
        func   = lambda p: J(p, pts, p_init, param)
        
        res = minimize( func,
                        p_init, 
                        method='SLSQP',  
                        bounds=bounds, 
                        #constraints=constraints,  
                        callback=callback, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        j_hat = res.fun
        
        return p_hat
    
    
    #####################################################
    def solve_zscan( self, pts , p_init , callback = None ):
        
        bounds = self.get_bounds()
        param  = self.get_cost_parameters()
        func   = lambda p: J(p, pts, p_init, param)
        
        n = 3
        zs = np.linspace( -50 , 50,  n)
        ps = np.zeros((self.n_p,n))
        js = np.zeros(n)
        
        for i in range(n):
            
            p    = p_init
            p[2] = p_init[2] + zs[i] # new z
        
            res = minimize( func,
                            p, 
                            method='SLSQP',  
                            bounds=bounds, 
                            #constraints=constraints,  
                            callback=callback, 
                            options={'disp':False,'maxiter':500})
            
            ps[:,i] = res.x
            js[i]   = res.fun
            
        i_star = js.argmin()
        p_hat  = ps[:,i_star]
        
        return p_hat
    
    
    #####################################################
    def is_target_aquired( self, p , pts , ):
        
        # generate a list of sample point on the model curve
        r_model  = self.p2r_w( p, self.n_sample , self.x_min , self.x_max )[0]
        
        # Minimum distances to model for all measurements
        d_min = catenary.compute_d_min( pts , r_model )
        
        # Group based on threshlod
        pts_in = pts[ : , d_min < self.d_th ]
        
        # Ratio of point in range of the model
        ratio = ( pts_in.shape[1] - pts.shape[1] ) / pts.shape[1]
        
        succes = ratio > self.succes_ratio
        
        return succes
    


# ###########################
# Plotting
# ###########################


###############################################################################
class EstimationPlot:
    """ 
    """
    
    ############################
    def __init__(self, p_true , p_hat , pts , p2r_w , n = 100 , xmin = -200, xmax = 200 ):
        
        
        fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
        ax  = fig.add_subplot(projection='3d')
        
        pts_true  = p2r_w( p_true , xmin , xmax , n )[1]
        pts_hat   = p2r_w( p_hat  , xmin , xmax , n )[1]
        pts_noisy = pts
        
        lines_true  = []
        lines_hat   = []
        
        self.n_line = pts_true.shape[2]
        
        for i in range(self.n_line):
            lines_true.append( ax.plot( pts_true[0,:,i]  , pts_true[1,:,i]  , pts_true[2,:,i] , '-k' ) ) #, label= 'True line %d ' %i ) )
            # lines_hat.append(   ax.plot( pts_hat[0,:,i]   , pts_hat[1,:,i]   , pts_hat[2,:,i]  , '--', label= 'Estimated line %d' %i) )
        
        if pts is not None:
            line_noisy = ax.plot( pts_noisy[0,:] , pts_noisy[1,:] , pts_noisy[2,:], 'x' , label= 'Measurements')
            self.line_noisy = line_noisy
        
        for i in range(self.n_line):
            # lines_true.append( ax.plot( pts_true[0,:,i]  , pts_true[1,:,i]  , pts_true[2,:,i] , '-k' ) ) #, label= 'True line %d ' %i ) )
            lines_hat.append(   ax.plot( pts_hat[0,:,i]   , pts_hat[1,:,i]   , pts_hat[2,:,i]  , '--', label= 'Estimated line %d' %i) )
        
            
        self.lines_true  = lines_true
        self.lines_hat   = lines_hat
        
        ax.axis('equal')
        ax.legend( loc = 'upper right' , fontsize = 5)
        ax.set_xlabel( 'x', fontsize = 5)
        ax.grid(True)
        
        self.fig = fig
        self.ax  = ax
        
        self.n    = n
        self.xmin = xmin
        self.xmax = xmax
        
        self.p2r_w = p2r_w
        
        
    ############################
    def update_estimation( self, p_hat ):
        
        pts_hat   = self.p2r_w( p_hat , self.xmin , self.xmax , self.n )[1]
        
        for i in range(self.n_line):
            self.lines_hat[i][0].set_data( pts_hat[0,:,i] , pts_hat[1,:,i]  )
            self.lines_hat[i][0].set_3d_properties( pts_hat[2,:,i] )
        
        plt.pause( 0.001 )
        
    ############################
    def update_pts( self, pts_noisy  ):
        
        self.line_noisy[0].set_data( pts_noisy[0,:] , pts_noisy[1,:]  )
        self.line_noisy[0].set_3d_properties( pts_noisy[2,:] )
        
        plt.pause( 0.001 )
        
    ############################
    def update_true( self, p_true ):
        
        pts_true   = self.p2r_w( p_true , self.xmin , self.xmax , self.n )[1]
        
        for i in range(self.n_line):
            self.lines_true[i][0].set_data( pts_true[0,:,i] , pts_true[1,:,i]  )
            self.lines_true[i][0].set_3d_properties( pts_true[2,:,i] )
        
        plt.pause( 0.001 )
        
    
    ############################
    def add_pts( self, pts ):
        
        self.ax.plot( pts[0,:] , pts[1,:] , pts[2,:], 'o' , label= 'Group')
        
        


############################
def estimator_test():
    
    from powerline32 import p2r_w
    from powerline32 import generate_test_data
    


    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    
    # p_ub   =
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 49.  , 25.  , 25    ])
    # p_lb   =
    
    pts = generate_test_data( p , partial_obs = True )
    
    plot = EstimationPlot( p , p_hat , pts , p2r_w )
    
    estimator = ArrayEstimator( p2r_w , p_hat )
    
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    for i in range(500):
        
        pts = generate_test_data( p , partial_obs = True )
        
        plot.update_pts( pts )
    
        p_hat  = estimator.solve( pts , p_hat ) 
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        print( " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
    return estimator


############################
def scan_z_test_test( zscan = True ):
    
    from powerline32 import p2r_w
    from powerline32 import generate_test_data
    

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 25.  , 50. ])
    p_hat  =  np.array([   0,   0,   0, 1.2, 500, 51.  , 25.  , 49  ])
    
    pts = generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = EstimationPlot( p , p_hat , pts , p2r_w )
    
    estimator = ArrayEstimator( p2r_w , p_hat )
    
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.000002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    
    for i in range(50):
        
        pts = generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -100, x_max = -70, n_out = 10 ,
                                             center = [-50,-50,-50] , w_o = 10 )
        
        plot.update_pts( pts )
    
        if zscan:
            p_hat  = estimator.solve_zscan( pts , p_hat ) 
            
        else:
            p_hat  = estimator.solve( pts , p_hat )
            
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        print( " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
    return estimator
    


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    # e = estimator_test()
    
    e = scan_z_test_test( False )
    e = scan_z_test_test( True )
    
    # from powerline_gazebo import p2r_w
    # from powerline_gazebo import generate_test_data
    
    # from scipy.optimize import minimize
    # import time

    # p      =  np.array([  1, -3, 4, 0.2,  500, 4.  , 4  , 9  ])
    # p_init =  np.array([  0,  0, 0, 0.0, 1000, 5.  , 5. , 10 ])
    
    # pts = generate_test_data(  p , n_obs = 10 , x_min = 0, x_max = 20, w_l = 0.1,
    #                         n_out = 10, center = [0,0,0] , w_o = 100,
    #                         partial_obs = False )
    
    
    # e = ArrayEstimator( p2r_w , p_init )
    
    # p_hat  = e.solve( pts , p_init )
    # pts_in = e.get_array_group( p_hat , pts )
    # target = e.is_target_aquired( p_hat , pts)
    
    # print('Target aquired is ', target)
    
    
    # plot2 = EstimationPlot( p , p_hat , pts , p2r_w )
    # plot2.add_pts( pts_in )







