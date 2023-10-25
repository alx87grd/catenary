#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:03:21 2023

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

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
        
        



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    from powerline32 import p2r_w
    from powerline32 import p2p5
    from powerline32 import generate_test_data
    
    
    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    p_init =  np.array([ 100, 100, 100, 1.0, 300, 40.  , 25.  , 25    ])
    
    # r , r5 , x = p2r_w( p , x_min = -200, x_max = 200 , n = 400, )
    p5 = p2p5( p )
    pts = generate_test_data( p , partial_obs = True )
    
    plot = EstimationPlot( p , p_init , pts , p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,1.14) , (100,2000) , (5,100), (5,100) , (5,100)]
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20, p2r_w ] 
    
    start_time = time.time()
    
    func = lambda p: J(p, pts, p_init, params)

    
    res = minimize( func,
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    #constraints=constraints,  
                    callback=plot.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
            f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
            f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n" + 
            f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )










