#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:52:04 2023

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

###########################
# Catanery function
###########################

############################
def cat( x , a = 1. ):
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
    
    z = a * ( np.cosh( x / a )  - 1. )
    
    return z


############################
def w_T_c( phi , x , y , z):
    """ 
    Tranformation matrix from catenery local frame to world frame
    
    inputs
    ----------
    phi : z rotation of catenary local basis with respect to world basis
    x   : x translation of catenary local frame orign in world frame
    y   : y translation of catenary local frame orign in world frame
    z   : z translation of catenary local frame orign in world frame
    
    outputs
    ----------
    world_T_catenery  : 4x4 Transformation Matrix
    
    """
    
    s = np.sin( phi )
    c = np.cos( phi )
    
    T = np.array([ [ c   , -s ,  0  , x ] , 
                   [ s   ,  c ,  0  , y ] ,
                   [ 0   ,  0 ,  1. , z ] ,
                   [ 0   ,  0 ,  0  , 1.] ])
    
    return T


############################
def p2r_w( p , x_min = -200, x_max = 200 , n = 400, ):
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
    x_c      : vector of x coord in catenery frame
    
    """
    
    x_c = np.linspace( x_min , x_max, n )
    
    # params
    x_0 = p[0]
    y_0 = p[1]
    z_0 = p[2]
    phi = p[3]
    a   = p[4]
    
    # catenery frame z
    z_c      = cat( x_c , a )
    
    r_c      = np.zeros((4,n))
    r_c[0,:] = x_c
    r_c[2,:] = z_c
    r_c[3,:] = np.ones((n))
    
    r_w = w_T_c( phi, x_0, y_0, z_0 ) @ r_c
    
    return ( r_w[0:3,:] , x_c ) 


# ###########################
# # Optimization
# ###########################
        
        
############################
def lorentzian( x , l = 1.0 , power = 2 , b = 1.0 ):
    """ 
    
    Cost shaping function that smooth out the cost of large distance to 
    minimize the effect of outliers.
    
    inputs
    --------
    x     : input vector of distances
    
    l
    
    outputs
    ----------
    c     : output vector of cost for each distances
        
    """
    
    c = np.log10( 1 + ( b * x ) ** power / l )
    
    return c


############################
def d_min_sample( p , pts , n_sample = 1000 , x_min = -200, x_max = 200 ):
    """ 
    
    Compute the minimum distance between 3D pts and the catanery curve defined
    by p based on brute force and sampling on the model curve
    
    """
    
    # generate a list of sample point on the model curve
    r_hat  = p2r_w( p, n_sample , x_min , x_max )[0]
    
    # Vectors between measurements and model sample pts
    e  = pts[:,:,np.newaxis] - r_hat[:,np.newaxis,:]
    
    # Distances between measurements and model sample pts
    d = np.linalg.norm( e , axis = 0 )
    
    # Minimum distances to model for all measurements
    d_min = d.min( axis = 1 )
    
    return d_min


default_cost_param = [ 'sample' , np.diag([ 1.0 , 1.0 , 1.0 , 10.0 , 0.01 ]) ,
                        1.0 , 1.0 , 2 , 1000 , -200 , 200] 

############################
def J( p , pts , p_nom , param = default_cost_param ):
    """ 
    Cost function for curve fitting a catanery model on a point cloud
    
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
    
    if method == 'sample':
    
        # Minimum distance to points
        d_min = d_min_sample( p , pts , n , x_min , x_max )
        
        # Cost shaping function
        c = lorentzian( d_min , l , power , b )
        
        # Average costs per measurement plus regulation
        pts_cost = c.sum() / m 
        
    elif method == 'x':
        
        pass
    
    # Regulation
    p_e = p_nom - p
    
    # Total cost with regulation
    cost = pts_cost + p_e.T @ Q @ p_e
    
    return cost

############################
def dJ_dp( p , pts , p_nom , param = default_cost_param , num = False ):
    """ """

    if num:
        
        dJ_dp = np.zeros( p.shape[0] )
        dp    = np.array([ 0.001 , 0.001, 0.001 , 0.001 , 0.001 ])
        
        for i in range(5):
        
            pp    = p.copy()
            pm    = p.copy()
            pp[i] = p[i] + dp[i]
            pm[i] = p[i] - dp[i]
            cp    = J( pp , pts , p_nom , param )
            cm    = J( pm , pts , p_nom , param )
            
            dJ_dp[i] = ( cp - cm ) / ( 2.0 * dp[i] )
    
    else:
        
        #########################
        # Analytical gratient
        #########################
        
        m      = pts.shape[1]  # number of measurements
        
        method = param[0]
        Q      = param[1]
        b      = param[2]
        l      = param[3]
        power  = param[4]
        n      = param[5]
        x_min  = param[6]
        x_max  = param[7]
        
        if method == 'sample':
        
           # generate a list of sample point on the model curve
           r_hat, x_l = p2r_w( p, n , x_min , x_max )

           # Vectors between measurements and model sample pts
           e = pts[:,:,np.newaxis] - r_hat[:,np.newaxis,:]
           
           # Distances between measurements and model sample pts
           d = np.linalg.norm( e , axis = 0 )
           
           # Minimum distances to model for all measurements
           d_min = d.min( axis = 1 )

           # Errors vector to closet model point
           j_min = d.argmin( axis = 1 )   # index of closet model point
           i     = np.arange( j_min.shape[0] )
           e_min = e[ : , i , j_min ]
           x     = x_l[ j_min  ]          # local x of closest pts on the model

           # Error Grad for each pts
           eT_de_dp = np.zeros( (5 , pts.shape[1] ) )
           
           
           phi = p[3]
           a   = p[4]
           
           eT_de_dp[0,:] = -e_min[0,:]
           eT_de_dp[1,:] = -e_min[1,:]
           eT_de_dp[2,:] = -e_min[2,:]
           eT_de_dp[3,:] = -( e_min[0,:] * ( - np.sin( phi ) * x ) + e_min[1,:] * ( + np.cos( phi ) * x ) )
           eT_de_dp[4,:] = -e_min[2,:] * ( np.cosh( x / a ) - x / a * np.sinh( x / a ) - 1 )
           
           # Norm grad
           dd_dp = eT_de_dp / d_min
           
            
        elif method == 'x':
            
            pass
        
        # Smoothing grad
        dc_dd = b * power * ( b * d_min ) ** ( power - 1 ) / ( np.log( 10 ) * ( l +  b * d_min ) ** power )
        
        dc_dp = dc_dd * dd_dp
        
        # Average grad per point
        dc_cp_average = dc_dp.sum( axis = 1 ) / m
    
        # Regulation
        p_e = p_nom - p
        
        # Total cost with regulation
        dJ_dp = dc_cp_average - 2 * p_e.T @ Q
    
    return dJ_dp






############################
def cost_local_yz( p_hat , pts ):
    """ """
    
    n = pts.shape[1]
    
    a_hat     = p_hat[0]
    w_T_local = w_T_c( p_hat[1] , p_hat[2] , p_hat[3] , p_hat[4])
    local_T_w = np.linalg.inv( w_T_local )
    
    pts_world = np.vstack( [pts , np.ones(n) ] )
    
    # Compute measurements points positions in local cable frame
    pts_local  = local_T_w @ pts_world
    
    # Compute expected z position based on x coord and catanery model
    zs_hat     = cat( pts_local[0,:] , a_hat )
    
    # Compute delta vector between measurements and expected position
    delta      = np.zeros((3,n))
    delta[1,:] = pts_local[1,:]
    delta[2,:] = pts_local[2,:] - zs_hat
    
    norm = np.linalg.norm( delta , axis = 0 )
    
    costs = lorentzian( norm , 1)
    
    return costs.sum()

############################
def cost_dmin_sample( p_hat , pts , l = 1.0 ):
    """ """
    
    d_min = d_min_sample( p_hat , pts )
    
    # Cost shaping function
    pts_cost = lorentzian( d_min , l )
    
    # Sum of all costs
    cost = pts_cost.sum()
    
    return cost


# ###########################
# # Segmentation
# ###########################

############################
def get_catanery_group( p_hat , pts , d_th = 1.0 , n_sample = 100 ):
    
    # Distance to catanery curves
    d_min = d_min_sample( p_hat , pts , n_sample , -50 , 50 )
    
    # Group based on threshlod
    pts_in = pts[ : , d_min < d_th ]
    
    return pts_in


# ###########################
# Plotting
# ###########################


###############################################################################
class CateneryEstimationPlot:
    """ 
    """
    
    ############################
    def __init__(self, p_true , p_hat , pts , n = 100 , xmin = -200, xmax = 200):
        
        
        fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
        ax  = fig.add_subplot(projection='3d')
        
        pts_true  = p2r_w( p_true , xmin , xmax , n )[0]
        pts_hat   = p2r_w( p_hat  , xmin , xmax , n )[0]
        pts_noisy = pts
    
        line_true  = ax.plot( pts_true[0,:]  , pts_true[1,:]  , pts_true[2,:] , label= 'True equation' )
        line_hat   = ax.plot( pts_hat[0,:]   , pts_hat[1,:]   , pts_hat[2,:]  , '--', label= 'Estimated equation' )
        line_noisy = ax.plot( pts_noisy[0,:] , pts_noisy[1,:] , pts_noisy[2,:], 'x' , label= 'Measurements')
        
        ax.axis('equal')
        ax.legend( loc = 'upper right' , fontsize = 5)
        ax.set_xlabel( 'x', fontsize = 5)
        ax.grid(True)
        
        self.fig = fig
        self.ax  = ax
        
        self.n    = n
        self.xmin = xmin
        self.xmax = xmax
        
        self.line_true  = line_true
        self.line_hat   = line_hat
        self.line_noisy = line_noisy
        
    ############################
    def update_estimation( self, p_hat ):
        
        pts_hat   = p2r_w( p_hat , self.xmin , self.xmax , self.n )[0]
        
        self.line_hat[0].set_data( pts_hat[0,:] , pts_hat[1,:]  )
        self.line_hat[0].set_3d_properties( pts_hat[2,:] )
        
        plt.pause( 0.001 )
        
    ############################
    def update_pts( self, pts_noisy  ):
        
        self.line_noisy[0].set_data( pts_noisy[0,:] , pts_noisy[1,:]  )
        self.line_noisy[0].set_3d_properties( pts_noisy[2,:] )
        
        plt.pause( 0.001 )
        
    ############################
    def update_true( self, p_true ):
        
        pts_true   = p2r_w( p_true , self.xmin , self.xmax , self.n )[0]
        
        self.line_true[0].set_data( pts_true[0,:] , pts_true[1,:]  )
        self.line_true[0].set_3d_properties( pts_true[2,:] )
        
        plt.pause( 0.001 )
        

############################
def plot_lorentzian( l = 10 , power = 2 , b = 1.0 , ax = None ):
    """ 
    x : input
    a : flattening parameter
    """
    
    x = np.linspace( -50.0 , 50.0 , 10000 ) 
    y = lorentzian( x , l , power )

    if ax is None:
        fig, ax = plt.subplots(2, figsize= (4, 3), dpi=300, frameon=True)
        
    x = np.linspace( -100.0 , 100.0 , 10000 ) 
    y = lorentzian( x , l , power )

    ax[0].plot( x , y , label= r'$l =$ %0.1f' % l )
    ax[0].set_xlabel( 'x', fontsize = 5)
    ax[0].grid(True)
    ax[0].legend()
    ax[0].tick_params( labelsize = 5 )
    ax[0].set_ylabel( 'lorentzian(x)', fontsize = 5 )
    
    x = np.linspace( -1.0 , 1.0 , 1000 ) 
    y = lorentzian( x , l , power )
    
    ax[1].plot( x , y , label= r'$l =$ %0.1f' % l )
    ax[1].set_xlabel( 'x', fontsize = 5)
    ax[1].grid(True)
    ax[1].legend()
    ax[1].tick_params( labelsize = 5 )
    ax[1].set_ylabel( 'lorentzian(x)', fontsize = 5 )
    
    return ax




###############################
# Data generation for testing
##############################


############################
def noisy_p2r_w( p , x_min = -200, x_max = 200 , n = 400, w = 0.5 ):
    """
    p2r_w but with added gaussian noise of standard deviation w
    
    """

    # true points on the curve
    r_line  = p2r_w( p, x_min, x_max, n )[0]
        
    # adding measurements noise
    rng     = np.random.default_rng( seed = None )
    r_noisy = r_line + w * rng.standard_normal(( 3 , n ))
    
    return r_noisy

############################
def multiples_noisy_p2r_w( p , x_min, x_max , n , w ):
    """
    Create 3D world pts for a list of catenery parameters vector

    """
    
    m = p.shape[1]
    
    pts = noisy_p2r_w( p[:,0] , x_min[0], x_max[0] , n[0], w[0] )
    
    for i in range(1,m):

        r = noisy_p2r_w( p[:,i] , x_min[i], x_max[i] , n[i], w[i] )
        
        pts = np.append( pts , r , axis = 1 )

    
    return pts


############################
def outliers( n = 10, center = [0,0,0] , w = 100 ):
    """
    Create random 3D world pts 

    """
    
    # Outliers randoms points (all pts are initialized as random outliers)
    rng   = np.random.default_rng( seed = None )
    noise = w * rng.standard_normal(( 3 , n ))
    pts   = np.zeros(( 3 , n ))
    
    pts[0,:] = center[0]
    pts[1,:] = center[1]
    pts[2,:] = center[2]
    
    pts = pts + noise   # noise randoms points
    
    return pts


############################
def generate_test_data( p , n_obs = 20 , x_min = -100, x_max = 100, w_l = 0.5,
                        n_out = 10, center = [0,0,0] , w_o = 100 ):
    """
    generate pts for a line and outliers
    
    """
    
    r_line = noisy_p2r_w( p , x_min , x_max , n_obs , w_l )
        
    r_out  = outliers( n_out, center , w_o )
        
    pts = np.append( r_line , r_out , axis = 1 )
    
    return pts


############################
def generate_test_data_sequence ( t, p0 , dp_dt , partial_obs = True, 
                                 n_obs = 20 , x_min = -100, x_max = 100, 
                                 w_l = 0.5, n_out = 10, center = [0,0,0] , 
                                 w_o = 100 ):
    """
    generate pts for a line and outliers with linear motion of p
    
    p = p0 + t * dp_dt
    
    """
    
    p = p0 + t * dp_dt
    
    if partial_obs:
        
        n_obs = np.random.randint(1,n_obs)
        x_min = np.random.randint(x_min,x_max)
        x_max = np.random.randint(x_min,x_max)
    
    r_line = noisy_p2r_w( p , x_min , x_max , n_obs , w_l )
        
    r_out  = outliers( n_out, center , w_o )
        
    pts = np.append( r_line , r_out , axis = 1 )
    
    return ( pts , p )


# ###########################
# Tests
# ###########################

    
def compare_gradient():
    
    p_true  =  np.array([  800.0, 1.0, 100.0, 100.0, 100.0])
    p_init  =  np.array([ 1000.0, 1.2, 107.0,  67.0,  98.0])
    
    pts  = generate_noisy_test_data( p_true , center = [50,50,50])
    
    plot = CataneryEstimationPlot( p_true , p_init , pts)
    
    # ###########################
    # # Optimization
    # ###########################
    
    bounds = [ (100,2000) , (0,3.14), (0,100), (0,100) , (0,100) ]
    
    start_time = time.time()
    
    res = minimize( lambda p: cost_dmin_sample(p, pts), 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    # jac= lambda p: grad_cost_dmin_sample(p, pts),
                    #constraints=constraints,  
                    callback=plot.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
    plot2 = CataneryEstimationPlot( p_true , p_init , pts)
    
    start_time = time.time()
    
    res = minimize( lambda p: cost_dmin_sample(p, pts), 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    jac= lambda p: grad_cost_dmin_sample(p, pts),
                    #constraints=constraints,  
                    callback=plot2.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
def compare_gradient_speed():
    
    p_true  =  np.array([  800.0, 1.0, 100.0, 100.0, 100.0])
    p_init  =  np.array([ 1000.0, 1.2, 107.0,  67.0,  98.0])
    
    pts  = generate_noisy_test_data( p_true , center = [50,50,50])
    
    # plot = CataneryEstimationPlot( p_true , p_init , pts)
    
    # ###########################
    # # Optimization
    # ###########################
    
    bounds = [ (100,2000) , (0,3.14), (0,100), (0,100) , (0,100) ]
    
    start_time = time.time()
    
    res = minimize( lambda p: cost_dmin_sample(p, pts), 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    # jac= lambda p: grad_cost_dmin_sample(p, pts),
                    #constraints=constraints,  
                    # callback=plot.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
    # plot2 = CataneryEstimationPlot( p_true , p_init , pts)
    
    start_time = time.time()
    
    res = minimize( lambda p: cost_dmin_sample(p, pts), 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    jac= lambda p: grad_cost_dmin_sample(p, pts),
                    #constraints=constraints,  
                    # callback=plot2.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
    
def tracking_test( use_grad = False , partial_obs = True ):
    
    p0      =  np.array([ 50 , 50 , 50 , 0.2 , 500 ])
    dp_dt   =  np.array([  1 ,  1 , 10 , 0.01 , 0  ])
    
    p_hat   =  np.array([  0 ,  0 ,  0 , 0   , 1000 ])
    
    t = np.linspace( 0 , 10, 101 )
    
    ( pts , p ) = generate_test_data_sequence( 0, p0 , dp_dt , partial_obs = True, 
                                     n_obs = 20 , x_min = -100, x_max = 100, 
                                     w_l = 0.5, n_out = 10, center = [0,0,0] , 
                                     w_o = 100 )

    plot  = CateneryEstimationPlot(  p , p_hat , pts )
    
    for i in range(101):
        
        ( pts , p )  = generate_test_data_sequence( t[i], p0 , dp_dt , 
                                                    partial_obs = True, 
                                         n_obs = 20 , x_min = -100, x_max = 100, 
                                         w_l = 0.5, n_out = 10, center = [0,0,0] , 
                                         w_o = 100 )
        plot.update_true( p )
        plot.update_pts( pts )
        
        bounds = [ (0,500) , (0,500), (0,500) , (0,3.14) , (100,2000) ]
        # params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
        #             1.0 , 1.0 , 2 , 25 , -20 , 20] 
        
        params = [ 'sample' , np.diag([ 0.00001 , 0.00001 , 0.00001 , 0.001 , 0.0 ]) ,
                    1.0 , 1.0 , 2 , 25 , -20 , 20] 
        
        start_time = time.time()
        
        func = lambda p: J(p, pts, p_hat, params)
        grad = lambda p: dJ_dp( p, pts, p_hat, params)
        
        res = minimize( func,
                        p_hat, 
                        method='SLSQP',  
                        bounds=bounds, 
                        jac=grad,
                        #constraints=constraints,  
                        callback=plot.update_estimation, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        
        print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
               f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n" + 
               f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        plot.update_estimation( p_hat )
        


def grouping_test():
    
    p_true  =  np.array([ 530.0, 1.3, 32.0, 43.0, 77.0])
    p_init  =  np.array([ 800.0, 2.0, 10.0, 10.0, 10.0])
    
    pts  = generate_noisy_test_data( p_true , center = [50,50,50])
    
    plot = CataneryEstimationPlot( p_true , p_init , pts)
    
    # ###########################
    # # Optimization
    # ###########################
    
    bounds = [ (100,2000) , (0,3.14), (0,100), (0,100) , (0,100) ]
    
    start_time = time.time()
    
    res = minimize( lambda p: cost_dmin_sample(p, pts), 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    #constraints=constraints,  
                    callback=plot.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
    pts_in = get_catanery_group( p_hat , pts , 2.0 )
    
    plot.ax.plot( pts_in[0,:] , pts_in[1,:] , pts_in[2,:], 'o' , label= 'Group')
    
    
def grouping_test2():
    
    p_1  =  np.array([ 530.0, 1.3, 32.0, 43.0, 77.0])
    p_2  =  np.array([ 530.0, 1.3, 28.0, 53.0, 37.0])
    p_init  =  np.array([ 800.0, 2.0, 10.0, 10.0, 10.0])
    
    pts1 = generate_noisy_test_data( p_1 , center = [50,50,50])
    pts2 = generate_noisy_test_data( p_2 , center = [50,50,50])
    pts2 = generate_noisy_test_data( p_2 , center = [50,50,50])
    
    pts = np.hstack( (pts1 , pts2 ))
    
    plot = CataneryEstimationPlot( p_1 , p_init , pts)
    
    # ###########################
    # # Optimization
    # ###########################
    
    bounds = [ (100,2000) , (0,3.14), (0,100), (0,100) , (0,100) ]
    
    start_time = time.time()
    
    res = minimize( lambda p: cost_dmin_sample(p, pts), 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    #constraints=constraints,  
                    callback=plot.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True 1: {np.array2string(p_1, precision=2, floatmode='fixed')} \n"  + 
           f" True 2: {np.array2string(p_2, precision=2, floatmode='fixed')} \n"  + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n"  )
    
    pts_in = get_catanery_group( p_hat , pts , 2.0 )
    
    plot.ax.plot( pts_in[0,:] , pts_in[1,:] , pts_in[2,:], 'o' , label= 'Group')



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    
    tracking_test( True )

    

