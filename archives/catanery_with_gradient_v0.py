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
def catanery( x , a = 1 ):
    """
    inputs
    --------
    x : position along cable in local frame ( x = 0 is z_min )
    a : slag parameter

    
    output
    -------
    z : elevation in local frame ( z = 0 is z_min )
    
    Note on local cable frame 
    --------------------------
    origin (0,0,0) is lowest elevation point on the cable
    x-axis is tangential to the cable at the origin
    y-axis is the normal of the plane formed by the cable
    z-axis is positive for greater elevation
    
    """
    
    z = a * ( np.cosh( x / a )  - 1 )
    
    return z


############################
def T( phi , x , y , z):
    """ 
    inputs
    ----------
    phi : z rotation of local frame basis in in world frame
    x   : x translation of local frame orign in world frame
    y   : y translation of local frame orign in world frame
    z   : z translation of local frame orign in world frame
    
    outputs
    ----------
    world_T_local  : 4x4 Transformation Matrix
    
    """
    
    s = np.sin( phi )
    c = np.cos( phi )
    
    T = np.array([ [ c   , -s ,  0  , x ] , 
                   [ s   ,  c ,  0  , y ] ,
                   [ 0   ,  0 ,  1. , z ] ,
                   [ 0   ,  0 ,  0  , 1.] ])
    
    return T


##################################
def catanery_x2r_world( xs , p ):
    """ 
    inputs
    --------
    xs  : positions along cable in local frame ( x = 0 is z_min )
    
    p   : vector of parameters [ a , phi , x_0 , y_0 , z_0 ]
        a   : slag parameter
        phi : z rotation of local frame basis in in world frame
        x_0 : x translation of local frame orign in world frame
        y_0 : y translation of local frame orign in world frame
        z_0 : z translation of local frame orign in world frame
    
    outputs
    ----------
    r[0,:] : x position in world frame
    r[1,:] : y position in world frame
    r[2,:] : z position in world frame
    
    """
    
    # params
    a   = p[0]
    phi = p[1]
    x_0 = p[2]
    y_0 = p[3]
    z_0 = p[4]
    
    # local frame pts
    zs      = catanery( xs , a )
    
    n            = xs.shape[0]
    r_local      = np.zeros((4,n))
    r_local[0,:] = xs
    r_local[2,:] = zs
    r_local[3,:] = np.ones((n))
    
    r_world = T( phi, x_0, y_0, z_0 ) @ r_local
    
    return r_world[0:3,:]



############################
def catanery_p2pts( p, n = 400, x_min = -200, x_max = 200):
    """ 
    inputs
    --------
    p      : vector of parameters [ a , phi , x_p , y_p , z_p ]
    n      : number of pts
    x_min  : start of pts points in cable local frame 
    x_max  : end   of pts points in cable local frame
    
    outputs
    ----------
    r[0,:] : xs position in world frame
    r[1,:] : ys position in world frame
    r[2,:] : zs position in world frame
    
    """
    
    xs_local  = np.linspace( x_min , x_max, n )
    pts_world = catanery_x2r_world( xs_local , p )
    
    return ( pts_world , xs_local ) 


############################
def generate_noisy_test_data( p, 
                             n_line = 20, x_min = -200, x_max = 200, w_line = 0.5,
                             n_out = 10, center = [0,0,0], w_out = 100):
    """  
    p      : vector of true parameters [ a , phi , x_p , y_p , z_p ]
    
    n_line : numbers of measurements point on the line
    x_min  : start of measurements points in cable local frame 
    x_max  : end   of measurements points in cable local frame
    w_line : variance of cable measurements noise
    
    n_out  : additionnal outliers points
    centre : center point of the distribution of outlier points
    w_out  : variance of outlier point distributions
    
    """
    
    # number of points
    n = n_line + n_out
    
    # Outliers randoms points (all pts are initialized as random outliers)
    rng   = np.random.default_rng( seed = None )
    noise = w_out * rng.standard_normal(( 3 , n ))
    pts   = np.zeros(( 3 , n ))
    
    pts[0,:] = center[0]
    pts[1,:] = center[1]
    pts[2,:] = center[2]
    
    pts = pts + noise   # noise randoms points
    
    # true points on the curve
    pts_line = catanery_p2pts( p, n_line, x_min, x_max)[0]
        
    # adding measurements noise
    pts_line   = pts_line + w_line * rng.standard_normal(( 3 , n_line ))
    
    # 
    pts[:,0:n_line] = pts_line
    
    
    return pts


# ###########################
# Plotting
# ###########################


###############################################################################
class CataneryEstimationPlot:
    """ 
    """
    
    ############################
    def __init__(self, p_true , p_hat , pts , n = 100 , xmin = -200, xmax = 200):
        
        
        fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
        ax  = fig.add_subplot(projection='3d')
        
        pts_true  = catanery_p2pts( p_true , n , xmin , xmax )[0]
        pts_hat   = catanery_p2pts( p_hat  , n , xmin , xmax )[0]
        pts_noisy = pts
    
        line_true  = ax.plot( pts_true[0,:]  , pts_true[1,:]  , pts_true[2,:] , label= 'True equation' )
        line_hat   = ax.plot( pts_hat[0,:]   , pts_hat[1,:]   , pts_hat[2,:]  , '--', label= 'Fitted equation' )
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
        
        pts_hat   = catanery_p2pts( p_hat ,  self.n , self.xmin , self.xmax  )[0]
        
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
        
        pts_true   = catanery_p2pts( p_true , self.n , self.xmin , self.xmax )[0]
        
        self.line_true[0].set_data( pts_true[0,:] , pts_true[1,:]  )
        self.line_true[0].set_3d_properties( pts_true[2,:] )
        
        plt.pause( 0.001 )
        

# ###########################
# # Optimization
# ###########################
        
        
############################
def lorentzian( x , l = 1.0 ):
    """ 
    x : input
    a : flattening parameter
    y : output
    """
    
    y = np.log10( 1 + x**2 / l )
    
    return y

############################
def plot_lorentzian( l = 10 , ax = None ):
    """ 
    x : input
    a : flattening parameter
    """
    
    x = np.linspace( -50.0 , 50.0 , 10000 ) 
    y = lorentzian( x , l )

    if ax is None:
        fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
        ax  = fig.add_subplot(1, 1, 1)

    ax.plot( x , y , label= r'$l =$ %0.1f' % l )
    ax.set_xlabel( 'x', fontsize = 5)
    ax.grid(True)
    ax.legend()
    ax.tick_params( labelsize = 5 )
    ax.set_ylabel( 'lorentzian(x)', fontsize = 5 )
    
    return ax


############################
def dmin_sample( p_hat , pts , n_sample = 1000 , x_min = -200, x_max = 200 ):
    """ """
    
    # generate a list of sample point on the model curve
    pts_hat = catanery_p2pts( p_hat, n_sample , x_min , x_max )[0]
    
    # ############################
    # Vectorized version
    
    # Vectors between measurements and model sample pts
    deltas    = pts[:,:,np.newaxis] - pts_hat[:,np.newaxis,:]
    
    # Distances between measurements and model sample pts
    distances = np.linalg.norm( deltas , axis = 0 )
    
    # Minimum distances to model for all measurements
    d_min = distances.min( axis = 1 )
    
    return d_min

############################
def cost_dmin_sample( p_hat , pts , l = 1.0 ):
    """ """
    
    d_min = dmin_sample( p_hat , pts )
    
    # Cost shaping function
    pts_cost = lorentzian( d_min , l )
    
    # Sum of all costs
    cost = pts_cost.sum()
    
    return cost


############################
def grad_cost_dmin_sample( p , pts , l = 1.0 ):
    """ """
    
    # generate a list of sample point on the model curve
    pts_hat, xs_local = catanery_p2pts( p )

    # Vectors between measurements and model sample pts
    deltas    = pts[:,:,np.newaxis] - pts_hat[:,np.newaxis,:]
    
    # Distances between measurements and model sample pts
    distances = np.linalg.norm( deltas , axis = 0 )
    
    # Minimum distances to model for all measurements
    d_min       = distances.min( axis = 1 )
    j_min_index = distances.argmin( axis = 1 ) # index of closet model point

    
    # Errors vector to closet model point
    
    i_index = np.arange( j_min_index.shape[0] )
    
    e_min   = deltas[ : , i_index , j_min_index ]
    xl_min  = xs_local[ j_min_index  ]
    
    # pts_cost = lorentzian( d_min , l )
    # cost     = pts_cost.sum()
    
    # print('jmin:', j_min_index )
    # print('dmin:', d_min )
    # print('emin:\n', e_min )
    # print('pts_cost:\n', pts_cost )
    # print('cost:', cost ,'\n')
    
    # Error Grad for each pts
    de_dp = np.zeros( (5 , pts.shape[1] ) )
    
    # a   : slag parameter
    # phi : z rotation of local frame basis in in world frame
    # x_0 : x translation of local frame orign in world frame
    # y_0 : y translation of local frame orign in world frame
    # z_0 : z translation of local frame orign in world frame
    
    a   = p[0]
    phi = p[1]
    
    de_dp[0,:] = e_min[2,:] * ( np.cosh( xl_min / a ) - xl_min / a * np.sinh( xl_min / a ) - 1 )
    de_dp[1,:] = e_min[0,:] * ( - np.sin( phi ) * xl_min ) +  e_min[1,:] * ( + np.cos( phi ) * xl_min )
    de_dp[2,:] = e_min[0,:]
    de_dp[3,:] = e_min[1,:]
    de_dp[4,:] = e_min[2,:]
    
    # print('de_dp:\n', de_dp ,'\n')
    
    # Cost shaping function
    dc_dp =  de_dp *  ( 2.0 / ( np.log( 10 ) * ( l + d_min ** 2 ) ) ) 
    
    # print('dc_dp:\n', dc_dp ,'\n')
    
    # Sum of all costs
    dCost_dp = -dc_dp.sum( axis = 1 )
    
    return dCost_dp

############################
def grad_cost_dmin_sample_numerical( p , pts , l = 1.0 ):
    """ """
    
    dCost_dp = np.zeros(5)
    dp = np.array([ 10.0 , 0.01, 0.5 , 0.5 , 0.5 ])
    # dp = np.array([ 0.001 , 0.001, 0.001 , 0.001 , 0.001 ])
    
    for i in range(5):
    
        pp    = p.copy()
        pm    = p.copy()
        pp[i] = p[i] + dp[i]
        pm[i] = p[i] - dp[i]
        cp    = cost_dmin_sample( pp , pts , l )
        cm    = cost_dmin_sample( pm , pts , l )
        
        dCost_dp[i] = ( cp - cm ) / ( 2.0 * dp[i] )

    return dCost_dp


############################
def cost_local_yz( p_hat , pts ):
    """ """
    
    n = pts.shape[1]
    
    a_hat     = p_hat[0]
    w_T_local = T( p_hat[1] , p_hat[2] , p_hat[3] , p_hat[4])
    local_T_w = np.linalg.inv( w_T_local )
    
    pts_world = np.vstack( [pts , np.ones(n) ] )
    
    # Compute measurements points positions in local cable frame
    pts_local  = local_T_w @ pts_world
    
    # Compute expected z position based on x coord and catanery model
    zs_hat     = catanery( pts_local[0,:] , a_hat )
    
    # Compute delta vector between measurements and expected position
    delta      = np.zeros((3,n))
    delta[1,:] = pts_local[1,:]
    delta[2,:] = pts_local[2,:] - zs_hat
    
    norm = np.linalg.norm( delta , axis = 0 )
    
    costs = lorentzian( norm , 1)
    
    return costs.sum()



# ###########################
# # Segmentation
# ###########################

############################
def get_catanery_group( p_hat , pts , d_th = 1.0 , n_sample = 100 ):
    
    # Distance to catanery curves
    d_min = dmin_sample( p_hat , pts , n_sample , -50 , 50 )
    
    # Group based on threshlod
    pts_in = pts[ : , d_min < d_th ]
    
    return pts_in


# ###########################
# Tests
# ###########################

def speed_test():
    
    p_true  =  np.array([ 530.0, 1.3, 32.0, 43.0, 77.0])
    p_init  =  np.array([ 800.0, 2.0, 10.0, 10.0, 10.0])
    
    pts  = generate_noisy_test_data( p_true , center = [50,50,50] )
    
    # plot = CataneryEstimationPlot( p_true , p_init , pts)
    
    # ###########################
    # # Optimization
    # ###########################
    
    bounds = [ (100,2000) , (0,3.14), (0,100), (0,100) , (0,100) ]
    
    start_time = time.time()
    
    res = minimize( lambda p: cost_dmin_sample(p, pts), 
    # res = minimize( lambda p: cost_local_yz(p, pts), #cost_dmin_sample(p, pts), 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    #constraints=constraints,  
                    # callback=plot.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )


def compare_cost_functions():
    
    p_true  =  np.array([  530.0, 1.3, 32.0, 43.0, 77.0])
    p_init  =  np.array([ 1000.0, 2.0, 10.0, 10.0, 10.0])
    
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
    
    plot2 = CataneryEstimationPlot( p_true , p_init , pts)
    
    start_time = time.time()
    
    res = minimize( lambda p: cost_local_yz(p, pts), 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    #constraints=constraints,  
                    callback=plot2.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
    
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
    
    
def tracking_test():
    
    
    p_true  =  np.array([ 530.0, 1.3, 32.0, 43.0, 77.0])
    p_init  =  np.array([ 800.0, 2.0, 10.0, 10.0, 10.0])

    pts  = generate_noisy_test_data( p_true , center = [50,50,50])

    plot = CataneryEstimationPlot( p_true , p_init , pts)

    for i in range(50):
        
        p_true  =  np.array([ 530.0, 1.3 + 0.1 * i, 32.0 + 5.0 * i, 43.0, 77.0])
        
        plot.update_true( p_true )
        
        pts  = generate_noisy_test_data( p_true , center = [50 + 5.0 * i,50,50])
        
        plot.update_pts( pts )
        
        bounds = [ (100,2000) , (0,3.14), (0,500), (0,500) , (0,500) ]
        
        start_time = time.time()
        
        res = minimize( lambda p: cost_local_yz(p, pts), 
                        p_init, 
                        method='SLSQP',  
                        bounds=bounds, 
                        #constraints=constraints,  
                        # callback=plot.update_estimation, 
                        options={'disp':False,'maxiter':500})
        
        p_hat = res.x
        
        print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
               f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
               f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
               f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        plot.update_estimation( p_hat )
        
        p_init = p_hat


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
    
    # speed_test()
    
    # compare_cost_functions()
    # compare_gradient()
    
    # compare_gradient_speed()
    
    # tracking_test()
    
    # grouping_test()
    # grouping_test2()
    
    # test_gradient()
    
    p_true  =  np.array([ 600.0, 1.0,  50.0,  50.0, 50.0])
    p       =  np.array([ 800.0, 0.0,  0.0,  0.0, 0.0])
    
    pts = catanery_p2pts( p_true , 35 )[0]
    
    plot = CataneryEstimationPlot( p_true , p , pts)
    
    # pts  = generate_noisy_test_data( p_true , center = [50,50,50])
    
    grad_analytical = grad_cost_dmin_sample( p , pts )
    grad_numerical  = grad_cost_dmin_sample_numerical( p , pts )
    
    print( grad_analytical )
    print( grad_numerical )
    
    # # p2 = p + np.array([1.0,0,0,0,0])
    
    # # delta = cost_dmin_sample( p2 , pts ) - cost_dmin_sample( p , pts)
    
    # # print( delta )
    
    # l = 1.0
    
    # # generate a list of sample point on the model curve
    # pts_hat, xs_local = catanery_p2pts( p )
    
    
    # # Vectors between measurements and model sample pts
    # deltas    = pts[:,:,np.newaxis] - pts_hat[:,np.newaxis,:]
    
    # # Distances between measurements and model sample pts
    # distances = np.linalg.norm( deltas , axis = 0 )
    
    # # Minimum distances to model for all measurements
    # d_min       = distances.min( axis = 1 )
    # j_min_index = distances.argmin( axis = 1 ) # index of closet model point

    
    
    # # Errors vector to closet model point
    
    # i_index = np.arange( j_min_index.shape[0] )
    
    # e_min   = deltas[ : , i_index , j_min_index ]
    # xl_min  = xs_local[ j_min_index  ]
    
    # pts_cost = lorentzian( d_min , l )
    # cost     = pts_cost.sum()
    
    # print('jmin:', j_min_index )
    # print('dmin:', d_min )
    # print('emin:\n', e_min )
    # print('pts_cost:\n', pts_cost )
    # print('cost:', cost ,'\n')
    
    
    # # Error Grad for each pts
    # de_dp = np.zeros( (5 , pts.shape[1] ) )
    
    # # a   : slag parameter
    # # phi : z rotation of local frame basis in in world frame
    # # x_0 : x translation of local frame orign in world frame
    # # y_0 : y translation of local frame orign in world frame
    # # z_0 : z translation of local frame orign in world frame
    
    # a   = p[0]
    # phi = p[1]
    
    # de_dp[0,:] = e_min[2,:] * ( np.cosh( xl_min / a ) - xl_min / a * np.sinh( xl_min / a ) - 1 )
    # de_dp[1,:] = e_min[0,:] * ( - np.sin( phi ) * xl_min ) +  e_min[1,:] * ( + np.cos( phi ) * xl_min )
    # de_dp[2,:] = e_min[0,:]
    # de_dp[3,:] = e_min[1,:]
    # de_dp[4,:] = e_min[2,:]
    
    # print('de_dp:\n', de_dp ,'\n')
    
    # # Cost shaping function
    # dc_dp =  de_dp *  ( 2.0 / ( np.log( 10 ) * ( l + d_min ** 2 ) ) ) 
    
    # print('dc_dp:\n', dc_dp ,'\n')
    
    # # Sum of all costs
    # dCost_dp = -dc_dp.sum( axis = 1 )
    
    # print( dCost_dp.T )
    # print( grad_numerical.T )
    
    # cost_nom = cost_dmin_sample( p , pts )
    # p2 = p.copy()
    # delta = 1.0
    # p2[4] = p2[4] + delta
    # cost_2 = cost_dmin_sample( p2 , pts )
    
    # print( cost_2 ,'\n', cost_nom ,'\n', (cost_2 - cost_nom) / delta )

