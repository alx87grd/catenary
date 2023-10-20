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

from catanery import catanery
from catanery import T
from catanery import catanery_p2pts
from catanery import generate_noisy_test_data
from catanery import CataneryEstimationPlot
from catanery import lorentzian


##################################
def triple_catanery_x2r_world( xs , p ):
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
        d   : distance between cables
    
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
    d   = p[5]
    
    # local frame pts
    zs      = catanery( xs , a )
    
    delta_y = np.array([ -d , 0 , d ])
    delta_z = np.array([ +0 , 0 , 0 ])
    
    n       = xs.shape[0]
    n_cable = delta_y.shape[0]
    
    r_local = np.zeros(( 4 , n_cable , n ))
    
    for cable_id in range( n_cable ):
    
        r_local[0,cable_id,:] = xs
        r_local[1,cable_id,:] = delta_y[cable_id]
        r_local[2,cable_id,:] = zs + delta_z[cable_id]
        r_local[3,cable_id,:] = np.ones((n))
        
    r_local_all = r_local.reshape( ( 4 , n_cable * n ) )
    
    r_world = T( phi, x_0, y_0, z_0 ) @ r_local_all
    
    return r_world[0:3,:]


##################################
def six_catanery_x2r_world( xs , p ):
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
        d   : distance between cables
        h   : height between cables
    
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
    d   = p[5]
    h   = p[6]
    
    # local frame pts
    zs      = catanery( xs , a )
    
    delta_y = np.array([ -d , 0 , d , -d , 0 , d ])
    delta_z = np.array([ +0 , 0 , 0 , +h , h , h ])
    
    n       = xs.shape[0]
    n_cable = delta_y.shape[0]
    
    r_local = np.zeros(( 4 , n_cable , n ))
    
    for cable_id in range( n_cable ):
    
        r_local[0,cable_id,:] = xs
        r_local[1,cable_id,:] = delta_y[cable_id]
        r_local[2,cable_id,:] = zs + delta_z[cable_id]
        r_local[3,cable_id,:] = np.ones((n))
        
    r_local_all = r_local.reshape( ( 4 , n_cable * n ) )
    
    r_world = T( phi, x_0, y_0, z_0 ) @ r_local_all
    
    return r_world[0:3,:]




############################
def triple_catanery_p2pts( p, n = 50, x_min = -50, x_max = 50):
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
    
    xs_local = np.linspace( x_min , x_max, n )
    pts      = triple_catanery_x2r_world( xs_local , p )
    
    return pts


############################
def six_catanery_p2pts( p, n = 50, x_min = -50, x_max = 50):
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
    
    xs_local = np.linspace( x_min , x_max, n )
    pts      = six_catanery_x2r_world( xs_local , p )
    
    return pts


############################
def dmin_sample3( p_hat , pts , n_sample = 55 , x_min = -50, x_max = 50 ):
    """ """
    
    # generate a list of sample point on the model curve
    pts_hat = triple_catanery_p2pts( p_hat, n_sample , x_min , x_max )
    
    #############################
    # Vectorized version
    #############################
    
    # Vectors between measurements and model sample pts
    deltas    = pts[:,:,np.newaxis] - pts_hat[:,np.newaxis,:]
    
    # Distances between measurements and model sample pts
    distances = np.linalg.norm( deltas , axis = 0 )
    
    # Minimum distances to model for all measurements
    d_min = distances.min( axis = 1 )
    
    return d_min


############################
def cost_dmin_sample3( p_hat , pts ):
    """ """
    
    d_min = dmin_sample3( p_hat , pts )
    
    # Cost shaping function
    pts_cost = lorentzian( d_min , 1.0 )
    
    # Sum of all costs
    cost = pts_cost.sum()
    
    return cost


############################
def dmin_sample6( p_hat , pts , n_sample = 55 , x_min = -50, x_max = 50 ):
    """ """
    
    # generate a list of sample point on the model curve
    pts_hat = six_catanery_p2pts( p_hat, n_sample , x_min , x_max )
    
    #############################
    # Vectorized version
    #############################
    
    # Vectors between measurements and model sample pts
    deltas    = pts[:,:,np.newaxis] - pts_hat[:,np.newaxis,:]
    
    # Distances between measurements and model sample pts
    distances = np.linalg.norm( deltas , axis = 0 )
    
    # Minimum distances to model for all measurements
    d_min = distances.min( axis = 1 )
    
    return d_min


############################
def cost_dmin_sample6( p_hat , pts ):
    """ """
    
    d_min = dmin_sample6( p_hat , pts )
    
    # Cost shaping function
    pts_cost = lorentzian( d_min , 1.0 )
    
    # Sum of all costs
    cost = pts_cost.sum()
    
    return cost





# ###########################
# Tests
# ###########################





'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    # speed_test()
    
    # compare_cost_functions()
    
    # tracking_test()
    
    # grouping_test()
    # grouping_test2()
    
    if True:
        
        p_1  =  np.array([ 53.0, 0.0, 28.0, 30.0, 77.0])
        p_2  =  np.array([ 53.0, 0.0, 28.0, 50.0, 77.0])
        p_3  =  np.array([ 53.0, 0.0, 28.0, 70.0, 77.0])
        p_4  =  np.array([ 53.0, 0.0, 28.0, 30.0, 97.0])
        p_5  =  np.array([ 53.0, 0.0, 28.0, 50.0, 97.0])
        p_6  =  np.array([ 53.0, 0.0, 28.0, 70.0, 97.0])
        
        pts1 = generate_noisy_test_data( p_1 , center = [50,50,50] , x_min = 0 , x_max = 10)
        pts2 = generate_noisy_test_data( p_2 , center = [50,50,50] , x_min = -20 , x_max = 30)
        pts3 = generate_noisy_test_data( p_3 , center = [50,50,50] , x_min = -10 , x_max = 15)
        pts4 = generate_noisy_test_data( p_4 , center = [50,50,50] , x_min = -30 , x_max = 30)
        pts5 = generate_noisy_test_data( p_5 , center = [50,50,50] , x_min = -20 , x_max = 20)
        pts6 = generate_noisy_test_data( p_6 , center = [50,50,50] , x_min = -30 , x_max = -10)
        
        pts = np.hstack( ( pts1 , pts2 , pts3 , pts4 , pts5 , pts6 ))
        
        p_init  =  np.array([ 80.0, 2.0, 10.0, 10.0, 10.0])
        p_init6 =  np.array([ 80.0, 2.0, 10.0, 10.0, 10.0, 16.0 , 16.0])
        
        
        
        # ###########################
        # # Optimization
        # ###########################
        
        
        start_time = time.time()
        
        # ###########################
        # # Optimization
        # ###########################
        
        # bounds = [ (10,200) , (-1.0,1.0), (0,100), (0,100) , (0,100) ]
        
        # plot1 = CataneryEstimationPlot( p_1 , p_init , pts )
        
        # start_time = time.time()
        
        # res = minimize( lambda p: cost_dmin_sample(p, pts), 
        #                 p_init, 
        #                 method='SLSQP',  
        #                 bounds=bounds, 
        #                 #constraints=constraints,  
        #                 callback=plot1.update_estimation, 
        #                 options={'disp':True,'maxiter':500})
        
        plot = CataneryEstimationPlot( p_1 , p_init6 , pts , xmin = -50, xmax = 50 )
        
        bounds = [ (10,200) , (-1.0,1.0), (0,100), (0,100) , (0,100) , (15,30) , (15,30) ]
        
        res = minimize( lambda p: cost_dmin_sample6(p, pts), 
                        p_init6, 
                        method='SLSQP',  
                        bounds=bounds, 
                        #constraints=constraints,  
                        callback=plot.update_estimation, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        
        print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
               f" Init: {np.array2string(p_init, precision=1, floatmode='fixed')} \n" +
               f" True 1: {np.array2string(p_1, precision=1, floatmode='fixed')} \n" + 
               f" True 2: {np.array2string(p_2, precision=1, floatmode='fixed')} \n" + 
               f" Hat : {np.array2string(p_hat, precision=1, floatmode='fixed')}  \n" )
        
        pts_hat = six_catanery_p2pts( p_hat )
        
        pts_six = pts_hat.reshape((3, 6 , -1 ))
        
        for i in range(6):
            pts = pts_six[:,i,:]
            plot.ax.plot( pts[0,:] , pts[1,:] , pts[2,:], '-' , label= 'Hat')
    

