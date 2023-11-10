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


import catenary
import powerline



############################
def basic_array32_estimator_test():
    
    
    model  = powerline.ArrayModel32()

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 49.  , 29.  , 49    ])
    
    pts = model.generate_test_data( p , partial_obs = True )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    for i in range(500):
        
        pts = model.generate_test_data( p , partial_obs = True )
        
        plot.update_pts( pts )
        
        start_time = time.time()
        p_hat      = estimator.solve( pts , p_hat ) 
        solve_time = time.time() - start_time 
        
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        print( " Solve time : " + str(solve_time) + '\n' + 
               " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
    return estimator





############################
def translation_search_test( search = True , n = 3 , var = 10 ):
    
    model  = powerline.ArrayModel32()
    
    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 25.  , 50. ])
    p_hat  =  np.array([   0,   0, 150, 1.2, 500, 51.  , 25.  , 49  ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 5 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    # estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.000002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    
    for i in range(25):
        
        pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -100, x_max = -70, n_out = 5 ,
                                             center = [-50,-50,-50] , w_o = 10 )
        
        plot.update_pts( pts )
    
        if search:
            p_hat  = estimator.solve_with_translation_search( pts , p_hat , n , var )
            
        else:
            p_hat  = estimator.solve( pts , p_hat )
            
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        print( " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
    return estimator


############################
def hard_test( search = True , method = 'x' , n = 2, var = 10 ):
    
    model  = powerline.ArrayModel32()
    
    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 25.  , 50. ])
    p_hat  =  np.array([   0,   0, 150, 1.2, 500, 51.  , 26.  , 49  ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                    w_l = 0.2 , x_min = -100, x_max = -50, 
                                    n_out = 3 , center = [0,0,0] , w_o = 20 )
    
    pts = pts[:,:30] #remover one cable
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    # estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.000002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    estimator.d_th         = 3.0
    estimator.succes_ratio = 0.7
    estimator.method       = method
    
    
    for i in range(250):
        
        pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -100, x_max = -70, n_out = 5 ,
                                             center = [-50,-50,-50] , w_o = 10 )
        
        pts = pts[:,:30] #remover one cable
        
        plot.update_pts( pts )
        
        start_time = time.time()
        if search:
            p_hat  = estimator.solve_with_translation_search( pts , p_hat , n , var )
            
        else:
            p_hat  = estimator.solve( pts , p_hat )
            
        solve_time = time.time() - start_time 
            
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        print(  " Solve time : " + str(solve_time) + '\n' + 
                " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
    return estimator


############################
def very_hard_test( search = True , method = 'x' , n = 5, var = 100 ):
    
    model  = powerline.ArrayModel32()
    
    p      =  np.array([  50,  50,  50, 0.0, 200, 50.  , 25.  , 150. ])
    p_hat  =  np.array([   0,   0, 150, 0.2, 300, 51.  , 26.  , 149  ])
    
    ps = model.p2ps( p )
    
    ps[4,3] = 2000
    ps[4,4] = 2000
    
    pts0 = catenary.generate_test_data( ps[:,0], n_obs = 10, n_out = 2 , x_min = -50 , x_max = 50)
    pts1 = catenary.generate_test_data( ps[:,1], n_obs = 10, n_out = 2 , x_min = -50 , x_max = 50)
    pts2 = catenary.generate_test_data( ps[:,2], n_obs = 10, n_out = 2 , x_min = -50 , x_max = 50)
    pts3 = catenary.generate_test_data( ps[:,3], n_obs = 10, n_out = 2 , x_min = -50 , x_max = 50)
    pts4 = catenary.generate_test_data( ps[:,4], n_obs = 10, n_out = 2 , x_min = -50 , x_max = 50)
    
    # pts = np.hstack( ( pts0 , pts1 , pts2 , pts3 , pts4 ))
    
    pts = np.hstack( ( pts1 , pts3 , pts4 ))
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    # estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.000002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    estimator.d_th         = 5.0
    estimator.succes_ratio = 0.5
    estimator.method       = method
    
    
    for i in range(100):
        
        pts0 = catenary.generate_test_data( ps[:,0], n_obs = 10, n_out = 2 , x_min = -50 , x_max = 50)
        pts1 = catenary.generate_test_data( ps[:,1], n_obs = 7, n_out = 2 , x_min = -50 , x_max = 50)
        pts2 = catenary.generate_test_data( ps[:,2], n_obs = 10, n_out = 2 , x_min = -50 , x_max = 50)
        pts3 = catenary.generate_test_data( ps[:,3], n_obs = 6, n_out = 2 , x_min = -50 , x_max = 50)
        pts4 = catenary.generate_test_data( ps[:,4], n_obs = 3, n_out = 2 , x_min = -30 , x_max = 20)
        
        # pts = np.hstack( ( pts0 , pts1 , pts2 , pts3 , pts4 ))
        
        pts = np.hstack( ( pts1 , pts3 , pts4 ))
        
        plot.update_pts( pts )
        
        start_time = time.time()
        
        if search:
            p_hat  = estimator.solve_with_translation_search( pts , p_hat , n , var )
            
        else:
            p_hat  = estimator.solve( pts , p_hat )
            
        solve_time = time.time() - start_time 
            
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        print(  " Solve time : " + str(solve_time) + '\n' + 
                " Target acquired: " + str(target) + '\n' +
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
    
    
    # basic_array32_estimator_test()
    
    # translation_search_test( False )
    # translation_search_test( True )
    
    # hard_test( method = 'sample' )
    # hard_test( method = 'x' )
    
    very_hard_test()

