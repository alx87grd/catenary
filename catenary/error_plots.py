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
def error_plot_test():
    
    xm = -200
    xp = 200
    
    # xm = 10
    # xp = 20
    
    
    model  = powerline.ArrayModel32()

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    p_hat  =  np.array([ 100, 100, 100, 0.6, 300, 49.  , 29.  , 49    ])
    
    pts = model.generate_test_data( p , partial_obs = True )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    estimator.Q = 10.0 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.01 , 0.00001 , 0.002 , 0.002 , 0.002])
    
    steps = 100
    
    frame = np.arange(steps)
    p_e   = np.zeros((8,steps))
    
    for i in range(steps):
        
        pts = model.generate_test_data( p , partial_obs = True , x_min = xm, x_max = xp , n_out = 100 , w_o = 50.0 , center = [0,0,-200])
        
        plot.update_pts( pts )
        
        start_time = time.time()
        p_hat      = estimator.solve_with_translation_search( pts , p_hat , n = 3 , var = 50 )
        solve_time = time.time() - start_time 
        
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        # print( " Solve time : " + str(solve_time) + '\n' + 
        #        " Target acquired: " + str(target) + '\n' +
        #         f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
        #         f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        p_e[:,i]   = p - p_hat
        # error[i] = np.linalg.norm( p_e[0] )
        # error[i] = np.linalg.norm( p_e[1] )
        # error[i] = np.linalg.norm( p_e[2] )
        # # error[i] = np.linalg.norm( p_e[4:] )
        
    fig, ax = plt.subplots(1, figsize= (4, 3), dpi=300, frameon=True)
    
    ax.plot( frame , p_e[0,:] , label= '$x_o$' )
    ax.plot( frame , p_e[1,:] , label= '$y_o$' )
    ax.plot( frame , p_e[2,:] , label= '$z_o$' )
    
    ax.legend( loc = 'upper right' , fontsize = 5)
    ax.set_xlabel( 'steps', fontsize = 5)
    ax.grid(True)
    
    
    
    fig, ax = plt.subplots(1, figsize= (4, 3), dpi=300, frameon=True)
    
    ax.plot( frame , p_e[3,:] , label= '$\psi$' )
    
    ax.legend( loc = 'upper right' , fontsize = 5)
    ax.set_xlabel( 'steps', fontsize = 5)
    ax.grid(True)
    
    fig, ax = plt.subplots(1, figsize= (4, 3), dpi=300, frameon=True)
    
    ax.plot( frame , p_e[4,:] , label= '$a$' )
    
    # ax.legend( loc = 'upper right' , fontsize = 5)
    ax.set_xlabel( 'steps', fontsize = 5)
    ax.grid(True)
    
    fig, ax = plt.subplots(1, figsize= (4, 3), dpi=300, frameon=True)
    
    ax.plot( frame , p_e[5,:] , label= '$d_1$' )
    ax.plot( frame , p_e[6,:] , label= '$d_2$' )
    ax.plot( frame , p_e[7,:] , label= '$h_1$' )
    
    # ax.legend( loc = 'upper right' , fontsize = 5)
    ax.set_xlabel( 'steps', fontsize = 5)
    ax.grid(True)
        
        
        
        
    return estimator


    

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    
    estimator = error_plot_test()
    
    

