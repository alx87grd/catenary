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
def global_convergence_test( n_steps = 100 , n_run = 20 ):
    
    xm = -200
    xp = 200
    
    # xm = 10
    # xp = 20
    
    
    model  = powerline.ArrayModel32()

    p      =  np.array([  50,  50,  50, 1.0,  300, 50.  , 30.  , 50. ])
    p_l    =  np.array([-100,-100,   0, 0.0,  200, 48.  , 28.  , 48. ])
    p_u    =  np.array([ 100, 100, 200, 2.0,  800, 60.  , 50.  , 60. ])
    
    
    plot  = powerline.EstimationPlot( p , p , None , model.p2r_w )
    
    
    
    
    estimator = powerline.ArrayEstimator( model , p )
    
    estimator.Q = 1.0 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.01 , 0.00001 , 0.002 , 0.002 , 0.002])
    
    estimator.p_lb = p_l
    estimator.p_ub = p_u
    
    
    for j in range(n_run):
        
        print('Run no' , j)
        
        p_hat  = np.random.uniform( p_l , p_u )
        
        plot.update_estimation( p_hat )
        
        if j == 0:
            plot2 = powerline.ErrorPlot( p , p_hat , n_steps , n_run )
        
        else:
            plot2.init_new_run( p_hat )
        
    
            for i in range(n_steps):
                
                pts = model.generate_test_data( p , partial_obs = False , x_min = xm, x_max = xp , n_out = 10 , w_o = 50.0 , center = [0,0,-200])
                
                # plot.update_pts( pts )
                
                start_time = time.time()
                p_hat      = estimator.solve_with_translation_search( pts , p_hat , n = 3 , var = 50 )
                solve_time = time.time() - start_time 
                
                # plot.update_estimation( p_hat )
                plot2.save_new_estimation( p_hat , solve_time )
                
            # plot2.plot_error_single_run()
            
    plot2.plot_error_all_run()
    
    

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    global_convergence_test()

    

