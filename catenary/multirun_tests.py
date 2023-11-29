#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:15:46 2023

@author: alex
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time


import catenary
import powerline


###############################################################################
def ArrayModelEstimatorTest(   save    = True,
                               plot    = True,
                               name    = 'test' , 
                               n_run   = 5,
                               n_steps = 10,
                               # Model
                               model   = powerline.ArrayModel32(),
                               p_hat   = np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ]),
                               p_ub    = np.array([ 150, 150, 150, 2.0, 900, 51.  , 31.  , 51. ]),
                               p_lb    = np.array([   0,   0,   0, 0.0, 300, 49.  , 29.  , 49. ]),
                               # Fake data Distribution param
                               n_obs = 20, 
                               n_out = 10,
                               x_min = -200, 
                               x_max = 200, 
                               w_l   = 0.5,  
                               w_o   = 100,
                               center = [0,0,0] , 
                               partial_obs = False,
                               # Solver param
                               n_sea    = 2, 
                               var      = 10 ,
                               Q        = 0.0 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002]),
                               l        = 1.0,
                               power    = 2.0,
                               b        = 1.0,
                               method   = 'x',
                               n_s      = 100,
                               x_min_s  = -200,
                               x_max_s  = +200,
                               use_grad = True):
    
    
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    estimator.Q         = Q
    estimator.p_lb      = p_lb
    estimator.p_ub      = p_ub
    estimator.method    = method
    estimator.Q         = Q
    estimator.b         = b
    estimator.l         = l
    estimator.power     = power
    estimator.x_min     = x_min_s
    estimator.x_max     = x_max_s
    estimator.n_sample  = n_s
    estimator.d_th      = w_l * 5.0
    
    
    for j in range(n_run):
        
        print('Run no' , j)
        
        # Alway plot the 3d graph for the last run
        if j == (n_run-1): plot = True
        
        # Random true line position
        p_true  = np.random.uniform( p_lb , p_ub )
        
        if plot: plot_3d = powerline.EstimationPlot( p_true , p_hat , None, model.p2r_w )
        
        if j == 0: e_plot = powerline.ErrorPlot( p_true , p_hat , n_steps , n_run )
        else:      e_plot.init_new_run( p_true , p_hat )
        
    
        for i in range(n_steps):
            
            # Generate fake noisy data
            pts = model.generate_test_data( p_true, 
                                            n_obs, 
                                            x_min, 
                                            x_max,
                                            w_l, 
                                            n_out, 
                                            center, 
                                            w_o, 
                                            partial_obs
                                            )
            
            if plot: plot_3d.update_pts( pts )
            
            start_time = time.time()
            ##################################################################
            p_hat      = estimator.solve_with_translation_search( pts, 
                                                                  p_hat, 
                                                                  n_sea, 
                                                                  var,
                                                                  use_grad)
            ##################################################################
            solve_time = time.time() - start_time 
            
            if plot: plot_3d.update_estimation( p_hat )
            
            ##################################################################
            n_tot  = pts.shape[1] - n_out
            pts_in = estimator.get_array_group( p_hat , pts )
            n_in   = pts_in.shape[1] /  n_tot * 100
            ##################################################################
            
            # print(pts.shape,pts_in.shape)
            
            e_plot.save_new_estimation( p_hat , solve_time , n_in )
        
        # Plot pts_in
        if plot : plot_3d.add_pts( pts_in )
            
    # Finalize figures
    if save: plot_3d.save( name = name )
    e_plot.plot_error_all_run( save = save , name = name )
    

###############################################################################
def GlobalConvergenceTest( n_run = 5 , plot = False , save = True ):
    
    # Baseline:
    save    = save
    plot    = plot
    name    = 'GlobalConvergence'
    n_run   = n_run
    n_steps = 200
    model   = powerline.ArrayModel32()
    p_hat   = np.array([  50,  50,  50, 1.0, 300, 50.  , 30.  , 50. ])
    p_ub    = np.array([ 150, 150, 150, 2.0, 900, 51.  , 31.  , 51. ])
    p_lb    = np.array([   0,   0,   0, 0.0, 300, 49.  , 29.  , 49. ])
    # Fake data Distribution param
    n_obs = 20 
    n_out = 100
    x_min = -200 
    x_max = 200 
    w_l   = 0.5  
    w_o   = 50.0 
    center = [0,0,-200]
    partial_obs = True
    # Solver param
    n_sea    = 3 
    var      = 50 
    Q        = 0.0001 * np.diag([ 20. , 20. , 20. , 1000.0 , 1.0, 200.0, 200.0 , 200.0 ])
    l        = 1.0
    power    = 2.0
    b        = 1.0
    method   = 'x'
    n_s      = 100
    x_min_s  = -200
    x_max_s  = +200
    use_grad = True

    
    ArrayModelEstimatorTest(save,
                            plot,
                            name, 
                            n_run,
                            n_steps,
                            model,
                            p_hat,
                            p_ub,
                            p_lb,
                            n_obs,
                            n_out,
                            x_min,
                            x_max,
                            w_l,
                            w_o,
                            center, 
                            partial_obs,
                            n_sea,
                            var,
                            Q,
                            l,
                            power,
                            b,
                            method,
                            n_s,
                            x_min_s,
                            x_max_s,
                            use_grad)
    
    
###############################################################################
def PartialObsTest( n_run = 5 , plot = False , save = True ):
    
    # Baseline:
    save    = save
    plot    = plot
    name    = 'PartialObs'
    n_run   = n_run
    n_steps = 200
    model   = powerline.ArrayModel32()
    p_hat   = np.array([  50,  50,  50, 1.0, 300, 50.  , 30.  , 50. ])
    p_ub    = np.array([ 150, 150, 150, 2.0, 900, 51.  , 31.  , 51. ])
    p_lb    = np.array([   0,   0,   0, 0.0, 300, 49.  , 29.  , 49. ])
    # Fake data Distribution param
    n_obs = 16 
    n_out = 5
    x_min = -100 
    x_max = -70 
    w_l   = 0.5  
    w_o   = 10.0 
    center = [-50,-50,-50]
    partial_obs = True
    # Solver param
    n_sea    = 3 
    var      = 50 
    Q        = 0.0001 * np.diag([ 20. , 20. , 20. , 1000.0 , 1.0, 200.0, 200.0 , 200.0 ])
    l        = 1.0
    power    = 2.0
    b        = 1.0
    method   = 'x'
    n_s      = 100
    x_min_s  = -200
    x_max_s  = +200
    use_grad = True

    
    ArrayModelEstimatorTest(save,
                            plot,
                            name, 
                            n_run,
                            n_steps,
                            model,
                            p_hat,
                            p_ub,
                            p_lb,
                            n_obs,
                            n_out,
                            x_min,
                            x_max,
                            w_l,
                            w_o,
                            center, 
                            partial_obs,
                            n_sea,
                            var,
                            Q,
                            l,
                            power,
                            b,
                            method,
                            n_s,
                            x_min_s,
                            x_max_s,
                            use_grad)
    
    
    
    

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    # ArrayModelEstimatorTest()
    
    # GlobalConvergenceTest( 2 , True , False )
    # PartialObsTest( 2 , True , False )
    
    GlobalConvergenceTest( 2 , False , True )
    PartialObsTest( 2 , False , True )