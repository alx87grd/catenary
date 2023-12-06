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


from catenary import singleline as catenary
from catenary import powerline


###########################
# Powerline Model
###########################


############################
def basic_array32_estimator_test( n_steps = 50 ):
    
    
    model  = powerline.ArrayModel32()

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 49.  , 29.  , 49    ])
    
    pts = model.generate_test_data( p , partial_obs = True )
    
    plot  = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    plot2 = powerline.ErrorPlot( p , p_hat , n_steps )
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    for i in range( n_steps ):
        
        pts = model.generate_test_data( p , partial_obs = True )
        
        plot.update_pts( pts )
        
        start_time = time.time()
        p_hat      = estimator.solve( pts , p_hat ) 
        solve_time = time.time() - start_time 
        
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        plot2.save_new_estimation( p_hat , solve_time )
        
        
        print( " Solve time : " + str(solve_time) + '\n' + 
               " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
    plot2.plot_error_mean_std()
        
        
    return estimator



############################
def basic_array_constant2221_estimator_test( n = 5 , var = 5. ):
    
    
    model  = powerline.ArrayModelConstant2221()

    p      =  np.array([  50,  50,  50, 1.3, 600, 4.  , 5.  , 6. ])
    p_hat  =  np.array([   0,   0,   0, 1.0, 800, 3.  , 4.  , 7. ])
    
    pts = model.generate_test_data( p , partial_obs = False )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.000001 , 0.002 , 0.002 , 0.002])
    estimator.n_search = n
    estimator.p_var    = np.array([ var, var, var , 0 , 0 , 0, 0 , 0])
    
    for i in range(100):
        
        pts = model.generate_test_data( p , partial_obs = True )
        
        plot.update_pts( pts )
        
        start_time = time.time()
        p_hat      = estimator.solve_with_search( pts , p_hat )
        solve_time = time.time() - start_time 
        
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        print( " Solve time : " + str(solve_time) + '\n' + 
               " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
    return estimator


############################
def hard_array_constant2221_estimator_test( n = 2 , var = 5. ):
    
    
    model  = powerline.ArrayModelConstant2221()

    p      =  np.array([  15, -20,  50, 0.3, 600, 5.  , 5.  , 6. ])
    p_hat  =  np.array([   0,   0,   0, 0.0, 800, 4.  , 4.  , 7. ])
    
    pts = model.generate_test_data( p , n_obs = 10 , x_min = -50, x_max = -40, 
                            w_l = 0.5, n_out = 3, center = [0,0,0] , 
                            w_o = 10, partial_obs = False)
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.000001 , 0.002 , 0.002 , 0.002])
    estimator.n_search = n
    estimator.p_var    = np.array([ var, var, var , 0 , 0 , 0, 0 , 0])
    
    for i in range(100):
        
        pts = model.generate_test_data(  p , n_obs = 6 , x_min = -50, x_max = -40, 
                                w_l = 0.5, n_out = 3, center = [0,0,0] , 
                                w_o = 10, partial_obs = False)
        
        plot.update_pts( pts )
        
        start_time = time.time()
        p_hat      = estimator.solve_with_search( pts , p_hat )
        solve_time = time.time() - start_time 
        
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        print( " Solve time : " + str(solve_time) + '\n' + 
               " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
    return estimator





############################
def translation_search_test( search = True , n = 3 , var = 10. ):
    
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
    estimator.n_search = n
    estimator.p_var    = np.array([ var, var, var, 0 , 0 , 0, 0 , 0])
    
    
    for i in range(50):
        
        pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -100, x_max = -70, n_out = 5 ,
                                             center = [-50,-50,-50] , w_o = 10 )
        
        plot.update_pts( pts )
    
        if search:
            p_hat  = estimator.solve_with_search( pts , p_hat )
            
        else:
            p_hat  = estimator.solve( pts , p_hat )
            
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        print( " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
    return estimator


############################
def hard_test( search = True , method = 'x' , n = 2, var = 10 ,  n_steps = 50 ):
    
    model  = powerline.ArrayModel32()
    
    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 25.  , 50. ])
    p_hat  =  np.array([   0,   0, 150, 1.2, 500, 51.  , 26.  , 49  ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                    w_l = 0.2 , x_min = -100, x_max = -50, 
                                    n_out = 3 , center = [0,0,0] , w_o = 20 )
    
    pts = pts[:,:30] #remover one cable
    
    plot  = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    plot2 = powerline.ErrorPlot( p , p_hat , n_steps )
    plot.plot_model( p_hat )
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    # estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.000002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    estimator.Q = 0 * np.diag([ 0.0002 , 0.0002 , 0.0 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    estimator.n_search = n
    estimator.p_var    = np.array([ var, var, var, 0 , 0 , 0, 0 , 0])
    
    estimator.d_th         = 3.0
    estimator.succes_ratio = 0.7
    estimator.method       = method
    
    
    for i in range( n_steps ):
        
        pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -100, x_max = -70, n_out = 5 ,
                                             center = [-50,-50,-50] , w_o = 10 )
        
        pts = pts[:,:30] #remover one cable
        
        plot.update_pts( pts )
        
        start_time = time.time()
        if search:
            p_hat  = estimator.solve_with_search( pts , p_hat )
            
        else:
            p_hat  = estimator.solve( pts , p_hat )
            
        solve_time = time.time() - start_time 
            
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        plot2.save_new_estimation( p_hat , solve_time )
        
        
        print(  " Solve time : " + str(solve_time) + '\n' + 
                " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
    plot2.plot_error_mean_std()
        
        
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
    
    estimator.n_search = n
    estimator.p_var    = np.array([ var, var, var, 0 , 0 , 0, 0 , 0])
    
    estimator.d_th         = 5.0
    estimator.succes_ratio = 0.5
    estimator.method       = method
    
    
    for i in range(50):
        
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
            p_hat  = estimator.solve_with_search( pts , p_hat )
            
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
def quad_test( search = True , method = 'x' , n = 5, var = 10 ):
    
    
    # 3 x quad powerlines
    quad  = powerline.Quad()
    
    p4_1  =  np.array([  50,  40,  50, 0.0, 500, 0.2  , 0.4  ])
    p4_2  =  np.array([  50,  50,  50, 0.0, 500, 0.2  , 0.4  ])
    p4_3  =  np.array([  50,  60,  50, 0.0, 500, 0.2  , 0.4  ])
    
    pts4_1 = quad.generate_test_data( p4_1 , n_obs = 16, w_l = 0.05 , n_out = 2 , x_min = +80 , x_max = 100)
    pts4_2 = quad.generate_test_data( p4_2 , n_obs = 16, w_l = 0.05 , n_out = 2 , x_min = +80 , x_max = 100)
    pts4_3 = quad.generate_test_data( p4_3 , n_obs = 16, w_l = 0.05 , n_out = 2 , x_min = +80 , x_max = 100)
    
    # 2x guard cables
    pg1    =  np.array([  50,  45,  60, 0.0, 800 ])
    pg2    =  np.array([  50,  55,  60, 0.0, 800 ])

    pts_g1 = catenary.generate_test_data( pg1, n_obs = 10, n_out = 2 , x_min = +80 , x_max = 100)
    pts_g2 = catenary.generate_test_data( pg2, n_obs = 10, n_out = 2 , x_min = +80 , x_max = 100)
    
    pts = np.hstack( ( pts4_1 , pts4_2 , pts4_3 , pts_g1 , pts_g2 ))
    
    #  Estimation Model
    model  = powerline.ArrayModel32()
    
    p_true =  np.array([  50,  50,  50, 0.0, 500, 10.  , 5.  , 10. ])
    p_hat  =  np.array([  0,  0,  0, 0.3, 800, 9.  , 4.  , 9. ])
    
    
    plot = powerline.EstimationPlot( p_true , p_hat , pts , model.p2r_w )
    
    callback = None #plot.update_estimation
    
    estimator   = powerline.ArrayEstimator( model , p_hat )
    
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    estimator.n_search = n
    estimator.p_var    = np.array([ var, var, var, 0 , 0 , 0, 0 , 0])
    
    estimator.method       = method
    
    for i in range(50):
        
        # 3 x quad powerlines
        quad  = powerline.Quad()
        
        p4_1  =  np.array([  50,  40,  50, 0.0, 500, 0.2  , 0.4  ])
        p4_2  =  np.array([  50,  50,  50, 0.0, 500, 0.2  , 0.4  ])
        p4_3  =  np.array([  50,  60,  50, 0.0, 500, 0.2  , 0.4  ])
        
        pts4_1 = quad.generate_test_data( p4_1 , n_obs = 16, w_l = 0.05 , n_out = 2 , x_min = +80 , x_max = 100)
        pts4_2 = quad.generate_test_data( p4_2 , n_obs = 16, w_l = 0.05 , n_out = 2 , x_min = +80 , x_max = 100)
        pts4_3 = quad.generate_test_data( p4_3 , n_obs = 16, w_l = 0.05 , n_out = 2 , x_min = +80 , x_max = 100)
        
        # 2x guard cables
        pg1    =  np.array([  50,  45,  60, 0.0, 800 ])
        pg2    =  np.array([  50,  55,  60, 0.0, 800 ])

        pts_g1 = catenary.generate_test_data( pg1, n_obs = 10, n_out = 2 , x_min = +80 , x_max = 100)
        pts_g2 = catenary.generate_test_data( pg2, n_obs = 10, n_out = 2 , x_min = +80 , x_max = 100)
        
        pts = np.hstack( ( pts4_1 , pts4_2 , pts4_3 , pts_g1 , pts_g2 ))
        
        plot.update_pts( pts )
    
        start_time = time.time()
        
        if search:
            p_hat  = estimator.solve_with_search( pts , p_hat , callback )
            
        else:
            p_hat  = estimator.solve( pts , p_hat , callback )
            
        solve_time = time.time() - start_time 
            
        target = estimator.is_target_aquired( p_hat , pts)
        
        plot.update_estimation( p_hat )
        
        
        print(  " Solve time : " + str(solve_time) + '\n' + 
                " Target acquired: " + str(target) + '\n' +
                f" p_true : {np.array2string(p_true, precision=2, floatmode='fixed')}  \n" +
                f" p_hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
############################
def global_convergence_test( n_steps = 100 ):
    
    xm = -200
    xp = 200
    
    model  = powerline.ArrayModel32()

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    p_hat  =  np.array([ 100, 100, 100, 0.6, 300, 49.  , 29.  , 49    ])
    
    pts = model.generate_test_data( p , partial_obs = True )
    
    plot  = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    plot2 = powerline.ErrorPlot( p , p_hat , n_steps )
    
    estimator = powerline.ArrayEstimator( model , p_hat )
    
    estimator.Q = 1.0 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.01 , 0.00001 , 0.002 , 0.002 , 0.002])
    
    estimator.n_search = 3
    estimator.p_var    = np.array([ 50., 50., 50., 0 , 0 , 0, 0 , 0])
    
    estimator.b = 200
    
    plot.plot_model( p_hat )
    
    for i in range(n_steps):
        
        pts = model.generate_test_data( p , partial_obs = True , x_min = xm, x_max = xp , n_out = 500 , w_o = 50.0 , center = [0,0,-200])
        
        plot.update_pts( pts )
        
        start_time = time.time()
        p_hat      = estimator.solve_with_search( pts , p_hat )
        solve_time = time.time() - start_time 
        
        plot.update_estimation( p_hat )
        
        n_tot  = pts.shape[1] - 100.0
        pts_in = estimator.get_array_group( p_hat , pts )
        n_in   = pts_in.shape[1] /  n_tot * 100.0
        
        plot2.save_new_estimation( p_hat , solve_time , n_in )
        
    plot2.plot_error_mean_std()
    
    

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    
    # basic_array32_estimator_test( 100 )
    
    # basic_array_constant2221_estimator_test()
    # hard_array_constant2221_estimator_test()
    
    # translation_search_test( False )
    translation_search_test( True )
    
    # hard_test( method = 'sample' )
    # hard_test( method = 'x' , n_steps = 100 )
    
    # very_hard_test()
    
    # quad_test()
    
    global_convergence_test()

