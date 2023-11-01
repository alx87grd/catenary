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


def basic_array3_convergence_test():
    
    p_1  =  np.array([ 28.0, 30.0, 77.0, 0.0, 53.0])
    p_2  =  np.array([ 28.0, 50.0, 77.0, 0.0, 53.0])
    p_3  =  np.array([ 28.0, 70.0, 77.0, 0.0, 53.0])
    # p_4  =  np.array([ 28.0, 35.0, 97.0, 0.0, 53.0])
    # p_5  =  np.array([ 28.0, 65.0, 97.0, 0.0, 53.0])
    
    pts1 = catenary.generate_test_data( p_1 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = 0 , x_max = 10)
    pts2 = catenary.generate_test_data( p_2 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = -20 , x_max = 30)
    pts3 = catenary.generate_test_data( p_3 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = -10 , x_max = 15)
    
    pts = np.hstack( ( pts1 , pts2 , pts3  ))
    
    p      =  np.array([  28.0, 50.0, 77.0, 0.0, 53, 20.0 ])
    
    p_init =  np.array([  10.0, 10.0, 10.0, 1.0, 80, 16. ])
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,0.3) , (10,200) , (15,30)]
    
    
    model  = powerline.ArrayModel()
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 55 , -50 , 50, model.p2r_w ] 
    
    start_time = time.time()
    plot = powerline.EstimationPlot( p , p_init , pts , model.p2r_w , 25, -50, 50)
    
    func = lambda p: powerline.J(p, pts, p_init, params)
    
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
    
    
############################
def basic_array32_convergence_test():
    
    model  = powerline.ArrayModel32()

    p_1  =  np.array([ 28.0, 30.0, 77.0, 0.0, 53.0])
    p_2  =  np.array([ 28.0, 50.0, 77.0, 0.0, 53.0])
    p_3  =  np.array([ 28.0, 70.0, 77.0, 0.0, 53.0])
    p_4  =  np.array([ 28.0, 35.0, 97.0, 0.0, 53.0])
    p_5  =  np.array([ 28.0, 65.0, 97.0, 0.0, 53.0])
    
    pts1 = catenary.generate_test_data( p_1 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = 0 , x_max = 10)
    pts2 = catenary.generate_test_data( p_2 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = -20 , x_max = 30)
    pts3 = catenary.generate_test_data( p_3 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = -10 , x_max = 15)
    pts4 = catenary.generate_test_data( p_4 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = -30 , x_max = 30)
    pts5 = catenary.generate_test_data( p_5 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = -20 , x_max = 20)
    
    pts = np.hstack( ( pts1 , pts2 , pts3 , pts4 , pts5 ))
    
    p      =  np.array([  28.0, 50.0, 77.0, 0.0, 53, 20.0, 15.0, 20.0 ])
    
    p_init =  np.array([  10.0, 10.0, 10.0, 1.0, 80, 16., 15., 16.0 ])
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,0.3) , (10,200) , (15,30), (15,15) , (15,30)]
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0]) ,
                1.0 , 1.0 , 2 , 55 , -50 , 50, model.p2r_w ] 
    
    start_time = time.time()
    plot = powerline.EstimationPlot( p , p_init , pts , model.p2r_w , 25, -50, 50)
    
    func = lambda p: powerline.J(p, pts, p_init, params)
    
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
    
    
############################
def basic_array32_tracking_test():
    
    model  = powerline.ArrayModel32()

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 40.  , 25.  , 25    ])
    
    pts = model.generate_test_data( p , partial_obs = True )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,3.14) , (100,2000) , (15,60), (15,50) , (15,50)]
    
    params = [ 'sample' , 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002]) ,
                1.0 , 1.0 , 2 , 501 , -200 , 200 , model.p2r_w ] 
    
    
    for i in range(500):
        
        pts = model.generate_test_data( p , partial_obs = True )
        
        plot.update_pts( pts )
    
        start_time = time.time()
        
        func = lambda p: powerline.J(p, pts, p_hat, params)
    
        
        res = minimize( func,
                        p_hat, 
                        method='SLSQP',  
                        bounds=bounds, 
                        #constraints=constraints,  
                        # callback=plot.update_estimation, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        
        plot.update_estimation( p_hat )
        
        
        print( f" Optimzation completed in : { time.time() - start_time } sec \n"     
                f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n" + 
                f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        

############################
def hard_array32_tracking_test():
    
    model  = powerline.ArrayModel32()

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 25.  , 50. ])
    p_hat  =  np.array([   0,   0,   0, 1.2, 500, 40.  , 25.  , 25    ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0.5,1.5) , (100,2000) , (30,60), (25,25) , (25,60)]
    
    params = [ 'sample' , 1 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002]) ,
                1.0 , 1.0 , 2 , 501 , -200 , 200, model.p2r_w ] 
    
    
    for i in range(500):
        
        pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -100, x_max = -70, n_out = 10 ,
                                             center = [-50,-50,-50] , w_o = 10 )
        
        plot.update_pts( pts )
    
        start_time = time.time()
        
        func = lambda p: powerline.J(p, pts, p_hat, params)
    
        
        res = minimize( func,
                        p_hat, 
                        method='SLSQP',  
                        bounds=bounds, 
                        #constraints=constraints,  
                        # callback=plot.update_estimation, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        
        plot.update_estimation( p_hat )
        
        
        print( f" Optimzation completed in : { time.time() - start_time } sec \n"     
                f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n" + 
                f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        

############################
def hard_array32_tracking_local_minima_analysis( model =  powerline.ArrayModel32() ):
    
    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 25.  , 50. ])
    p_hat  =  np.array([ 3.56, 26.8, 25.82, 1.05, 499.95, 44.12, 25.00, 28.1 ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -70, n_out = 10 ,
                                         center = [-50,-50,-50] , w_o = 10 )
    
    params = [ 'sample' , 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002]) ,
                1.0 , 1.0 , 2 , 501 , -200 , 200, model.p2r_w ] 
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    
    n = 200
    zs = np.linspace( -100 , 100, n )
    cs = np.zeros(n)
    
    
    for i in range(n):
        
        p_hat[2] = p[2] + zs[i]
        
        cs[i]    = powerline.J( p_hat, pts, p_hat, params)
    
        plot.update_estimation( p_hat )
        
    
    fig, ax = plt.subplots(1, figsize= (4, 3), dpi=300, frameon=True)
    
    ax = [ax]
    ax[0].plot( zs , cs  )
    ax[0].set_xlabel( 'z_hat', fontsize = 5)
    ax[0].set_ylabel( 'J(p)', fontsize = 5)
    ax[0].grid(True)
    ax[0].legend()
    ax[0].tick_params( labelsize = 5 )
    
    
############################
def basic_array2221_tracking_test():
    
    model = powerline.ArrayModel2221()

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 70.  , 50. , 30 , 30 , 30 ])
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 40.  , 25.  , 25  , 25 , 25 , 25 ])
    
    pts = model.generate_test_data( p , partial_obs = True )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,3.14) , (100,2000) ,
              (40,60), (40,80) , (40,60) , (20,40), (20,40) , (20,40) ]
    
    params = [ 'sample' , 2 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 ,
                                        0.0001 , 0.002 , 0.002 , 0.002 , 
                                        0.002 , 0.002 , 0.002 ]) , 
              1.0 , 1.0 , 2 , 501 , -200 , 200 , model.p2r_w ] 
    
    
    for i in range(50):
        
        pts = model.generate_test_data( p , partial_obs = True )
        
        plot.update_pts( pts )
    
        start_time = time.time()
        
        func = lambda p: powerline.J(p, pts, p_hat, params)
    
        
        res = minimize( func,
                        p_hat, 
                        method='SLSQP',  
                        bounds=bounds, 
                        #constraints=constraints,  
                        # callback=plot.update_estimation, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        
        plot.update_estimation( p_hat )
        
        
        print( f" Optimzation completed in : { time.time() - start_time } sec \n"     
                f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n" + 
                f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        

############################
def hard_array2221_tracking_test():
    
    model = powerline.ArrayModel2221()

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 70.  , 50. , 30 , 30 , 30 ])
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 40.  , 25.  , 25  , 25 , 25 , 25 ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,3.14) , (100,2000) ,
              (40,60), (40,80) , (40,60) , (20,40), (20,40) , (20,40) ]
    
    params = [ 'sample' , 1 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 ,
                                        0.0001 , 0.002 , 0.002 , 0.002 , 
                                        0.002 , 0.002 , 0.002 ]) , 
              1.0 , 1.0 , 2 , 501 , -200 , 200 , model.p2r_w ] 
    
    
    for i in range(100):
        
        pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -100, x_max = -70, n_out = 10 ,
                                             center = [-50,-50,-50] , w_o = 10 )
        
        plot.update_pts( pts )
    
        start_time = time.time()
        
        func = lambda p: powerline.J(p, pts, p_hat, params)
    
        
        res = minimize( func,
                        p_hat, 
                        method='SLSQP',  
                        bounds=bounds, 
                        #constraints=constraints,  
                        # callback=plot.update_estimation, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        
        plot.update_estimation( p_hat )
        
        
        print( f" Optimzation completed in : { time.time() - start_time } sec \n"     
                f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n" + 
                f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
        
############################
def hard_arrayconstant2221_tracking_test():
    
    model = powerline.ArrayModelConstant2221()

    p      =  np.array([  1, -3, 14, 0.2,  500, 4.  , 4  , 9  ])
    p_hat  =  np.array([  0,  0,  0, 0.0, 1000, 5.  , 5. , 10 ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    bounds = [ (-5,5), (-5,5) , (-15,15) , (-0.3,0.3) , (100,2000) ,
              (3,6), (3,6) , (5,15) ]
    
    params = [ 'sample' , 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002]) ,
                1.0 , 1.0 , 2 , 501 , -200 , 200, model.p2r_w ] 
    
    
    for i in range(500):
        
        pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -10, x_max = +20, n_out = 10 ,
                                             center = [-50,-50,-50] , w_o = 10 )
        
        plot.update_pts( pts )
    
        start_time = time.time()
        
        func = lambda p: powerline.J(p, pts, p_hat, params)
    
        
        res = minimize( func,
                        p_hat, 
                        method='SLSQP',  
                        bounds=bounds, 
                        #constraints=constraints,  
                        # callback=plot.update_estimation, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        
        plot.update_estimation( p_hat )
        
        
        print( f" Optimzation completed in : { time.time() - start_time } sec \n"     
                f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n" + 
                f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        
############################
def arrayconstant2221_cost_shape_analysis():
    
    model = powerline.ArrayModelConstant2221()
    
    p_hat =  np.array([  0,  0, 0, 0.0,  500, 5.  , 5  , 10  ])
    p     =  np.array([  0,  0, 0, 0.0,  500, 5.  , 5. , 10 ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20, model.p2r_w ] 
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    
    n = 100
    zs = np.linspace( -25 , 25, n )
    cs = np.zeros(n)
    
    
    for i in range(n):
        
        p_hat[2] = zs[i]
        
        cs[i]    = powerline.J( p_hat, pts, p_hat, params)
    
        plot.update_estimation( p_hat )
        
    
    fig, ax = plt.subplots(1, figsize= (4, 3), dpi=300, frameon=True)
    
    ax = [ax]
    ax[0].plot( zs , cs  )
    ax[0].set_xlabel( 'z_hat', fontsize = 5)
    ax[0].set_ylabel( 'J(p)', fontsize = 5)
    ax[0].grid(True)
    ax[0].legend()
    ax[0].tick_params( labelsize = 5 )
        


############################
def basic_array32_estimator_test():
    
    
    model  = powerline.ArrayModel32()

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 49.  , 29.  , 49    ])
    
    pts = model.generate_test_data( p , partial_obs = True )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    estimator = powerline.ArrayEstimator( model.p2r_w , p_hat )
    
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    for i in range(500):
        
        pts = model.generate_test_data( p , partial_obs = True )
        
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
    
    model  = powerline.ArrayModel32()
    
    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 25.  , 50. ])
    p_hat  =  np.array([   0,   0, 150, 1.2, 500, 51.  , 25.  , 49  ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 5 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    estimator = powerline.ArrayEstimator( model.p2r_w , p_hat )
    
    # estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.000002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    
    for i in range(50):
        
        pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -100, x_max = -70, n_out = 5 ,
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



############################
def array32_cost_shape_analysis( model =  powerline.ArrayModel32() ):
    
    p_hat =  np.array([  0,  0, 0, 0.0,  500, 5.  , 3  , 5  ])
    p     =  np.array([  0,  0, 0, 0.0,  500, 5.  , 3. , 5 ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20, model.p2r_w ] 
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    
    n = 100
    zs = np.linspace( -25 , 25, n )
    cs = np.zeros(n)
    
    
    for i in range(n):
        
        p_hat[2] = zs[i]
        
        cs[i]    = powerline.J( p_hat, pts, p_hat, params)
    
        plot.update_estimation( p_hat )
        
    
    fig, ax = plt.subplots(1, figsize= (4, 3), dpi=300, frameon=True)
    
    ax = [ax]
    ax[0].plot( zs , cs  )
    ax[0].set_xlabel( 'z_hat', fontsize = 5)
    ax[0].set_ylabel( 'J(p)', fontsize = 5)
    ax[0].grid(True)
    ax[0].legend()
    ax[0].tick_params( labelsize = 5 )
    
    

############################
def translation_search_test( search = True , var = 10 ):
    
    model  = powerline.ArrayModel32()
    
    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 25.  , 50. ])
    p_hat  =  np.array([   0,   0, 150, 1.2, 500, 51.  , 25.  , 49  ])
    
    pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 5 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = powerline.EstimationPlot( p , p_hat , pts , model.p2r_w )
    
    estimator = powerline.ArrayEstimator( model.p2r_w , p_hat )
    
    # estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.000002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    estimator.Q = 10 * np.diag([ 0.0002 , 0.0002 , 0.0 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002])
    
    
    for i in range(25):
        
        pts = model.generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -100, x_max = -70, n_out = 5 ,
                                             center = [-50,-50,-50] , w_o = 10 )
        
        plot.update_pts( pts )
    
        if search:
            p_hat  = estimator.solve_with_translation_search( pts , p_hat , var )
            
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
    
    
    # basic_array3_convergence_test()
    
    # basic_array32_convergence_test()
    
    # basic_array32_tracking_test()
    
    # hard_array32_tracking_test()
    
    # hard_array32_tracking_local_minima_analysis()
    
    # basic_array2221_tracking_test()
    
    # hard_array2221_tracking_test()
    
    # hard_arrayconstant2221_tracking_test()
    
    # arrayconstant2221_cost_shape_analysis()
    
    # basic_array32_estimator_test()
    
    # scan_z_test_test( False )
    
    # scan_z_test_test( True )
    
    # array32_cost_shape_analysis()
    
    translation_search_test( False )
    translation_search_test( True )

