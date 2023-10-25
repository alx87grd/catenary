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
from powerline32 import generate_test_data
from powerline import J
from powerline import EstimationPlot


############################
def basic_concergence_test():

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
    # p_init =  np.array([  20.0, 40.0, 60.0, 0.2, 70, 21., 21., 21.0 ])
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,0.3) , (10,200) , (15,30), (15,15) , (15,30)]
    
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0]) ,
                1.0 , 1.0 , 2 , 55 , -50 , 50, p2r_w ] 
    
    start_time = time.time()
    plot = EstimationPlot( p , p_init , pts , p2r_w , 25, -50, 50)
    
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
    
    
############################
def basic_tracking_test():

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 40.  , 25.  , 25    ])
    
    pts = generate_test_data( p , partial_obs = True )
    
    plot = EstimationPlot( p , p_hat , pts , p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,3.14) , (100,2000) , (5,100), (5,100) , (5,100)]
    
    params = [ 'sample' , 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002]) ,
                1.0 , 1.0 , 2 , 501 , -200 , 200 , p2r_w ] 
    
    
    for i in range(500):
        
        pts = generate_test_data( p , partial_obs = True )
        
        plot.update_pts( pts )
    
        start_time = time.time()
        
        func = lambda p: J(p, pts, p_hat, params)
    
        
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
def hard_tracking_test():

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 25.  , 50. ])
    p_hat  =  np.array([   0,   0,   0, 1.2, 300, 40.  , 25.  , 25    ])
    
    pts = generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = EstimationPlot( p , p_hat , pts , p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0.5,1.5) , (100,2000) , (5,100), (25,25) , (5,100)]
    
    params = [ 'sample' , 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002]) ,
                1.0 , 1.0 , 2 , 501 , -200 , 200, p2r_w ] 
    
    
    for i in range(500):
        
        pts = generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -100, x_max = -70, n_out = 10 ,
                                             center = [-50,-50,-50] , w_o = 10 )
        
        plot.update_pts( pts )
    
        start_time = time.time()
        
        func = lambda p: J(p, pts, p_hat, params)
    
        
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
    
    
    



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    # basic_concergence_test()
    
    basic_tracking_test()
    
    # hard_tracking_test()
    










