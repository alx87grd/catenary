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
from powerline2221 import p2r_w
from powerline2221 import generate_test_data
from powerline import J
from powerline import EstimationPlot

    
############################
def basic_tracking_test():

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 70.  , 50. , 30 , 30 , 30 ])
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 40.  , 25.  , 25  , 25 , 25 , 25 ])
    
    pts = generate_test_data( p , partial_obs = True )
    
    plot = EstimationPlot( p , p_hat , pts , p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,3.14) , (100,2000) ,
              (40,60), (40,80) , (40,60) , (20,40), (20,40) , (20,40) ]
    
    params = [ 'sample' , 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 ,
                                        0.0001 , 0.002 , 0.002 , 0.002 , 
                                        0.002 , 0.002 , 0.002 ]) , 
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

    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 70.  , 50. , 30 , 30 , 30 ])
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 40.  , 25.  , 25  , 25 , 25 , 25 ])
    
    pts = generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = EstimationPlot( p , p_hat , pts , p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,3.14) , (100,2000) ,
              (40,60), (40,80) , (40,60) , (20,40), (20,40) , (20,40) ]
    
    params = [ 'sample' , 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 ,
                                        0.0001 , 0.002 , 0.002 , 0.002 , 
                                        0.002 , 0.002 , 0.002 ]) , 
              1.0 , 1.0 , 2 , 501 , -200 , 200 , p2r_w ] 
    
    
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
    
    
    # basic_tracking_test()
    
    hard_tracking_test()
    










