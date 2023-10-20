#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:44:24 2023

@author: alex
"""
import numpy as np
import time
from scipy.optimize import minimize

import catenery


###########################
# Unit tests
###########################

def cat_test():
    
    t1 = ( catenery.cat( 0 , 1 )   == 0.0 )
    t2 = ( catenery.cat( 1.0 , 1 ) == 0.5430806348152437 )
    
    return (t1 & t2)


def w_T_c_test():
    
    T = catenery.w_T_c( 0 , 0 , 0 , 0 )
    
    return T.shape == (4,4)


def p2r_w_test():
    
    p = np.array([0,0,0,0,1])
    
    r_w, x_c = catenery.p2r_w( p , -50 , 50 , 101 )
    
    return (r_w.shape == (3,101) ) & ( x_c.shape == (101,) )



#######################







#######################

def noisy_p2r_w_test():
    
    p = np.array([0,0,0,0,1])
    
    r = catenery.noisy_p2r_w( p , -50 , 50 , 101 , 0.5  )
    
    return (r.shape == (3,101) ) 


def multiples_noisy_p2r_w_test():
    
    p = np.array([ [0,0,0,0,1] , [0,0,0,0,2] ] ).T
    
    r = catenery.multiples_noisy_p2r_w( p , [-10, -5], [10,-5] , [30,60] , [ 0.5 , 0.0 ] )
    
    return (r.shape == (3,90) ) 


def outliers_test():
    
    r = catenery.outliers( 10 , [1,2,3] , 50. )
    
    return (r.shape == (3,10) ) 


def outliers_test():
    
    r = catenery.outliers( 10 , [1,2,3] , 50. )
    
    return (r.shape == (3,10) ) 


def gradient_test():
    
    p_true  =  np.array([ 50.0,  50.0, 50.0, 1.0, 600.0])
    p       =  np.array([  0.0,   0.0,  0.0, 2.0, 800.0])
    
    pts = catenery.p2r_w( p_true , 50 )[0]
    
    # plot = CateneryEstimationPlot( p_true , p , pts)
    
    # pts  = generate_test_data( p_true , n_obs = 20 , x_min = -30,
    #                                    x_max = 30, w_l = 0.5, n_out = 10, 
    #                                    center = [50,50,50] , w_o = 100 )
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                            1.0 , 1.0 , 2 , 1000 , -200 , 200] 
    
    p_nom = p
    
    grad_analytical = catenery.dJ_dp( p , pts, p_nom, params , False )
    grad_numerical  = catenery.dJ_dp( p , pts, p_nom, params , True  )
    
    e = np.linalg.norm( grad_analytical - grad_numerical ) / np.linalg.norm( grad_numerical )
    
    succes = ( e < 0.1 )
    
    return succes





###########################
# Ploting tests
###########################

def plot_test():
    
    
    p_true  =  np.array([ 50 , 50 , 50 , 0.2 , 500 ])
    p_hat   =  np.array([  0 ,  0 ,  0 , 0   , 1000 ])
    
    pts   = catenery.generate_test_data( p_true , n_obs = 20 , x_min = -100, 
                                        x_max = 100, w_l = 0.5, n_out = 10, 
                                        center = [0,0,0] , w_o = 100 )

    plot  = catenery.CateneryEstimationPlot(  p_true , p_hat , pts )
    
    return True
    
    
    
def animation_test():
    
    
    p0      =  np.array([ 50 , 50 , 50 , 0.2 , 500 ])
    dp_dt   =  np.array([  1 ,  1 , 10 , 0.01 , 0  ])
    
    p_hat   =  np.array([  0 ,  0 ,  0 , 0   , 1000 ])
    
    t = np.linspace( 0 , 10, 101 )
    
    ( pts , p ) = catenery.generate_test_data_sequence( 0, p0 , dp_dt , partial_obs = True, 
                                     n_obs = 20 , x_min = -100, x_max = 100, 
                                     w_l = 0.5, n_out = 10, center = [0,0,0] , 
                                     w_o = 100 )

    plot  = catenery.CateneryEstimationPlot(  p , p_hat , pts )
    
    for i in range(101):
        
        ( pts , p )  = catenery.generate_test_data_sequence( t[i], p0 , dp_dt , 
                                                    partial_obs = True, 
                                         n_obs = 20 , x_min = -100, x_max = 100, 
                                         w_l = 0.5, n_out = 10, center = [0,0,0] , 
                                         w_o = 100 )
        plot.update_true( p )
        plot.update_pts( pts )
        
    
    return True
        

###########################
# Convergence tests
###########################
        


def convergence_basic_test( method = 'sample' ,  grad = False ):
    
    
    p_true  =  np.array([ 32.0, 43.0, 77.0, 1.3, 53.0])
    p_init  =  np.array([ 10.0, 10.0, 10.0, 2.0, 80.0])
    
    pts  = catenery.generate_test_data( p_true , n_obs = 20 , x_min = -30,
                                       x_max = 30, w_l = 0.5, n_out = 10, 
                                       center = [50,50,50] , w_o = 100 )
    
    plot  = catenery.CateneryEstimationPlot(  p_true , p_init , pts , 50 , -50 , +50 )
    
    bounds = [ (0,100), (0,100) , (0,100) , (0,3.14) , (10,200) ]
    
    params = [ method , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20] 
    
    start_time = time.time()
    
    func = lambda p: catenery.J(p, pts, p_init, params)
    
    if grad:
        jac = lambda p: catenery.dJ_dp( p, pts, p_init, params)
    else:
        jac = None
    
    res = minimize( func,
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    jac = jac,
                    #constraints=constraints,  
                    callback=plot.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
    
def tracking_basic_test( method = 'sample' ,  grad = False , partial_obs = False ):
    
    p0      =  np.array([ 32 , 43 , 77 , 1.3 ,  53 ])
    dp_dt   =  np.array([ 50 ,  0 ,  0 , 1.0 ,  0  ])
    
    p_hat   =  np.array([ 10 , 10 , 10 , 2.0 ,  80. ])
    
    t = np.linspace( 0 , 5, 51 )
    
    ( pts , p ) = catenery.generate_test_data_sequence( 0, p0 , dp_dt , partial_obs, 
                                     n_obs = 20 , x_min = -30, x_max = 30, 
                                     w_l = 0.5, n_out = 10, center = [50,50,50] , 
                                     w_o = 100 )
    
    plot  = catenery.CateneryEstimationPlot( p , p_hat , pts , 50 , -50 , +50 )
    
    for i in range(51):
        
        ( pts , p )  = catenery.generate_test_data_sequence( t[i], p0 , dp_dt , 
                                                    partial_obs, 
                                         n_obs = 20 , x_min = -30, x_max = 30, 
                                         w_l = 0.5, n_out = 10, center = [50,50,50] , 
                                         w_o = 100 )
        plot.update_true( p )
        plot.update_pts( pts )
        
        bounds = [ (0,500) , (0,500), (0,500) , (0,3.14) , (10,200) ]
        params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                    1.0 , 1.0 , 2 , 25 , -20 , 20] 
        
        start_time = time.time()
        
        func = lambda p: catenery.J(p, pts, p_hat, params)
        
        if grad:
            jac = lambda p: catenery.dJ_dp( p, pts, p_hat, params)
        else:
            jac = None
        
        res = minimize( func,
                        p_hat, 
                        method='SLSQP',  
                        bounds=bounds, 
                        jac=jac,
                        #constraints=constraints,  
                        # callback=plot.update_estimation, 
                        options={'disp':True,'maxiter':500})
        p_init = p_hat 
        p_hat  = res.x
        
        print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
                f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
                f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n" + 
                f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        plot.update_estimation( p_hat )
    
        
    
###########################
# Convergence test
###########################



if __name__ == "__main__":     
    """ MAIN TEST """
    
    print('catenery.cat: ', cat_test())
    print('catenery.w_T_c: ', w_T_c_test())
    print('catenery.p2r_w: ', p2r_w_test())
    print('catenery.noisy_p2r_w: ', noisy_p2r_w_test())
    print('catenery.multiples_noisy_p2r_w: ', multiples_noisy_p2r_w_test())
    print('catenery.outliers: ', outliers_test())
    print('catenery.dJ_dp: ', gradient_test())
    
    
    # plot_test()
    # animation_test()
    
    # convergence_basic_test()
    # convergence_basic_test( grad = True )
    
    # tracking_basic_test()
    tracking_basic_test( grad = True )
