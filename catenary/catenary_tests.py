#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:44:24 2023

@author: alex
"""
import numpy as np
import time
from scipy.optimize import minimize

import catenary


###########################
# Unit tests
###########################

def cat_test():
    
    t1 = ( catenary.cat( 0 , 1 )   == 0.0 )
    t2 = ( catenary.cat( 1.0 , 1 ) == 0.5430806348152437 )
    
    return (t1 & t2)


def w_T_c_test():
    
    T = catenary.w_T_c( 0 , 0 , 0 , 0 )
    
    return T.shape == (4,4)


def p2r_w_test():
    
    p = np.array([0,0,0,0,1])
    
    r_w, x_c = catenary.p2r_w( p , -50 , 50 , 101 )
    
    return (r_w.shape == (3,101) ) & ( x_c.shape == (101,) )



#######################







#######################

def noisy_p2r_w_test():
    
    p = np.array([0,0,0,0,1])
    
    r = catenary.noisy_p2r_w( p , -50 , 50 , 101 , 0.5  )
    
    return (r.shape == (3,101) ) 


def multiples_noisy_p2r_w_test():
    
    p = np.array([ [0,0,0,0,1] , [0,0,0,0,2] ] ).T
    
    r = catenary.multiples_noisy_p2r_w( p , [-10, -5], [10,-5] , [30,60] , [ 0.5 , 0.0 ] )
    
    return (r.shape == (3,90) ) 


def outliers_test():
    
    r = catenary.outliers( 10 , [1,2,3] , 50. )
    
    return (r.shape == (3,10) ) 


def outliers_test():
    
    r = catenary.outliers( 10 , [1,2,3] , 50. )
    
    return (r.shape == (3,10) ) 


def gradient_test( method = 'sample' ):
    
    p_true  =  np.array([ 50.0,  50.0, 50.0, 1.0, 600.0])
    p       =  np.array([  0.0,   0.0,  0.0, 2.0, 800.0])
    
    pts = catenary.p2r_w( p_true , 50 )[0]
    
    # plot = catenaryEstimationPlot( p_true , p , pts)
    
    # pts  = generate_test_data( p_true , n_obs = 20 , x_min = -30,
    #                                    x_max = 30, w_l = 0.5, n_out = 10, 
    #                                    center = [50,50,50] , w_o = 100 )
    
    params = [ method, np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                            1.0 , 1.0 , 2 , 1000 , -200 , 200] 
    
    p_nom = p
    
    grad_analytical = catenary.dJ_dp( p , pts, p_nom, params , False )
    grad_numerical  = catenary.dJ_dp( p , pts, p_nom, params , True  )
    
    print( 'Analytical grad:\n' , grad_analytical )
    print( 'Numerical  grad:\n' , grad_numerical )
    
    e = np.linalg.norm( grad_analytical - grad_numerical ) / np.linalg.norm( grad_numerical )
    
    succes = ( e < 0.1 )
    
    return succes




###########################
# Ploting tests
###########################

def plot_test():
    
    
    p_true  =  np.array([ 50 , 50 , 50 , 0.2 , 500 ])
    p_hat   =  np.array([  0 ,  0 ,  0 , 0   , 1000 ])
    
    pts   = catenary.generate_test_data( p_true , n_obs = 20 , x_min = -100, 
                                        x_max = 100, w_l = 0.5, n_out = 10, 
                                        center = [0,0,0] , w_o = 100 )

    plot  = catenary.CatenaryEstimationPlot(  p_true , p_hat , pts )
    
    return True
    
    
    
def animation_test():
    
    
    p0      =  np.array([ 50 , 50 , 50 , 0.2 , 500 ])
    dp_dt   =  np.array([  1 ,  1 , 10 , 0.01 , 0  ])
    
    p_hat   =  np.array([  0 ,  0 ,  0 , 0   , 1000 ])
    
    t = np.linspace( 0 , 10, 101 )
    
    ( pts , p ) = catenary.generate_test_data_sequence( 0, p0 , dp_dt , partial_obs = True, 
                                     n_obs = 20 , x_min = -100, x_max = 100, 
                                     w_l = 0.5, n_out = 10, center = [0,0,0] , 
                                     w_o = 100 )

    plot  = catenary.CatenaryEstimationPlot(  p , p_hat , pts )
    
    for i in range(101):
        
        ( pts , p )  = catenary.generate_test_data_sequence( t[i], p0 , dp_dt , 
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
    
    pts  = catenary.generate_test_data( p_true , n_obs = 20 , x_min = -30,
                                       x_max = 30, w_l = 0.5, n_out = 10, 
                                       center = [50,50,50] , w_o = 100 )
    
    plot  = catenary.CatenaryEstimationPlot(  p_true , p_init , pts , 50 , -50 , +50 )
    
    bounds = [ (0,100), (0,100) , (0,100) , (0,3.14) , (10,200) ]
    
    params = [ method , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20] 
    
    start_time = time.time()
    
    func = lambda p: catenary.J(p, pts, p_init, params)
    
    if grad:
        jac = lambda p: catenary.dJ_dp( p, pts, p_init, params)
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
    
    ( pts , p ) = catenary.generate_test_data_sequence( 0, p0 , dp_dt , partial_obs, 
                                     n_obs = 20 , x_min = -30, x_max = 30, 
                                     w_l = 0.5, n_out = 10, center = [50,50,50] , 
                                     w_o = 100 )
    
    plot  = catenary.CatenaryEstimationPlot( p , p_hat , pts , 50 , -50 , +50 )
    
    for i in range(51):
        
        ( pts , p )  = catenary.generate_test_data_sequence( t[i], p0 , dp_dt , 
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
        
        func = lambda p: catenary.J(p, pts, p_hat, params)
        
        if grad:
            jac = lambda p: catenary.dJ_dp( p, pts, p_hat, params)
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
        
        
        
def tracking_advanced_test( method = 'sample' ,  grad = False , partial_obs = False ):
    
    p0      =  np.array([  0 ,  0 ,  0 , 0.2 , 500 ])
    dp_dt   =  np.array([  1 ,  1 , 10 , 0.01 , 0  ])
    
    p_hat   =  np.array([  -100 ,  -100 ,  -100 , 0.5   , 400 ])
    
    t = np.linspace( 0 , 2, 21 )
    
    ( pts , p ) = catenary.generate_test_data_sequence( 0, p0 , dp_dt , partial_obs = True, 
                                     n_obs = 20 , x_min = -100, x_max = 100, 
                                     w_l = 0.5, n_out = 10, center = [0,0,0] , 
                                     w_o = 100 )

    plot  = catenary.CatenaryEstimationPlot(  p , p_hat , pts )
    
    for i in range(21):
        
        ( pts , p )  = catenary.generate_test_data_sequence( t[i], p0 , dp_dt , 
                                                    partial_obs = True, 
                                         n_obs = 20 , x_min = -100, x_max = 100, 
                                         w_l = 0.5, n_out = 10, center = [0,0,0] , 
                                         w_o = 100 )
        plot.update_true( p )
        plot.update_pts( pts )
        
        bounds = [ (0,500) , (0,500), (0,500) , (0,3.14) , (100,2000) ]
        # params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
        #             1.0 , 1.0 , 2 , 25 , -20 , 20] 
        
        params = [ 'sample' , 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 ]) ,
                    1.0 , 1.0 , 2 , 25 , -20 , 20] 
        
        start_time = time.time()
        
        func = lambda p: catenary.J(p, pts, p_hat, params)
        grad = lambda p: catenary.dJ_dp( p, pts, p_hat, params)
        
        res = minimize( func,
                        p_hat, 
                        method='SLSQP',  
                        bounds=bounds, 
                        jac=grad,
                        #constraints=constraints,  
                        callback=plot.update_estimation, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        
        print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
               f" True: {np.array2string(p, precision=2, floatmode='fixed')} \n" + 
               f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        plot.update_estimation( p_hat )
    
        
    
###########################
# Grouping test
###########################



def grouping_test():
    
    
    p_true  =  np.array([ 32.0, 43.0, 77.0, 1.3, 53.0])
    p_init  =  np.array([ 10.0, 10.0, 10.0, 2.0, 80.0])
    
    pts  = catenary.generate_test_data( p_true , n_obs = 20 , x_min = -30,
                                       x_max = 30, w_l = 0.5, n_out = 10, 
                                       center = [50,50,50] , w_o = 100 )
    
    plot  = catenary.CatenaryEstimationPlot(  p_true , p_init , pts , 50 , -50 , +50 )
    
    bounds = [ (0,100), (0,100) , (0,100) , (0,3.14) , (10,200) ]
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20] 
    
    start_time = time.time()
    
    func = lambda p: catenary.J(p, pts, p_init, params)
    # grad = lambda p: catenary.dJ_dp( p, pts, p_init, params)
    
    res = minimize( func, 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    # jac=grad,
                    #constraints=constraints,  
                    callback=plot.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
    pts_in = catenary.get_catanery_group( p_hat , pts , 10.0 , n_sample = 600 , x_min = -30, x_max = 30)
    
    plot.ax.plot( pts_in[0,:] , pts_in[1,:] , pts_in[2,:], 'o' , label= 'Group')
    
    
###########################
# Speed tests
###########################
        


def speed_test():
    
    
    p_true  =  np.array([ 32.0, 43.0, 77.0, 1.3, 53.0])
    p_init  =  np.array([ 10.0, 10.0, 10.0, 2.0, 80.0])
    
    pts  = catenary.generate_test_data( p_true , n_obs = 20 , x_min = -30,
                                       x_max = 30, w_l = 0.5, n_out = 10, 
                                       center = [50,50,50] , w_o = 100 )

    
    
    bounds = [ (0,100), (0,100) , (0,100) , (0,3.14) , (10,200) ]
    
    ###########################
    # 1
    ###########################
    
    method = 'sample'
    grad   = False
    
    params = [ method , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20] 
    
    func = lambda p: catenary.J(p, pts, p_init, params)
    
    if grad:
        jac = lambda p: catenary.dJ_dp( p, pts, p_init, params)
    else:
        jac = None
    
    start_time = time.time()
    
    res = minimize( func,
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    jac = jac,
                    #constraints=constraints,  
                    # callback=plot.update_estimation, 
                    options={'disp':False,'maxiter':500})
    
    t1 = time.time() - start_time
    p1 = res.x
    
    ###########################
    # 2
    ###########################
    
    method = 'sample'
    grad   = True
    
    params = [ method , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20] 
    
    func = lambda p: catenary.J(p, pts, p_init, params)
    
    if grad:
        jac = lambda p: catenary.dJ_dp( p, pts, p_init, params)
    else:
        jac = None
    
    start_time = time.time()
    
    res = minimize( func,
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    jac = jac,
                    #constraints=constraints,  
                    # callback=plot.update_estimation, 
                    options={'disp':False,'maxiter':500})
    
    t2 = time.time() - start_time
    p2 = res.x
    
    ###########################
    # 3
    ###########################
    
    method = 'x'
    grad   = False
    
    params = [ method , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20] 
    
    func = lambda p: catenary.J(p, pts, p_init, params)
    
    if grad:
        jac = lambda p: catenary.dJ_dp( p, pts, p_init, params)
    else:
        jac = None
    
    start_time = time.time()
    
    res = minimize( func,
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    jac = jac,
                    #constraints=constraints,  
                    # callback=plot.update_estimation, 
                    options={'disp':False,'maxiter':500})
    
    t3 = time.time() - start_time
    p3 = res.x
    
    ###########################
    # 4
    ###########################
    
    method = 'x'
    grad   = True
    
    params = [ method , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20] 
    
    func = lambda p: catenary.J(p, pts, p_init, params)
    
    if grad:
        jac = lambda p: catenary.dJ_dp( p, pts, p_init, params)
    else:
        jac = None
    
    start_time = time.time()
    
    res = minimize( func,
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    jac = jac,
                    #constraints=constraints,  
                    # callback=plot.update_estimation, 
                    options={'disp':False,'maxiter':500})
    
    t4 = time.time() - start_time
    p4 = res.x
    
    
    
    print( f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" )
    
    print('Sample no grad   t=',t1, 'p1_hat =' , p1 )
    print('Sample with grad t=',t2, 'p1_hat =' , p2 )
    print('x no grad        t=',t3, 'p1_hat =' , p3 )
    print('x with grad      t=',t4, 'p1_hat =' , p4 )
    
    
###########################
# Speed tests
###########################
        


def drake_test():
    
    from pydrake.solvers import MathematicalProgram
    from pydrake.solvers import Solve
    
    p_true  =  np.array([ 32.0, 43.0, 77.0, 1.3, 53.0])
    p_init  =  np.array([ 10.0, 10.0, 10.0, 2.0, 80.0])
    
    pts  = catenary.generate_test_data( p_true , n_obs = 20 , x_min = -30,
                                       x_max = 30, w_l = 0.5, n_out = 10, 
                                       center = [50,50,50] , w_o = 100 )
    
    # plot  = catenary.CatenaryEstimationPlot(  p_true , p_init , pts , 50 , -50 , +50 )
    
    bounds = [ (0,100), (0,100) , (0,100) , (0,3.14) , (10,200) ]
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20] 
    
    start_time = time.time()
    
    func = lambda p: catenary.J(p, pts, p_init, params)
    
    jac = lambda p: catenary.dJ_dp( p, pts, p_init, params)
    
    res = minimize( func,
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    # jac = jac,
                    #constraints=constraints,  
                    # callback=plot.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    
    print( f" Optimzation completed in : { time.time() - start_time } sec \n"     +
           f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
    
    ## Drake
    
    prog = MathematicalProgram()
    
    x = prog.NewContinuousVariables( 5, 'x' )
    
    cost = prog.AddCost( func, vars=[ x[0], x[1], x[2] ,x[3], x[4] ] ) 
    
    prog.AddBoundingBoxConstraint(0, 100, x[:3])
    prog.AddBoundingBoxConstraint(0, 3.14, x[3])
    prog.AddBoundingBoxConstraint(10, 200, x[4])

    result = Solve(prog)


    # print out the result.
    print("Success? ", result.is_success())
    # Print the solution to the decision variables.
    print('x* = ', result.GetSolution(x))
    # Print the optimal cost.
    print('optimal cost = ', result.get_optimal_cost())
    # Print the name of the solver that was called.
    print('solver is: ', result.get_solver_id().name())


"""
######################################################
# Main
######################################################
"""

if __name__ == "__main__":     
    """ MAIN TEST """
    
    print('catenary.cat: ', cat_test())
    print('catenary.w_T_c: ', w_T_c_test())
    print('catenary.p2r_w: ', p2r_w_test())
    print('catenary.noisy_p2r_w: ', noisy_p2r_w_test())
    print('catenary.multiples_noisy_p2r_w: ', multiples_noisy_p2r_w_test())
    print('catenary.outliers: ', outliers_test())
    print('catenary.dJ_dp (samples) : ', gradient_test())
    print('catenary.dJ_dp (x) : ', gradient_test( 'x' ) )
    
    
    # plot_test()
    # # animation_test()
    
    convergence_basic_test()
    convergence_basic_test( method = 'x' )
    convergence_basic_test( grad = True )
    convergence_basic_test( method = 'x' , grad = True )
    
    # # # tracking_basic_test()
    tracking_basic_test( grad = True )
    
    tracking_advanced_test()
    
    grouping_test()
    
    speed_test()
    
    
    
    # p_true  =  np.array([ 32.0, 43.0, 77.0, 1.3, 53.0])
    # p_init  =  np.array([ 10.0, 10.0, 10.0, 2.0, 80.0])
    
    # pts  = catenary.generate_test_data( p_true , n_obs = 20 , x_min = -30,
    #                                    x_max = 30, w_l = 0.5, n_out = 10, 
    #                                    center = [50,50,50] , w_o = 100 )
    
    # plot  = catenary.CatenaryEstimationPlot(  p_true , p_init , pts , 50 , -50 , +50 )
    
    # bounds = [ (0,100), (0,100) , (0,100) , (0,3.14) , (10,200) ]
    
    # params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
    #             1.0 , 1.0 , 2 , 25 , -20 , 20] 
    
    # start_time = time.time()
    
    # func = lambda p: catenary.J(p, pts, p_init, params)
    
    
    # res = minimize( func,
    #                 p_init, 
    #                 method='SLSQP',  
    #                 bounds=bounds, 
    #                 #constraints=constraints,  
    #                 callback=plot.update_estimation, 
    #                 options={'disp':True,'maxiter':500})
    
    # p_hat = res.x