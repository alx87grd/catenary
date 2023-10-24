#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:44:24 2023

@author: alex
"""
import numpy as np
import time
from scipy.optimize import minimize

from pydrake.solvers import MathematicalProgram
from pydrake.solvers import Solve

import catenary



        

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
    



if __name__ == "__main__":     
    """ MAIN TEST """
    
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
    
    