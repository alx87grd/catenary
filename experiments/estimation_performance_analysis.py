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


from catenary import powerline

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
    Q        = 0.1 * 0.0001 * np.diag([ 20. , 20. , 20. , 1000.0 , 1.0, 200.0, 200.0 , 200.0 ])
    l        = 1.0
    power    = 2.0
    b        = 1.0
    method   = 'x'
    n_s      = 100
    x_min_s  = -200
    x_max_s  = +200
    use_grad = True

    
    powerline.ArrayModelEstimatorTest(save,
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

    
    powerline.ArrayModelEstimatorTest(save,
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
def RegulationVsNoRegulation( n_run = 5 , plot = False , save = True ):
    
    # Baseline:
    save    = save
    plot    = plot
    name    = 'WithRegulation'
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

    
    powerline.ArrayModelEstimatorTest(  save,
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
    
    name    = 'WithoutRegulation'
    Q        = 0.00 * np.diag([ 20. , 20. , 20. , 1000.0 , 1.0, 200.0, 200.0 , 200.0 ])

    
    powerline.ArrayModelEstimatorTest(  save,
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
def MethodTests( n_run = 5 , plot = False , save = True ):
    
    # Baseline:
    save    = save
    plot    = plot
    name    = 'MethodX'
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

    
    powerline.ArrayModelEstimatorTest(save,
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
    
    
    # Baseline:
    name    = 'MethodSample'
    method  = 'sample'

    
    powerline.ArrayModelEstimatorTest(save,
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
def GradientTests( n_run = 5 , plot = False , save = True ):
    
    # Baseline:
    save    = save
    plot    = plot
    name    = 'GradOn'
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

    
    powerline.ArrayModelEstimatorTest(save,
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
    
    
    # Baseline:
    name    = 'GradOff'
    use_grad = False

    
    powerline.ArrayModelEstimatorTest(save,
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
def SearchTests( n_run = 5 , plot = False , save = True ):
    
    # Baseline:
    save    = save
    plot    = plot
    name    = 'n3'
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

    
    powerline.ArrayModelEstimatorTest(save,
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
    
    
    # Baseline:
    name    = 'n1'
    n_sea   = 1

    
    powerline.ArrayModelEstimatorTest(save,
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
    
    # Demo
    # GlobalConvergenceTest( 2 , True , False )
    # PartialObsTest( 2 , True , False )
    
    # Baseline Plot
    GlobalConvergenceTest( 20 , False , True )
    # PartialObsTest( 20 , False , True )
    
    # Regulation
    # RegulationVsNoRegulation( 5 , False, True)
    
    # X vs sample
    # MethodTests( 5 , False , True )
    
    # Gradient
    # GradientTests( 5 , False , True )
    
    # Search
    # SearchTests( 5 , False , True )