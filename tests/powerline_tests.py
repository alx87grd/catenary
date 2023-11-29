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


def J1_vs_J2():
    
    p      =  np.array([  -10.0, -10.0, -10.0, 0.0, 100, 15.0, 5.0, 10.0 ])
    p_nom  =  np.array([  -11.0, -11.0, -10.1, 0.5, 200, 16.0, 1.0, 1.0 ])
    
    model  = powerline.ArrayModel32()

    pts    = model.generate_test_data( p , partial_obs = True )

    
    m   = pts.shape[1]

    n_p = model.l
        
    n        = 100
    x_min    = -200
    x_max    = +200

    R      = np.ones( ( m ) ) * 1 / m 
    Q      = np.diag( np.ones( (n_p) ) ) * 0.0000

    b      = 1.0
    l      = 1.0
    power  = 2.0

    params = [ 'sample' , Q , b , l , power , n , x_min , x_max, model.p2r_w ]

    params = [ model , R , Q , l , power , b , 'sample' , n , x_min , x_max ]

    J = powerline.J(p, pts, p_nom, params )
    
    ###########################################
    
    
    # generate a list of sample point on the model curve
    r_model  = model.p2r_w( p, x_min , x_max , n )[0]
    
    # Minimum distances to model for all measurements
    d_min = catenary.compute_d_min( pts , r_model )

    # Cost shaping function
    c = catenary.lorentzian( d_min , l , power , b )
    
    # Average costs per measurement plus regulation
    pts_cost = c.sum() / m 
    
    # Regulation
    p_e = p_nom - p
    
    # Total cost with regulation
    J_valid  = pts_cost + p_e.T @ Q @ p_e
    
    
    #######################
    
    print( J , J_valid )
    
    return ( np.abs(J-J_valid) < 0.00001 )


def J_x_vs_sample():
    
    p      =  np.array([  -10.0, -10.0, -10.0, 0.0, 5000, 15.0, 5.0, 10.0 ])
    p_nom  =  np.array([  -11.0, -11.0, -10.1, 0.5, 5000, 16.0, 1.0, 1.0 ])
    
    model  = powerline.ArrayModel32()

    pts    = model.generate_test_data( p , partial_obs = True )

    
    m   = pts.shape[1]

    n_p = model.l
        
    n        = 10000
    x_min    = -200
    x_max    = +200

    R      = np.ones( ( m ) ) * 1 / m 
    Q      = np.diag( np.ones( (n_p) ) ) * 0.0000

    b      = 1.0
    l      = 1.0
    power  = 2.0

    params1 = [ model , R , Q , l , power , b , 'sample' , n , x_min , x_max ]
    params2 = [ model , R , Q , l , power , b , 'x'      , None , None , None ]

    J1 = powerline.J(p, pts, p_nom, params1 )

    J2 = powerline.J(p, pts, p_nom, params2 )

    print( J1 , J2 )
    
    return ( np.abs(J1-J2) < 0.01 )


def gradient_test():

    model  = powerline.ArrayModel32()
    
    p      =  np.array([  -10.0, -10.0, -10.0, 0.0, 100, 15.0, 5.0, 10.0 ])
    p_nom  = np.array([ -9.0, -11.0, -12.0, 1.0, 120, 17.0, 4.0, 1.0])
    
    # pts    = np.zeros((3,1))
    pts    = model.generate_test_data( p , partial_obs = True )
    
    # plot = powerline.EstimationPlot( p , p, pts , model.p2r_w , 25, -50, 50)
    
    m   = pts.shape[1]
    n_p = model.l
        
    n        = 100
    x_min    = -200
    x_max    = +200
    
    R      = np.ones( ( m ) ) * 1 / m 
    Q      = np.diag( np.ones( (n_p) ) ) * 0.001
    
    b      = 1.0
    l      = 1.0
    power  = 2.0
    
    params = [ model , R , Q , l , power , b , 'sample' , n , x_min , x_max ]
    
    J1   = powerline.J(p, pts, p_nom, params )
    
    dJ1 = powerline.dJ_dp( p, pts, p_nom, params , num = True  )
    dJ2 = powerline.dJ_dp( p, pts, p_nom, params , num = False )
    
    print( J1 )
    print( dJ2[0:4] )
    print( dJ1[0:4] )
    print( dJ2[4:] )
    print( dJ1[4:] )
    
    err = np.linalg.norm( dJ1 - dJ2 )
    
    print( 'sample grad error:' , err )
    
    params = [ model , R , Q , l , power , b , 'x' , n , x_min , x_max ]
    
    J3   = powerline.J(p, pts, p_nom, params )
    
    dJ3 = powerline.dJ_dp( p, pts, p_nom, params , num = True  )
    dJ4 = powerline.dJ_dp( p, pts, p_nom, params , num = False )
    
    print( J3 )
    print( dJ3[0:4] )
    print( dJ4[0:4] )
    print( dJ3[4:] )
    print( dJ4[4:] )
    
    err2 = np.linalg.norm( dJ4 - dJ3 )
    
    print( 'x grad error:' , err2 )
    
    
    return ( err2 < 0.01 )

    

    
    

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    print('J2 test : ', J1_vs_J2())
    print('J x vs sample test : ', J_x_vs_sample())
    print('gradient test : ', gradient_test())


