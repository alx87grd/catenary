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
    Q      = np.diag( np.ones( (n_p) ) ) * 1.0

    b      = 1.0
    l      = 1.0
    power  = 2.0

    params_J1 = [ 'sample' , Q , b , l , power , n , x_min , x_max, model.p2r_w ]

    params_J2 = [ model , R , Q , l , power , b , 'sample' , n , x_min , x_max ]

    J1 = powerline.J(p, pts, p_nom, params_J1 )

    J2 = powerline.J2(p, pts, p_nom, params_J2 )

    print( J1 , J2 )
    
    return ( np.abs(J1-J2) < 0.00001 )
    

    
    

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    
    J1_vs_J2()

