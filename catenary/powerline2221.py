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


############################
def p2r_w( p , x_min = -200, x_max = 200 , n = 400, ):
    """ 
    Compute n pts coord in world frame based on a parameter vector
    
    powerline 32 is a model for 5 catenary with offsets
    
    ----------------------------------------------------------
    
     ^                       6
     |
     h3
     |
     _          4                 d3     5
     ^                       |----------->   
     |
     h2
     |                          
     _     2                                     3
     ^                                 d2
     |                       |------------------->   
     h1          
     |
     _         0                           1             
       
                                  d1
                            |------------->          
    
    
    ----------------------------------------------------------
    
    inputs
    --------
    p      : vector of parameters 
    
        x_0 : x translation of local frame orign in world frame
        y_0 : y translation of local frame orign in world frame
        z_0 : z translation of local frame orign in world frame
        phi : z rotation of local frame basis in in world frame
        a   : sag parameter
        d1  : horizontal distance between power lines
        d2  : horizontal distance between power lines
        d3  : horizontal distance between power lines
        h1  : vertical distance between power lines 
        h2  : vertical distance between power lines 
        h3  : vertical distance between power lines 
        
    x_min  : start of points in cable local frame 
    x_max  : end   of points in cable local frame
    n      : number of pts
    
    outputs
    ----------
    r_w_flat : dim (3,5*n)  all world pts
    r_w      : dim (3,n,5)  all world pts splitted by line id
    x_c      : dim (n) array of x coord in catenary frame
    
    """
    
    x_c = np.linspace( x_min , x_max, n )
    
    # params
    x_0 = p[0]
    y_0 = p[1]
    z_0 = p[2]
    phi = p[3]
    a   = p[4]
    d1  = p[5]
    d2  = p[6]
    d3  = p[7]
    h1  = p[8]
    h2  = p[9]
    h3  = p[10]
    
    # catenary frame z
    z_c      = catenary.cat( x_c , a )
    
    # Offset in local catenary frame
    delta_y = np.array([ -d1 , d1 , -d2 , d2 , -d3   , d3      , 0 ])
    delta_z = np.array([   0 , 0 ,  h1 , h1  , h1+h2 , h1+h2   , h1+h2+h3])
    
    r_c  = np.zeros((4,n,7))
    r_w  = np.zeros((4,n,7))
    
    for i in range(7):
    
        r_c[0,:,i] = x_c
        r_c[1,:,i] = delta_y[i]
        r_c[2,:,i] = z_c + delta_z[i]
        r_c[3,:,i] = np.ones((n))
        
        r_w[:,:,i] = catenary.w_T_c( phi, x_0, y_0, z_0 ) @ r_c[:,:,i]
        
    r_w_flat = r_w.reshape( (4 , n * 7 ) , order =  'F')
    
    return ( r_w_flat[0:3,:] , r_w[0:3,:,:] , x_c ) 


############################
def flat2five( r ):
    
    return r.reshape( (3,-1,7) , order =  'F' )

############################
def five2flat( r ):
    
    return r.reshape( (3, -1) , order =  'F' )


############################
def p2p7( p ):
    """
    Input: powerline32 parameter vector  ( 8 x 1 array )
    Ouput: list of  5 catenary parameter vector ( 5 x 5 array )

    """
    
    # params
    x_0 = p[0]
    y_0 = p[1]
    z_0 = p[2]
    phi = p[3]
    a   = p[4]
    d1  = p[5]
    d2  = p[6]
    d3  = p[7]
    h1  = p[8]
    h2  = p[9]
    h3  = p[10]
    
    # Offset in local catenary frame
    delta_y = np.array([ -d1 , d1 , -d2 , d2 , -d3   , d3      , 0 ])
    delta_z = np.array([   0 , 0 ,  h1 , h1  , h1+h2 , h1+h2   , h1+h2+h3])
    
    p7  = np.zeros((5,7))
    
    for i in range(7):
        
        r0_c = np.array([ 0 , delta_y[i] , delta_z[i] , 1.0 ])
        
        r0_w = catenary.w_T_c( phi, x_0, y_0, z_0 ) @ r0_c
    
        p7[0,i] = r0_w[0]
        p7[1,i] = r0_w[1]
        p7[2,i] = r0_w[2]
        p7[3,i] = p[3]
        p7[4,i] = p[4]
        
    return p7


############################
def generate_test_data( p , n_obs = 20 , x_min = -200, x_max = 200, w_l = 0.5,
                        n_out = 10, center = [0,0,0] , w_o = 100,
                        partial_obs = False ):
    """
    generate pts for a line and outliers
    
    """
    
    #outliers
    pts  = catenary.outliers( n_out, center , w_o )
    
    for i in range(7):
        
        p_line = p2p7( p )[:,i]  # parameter vector of ith line
    
        if partial_obs:
            
            xn = np.random.randint(1,n_obs)
            xm = np.random.randint(x_min,x_max)
            xp = np.random.randint(x_min,x_max)
            
            r_line = catenary.noisy_p2r_w( p_line , xm, xp, xn, w_l)
            
        else:

            r_line = catenary.noisy_p2r_w( p_line , x_min, x_max, n_obs, w_l)

        pts = np.append( pts , r_line , axis = 1 )
    
    return pts



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    from powerline import EstimationPlot
    from powerline import J
    
    p      =  np.array([  50,  50,  50, 1.0, 600, 30.  , 40.  , 30. , 30 , 30 , 30 ])
    p_init =  np.array([ 100, 100, 100, 1.0, 300, 30.  , 40.  , 30. , 30 , 30 , 30 ])
    
    pts = generate_test_data( p , partial_obs = True )
    
    plot = EstimationPlot( p , p_init , pts , p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,1.14) , (100,2000) ,
              (5,100), (5,100) , (5,100) , (5,100), (5,100) , (5,100)]
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0, 0 , 0 , 0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20, p2r_w ] 
    
    start_time = time.time()
    
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










