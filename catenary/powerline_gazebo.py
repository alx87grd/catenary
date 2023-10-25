#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:03:21 2023

@author: alex
"""

import numpy as np

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
     _          4                 d1     5
     ^                       |----------->   
     |
     h1
     |                          
     _     2                                     3
     ^                                 d1
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
        h1  : vertical distance between power lines 
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
    h1  = p[6]
    h3  = p[7]
    
    # catenary frame z
    z_c      = catenary.cat( x_c , a )
    
    # Offset in local catenary frame
    delta_y = np.array([ -d1 , d1 , -d1 , d1 , -d1   , d1      , 0 ])
    delta_z = np.array([   0 , 0 ,  h1 , h1  , h1+h1 , h1+h1   , h1+h1+h3])
    
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
    h1  = p[6]
    h3  = p[7]
    
    # Offset in local catenary frame
    delta_y = np.array([ -d1 , d1 , -d1 , d1 , -d1   , d1      , 0 ])
    delta_z = np.array([   0 , 0 ,  h1 , h1  , h1+h1 , h1+h1   , h1+h1+h3])
    
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



############################
def basic_convergence_test():
    
    from powerline import J
    from powerline import EstimationPlot
    from scipy.optimize import minimize
    import time

    p      =  np.array([  1, -3, 4, 0.2,  500, 4.  , 4  , 9  ])
    p_init =  np.array([  0,  0, 0, 0.0, 1000, 5.  , 5. , 10 ])
    
    pts = generate_test_data(  p , n_obs = 10 , x_min = 0, x_max = 20, w_l = 0.1,
                            n_out = 10, center = [0,0,0] , w_o = 100,
                            partial_obs = False )
    
    plot = EstimationPlot( p , p_init , pts , p2r_w )
    
    bounds = [ (-5,5), (-5,5) , (-5,5) , (-0.3,0.3) , (100,2000) ,
              (3,10), (3,10) , (5,15) ]
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
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
        

############################
def hard_tracking_test():
    
    from powerline import J
    from powerline import EstimationPlot
    from scipy.optimize import minimize
    import time

    p      =  np.array([  1, -3, 4, 0.2,  500, 4.  , 4  , 9  ])
    p_hat  =  np.array([  0,  0, 0, 0.0, 1000, 5.  , 5. , 10 ])
    
    pts = generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = EstimationPlot( p , p_hat , pts , p2r_w )
    
    bounds = [ (-5,5), (-5,5) , (-5,5) , (-0.3,0.3) , (100,2000) ,
              (3,10), (3,10) , (5,15) ]
    
    params = [ 'sample' , 10 * np.diag([ 0.0002 , 0.0002 , 0.0002 , 0.001 , 0.0001 , 0.002 , 0.002 , 0.002]) ,
                1.0 , 1.0 , 2 , 501 , -200 , 200, p2r_w ] 
    
    
    for i in range(500):
        
        pts = generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                             x_min = -10, x_max = +20, n_out = 10 ,
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
        
        
############################
def cost_shape_plot():
    
    from powerline import J
    from powerline import EstimationPlot
    import matplotlib.pyplot as plt
    import time

    p_hat =  np.array([  0,  0, 0, 0.0,  500, 5.  , 5  , 10  ])
    p     =  np.array([  0,  0, 0, 0.0,  500, 5.  , 5. , 10 ])
    
    pts = generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20, p2r_w ] 
    
    plot = EstimationPlot( p , p_hat , pts , p2r_w )
    
    
    n = 100
    zs = np.linspace( -25 , 25, n )
    cs = np.zeros(n)
    
    
    for i in range(n):
        
        p_hat[2] = zs[i]
        
        cs[i]    = J( p_hat, pts, p_hat, params)
    
        plot.update_estimation( p_hat )
        
    
    fig, ax = plt.subplots(1, figsize= (4, 3), dpi=300, frameon=True)
    
    ax = [ax]
    ax[0].plot( zs , cs  )
    ax[0].set_xlabel( 'z_hat', fontsize = 5)
    ax[0].set_ylabel( 'J(p)', fontsize = 5)
    ax[0].grid(True)
    ax[0].legend()
    ax[0].tick_params( labelsize = 5 )
        
    
    
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''



if __name__ == "__main__":     
    """ MAIN TEST """
    
    # basic_convergence_test()
    
    hard_tracking_test()
    
    cost_shape_plot()










