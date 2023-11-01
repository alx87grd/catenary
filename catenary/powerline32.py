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
    
                                d2
                             |---->
     ^                 3          4 
     |          
     h          
     |
     _          0            1             2 
       
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
        d2  : horizontal distance between guard cable
        h   : vertical distance between power lines and guard cables
        
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
    h   = p[7]
    
    # catenary frame z
    z_c      = catenary.cat( x_c , a )
    
    # Offset in local catenary frame
    delta_y = np.array([ -d1 , 0 , d1 , -d2 , d2 ])
    delta_z = np.array([   0 , 0 ,  0 , +h  , +h ])
    
    r_c  = np.zeros((4,n,5))
    r_w  = np.zeros((4,n,5))
    
    for i in range(5):
    
        r_c[0,:,i] = x_c
        r_c[1,:,i] = delta_y[i]
        r_c[2,:,i] = z_c + delta_z[i]
        r_c[3,:,i] = np.ones((n))
        
        r_w[:,:,i] = catenary.w_T_c( phi, x_0, y_0, z_0 ) @ r_c[:,:,i]
        
    r_w_flat = r_w.reshape( (4 , n * 5 ) , order =  'F')
    
    return ( r_w_flat[0:3,:] , r_w[0:3,:,:] , x_c ) 


############################
def flat2five( r ):
    
    return r.reshape( (3,-1,5) , order =  'F' )

############################
def five2flat( r ):
    
    return r.reshape( (3, -1) , order =  'F' )


############################
def p2p5( p ):
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
    h   = p[7]
    
    # Offset in local catenary frame
    delta_y = np.array([ -d1 , 0 , d1 , -d2 , d2 ])
    delta_z = np.array([   0 , 0 ,  0 , +h  , +h ])
    
    p5  = np.zeros((5,5))
    
    for i in range(5):
        
        r0_c = np.array([ 0 , delta_y[i] , delta_z[i] , 1.0 ])
        
        r0_w = catenary.w_T_c( phi, x_0, y_0, z_0 ) @ r0_c
    
        p5[0,i] = r0_w[0]
        p5[1,i] = r0_w[1]
        p5[2,i] = r0_w[2]
        p5[3,i] = p[3]
        p5[4,i] = p[4]
        
    return p5


############################
def generate_test_data( p , n_obs = 20 , x_min = -200, x_max = 200, w_l = 0.5,
                        n_out = 10, center = [0,0,0] , w_o = 100,
                        partial_obs = False ):
    """
    generate pts for a line and outliers
    
    """
    
    #outliers
    pts  = catenary.outliers( n_out, center , w_o )
    
    for i in range(5):
        
        p_line = p2p5( p )[:,i]  # parameter vector of ith line
    
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
# Test
############################

############################
def basic_concergence_test():
    
    from powerline import J
    from powerline import EstimationPlot
    from scipy.optimize import minimize
    import time


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
    
    from powerline import J
    from powerline import EstimationPlot
    from scipy.optimize import minimize
    import time


    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    p_hat  =  np.array([ 100, 100, 100, 1.0, 300, 40.  , 25.  , 25    ])
    
    pts = generate_test_data( p , partial_obs = True )
    
    plot = EstimationPlot( p , p_hat , pts , p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,3.14) , (100,2000) , (15,60), (15,50) , (15,50)]
    
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
    
    from powerline import J
    from powerline import EstimationPlot
    from scipy.optimize import minimize
    import time


    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 25.  , 50. ])
    p_hat  =  np.array([   0,   0,   0, 1.2, 500, 40.  , 25.  , 25    ])
    
    pts = generate_test_data( p , partial_obs = True , n_obs = 16 , 
                                         x_min = -100, x_max = -50, n_out = 10 ,
                                         center = [0,0,0] , w_o = 20 )
    
    plot = EstimationPlot( p , p_hat , pts , p2r_w )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0.5,1.5) , (100,2000) , (30,60), (25,25) , (25,60)]
    
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
    
    # basic_tracking_test()
    
    hard_tracking_test()










