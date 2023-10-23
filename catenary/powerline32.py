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




# ###########################
# Optimization
# ###########################

default_cost_param = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                        1.0 , 1.0 , 2 , 1000 , -200 , 200] 

############################
def J( p , pts , p_nom , param = default_cost_param ):
    """ 
    Cost function for curve fitting a catenary model on a point cloud
    
    J = average_cost_per_measurement + regulation_term
    
    see attached notes.
    
    inputs
    --------
    p     : 8x1 parameter vector
    pts   : 3xm cloud point
    p_nom : 8x1 expected parameter vector ( optionnal for regulation )
    param : list of cost function parameter and options
    
    default_param = [ method = 'sample' , 
                     Q = np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) ,
                     b = 1.0 , 
                     l = 1.0 , 
                     power = 2 , 
                     n = 1000 , 
                     x_min = -200 , 
                     x_max = 200    ] 
    
    method = 'sample' : sample-based brute force scheme
    method = 'x'      : data association is based on local x in cat frame
    
    Q      : 8x8 regulation weight matrix
    b      : scalar parameter in lorentzian function
    l      : scalar parameter in lorentzian function
    power  : scalar parameter in lorentzian function
    n      : number of sample (only used with method = 'sample' )
    x_min  : x start point of model catenary (only used with method = 'sample')
    x_max  : x end point of model catenary (only used with method = 'sample' )
    
    outputs
    ----------
    J : cost scalar
    
    """
    
    m      = pts.shape[1]  # number of measurements
    
    method = param[0]
    Q      = param[1]
    b      = param[2]
    l      = param[3]
    power  = param[4]
    n      = param[5]
    x_min  = param[6]
    x_max  = param[7]
    
    ###################################################
    if method == 'sample':
        """ data association is sampled-based """
    
        # generate a list of sample point on the model curve
        r_model  = p2r_w( p, n , x_min , x_max )[0]
        
        # Minimum distances to model for all measurements
        d_min = catenary.compute_d_min( pts , r_model )
    
    ###################################################
    elif method == 'x':
        """ data association is based on local x-coord in cat frame """
        
        raise NotImplementedError
        
    ###################################################
        
    # Cost shaping function
    c = catenary.lorentzian( d_min , l , power , b )
    
    # Average costs per measurement plus regulation
    pts_cost = c.sum() / m 
    
    # Regulation
    p_e = p_nom - p
    
    # Total cost with regulation
    cost = pts_cost + p_e.T @ Q @ p_e
    
    return cost


# ###########################
# Plotting
# ###########################


###############################################################################
class Powerline32EstimationPlot:
    """ 
    """
    
    ############################
    def __init__(self, p_true , p_hat , pts , n = 100 , xmin = -200, xmax = 200):
        
        
        fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
        ax  = fig.add_subplot(projection='3d')
        
        pts_true  = p2r_w( p_true , xmin , xmax , n )[1]
        pts_hat   = p2r_w( p_hat  , xmin , xmax , n )[1]
        pts_noisy = pts
        
        lines_true  = []
        lines_hat   = []
        
        for i in range(5):
            lines_true.append( ax.plot( pts_true[0,:,i]  , pts_true[1,:,i]  , pts_true[2,:,i] , '-k' ) ) #, label= 'True line %d ' %i ) )
            # lines_hat.append(   ax.plot( pts_hat[0,:,i]   , pts_hat[1,:,i]   , pts_hat[2,:,i]  , '--', label= 'Estimated line %d' %i) )
        
        if pts is not None:
            line_noisy = ax.plot( pts_noisy[0,:] , pts_noisy[1,:] , pts_noisy[2,:], 'x' , label= 'Measurements')
            self.line_noisy = line_noisy
        
        for i in range(5):
            # lines_true.append( ax.plot( pts_true[0,:,i]  , pts_true[1,:,i]  , pts_true[2,:,i] , '-k' ) ) #, label= 'True line %d ' %i ) )
            lines_hat.append(   ax.plot( pts_hat[0,:,i]   , pts_hat[1,:,i]   , pts_hat[2,:,i]  , '--', label= 'Estimated line %d' %i) )
        
            
        self.lines_true  = lines_true
        self.lines_hat   = lines_hat
        
        ax.axis('equal')
        ax.legend( loc = 'upper right' , fontsize = 5)
        ax.set_xlabel( 'x', fontsize = 5)
        ax.grid(True)
        
        self.fig = fig
        self.ax  = ax
        
        self.n    = n
        self.xmin = xmin
        self.xmax = xmax
        
        
    ############################
    def update_estimation( self, p_hat ):
        
        pts_hat   = p2r_w( p_hat , self.xmin , self.xmax , self.n )[1]
        
        for i in range(5):
            self.lines_hat[i][0].set_data( pts_hat[0,:,i] , pts_hat[1,:,i]  )
            self.lines_hat[i][0].set_3d_properties( pts_hat[2,:,i] )
        
        plt.pause( 0.001 )
        
    ############################
    def update_pts( self, pts_noisy  ):
        
        self.line_noisy[0].set_data( pts_noisy[0,:] , pts_noisy[1,:]  )
        self.line_noisy[0].set_3d_properties( pts_noisy[2,:] )
        
        plt.pause( 0.001 )
        
    ############################
    def update_true( self, p_true ):
        
        pts_true   = p2r_w( p_true , self.xmin , self.xmax , self.n )[1]
        
        for i in range(5):
            self.lines_true[i][0].set_data( pts_true[0,:,i] , pts_true[1,:,i]  )
            self.lines_true[i][0].set_3d_properties( pts_true[2,:,i] )
        
        plt.pause( 0.001 )
        
        
        

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



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    p      =  np.array([  50,  50,  50, 1.0, 600, 50.  , 30.  , 50. ])
    p_init =  np.array([ 100, 100, 100, 1.0, 300, 40.  , 25.  , 25    ])
    
    # r , r5 , x = p2r_w( p , x_min = -200, x_max = 200 , n = 400, )
    p5 = p2p5( p )
    pts = generate_test_data( p , partial_obs = True )
    
    plot = Powerline32EstimationPlot( p , p_init , pts )
    
    bounds = [ (0,200), (0,200) , (0,200) , (0,1.14) , (100,2000) , (5,100), (5,100) , (5,100)]
    
    params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0]) ,
                1.0 , 1.0 , 2 , 25 , -20 , 20] 
    
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










