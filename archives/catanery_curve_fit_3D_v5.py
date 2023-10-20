#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:52:04 2023

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

###########################
# Catanery function
###########################

############################
def catanery( x , a = 1 ):
    """
    inputs
    --------
    x : position along cable in local frame ( x = 0 is z_min )
    a : slag parameter

    
    output
    -------
    z : elevation in local frame ( z = 0 is z_min )
    
    Note on local cable frame 
    --------------------------
    origin (0,0,0) is lowest elevation point on the cable
    x-axis is tangential to the cable at the origin
    y-axis is the normal of the plane formed by the cable
    z-axis is positive for greater elevation
    
    """
    
    z = a * ( np.cosh( x / a )  - 1 )
    
    return z


############################
def T( phi , x , y , z):
    """ 
    inputs
    ----------
    phi : z rotation of local frame basis in in world frame
    x   : x translation of local frame orign in world frame
    y   : y translation of local frame orign in world frame
    z   : z translation of local frame orign in world frame
    
    outputs
    ----------
    world_T_local  : 4x4 Transformation Matrix
    
    """
    
    s = np.sin( phi )
    c = np.cos( phi )
    
    T = np.array([ [ c   , -s ,  0  , x ] , 
                   [ s   ,  c ,  0  , y ] ,
                   [ 0   ,  0 ,  1. , z ] ,
                   [ 0   ,  0 ,  0  , 1.] ])
    
    return T


############################
def catanery_x2world( x , a = 1, phi = 0, x_0 = 0, y_0 = 0, z_0 = 0):
    """ 
    inputs
    --------
    x   : position along cable in local frame ( x = 0 is z_min )
    a   : slag parameter
    phi : z rotation of local frame basis in in world frame
    x_0 : x translation of local frame orign in world frame
    y_0 : y translation of local frame orign in world frame
    z_0 : z translation of local frame orign in world frame
    
    outputs
    ----------
    r[0] : x position in world frame
    r[1] : y position in world frame
    r[2] : z position in world frame
    
    """
    
    z       = catanery( x , a )
    r_local = np.array([ x , 0 , z , 1. ])
    r_world = T( phi, x_0, y_0, z_0 ) @ r_local
    
    return r_world[0:3]


############################
def catanery_p2pts( p, n = 25, x_min = -20, x_max = 20):
    """ 
    inputs
    --------
    p      : vector of parameters [ a , phi , x_p , y_p , z_p ]
    n      : number of pts
    x_min  : start of pts points in cable local frame 
    x_max  : end   of pts points in cable local frame
    
    outputs
    ----------
    r[0,:] : xs position in world frame
    r[1,:] : ys position in world frame
    r[2,:] : zs position in world frame
    
    """
    
    xs_local = np.linspace( x_min , x_max, n )
    pts      = np.zeros(( 3 , n ))
    
    # params
    a   = p[0]
    phi = p[1]
    x_p = p[2]
    y_p = p[3]
    z_p = p[4]
    
    for i in range( n ):
        
        pts[:,i] = catanery_x2world( xs_local[i] , a, phi, x_p, y_p, z_p)
    
    return pts


############################
def generate_noisy_test_data( p, 
                             n_line = 10, x_min = -30, x_max = 30, w_line = 1,
                             n_out = 5, center = [0,0,0], w_out = 100):
    """  
    p      : vector of true parameters [ a , phi , x_p , y_p , z_p ]
    
    n_line : numbers of measurements point on the line
    x_min  : start of measurements points in cable local frame 
    x_max  : end   of measurements points in cable local frame
    w_line : variance of cable measurements noise
    
    n_out  : additionnal outliers points
    centre : center point of the distribution of outlier points
    w_out  : variance of outlier point distributions
    
    """
    
    # number of points
    n = n_line + n_out
    
    # Outliers randoms points (all pts are initialized as random outliers)
    rng   = np.random.default_rng( seed = None )
    noise = w_out * rng.standard_normal(( 3 , n ))
    pts   = np.zeros(( 3 , n ))
    
    pts[0,:] = center[0]
    pts[1,:] = center[1]
    pts[2,:] = center[2]
    
    pts = pts + noise   # noise randoms points
    
    # true points on the curve
    pts_line = catanery_p2pts( p, n_line, x_min, x_max)
        
    # adding measurements noise
    pts_line   = pts_line + w_line * rng.standard_normal(( 3 , n_line ))
    
    # 
    pts[:,0:n_line] = pts_line
    
    
    return pts


# ###########################
# Plotting
# ###########################


###############################################################################
class CataneryEstimationPlot:
    """ 
    """
    
    ############################
    def __init__(self, p_true , p_hat , pts ):
        
        
        fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
        ax  = fig.add_subplot(projection='3d')
        
        pts_true  = catanery_p2pts( p_true , 50 , -50 , +50)
        pts_hat   = catanery_p2pts( p_hat  , 50 , -50 , +50 )
        pts_noisy = pts
    
        line_true  = ax.plot( pts_true[0,:]  , pts_true[1,:]  , pts_true[2,:] , label= 'True equation' )
        line_hat   = ax.plot( pts_hat[0,:]   , pts_hat[1,:]   , pts_hat[2,:]  , '--', label= 'Fitted equation' )
        line_noisy = ax.plot( pts_noisy[0,:] , pts_noisy[1,:] , pts_noisy[2,:], 'x' , label= 'Measurements')
        
        ax.axis('equal')
        ax.legend( loc = 'upper right' , fontsize = 5)
        ax.set_xlabel( 'x', fontsize = 5)
        ax.grid(True)
        
        self.fig = fig
        self.ax  = ax
        
        self.line_true  = line_true
        self.line_hat   = line_hat
        self.line_noisy = line_noisy
        
    ############################
    def update_estimation( self, p_hat ):
        
        pts_hat   = catanery_p2pts( p_hat , 50 , -50 , +50  )
        
        self.line_hat[0].set_data( pts_hat[0,:] , pts_hat[1,:]  )
        self.line_hat[0].set_3d_properties( pts_hat[2,:] )
        
        plt.pause( 0.001 )
        
    ############################
    def update_pts( self, pts_noisy  ):
        
        self.line_noisy[0].set_data( pts_noisy[0,:] , pts_noisy[1,:]  )
        self.line_noisy[0].set_3d_properties( pts_noisy[2,:] )
        
        plt.pause( 0.001 )
        
    ############################
    def update_true( self, p_true ):
        
        pts_true   = catanery_p2pts( p_true , 50 , -50 , +50  )
        
        self.line_true[0].set_data( pts_true[0,:] , pts_true[1,:]  )
        self.line_true[0].set_3d_properties( pts_true[2,:] )
        
        plt.pause( 0.001 )
        

# ###########################
# # Optimization
# ###########################
        
        
############################
def lorentzian( x , a = 1):
    """ 
    x : input
    a : flattening parameter
    y : output
    """
    
    y = np.log10( 1 +  x**2 / a)
    
    return y

############################
def cost_dmin_sample( p_hat , pts ):
    """ """

    e_sum = 0
    
    # generate a list of sample point on the model curve
    n_catanery_model = 25
    pts_hat = catanery_p2pts( p_hat, n = n_catanery_model, x_min = -20, x_max = 20)
        
    # for all measurements points
    for i in range( pts.shape[1] ):

          distances_to_model = np.zeros(( n_catanery_model ))
         
          # for all model points
          for j in range( n_catanery_model ):
             
              #errors to all model points
              distances_to_model[j] = np.linalg.norm( pts_hat[:,j] - pts[:,i] )
             
          # Minimum distance from pts i to model
          d_min = distances_to_model.min()
          
          # Truncation for outlier robustness
          d_min = lorentzian( d_min , 1.0 )
          # d_min = np.clip( d_min , 0 , 20.0 )
             
          e_sum = e_sum + d_min
    
    return e_sum


############################
def cost_local_yz( p_hat , pts ):
    """ """
    
    n = pts.shape[1]
    
    a_hat     = p_hat[0]
    w_T_local = T( p_hat[1] , p_hat[2] , p_hat[3] , p_hat[4])
    local_T_w = np.linalg.inv( w_T_local )
    
    pts_global = np.vstack( [pts , np.ones(n) ] )
    pts_local  = np.zeros((4,n))
        
    e_sum = 0
    
    # for all measurements points
    for i in range( n ):
        
        pt_global = pts_global[:,i]
        pt_local  = local_T_w @ pt_global
        
        z_hat = catanery( pt_local[0] , a_hat )
        
        ey    = pt_local[1]**2
        ez    = ( pt_local[2] - z_hat )**2
        
        e2    = ey + ez
        
        e2_trunc = lorentzian( e2 , 1)
        
        # e_sum = e_sum + e2
        e_sum = e_sum + e2_trunc
    
    return e_sum


# ###########################
# Tests
# ###########################

def compare_cost_functions():
    
    p_true  =  np.array([ 53.0, 1.3, 32.0, 43.0, 77.0])
    p_init  =  np.array([ 80.0, 2.0, 10.0, 10.0, 10.0])
    
    pts  = generate_noisy_test_data( p_true , center = [50,50,50])
    
    plot = CataneryEstimationPlot( p_true , p_init , pts)
    
    # ###########################
    # # Optimization
    # ###########################
    
    bounds = [ (10,200) , (0,3.14), (0,100), (0,100) , (0,100) ]
    
    res = minimize( lambda p: cost_dmin_sample(p, pts), 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    #constraints=constraints,  
                    callback=plot.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    p_hat = res.x
    
    print( f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
    plot2 = CataneryEstimationPlot( p_true , p_init , pts)
    
    es = minimize( lambda p: cost_local_yz(p, pts), 
                    p_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    #constraints=constraints,  
                    callback=plot2.update_estimation, 
                    options={'disp':True,'maxiter':500})
    
    
    print( f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
           f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
           f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
    
    
    
def tracking_test():
    
    
    p_true  =  np.array([ 53.0, 1.3, 32.0, 43.0, 77.0])
    p_init  =  np.array([ 80.0, 2.0, 10.0, 10.0, 10.0])

    pts  = generate_noisy_test_data( p_true , center = [50,50,50])

    plot = CataneryEstimationPlot( p_true , p_init , pts)

    for i in range(20):
        
        p_true  =  np.array([ 53.0, 1.3 + 0.1 * i, 32.0 + 5.0 * i, 43.0, 77.0])
        
        plot.update_true( p_true )
        
        pts  = generate_noisy_test_data( p_true , center = [50 + 5.0 * i,50,50])
        
        plot.update_pts( pts )
        
        bounds = [ (10,200) , (0,3.14), (0,500), (0,500) , (0,500) ]
        
        res = minimize( lambda p: cost_dmin_sample(p, pts), 
                        p_init, 
                        method='SLSQP',  
                        bounds=bounds, 
                        #constraints=constraints,  
                        # callback=plot.update_estimation, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        
        plot.update_estimation( p_hat )
        
        print( f" Init: {np.array2string(p_init, precision=2, floatmode='fixed')} \n" +
               f" True: {np.array2string(p_true, precision=2, floatmode='fixed')} \n" + 
               f" Hat : {np.array2string(p_hat, precision=2, floatmode='fixed')}  \n" )
        
        p_init = p_hat
    



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    # compare_cost_functions()
    
    tracking_test()
    

