#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:03:21 2023

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

###########################
# catenary function
###########################

import catenary


###########################
# Powerline Model
###########################

###############################################################################
class ArrayModel:
    """
    A class wraping up usefull functions related to model of array of catenary 
    line where each line as the same sag parameter and the same orientation
    
    """
    
    #################################################
    def __init__(self, l = 6 , q = 3 ):
        """
        l : int
            The number of model parameters
        q : int
            The number of discret lines in the array

        """
        
        self.q = q  # number of lines
        self.l = l  # number of model parameters
        
        
    ############################
    def p2deltas( self, p ):
        """ 
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame
        
        this is a model for 3 catenary with  equal horizontal d1 offset
        
        ----------------------------------------------------------
       
        
                    0            1             2 
           
                                      d1
                                |------------->          
        
        ----------------------------------------------------------
        """
        
        d1  = p[5]
        
        delta = np.array([[ 0.,  0.,  0. ],
                          [-d1,  0.,  +d1],
                          [ 0.,  0.,  0. ]])
        
        return delta
    
    ############################
    def deltas_grad( self ):
        """ 
        Compute the gradient of deltas with respect to offset parameters
        """
        
        grad = np.zeros(( 3 , self.q , ( self.l - 5 ) ))
        
        grad[:,:,0] = np.array([[ 0.,  0.,  0. ],
                                [-1.,  0.,  +1.],
                                [ 0.,  0.,  0. ]])
        
        return grad
        
        
    ############################
    def p2r_w( self, p , x_min = -200, x_max = 200 , n = 400, ):
        """ 
        Compute n pts coord in world frame based on a parameter vector p
        
        
        
        inputs
        --------
        p      : vector of parameters 
        
            x_0 : x translation of local frame orign in world frame
            y_0 : y translation of local frame orign in world frame
            z_0 : z translation of local frame orign in world frame
            phi : z rotation of local frame basis in in world frame
            a   : sag parameter
            d1  : horizontal distance between power lines
            
        x_min  : start of points in cable local frame 
        x_max  : end   of points in cable local frame
        n      : number of pts
        
        outputs
        ----------
        r_w_flat : dim (3,n * q)  all world pts
        r_w      : dim (3,n,q)    all world pts splitted by line id
        x_c      : dim (n)        array of x coord in catenary frame
        
        """
        
        x_c = np.linspace( x_min , x_max, n )
        
        # params
        x_0 = p[0]
        y_0 = p[1]
        z_0 = p[2]
        phi = p[3]
        a   = p[4]
        
        # catenary frame z
        z_c      = catenary.cat( x_c , a )
        
        # Offset in local catenary frame
        delta = self.p2deltas( p )
        
        r_c  = np.zeros((4,n,self.q))
        r_w  = np.zeros((4,n,self.q))
        
        # Foward kinematic for all lines in the array
        for i in range( self.q ):
        
            r_c[0,:,i] = x_c + delta[0,i]
            r_c[1,:,i] = 0.0 + delta[1,i]
            r_c[2,:,i] = z_c + delta[2,i]
            r_c[3,:,i] = np.ones((n))
            
            r_w[:,:,i] = catenary.w_T_c( phi, x_0, y_0, z_0 ) @ r_c[:,:,i]
            
        r_w_flat = r_w.reshape( (4 , n * self.q ) , order =  'F')
        
        return ( r_w_flat[0:3,:] , r_w[0:3,:,:] , x_c ) 


    ############################
    def flat2line(self, r ):
        """ split a list of pts by line """
        
        return r.reshape( (3,-1,self.q) , order =  'F' )

    ############################
    def line2flat(self, r ):
        """ flatten the list of pts by line """
        
        return r.reshape( (3, -1) , order =  'F' )


    ############################
    def p2ps(self, p ):
        """
        Input: model parameter vector  ( l x 1 array )
        Ouput: list of q catenary parameter vector ( 5 x q array )

        """
        
        # params
        x_0 = p[0]
        y_0 = p[1]
        z_0 = p[2]
        phi = p[3]
        a   = p[4]
        
        # Offset in local catenary frame
        delta = self.p2deltas( p )
        
        ps  = np.zeros(( 5 , self.q ))
        
        for i in range( self.q ):
            
            r0_c = np.hstack( ( delta[:,i] , np.array([1.0])  ) )
            
            r0_w = catenary.w_T_c( phi, x_0, y_0, z_0 ) @ r0_c
        
            ps[0,i] = r0_w[0]
            ps[1,i] = r0_w[1]
            ps[2,i] = r0_w[2]
            ps[3,i] = phi
            ps[4,i] = a
            
        return ps


    ############################
    def generate_test_data( self, p , n_obs = 20 , x_min = -200, x_max = 200, 
                            w_l = 0.5, n_out = 10, center = [0,0,0] , 
                            w_o = 100, partial_obs = False ):
        """
        generate pts for a line and outliers
        
        """
        
        #outliers
        pts  = catenary.outliers( n_out, center , w_o )
        
        # Individual catenary parameters
        ps = self.p2ps( p )
        
        for i in range( self.q ):
            
            p_line = ps[:,i]  # parameter vector of ith line
        
            if partial_obs:
                
                xn = np.random.randint(1,n_obs)
                xm = np.random.randint(x_min,x_max)
                xp = np.random.randint(x_min,x_max)
                
                r_line = catenary.noisy_p2r_w( p_line , xm, xp, xn, w_l)
                
            else:

                r_line = catenary.noisy_p2r_w( p_line , x_min, x_max, n_obs, w_l)

            pts = np.append( pts , r_line , axis = 1 )
        
        return pts


    
###############################################################################
class ArrayModel32( ArrayModel ):
    """
    ArrayModel 32 is a model for 5 catenary with 3 offsets variables
    
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
    
    p      :  8 x 1 array of parameters 
    
        x_0 : x translation of local frame orign in world frame
        y_0 : y translation of local frame orign in world frame
        z_0 : z translation of local frame orign in world frame
        phi : z rotation of local frame basis in in world frame
        a   : sag parameter
        d1  : horizontal distance between power lines
        d2  : horizontal distance between guard cable
        h   : vertical distance between power lines and guard cables
        
    """
    
    #################################################
    def __init__(self):

        ArrayModel.__init__( self, l = 8 , q = 5 )
        
        
    ############################
    def p2deltas( self, p ):
        """ 
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame
        """
        
        delta = np.zeros((3, self.q ))

        d1  = p[5]
        d2  = p[6]
        h   = p[7]
        
        # Offset in local catenary frame
        
        delta[1,0] = -d1    # y offset of cable 0
        delta[1,2] = +d1    # y offset of cable 2
        delta[1,3] = -d2    # y offset of cable 3
        delta[1,4] = +d2    # y offset of cable 4
        
        delta[2,3] = +h    # z offset of cable 3
        delta[2,4] = +h    # z offset of cable 4
        
        return delta
    
    ############################
    def deltas_grad( self ):
        """ 
        Compute the gradient of deltas with respect to offset parameters
        """
        
        grad = np.zeros(( 3 , self.q , ( self.l - 5 ) ))
        
        grad[:,:,0] = np.array([[ 0.,  0.,  0. ,  0.,  0.],
                                [-1.,  0.,  +1.,  0.,  0.],
                                [ 0.,  0.,  0. ,  0.,  0.]])
        
        grad[:,:,1] = np.array([[ 0.,  0.,  0. ,  0.,  0.],
                                [ 0.,  0.,  0. , -1., +1.],
                                [ 0.,  0.,  0. ,  0.,  0.]])
        
        grad[:,:,2] = np.array([[ 0.,  0.,  0. ,  0.,  0.],
                                [ 0.,  0.,  0. ,  0.,  0.],
                                [ 0.,  0.,  0. , +1., +1.]])
        
        return grad
    
    
    
###############################################################################
class ArrayModel2221( ArrayModel ):
    """
    ArrayModel 32 is a model for 7 catenary with 6 offsets variables
    
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
        
    """
    
    #################################################
    def __init__(self):

        ArrayModel.__init__( self, l = 11 , q = 7 )
        
        
    ############################
    def p2deltas( self, p ):
        """ 
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame
        """
        
        delta = np.zeros((3, self.q ))
        
        d1  = p[5]
        d2  = p[6]
        d3  = p[7]
        h1  = p[8]
        h2  = p[9]
        h3  = p[10]
        
        # Offset in local catenary frame
        delta[1,0] = -d1    # y offset of cable 0
        delta[1,1] = +d1    # y offset of cable 1
        delta[1,2] = -d2    # y offset of cable 2
        delta[1,3] = +d2    # y offset of cable 3
        delta[1,4] = -d3    # y offset of cable 4
        delta[1,5] = +d3    # y offset of cable 5
        
        delta[2,2] = +h1    # z offset of cable 2
        delta[2,3] = +h1    # z offset of cable 3
        delta[2,4] = +h1+h2    # z offset of cable 4
        delta[2,5] = +h1+h2    # z offset of cable 5
        delta[2,6] = +h1+h2+h3    # z offset of cable 2
        
        return delta
    
    
    ############################
    def deltas_grad( self ):
        """ 
        Compute the gradient of deltas with respect to offset parameters
        """
        
        grad = np.zeros(( 3 , self.q , ( self.l - 5 ) ))
        
        grad[:,:,0] = np.array([[ 0.,  0.,  0 ,  0.,  0.,  0.,  0.],
                                [-1., +1.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        
        grad[:,:,1] = np.array([[ 0.,  0.,  0 ,  0.,  0.,  0.,  0.],
                                [ 0.,  0., -1., +1.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        
        grad[:,:,2] = np.array([[ 0.,  0.,  0 ,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0., -1., +1.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        
        grad[:,:,3] = np.array([[ 0.,  0.,  0 ,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0., +1., +1., +1., +1., +1.]])
        
        grad[:,:,4] = np.array([[ 0.,  0.,  0 ,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0., +1., +1., +1.]])
        
        grad[:,:,5] = np.array([[ 0.,  0.,  0 ,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  +1.]])
        
        return grad
    
    
###############################################################################
class ArrayModelConstant2221( ArrayModel ):
    """
    ArrayModel 32 is a model for 7 catenary with 6 offsets variables
    
    ----------------------------------------------------------
    
     ^                       6
     |
     h3
     |
     _          4                 d1       5
     ^                       |------------->   
     |
     h1
     |                          
     _          2                          3
     ^                                 d1
     |                       |------------>   
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
    
    
    ----------------------------------------------------------
        
    """
    
    #################################################
    def __init__(self):

        ArrayModel.__init__( self, l = 8 , q = 7 )
        
        
    ############################
    def p2deltas( self, p ):
        """ 
        Compute the translation vector of each individual catenary model origin
        with respect to the model origin in the model frame
        """
        
        delta = np.zeros((3, self.q ))
        
        d1  = p[5]
        h1  = p[6]
        h3  = p[7]
        
        # Offset in local catenary frame
        delta[1,0] = -d1    # y offset of cable 0
        delta[1,1] = +d1    # y offset of cable 1
        delta[1,2] = -d1    # y offset of cable 2
        delta[1,3] = +d1    # y offset of cable 3
        delta[1,4] = -d1    # y offset of cable 4
        delta[1,5] = +d1    # y offset of cable 5
        
        delta[2,2] = +h1    # z offset of cable 2
        delta[2,3] = +h1    # z offset of cable 3
        delta[2,4] = +h1+h1    # z offset of cable 4
        delta[2,5] = +h1+h1    # z offset of cable 5
        delta[2,6] = +h1+h1+h3    # z offset of cable 2
        
        return delta
    
    ############################
    def deltas_grad( self ):
        """ 
        Compute the gradient of deltas with respect to offset parameters
        """
        
        grad = np.zeros(( 3 , self.q , ( self.l - 5 ) ))
        
        grad[:,:,0] = np.array([[ 0.,  0.,  0 ,  0.,  0.,  0.,  0.],
                                [-1., +1., -1., +1., -1., +1.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        
        grad[:,:,1] = np.array([[ 0.,  0.,  0 ,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0., +1., +1., +2., +2., +2.]])
        
        grad[:,:,2] = np.array([[ 0.,  0.,  0 ,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  +1.]])
        
        return grad
    


# ###########################
# Optimization
# ###########################

#######################################
def find_closest_distance( p , pts , p2r_w , x_min = -200 , x_max = +200 , n = 100 ):
    
    r_model_flat  = p2r_w( p, x_min , x_max , n )[0]
   
    # Vectors between measurements and all model pts
    e_flat  = pts[:,:,np.newaxis] - r_model_flat[:,np.newaxis,:]
   
    # Distances between measurements and model sample pts
    d_flat = np.linalg.norm( e_flat , axis = 0 )
   
    # Minimum distances to model for all measurements
    d_min_flat = d_flat.min( axis = 1 )
    
    return d_min_flat

#######################################
def find_closest_distance_cable_point( p , pts , p2r_w , x_min = -200 , x_max = +200 , n = 100 ):
   
    # number of measurements
    m   = pts.shape[1]
    ind = np.arange(0,m)
   
    r_model  = p2r_w( p , x_min , x_max , n )[1]
   
    # Vectors between measurements and all model pts of all cables
    e = pts[:,:,np.newaxis,np.newaxis] - r_model[:,np.newaxis,:,:]
   
    # Distances between measurements and model sample pts
    d = np.linalg.norm( e , axis = 0 )
   
    # Minimum distances to all cable and closet model points index j
    d_min = d.min( axis = 1 )
    j_min = d.argmin( axis = 1 )
   
    # Closest cable
    k = d_min.argmin( axis = 1 )  # closest cable index
    j = j_min[ ind , k ]          # closest point index on closest cable
   
    # d_min_min = d_min.min( axis = 1 )
    d_min_min = d[ ind , j , k ]   # Alternative computation 
    
    return ( d_min_min  , j , k )






###############################################################################
# Cost function of match between pts cloud and an ArrayModel parameter vector
###############################################################################

def J( p , pts , p_nom , param  ):
    """ 
    cost function for curve fitting a catenary model on a point cloud
    
    J = sum_of_cost_per_measurement + regulation_term
    
    see attached notes.
    
    inputs
    --------
    p     : dim (n_p) parameter vector
    pts   : dim (3,m) list of m 3d measurement points
    p_nom : dim (n_p) expected parameter vector ( optionnal for regulation )
    
    param : list of cost function parameter and options
        
    [ model , R , Q , l , power , b , method , n , x_min , x_max ]
    
    model  : instance of ArrayModel
    
    R      : dim (m) weight for each measurement points
    Q      : dim (n_p,n_p) regulation weight matrix
    
    b      : scalar parameter in lorentzian function
    l      : scalar parameter in lorentzian function
    power  : scalar parameter in lorentzian function
    
    method = 'sample' : sample-based brute force scheme
    method = 'x'      : data association is based on local x in cat frame
    
    n      : number of sample (only used with method = 'sample' )
    x_min  : x start point of model catenary (only used with method = 'sample')
    x_max  : x end point of model catenary (only used with method = 'sample' )
    
    outputs
    ----------
    J : cost scalar
    
    """
    
    m      = pts.shape[1]    # number of measurements
    n_p    = p.shape[0]      # number of model parameters
    model  = param[0]        # array model
    R      = param[1]        # vector of pts weight
    Q      = param[2]        # regulation matrix
    l      = param[3]        # cost function shaping param
    power  = param[4]        # cost function shaping param
    b      = param[5]        # cost function shaping param
    method = param[6]        # data association method
    n      = param[7]        # number of model pts (for sample method)
    x_min  = param[8]        # start of sample points (for sample method)
    x_max  = param[9]        # end of sample points (for sample method)
    
    ###################################################
    if method == 'sample':
        """ data association is sampled-based """
        
        p2r_w = model.p2r_w
        
        # Minimum distances to model for all measurements
        d_min = find_closest_distance( p , pts , p2r_w , x_min ,  x_max , n )
    
    ###################################################
    elif method == 'x':
        """ data association is based on local x-coord in cat frame """
        
        x0  = p[0]
        y0  = p[1]
        z0  = p[2]
        psi = p[3]
        a   = p[4]
        
        # Array offsets
        r_k = model.p2deltas( p )
        
        # Transformation Matrix
        c_T_w = np.linalg.inv( catenary.w_T_c( psi , x0 , y0 , z0 ) )
        
        # Compute measurements points positions in local cable frame
        r_w = np.vstack( [ pts , np.ones(m) ] )
        r_i = ( c_T_w @ r_w )[0:3,:]
        
        # Catenary elevation at ref point
        x_j = r_i[0,:,np.newaxis] - r_k[0,np.newaxis,:]
        z_j = catenary.cat( x_j , a )
        
        # Reference model points
        r_j = np.zeros( ( 3 , m , model.q ) )

        r_j[0,:,:] = r_i[0,:,np.newaxis]
        r_j[1,:,:] = r_k[1,np.newaxis,:]
        r_j[2,:,:] = z_j + r_k[2,np.newaxis,:]
        
        # All error vectors
        E = r_i[:,:,np.newaxis] - r_j
        
        # Distances between measurements and model ref 
        D = np.linalg.norm( E , axis = 0 )
       
        # Minimum distances to model for all measurements
        d_min = D.min( axis = 1 )
        # k_min = D.argmin( axis = 1 ) 
        
    ###################################################
        
    
    # From distance to individual pts cost
    if not ( l == 0 ):
        # lorentzian cost shaping function
        c = catenary.lorentzian( d_min , l , power , b )
        
    else:
        # No cost shaping
        c = d_min 
    
    # Regulation error
    p_e = p_nom - p
    
    # Total cost with regulation
    cost = R.T @ c + p_e.T @ Q @ p_e
    
    return cost


######################################################
def dJ_dp( p , pts , p_nom , param , num = False ):
    """ 
    
    Gradient of J with respect to parameters p
    
    inputs
    --------
    see J function doc
    
    outputs
    ----------
    dJ_dp : 1 x n_p gradient evaluated at p
    
    """
    
    m      = pts.shape[1]    # number of measurements
    n_p    = p.shape[0]      # number of model parameters
    

    if num:
        
        dJ_dp = np.zeros( n_p )
        dp    =  0.0001 * np.ones( n_p )
        
        for i in range( n_p ):
        
            pp    = p.copy()
            pm    = p.copy()
            pp[i] = p[i] + dp[i]
            pm[i] = p[i] - dp[i]
            cp    = J( pp , pts , p_nom , param )
            cm    = J( pm , pts , p_nom , param )
            
            dJ_dp[i] = ( cp - cm ) / ( 2.0 * dp[i] )
    
    else:
        
        #########################
        # Analytical gratient
        #########################
        
        model  = param[0]        # array model
        R      = param[1]        # vector of pts weight
        Q      = param[2]        # regulation matrix
        l      = param[3]        # cost function shaping param
        power  = param[4]        # cost function shaping param
        b      = param[5]        # cost function shaping param
        method = param[6]        # data association method
        n      = param[7]        # number of model pts (for sample method)
        x_min  = param[8]        # start of sample points (for sample method)
        x_max  = param[9]        # end of sample points (for sample method)
        
        x0  = p[0]
        y0  = p[1]
        z0  = p[2]
        phi = p[3]
        a   = p[4]
        
        if method == 'sample':
            
            # number of measurements
            ind = np.arange(0,m)
               
            # generate a list of sample point on the model curve
            r_flat, r , xs = model.p2r_w( p , x_min , x_max , n )
               
            # Vectors between measurements and all model pts of all cables
            E = pts[:,:,np.newaxis,np.newaxis] - r[:,np.newaxis,:,:]
               
            # Distances between measurements and model sample pts
            D = np.linalg.norm( E , axis = 0 )
               
            # Minimum distances to all cable and closet model points index j
            D_min = D.min( axis = 1 )
            j_min = D.argmin( axis = 1 )
               
            # Closest cable
            k  = D_min.argmin( axis = 1 )  # closest cable index
            j  = j_min[ ind , k ]          # closest point index on closest cable
            xj = xs[ j ]                   # local x of closest pts on the model
            d  = D[ ind , j , k ]          # Closest distnace
            e  = E[ : , ind , j , k ]      # Closest error vector
            
            # Array offsets
            deltas      = model.p2deltas( p )
            deltas_grad = model.deltas_grad()
            
            xk = deltas[0,k]
            yk = deltas[1,k]
            zk = deltas[2,k]
            
            # pre-computation
            s  = np.sin( phi )
            c  = np.cos( phi )
            sh = np.sinh( xj / a )
            ch = np.cosh( xj / a )
            ex = e[0,:]
            ey = e[1,:]
            ez = e[2,:]
    
            # Error Grad for each pts
            eT_de_dp = np.zeros( ( n_p , pts.shape[1] ) )
            
            eT_de_dp[0,:] = -ex
            eT_de_dp[1,:] = -ey
            eT_de_dp[2,:] = -ez
            eT_de_dp[3,:] =  ( ex * ( ( xj + xk ) * s + yk * c ) +
                               ey * (-( xj + xk ) * c + yk * s ) ) 
            eT_de_dp[4,:] = ez * ( 1 + ( xj / a ) * sh - ch  )
            
            
            
            # for all offset parameters
            for i_p in range(5, n_p):
                
                dxk_dp = deltas_grad[0,k,i_p-5]
                dyk_dp = deltas_grad[1,k,i_p-5]
                dzk_dp = deltas_grad[2,k,i_p-5]
                
                eT_de_dp[i_p,:] = ( ex * ( -c * dxk_dp + s * dyk_dp ) + 
                                    ey * ( -s * dxk_dp - c * dyk_dp ) +
                                    ez * ( - dzk_dp                 ) )
                
                
            # Norm grad
            dd_dp = eT_de_dp / d
           
            
        elif method == 'x':
            
            ind = np.arange(0,m)
            
            # Array offsets
            r_k = model.p2deltas( p )
            
            # Transformation Matrix
            c_T_w = np.linalg.inv( catenary.w_T_c( phi , x0 , y0 , z0 ) )
            
            # Compute measurements points positions in local cable frame
            r_w = np.vstack( [ pts , np.ones(m) ] )
            r_i = ( c_T_w @ r_w )[0:3,:]
            
            # Catenary elevation at ref point
            x_j = r_i[0,:,np.newaxis] - r_k[0,np.newaxis,:]
            z_j = catenary.cat( x_j , a )
            
            # Reference model points
            r_j = np.zeros( ( 3 , m , model.q ) )

            r_j[0,:,:] = r_i[0,:,np.newaxis]
            r_j[1,:,:] = r_k[1,np.newaxis,:]
            r_j[2,:,:] = z_j + r_k[2,np.newaxis,:]
            
            # All error vectors
            E = r_i[:,:,np.newaxis] - r_j
            
            # Distances between measurements and model ref 
            D = np.linalg.norm( E , axis = 0 )
           
            # Minimum distances to model for all measurements
            d = D.min( axis = 1 )
            k = D.argmin( axis = 1 ) 
            e = E[:,ind,k]
            
            # Array offsets
            deltas      = model.p2deltas( p )
            deltas_grad = model.deltas_grad()
            
            xk = deltas[0,k]
            yk = deltas[1,k]
            zk = deltas[2,k]
            xj = x_j[ind,k]
            
            # pre-computation
            s  = np.sin( phi )
            c  = np.cos( phi )
            sh = np.sinh( xj / a )
            ch = np.cosh( xj / a )
            ex = e[0,:]
            ey = e[1,:]
            ez = e[2,:]
            
            xi = pts[0,:]
            yi = pts[1,:]
            
            dz_da    = ch - xj / a * sh - 1
            dz_dxj   = sh
            dxj_dx0  = -c
            dxj_dy0  = -s
            dxj_dpsi = -s * ( xi - x0 ) + c * ( yi - y0 )
            dxj_dxk  = -1

            # Error Grad for each pts
            eT_de_dp = np.zeros( ( n_p , pts.shape[1] ) )
            
            eT_de_dp[0,:] = ey * +s - ez * dz_dxj * dxj_dx0
            eT_de_dp[1,:] = ey * -c - ez * dz_dxj * dxj_dy0
            eT_de_dp[2,:] = ez * -1
            eT_de_dp[3,:] = ey * ( c * ( x0 - xi ) + s * ( y0 - yi ) ) - ez * dz_dxj * dxj_dpsi
            eT_de_dp[4,:] = ez * -dz_da
            
            # for all offset parameters
            for i_p in range(5, n_p):
                
                dxk_dp = deltas_grad[0,k,i_p-5]
                dyk_dp = deltas_grad[1,k,i_p-5]
                dzk_dp = deltas_grad[2,k,i_p-5]
                
                eT_de_dp[i_p,:] = ( ex * ( 0.0 * dxk_dp + 0.0 * dyk_dp + 0.0 * dzk_dp )  + 
                                    ey * ( 0.0 * dxk_dp - 1.0 * dyk_dp + 0.0 * dzk_dp )  +
                                    ez * ( sh  * dxk_dp - 0.0 * dyk_dp - 1.0 * dzk_dp )  )
                
            
            # Norm grad
            dd_dp = eT_de_dp / d
        
        ################################
        
        # From distance to individual pts cost
        if not ( l == 0 ):
            # Cost shaping function
            dc_dd = b ** power * power * d ** ( power - 1 ) / ( np.log( 10 ) * ( l + ( b * d ) ** power ) )
            
        else:
            # No cost shaping
            dc_dd = 1.0
        
        # Smoothing grad
        dc_dp = dc_dd * dd_dp
    
        # Regulation
        p_e = p_nom - p
        
        # Total cost with regulation
        dJ_dp = R.T @ dc_dp.T - 2 * p_e.T @ Q
    
    return dJ_dp





############################
# Estimation Class
############################



###############################################################################
class ArrayEstimator:
    """ 
    """
    
    #################################################
    def __init__(self, model , p_0 ):
        
        self.model  = model
        # self.p2r_w  = model.p2r_w        # forward kinematic
        self.n_p    = model.l            # number of parameters
        self.n_line = model.q            # number of lines in the array
        
        
        # default parameter range
        self.p_ub      = np.zeros( self.n_p )
        self.p_lb      = np.zeros( self.n_p )
        #translation
        self.p_ub[0:2] = p_0[0:2] + 50.0
        self.p_lb[0:2] = p_0[0:1] - 50.0
        self.p_ub[2] = p_0[2] + 100.0
        self.p_lb[3] = p_0[3] - 100.0
        # rotation
        self.p_ub[3] = p_0[3] + 0.5
        self.p_lb[3] = p_0[3] - 0.5
        # sag
        self.p_ub[4] = p_0[4] + 500
        self.p_lb[4] = 100
        # intercablw distance
        self.p_ub[5:] = p_0[5:] + 2.0
        self.p_lb[5:] = p_0[5:] - 2.0
        
        # default sampling parameters
        self.x_min     = -200
        self.x_max     = +200
        self.n_sample  =  201
        
        # default cost function parameters
        self.method = 'x'
        self.Q      = np.diag( np.zeros( self.n_p ) )
        self.b      = 1.0
        self.l      = 1.0
        self.power  = 2
        
        # grouping param
        self.d_th         = 1.0
        self.succes_ratio = 0.8
        
        
    #####################################################
    def compute_average_dmin( self, p , pts ):
        
        # number of measurements
        m      = pts.shape[1]
        
        # generate a list of sample point on the model curve
        r_model  = self.p2r_w( p,  self.x_min , self.x_max , self.n_sample  )[0]
        
        # Minimum distances to model for all measurements
        d_min = catenary.compute_d_min( pts , r_model )
        
        # Average
        d_min_average = d_min.sum() / m 
        
        return d_min_average

            
    #####################################################
    def get_array_group( self, p , pts ):
        
        # generate a list of sample point on the model curve
        r_model  = self.p2r_w( p , self.x_min , self.x_max , self.n_sample )[0]
        
        # Minimum distances to model for all measurements
        d_min = catenary.compute_d_min( pts , r_model )
        
        # Group based on threshlod
        pts_in = pts[ : , d_min < self.d_th ]
        
        return pts_in
    
    
    #####################################################
    def get_bounds( self):
        
        bounds = []
        
        for i in range(self.n_p):
            
            bounds.append( ( self.p_lb[i] , self.p_ub[i] ) )
            
        return bounds
    
    
    #####################################################
    def get_cost_parameters(self, m ):
        
        # For J1
        # param = [self.method,
        #          self.Q,
        #          self.b,
        #          self.l,
        #          self.power,
        #          self.n_sample,
        #          self.x_min,
        #          self.x_max,
        #          self.p2r_w   ]
        
        R      = np.ones( ( m ) ) * 1 / m 
        
        param = [self.model,
                 R,
                 self.Q,
                 self.l,
                 self.power,
                 self.b,
                 self.method,
                 self.n_sample,
                 self.x_min,
                 self.x_max]
            
        return param
            
    
    
    #####################################################
    def solve( self, pts , p_init , callback = None ):
        
        bounds = self.get_bounds()
        param  = self.get_cost_parameters( m = pts.shape[1] )
        func   = lambda p: J(p, pts, p_init, param)
        grad   = lambda p: dJ_dp( p, pts, p_init, param , num = False )
        
        res = minimize( func,
                        p_init, 
                        method='SLSQP',  
                        bounds=bounds, 
                        jac = grad,
                        #constraints=constraints,  
                        callback=callback, 
                        options={'disp':True,'maxiter':500})
        
        p_hat = res.x
        j_hat = res.fun
        
        return p_hat
    
    
    #####################################################
    def solve_with_translation_search( self, pts , p_init , n = 10 , var = 10 , callback = None ):
        
        bounds = self.get_bounds()
        param  = self.get_cost_parameters( m = pts.shape[1] )
        func   = lambda p: J(p, pts, p_init, param)
        grad   = lambda p: dJ_dp( p, pts, p_init, param , num = False )
        
        # variation to params
        rng    = np.random.default_rng( seed = None )
        deltas = var * rng.standard_normal(( 3 , n ))
        
        # keep original solution
        deltas[:,0] = np.zeros((3))
        
        # solutions
        ps = np.zeros(( self.n_p , n ))
        js = np.zeros(n)
        
        for i in range(n):
            
            p      = p_init
            p[0:3] = p_init[0:3] + deltas[:,i]
        
            res = minimize( func,
                            p, 
                            method='SLSQP',  
                            bounds=bounds, 
                            jac = grad,
                            #constraints=constraints,  
                            callback=callback, 
                            options={'disp':False,'maxiter':500})
            
            ps[:,i] = res.x
            js[i]   = res.fun
            
        i_star = js.argmin()
        p_hat  = ps[:,i_star]
        
        return p_hat
    
    
    #####################################################
    def is_target_aquired( self, p , pts ):
        
        # generate a list of sample point on the model curve
        r_model  = self.model.p2r_w( p , self.x_min , self.x_max , self.n_sample )[0]
        
        # Minimum distances to model for all measurements
        d_min = catenary.compute_d_min( pts , r_model )
        
        # Group based on threshlod
        pts_in = pts[ : , d_min < self.d_th ]
        
        # Ratio of point in range of the model
        ratio = pts_in.shape[1] / pts.shape[1]
        
        succes = ratio > self.succes_ratio
        
        return succes
    


# ###########################
# Plotting
# ###########################


###############################################################################
class EstimationPlot:
    """ 
    """
    
    ############################
    def __init__(self, p_true , p_hat , pts , p2r_w , n = 100 , xmin = -200, xmax = 200 ):
        
        
        fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
        ax  = fig.add_subplot(projection='3d')
        
        pts_true  = p2r_w( p_true , xmin , xmax , n )[1]
        pts_hat   = p2r_w( p_hat  , xmin , xmax , n )[1]
        pts_noisy = pts
        
        lines_true  = []
        lines_hat   = []
        
        self.n_line = pts_true.shape[2]
        
        for i in range(self.n_line):
            lines_true.append( ax.plot( pts_true[0,:,i]  , pts_true[1,:,i]  , pts_true[2,:,i] , '-k' ) ) #, label= 'True line %d ' %i ) )
            # lines_hat.append(   ax.plot( pts_hat[0,:,i]   , pts_hat[1,:,i]   , pts_hat[2,:,i]  , '--', label= 'Estimated line %d' %i) )
        
        if pts is not None:
            line_noisy = ax.plot( pts_noisy[0,:] , pts_noisy[1,:] , pts_noisy[2,:], 'x' , label= 'Measurements')
            self.line_noisy = line_noisy
        
        for i in range(self.n_line):
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
        
        self.p2r_w = p2r_w
        
        
    ############################
    def update_estimation( self, p_hat ):
        
        pts_hat   = self.p2r_w( p_hat , self.xmin , self.xmax , self.n )[1]
        
        for i in range(self.n_line):
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
        
        pts_true   = self.p2r_w( p_true , self.xmin , self.xmax , self.n )[1]
        
        for i in range(self.n_line):
            self.lines_true[i][0].set_data( pts_true[0,:,i] , pts_true[1,:,i]  )
            self.lines_true[i][0].set_3d_properties( pts_true[2,:,i] )
        
        plt.pause( 0.001 )
        
    
    ############################
    def add_pts( self, pts ):
        
        self.ax.plot( pts[0,:] , pts[1,:] , pts[2,:], 'o' , label= 'Group')
        
        


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    pass
    







