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




model  = powerline.ArrayModel32()

p_1  =  np.array([ 28.0, 30.0, 77.0, 0.0, 53.0])
p_2  =  np.array([ 28.0, 50.0, 77.0, 0.0, 53.0])
p_3  =  np.array([ 28.0, 70.0, 77.0, 0.0, 53.0])
p_4  =  np.array([ 28.0, 35.0, 97.0, 0.0, 53.0])
p_5  =  np.array([ 28.0, 65.0, 97.0, 0.0, 53.0])

pts1 = catenary.generate_test_data( p_1 , n_obs = 10, w_l = 0.01 , n_out = 1, center = [50,50,50] , x_min = 0 , x_max = 10)
pts2 = catenary.generate_test_data( p_2 , n_obs = 10, w_l = 0.01 , n_out = 1, center = [50,50,50] , x_min = -20 , x_max = 30)
pts3 = catenary.generate_test_data( p_3 , n_obs = 10, w_l = 0.01 , n_out = 1, center = [50,50,50] , x_min = -10 , x_max = 15)
pts4 = catenary.generate_test_data( p_4 , n_obs = 10, w_l = 0.01 , n_out = 1, center = [50,50,50] , x_min = -30 , x_max = 30)
pts5 = catenary.generate_test_data( p_5 , n_obs = 10, w_l = 0.01 , n_out = 1, center = [50,50,50] , x_min = -20 , x_max = 20)

pts = np.hstack( ( pts1 , pts2 , pts3 , pts4 , pts5 ))


p_true =  np.array([  28.0, 50.0, 77.0, 0.0, 53, 20.0, 15.0, 20.0 ])

p_hat  =  np.array([  27.0, 49.0, 76.0, 0.1, 50, 19., 14., 19.0 ])
# p_hat  =  np.array([  27.0, 49.0, 6.0, 2.1, 150, 1., 1., 1.0 ])

bounds = [ (0,200), (0,200) , (0,200) , (0,0.3) , (10,200) , (15,30), (15,15) , (15,30)]

 

## Simple
pts = np.zeros((3,1))
p_hat  =  np.array([  -10.0, -10.0, -10.0, 0.0, 100, 15.0, 5.0, 10.0 ])
# p_hat  =  np.array([  -9.0, -10.0, -10.0, 0.0, 100, 15.0, 5.0, 10.0 ])
pts    = np.zeros((3,1))

plot = powerline.EstimationPlot( p_hat , p_hat, pts , model.p2r_w , 25, -50, 50)

p     = p_hat
p_nom = np.array([ 0.,0,0,0,0,0,0,0 ])

m   = pts.shape[1]
ind = np.arange(0,m)

n_p = model.l
    
# generate a list of sample point on the model curve
n        = 100
x_min    = -200
x_max    = +200

R      = np.ones( ( m ) ) * 1 / m 
Q      = np.diag( np.ones( (n_p) ) ) * 0.0

b      = 1.0
l      = 1.0
power  = 2.0

# generate a list of sample point on the model curve
r_flat, r , xs = model.p2r_w( p , x_min , x_max , n )

# Vectors between measurements and all model pts
e_flat  = pts[:,:,np.newaxis] - r_flat[:,np.newaxis,:]

# Distances between measurements and model sample pts
d_flat = np.linalg.norm( e_flat , axis = 0 )

# Minimum distances to model for all measurements
d_min_flat = d_flat.min( axis = 1 )

   
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
e  = E[ : , ind , j , k ]        # Closest error vector


# d2 = powerline.find_closest_distance( p_init , pts , model.p2r_w )
# (d3,j3,k3) = powerline.find_closest_distance_cable_point( p_init , pts , model.p2r_w )

c = catenary.lorentzian( d , l , power , b )

# Regulation
p_e = p_nom - p

# Total cost with regulation
R      = np.ones( ( m ) ) * 1 / m 
Q      = np.diag( np.ones( (n_p) ) ) * 0.0
J3 = R.T @ c + p_e.T @ Q @ p_e



params = [ 'sample' , Q ,
            1.0 , 1.0 , 2 , 100 , -200 , 200, model.p2r_w ]

J1 = powerline.J(p, pts, p_nom, params)

J2 = powerline.J2(p, pts, p_nom, model.p2r_w )

print( J1 , J2 , J3)
# print( J3 )
print( e )
print( d )
print( c )

# # Array offsets
# deltas      = model.p2deltas( p )
# deltas_grad = model.deltas_grad()

# xk = deltas[0,k]
# yk = deltas[1,k]
# zk = deltas[2,k]

# x0  = p[0]
# y0  = p[1]
# z0  = p[2]
# phi = p[3]
# a   = p[4]

# # pre-computation
# s  = np.sin( phi )
# c  = np.cos( phi )
# sh = np.sinh( xj / a )
# ch = np.cosh( xj / a )
# ex = e[0,:]
# ey = e[1,:]
# ez = e[2,:]

# # Error Grad for each pts
# eT_de_dp = np.zeros( ( n_p , pts.shape[1] ) )

# eT_de_dp[0,:] = -ex
# eT_de_dp[1,:] = -ey
# eT_de_dp[2,:] = -ez
# eT_de_dp[3,:] =  ( ex * ( ( xj + xk ) * s + yk * c ) +
#                     ey * (-( xj + xk ) * c + yk * s ) ) 
# eT_de_dp[4,:] = ez * ( 1 + ( xj / a ) * sh - ch  )



# # for all offset parameters
# for i_p in range(5, n_p):
    
#     dxk_dp = deltas_grad[0,k,i_p-5]
#     dyk_dp = deltas_grad[1,k,i_p-5]
#     dzk_dp = deltas_grad[2,k,i_p-5]
    
    
#     eT_de_dp[i_p,:] = ( ex * ( -c * dxk_dp + s * dyk_dp ) + 
#                         ey * ( -s * dxk_dp - c * dyk_dp ) +
#                         ez * ( - dzk_dp                 ) )
    

# # Norm grad
# dd_dp = eT_de_dp / d

# # Smoothing grad
# dc_dd = b * power * ( b * d ) ** ( power - 1 ) / ( np.log( 10 ) * ( l +  b * d ) ** power )

# dc_dp = dc_dd * dd_dp

# # Regulation
# p_e = p_nom - p

# # Total cost with regulation
# dJ_dp = R.T @ dc_dp.T - 2 * p_e.T @ Q

# print( dJ_dp )

dJ2 = powerline.dJ2_dp( p, pts, p_nom, model , num = False )


dJ1 = powerline.dJ2_dp( p, pts, p_nom, model , num = True )
print( dJ2[0:4] )
print( dJ1[0:4] )

print( dJ2[4:] )
print( dJ1[4:] )

