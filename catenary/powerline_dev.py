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

p      =  np.array([  28.0, 50.0, 77.0, 0.0, 53, 20.0, 15.0, 20.0 ])

p_init =  np.array([  27.0, 49.0, 76.0, 0.1, 50, 19., 14., 19.0 ])

bounds = [ (0,200), (0,200) , (0,200) , (0,0.3) , (10,200) , (15,30), (15,15) , (15,30)]

 

plot = powerline.EstimationPlot( p , p_init , pts , model.p2r_w , 25, -50, 50)



m   = pts.shape[1]
ind = np.arange(0,m)
    
# generate a list of sample point on the model curve
n        = 100
x_min    = -40
x_max    = +40

r_model_flat  = model.p2r_w( p_init, x_min , x_max , n )[0]

# Vectors between measurements and all model pts
e_flat  = pts[:,:,np.newaxis] - r_model_flat[:,np.newaxis,:]

# Distances between measurements and model sample pts
d_flat = np.linalg.norm( e_flat , axis = 0 )

# Minimum distances to model for all measurements
d_min_flat = d_flat.min( axis = 1 )

r_model  = model.p2r_w( p_init, x_min , x_max , n )[1]

# Vectors between measurements and all model pts
e = pts[:,:,np.newaxis,np.newaxis] - r_model[:,np.newaxis,:,:]

# Distances between measurements and model sample pts
d = np.linalg.norm( e , axis = 0 )

# # Minimum distances to all cable and closet model points index j
d_min = d.min( axis = 1 )
j_min = d.argmin( axis = 1 )

d_min_min = d_min.min( axis = 1 )
k         = d_min.argmin( axis = 1 )  # closest cable index
j         = j_min[ ind , k ]          # closest point indev on closest cable

d_min_min2 = d[ ind , j , k ]


a = powerline.find_closest_distance( p_init , pts , model.p2r_w )
(b,c,d) = powerline.find_closest_distance_cable_point( p_init , pts , model.p2r_w )

params = [ 'sample' , np.diag([ 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0]) ,
            1.0 , 1.0 , 2 , 200 , -200 , 200, model.p2r_w ]

J1 = powerline.J(p_init, pts, p, params)

J2 = powerline.J2(p_init, pts, p, model.p2r_w )
