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

pts1 = catenary.generate_test_data( p_1 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = 0 , x_max = 10)
pts2 = catenary.generate_test_data( p_2 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = -20 , x_max = 30)
pts3 = catenary.generate_test_data( p_3 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = -10 , x_max = 15)
pts4 = catenary.generate_test_data( p_4 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = -30 , x_max = 30)
pts5 = catenary.generate_test_data( p_5 , n_obs = 10, n_out = 5, center = [50,50,50] , x_min = -20 , x_max = 20)

pts = np.hstack( ( pts1 , pts2 , pts3 , pts4 , pts5 ))

p      =  np.array([  28.0, 50.0, 77.0, 0.0, 53, 20.0, 15.0, 20.0 ])

p_init =  np.array([  10.0, 10.0, 10.0, 1.0, 80, 16., 15., 16.0 ])

bounds = [ (0,200), (0,200) , (0,200) , (0,0.3) , (10,200) , (15,30), (15,15) , (15,30)]

params = [ 'sample' , np.diag([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0]) ,
            1.0 , 1.0 , 2 , 55 , -50 , 50, model.p2r_w ] 

# plot = powerline.EstimationPlot( p , p_init , pts , model.p2r_w , 25, -50, 50)




    
# generate a list of sample point on the model curve
n        = 100
x_min    = -10
x_max    = +10
r_model  = model.p2r_w( p_init, x_min , x_max , n )[0]

# Vectors between measurements and all model pts
e  = pts[:,:,np.newaxis] - r_model[:,np.newaxis,:]

# Distances between measurements and model sample pts
d = np.linalg.norm( e , axis = 0 )

# Minimum distances to model for all measurements
d_min = d.min( axis = 1 )