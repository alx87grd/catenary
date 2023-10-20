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

def cat( x , a , x_min , y_min ):
    
    z = a * ( np.cosh( ( x - x_min) / a )  - 1 ) + y_min
    
    return z


###########################
# Generate fake noisy data
###########################

# true params
a     = 50.0
x_min = 30.0
z_min = 50.0

#  domain
x_lb  = 0
x_ub  = 100

# discretization
n     = 20

x       = np.linspace( x_lb, x_ub, n) 
z       = np.zeros(n)

for i in range(n):
    z[i] = cat( x[i] , a , x_min , z_min )

# noise amplitude
noise_a = 5.0

x_noisy = x + noise_a * (np.random.rand(n) - 0.5 )
z_noisy = z + noise_a * (np.random.rand(n) - 0.5 )


###########################
# Optimization
###########################

bounds = [ (10,100) , (0,100) , (0,100) ]

def cost( theta ):
    
    error_sum = 0
    
    a     = theta[0]
    x_min = theta[1]
    y_min = theta[2]
    
    for i in range(n):
    
        z_hat = a * ( np.cosh( ( x_noisy[i] - x_min) / a )  - 1 ) + y_min
        e     = (z_hat - z_noisy[i] )**2
        
        error_sum = error_sum + e
    
    return error_sum


theta_init = np.array([ 100.0, 50.0, 100.0])

#theta_init = np.array([ 20.0, 0.0, 0.0])

res = minimize(cost, 
               theta_init, 
               method='SLSQP',  
               bounds=bounds, 
               #constraints=constraints,  
               options={'disp':True,'maxiter':500})


theta_hat = res.x

###########################
# Approx curve
###########################

print(theta_hat)

a     = theta_hat[0]
x_min = theta_hat[1]
y_min = theta_hat[2]


z_hat   = np.zeros(n)

for i in range(n):

    z_hat[i] = a * ( np.cosh( ( x[i] - x_min) / a )  - 1 ) + y_min
    
    
###########################
# Plot
###########################

fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
ax  = fig.add_subplot(1, 1, 1)

ax.plot( x , z  , label= 'True equation' )
ax.plot( x , z_hat  , '--' , label= 'Fitted equation' )
ax.plot( x_noisy , z_noisy, 'x' , label= 'Measurements')
ax.plot( x_min , z_min , 'o')
ax.set_xlim([ x_lb, x_ub ])
ax.axis('equal')
ax.legend( loc = 'upper right' , fontsize = 5)
ax.set_xlabel( 'x', fontsize = 5)
ax.grid(True)
ax.tick_params( labelsize = 5 )
ax.set_ylabel( 'z(x)', fontsize = 5 )

