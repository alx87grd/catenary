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
def catanery( x , a ):
    
    z = a * ( np.cosh( x / a )  - 1 )
    
    return z

############################
def T( phi , x , y , z):
    """ Rotation arround z + translation x,y,z """
    
    s = np.sin( phi )
    c = np.cos( phi )
    
    T = np.array([ [ c   , -s ,  0  , x ] , 
                   [ s   ,  c ,  0  , y ] ,
                   [ 0   ,  0 ,  1. , z ] ,
                   [ 0   ,  0 ,  0  , 1.] ])
    
    return T


############################
def catanery_model( x , a, phi, x_p, y_p, z_p):
    """ x is position along cable in local frame """
    
    # local z
    z = a * ( np.cosh( x / a )  - 1 )
    
    r_local = np.array([ x , 0 , z , 1. ])
    r_world = T( phi, x_p, y_p, z_p ) @ r_local
    
    return r_world


###########################
# Generate fake noisy data
###########################

# true params
a   = 53.0
phi = 1.3
x_p = 32.0
y_p = 43.0
z_p = 77.0

#  domain 
x_lb  = -50
x_ub  = 50

# discretization
n     = 20

x       = np.linspace( x_lb, x_ub, n) 
r       = np.zeros((n,4))

for i in range(n):
    r[i,:] = catanery_model( x[i] , a , phi, x_p, y_p, z_p)

# noise amplitude
noise_a = 20.0

r_noisy      = np.zeros((n,4))
r_noisy[:,0] = r[:,0] + noise_a * (np.random.rand(n) - 0.5 )
r_noisy[:,1] = r[:,1] + noise_a * (np.random.rand(n) - 0.5 )
r_noisy[:,2] = r[:,2] + noise_a * (np.random.rand(n) - 0.5 )


###########################
# Optimization
###########################

bounds = [ (10,100) , (0,3.14), (0,100), (0,100) , (0,100) ]

def cost( theta ):
    
    error_sum = 0
    
    a     = theta[0]
    phi   = theta[1]
    x_p   = theta[2]
    y_p   = theta[3]
    z_p   = theta[4]
    
    
    # Prelim test: brute force closest point
    r_hat = np.zeros((n,4))
    # e     = np.zeros((n,n,4))
    e_min = np.zeros((n,4))
    
    e_sum = 0
    
    # generate a list of point on the model curve
    for j in range(n):
        r_hat[j,:] = catanery_model( x[j] , a , phi, x_p, y_p, z_p)
        
    # for all measurements points
    for i in range(n):

         errors = np.zeros((n))
         
         # for all model points
         for j in range(n):
             
             #errors to all model points
             errors[j] = np.linalg.norm( r_hat[j,:] - r[i,:] )
             
         e_min = errors.min()
             
         e_sum = e_sum + e_min
    
    return e_sum


theta_init = np.array([ 100.0, 2.0, 10.0, 10.0, 10.0])

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
phi   = theta_hat[1]
x_p   = theta_hat[2]
y_p   = theta_hat[3]
z_p   = theta_hat[4]


r_hat     = np.zeros((n,4))

for i in range(n):
    r_hat[i,:] = catanery_model( x[i] , a , phi, x_p, y_p, z_p)
    
###########################
# Plot
###########################

fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
ax  = fig.add_subplot(projection='3d')

ax.plot( r[:,0] , r[:,1] , r[:,2] ,  label= 'True equation' )
ax.plot( r_hat[:,0] , r_hat[:,1] , r_hat[:,2] , '--', label= 'Fitted equation' )
ax.plot( r_noisy[:,0] , r_noisy[:,1] , r_noisy[:,2], 'x' , label= 'Measurements')
ax.plot( x_p , y_p, z_p, 'o')
# ax.set_xlim([ x_lb, x_ub ])
ax.axis('equal')
ax.legend( loc = 'upper right' , fontsize = 5)
ax.set_xlabel( 'x', fontsize = 5)
ax.grid(True)
ax.tick_params( labelsize = 5 )
ax.set_ylabel( 'z(x)', fontsize = 5 )

