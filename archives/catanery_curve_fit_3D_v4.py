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
def catanery_model( x , a, phi, x_p, y_p, z_p ):
    """ x is position along cable in local frame """
    
    # local z
    z = a * ( np.cosh( x / a )  - 1 )
    
    r_local = np.array([ x , 0 , z , 1. ])
    r_world = T( phi, x_p, y_p, z_p ) @ r_local
    
    return r_world[0:3]



############################
def lorentzian( x , a = 1):
    
    y = np.log10( 1 +  x**2 / a)
    
    return y


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
x_ub  = +50

# discretization
n_line_measurements = 20
n_else_measurements = 20
n_measurements      = n_line_measurements + n_else_measurements

x       = np.linspace( x_lb, x_ub, n_line_measurements ) 
r       = np.zeros(( n_line_measurements , 3 ))

for i in range( n_line_measurements ):
    r[i,:] = catanery_model( x[i] , a , phi, x_p, y_p, z_p)

# noise amplitude
line_noise_x = 2.0
line_noise_y = 2.0
line_noise_z = 2.0

a = 100
b = 0

r_noisy      = a * np.random.rand( n_measurements , 3 ) + b
r_noisy[0:n_line_measurements,0] = r[:,0] + line_noise_x * (np.random.rand(n_line_measurements) - 0.5 )
r_noisy[0:n_line_measurements,1] = r[:,1] + line_noise_y * (np.random.rand(n_line_measurements) - 0.5 )
r_noisy[0:n_line_measurements,2] = r[:,2] + line_noise_z * (np.random.rand(n_line_measurements) - 0.5 )


###########################
# Optimization
###########################

n_catanery_model = 25
x_hat            = np.linspace( x_lb, x_ub, n_catanery_model ) 

bounds = [ (10,200) , (0,3.14), (0,100), (0,100) , (0,100) ]

def cost( theta ):
    
    error_sum = 0
    
    a     = theta[0]
    phi   = theta[1]
    x_p   = theta[2]
    y_p   = theta[3]
    z_p   = theta[4]
    
    
    # Prelim test: brute force closest point
    r_hat = np.zeros(( n_catanery_model , 3 ))
    
    e_sum = 0
    
    # generate a list of point on the model curve
    for j in range( n_catanery_model ):
        r_hat[j,:] = catanery_model( x_hat[j] , a , phi, x_p, y_p, z_p)
        
    # for all measurements points
    for i in range( n_measurements ):

         distances_to_model = np.zeros(( n_catanery_model ))
         
         # for all model points
         for j in range( n_catanery_model ):
             
             #errors to all model points
             distances_to_model[j] = np.linalg.norm( r_hat[j,:] - r_noisy[i,:] )
             
         # Closest point distance
         d_min = distances_to_model.min()
         
         # print( d_min )
         
         d_min = lorentzian( d_min , 1.0 )
         
         # d_min = np.clip( d_min , 0 , 20.0 )
             
         e_sum = e_sum + d_min
    
    return e_sum


theta_init = np.array([ 100.0, 2.0, 10.0, 10.0, 10.0])

res = minimize(cost, 
                theta_init, 
                method='SLSQP',  
                bounds=bounds, 
                #constraints=constraints,  
                options={'disp':True,'maxiter':500})


# theta_hat = res.x

###########################
# Approx curve
###########################

theta_hat = theta_init

a     = theta_hat[0]
phi   = theta_hat[1]
x_p   = theta_hat[2]
y_p   = theta_hat[3]
z_p   = theta_hat[4]


r_hat     = np.zeros(( n_catanery_model , 3 ))

for i in range( n_catanery_model ):
    r_hat[i,:] = catanery_model( x_hat[i] , a , phi, x_p, y_p, z_p)
    
###########################
# Plot
###########################

fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
ax  = fig.add_subplot(projection='3d')

ax.plot( r[:,0] , r[:,1] , r[:,2] ,  label= 'True equation' )
line = ax.plot( r_hat[:,0] , r_hat[:,1] , r_hat[:,2] , '--', label= 'Fitted equation' )
ax.plot( r_noisy[:,0] , r_noisy[:,1] , r_noisy[:,2], 'x' , label= 'Measurements')
# ax.plot( x_p , y_p, z_p, 'o')
# ax.set_xlim([ x_lb, x_ub ])
ax.axis('equal')
ax.legend( loc = 'upper right' , fontsize = 5)
ax.set_xlabel( 'x', fontsize = 5)
ax.grid(True)
ax.tick_params( labelsize = 5 )
ax.set_ylabel( 'z(x)', fontsize = 5 )


succes = False

while not succes:
    
    theta_init = theta_hat
    
    n = 5
    
    res = minimize(cost, 
                    theta_init, 
                    method='SLSQP',  
                    bounds=bounds, 
                    #constraints=constraints,  
                    options={'disp':True,'maxiter':n})
    
    
    theta_hat = res.x
    succes    = res.success
    
    
    #Plot
    
    a     = theta_hat[0]
    phi   = theta_hat[1]
    x_p   = theta_hat[2]
    y_p   = theta_hat[3]
    z_p   = theta_hat[4]


    r_hat     = np.zeros(( n_catanery_model , 3 ))

    for i in range( n_catanery_model ):
        r_hat[i,:] = catanery_model( x_hat[i] , a , phi, x_p, y_p, z_p)
    
    line[0].set_data( r_hat[:,0] , r_hat[:,1]  )
    line[0].set_3d_properties( r_hat[:,2] )
    
    plt.pause( 0.001 )



