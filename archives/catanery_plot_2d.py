#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:52:04 2023

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

def cat( x , a , x_min , y_min ):
    
    z = a * ( np.cosh( ( x - x_min) / a )  - 1 ) + y_min
    
    return z


a     = 500.0
x_min = 30.0
z_min = 50.0

x_lb  = -200
x_ub  = 200
n     = 1000

x = np.linspace( x_lb, x_ub, n) 
z = np.zeros(n)

for i in range(n):
    z[i] = cat( x[i] , a , x_min , z_min )



fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
ax  = fig.add_subplot(1, 1, 1)

ax.plot( x , z , label= r'$a =$ %0.1f' % a )
ax.plot( x_min , z_min , 'x')
ax.set_xlim([ x_lb, x_ub ])
ax.axis('equal')
ax.set_xlabel( 'x', fontsize = 5)
ax.grid(True)
ax.tick_params( labelsize = 5 )
ax.set_ylabel( 'z(x)', fontsize = 5 )