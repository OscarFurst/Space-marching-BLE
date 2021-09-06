"""
Created on Fri Sep  3 17:09:06 2021

@author: Oscar

This script is a proof of concept. It shows that it is possible to solve the
boundary layer equations with a space marching algorithm: the first axial step
is solved for the whole simulation time before the computation of the next
axial step begins.
The nondimensional boundary layer equations are solved for a 2D (infinitely
wide) channel flow.

Useful reference explaining the computation of the pressure gradient: 
http://inis.jinr.ru/sl/Simulation/Tannehill,_CFM_and_Heat_Transfer,2_ed/chap07.pdf 
"""

import numpy as np
import matplotlib.pyplot as plt
from BLE_FVM import *

# computational domain
nx = 15
ny = 21
h = 1/ny
dt = 0.001
t_end = 1

# starting and boundary conditions
Re = 5
u0 = np.ones(ny)
u0[[0, -1]] = 0
v0 = np.zeros(ny)
u_min = u0
flux = 0.1

# u[axial node][time step][radial node]
u = []
v = []
for k in range(nx): # space marching (in axial direction)
    u.append([])
    v.append([])
    u[k].append(u0)
    v[k].append(v0)
    for timestep in range(1, int(t_end/dt)): # time marching
        if k > 0:
            u_min = u[k-1][timestep]
        u[k].append(momentum(
                u[k][timestep-1], u_min, 
                v[k][timestep-1],
                h, dt, nx, ny, Re, flux))
        v[k].append(compute_v(u[k][timestep-1], u_min, h, nx, ny, flux))

for i in range(nx):
    plt.plot(u[i][-1])
plt.show()