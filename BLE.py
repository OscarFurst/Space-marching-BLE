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
    
def momentum(u, u_min, v, h, dt, nx, ny, Re):
    # computation of the pressure gradient (see reference)
    q = np.zeros(ny)
    Q = np.zeros(ny)
    R = np.zeros(ny)
    vdot = np.trapz(u_min)*h
    for j in range(1,ny-1):
        # q is rhs without pressure gradient dp/dx
        q[j] = \
            - u[j] * (u[j]-u_min[j])/h          \
            - v[j] * dv_upwind(v, j)/h          \
            + 1/Re * (u[j+1] -2*u[j] + u[j-1])/h**2
        Q[j] = u[j] + dt * q[j]
        R[j] = - dt
    dpdx = (vdot - np.trapz(Q*h)) / np.trapz(R*h)
    
    # computation of du/dt
    dudt = np.zeros(ny)
    for j in range(1,ny-1):
        dudt[j] = q[j] - dpdx
        
    # next step velocity
    return u + dt * dudt
        
def compute_v(u, u_min, h, nx, ny):
    # computation of radial velocity using continuity equation
    v = np.zeros(ny)
    for j  in range(ny):
        v[j] = - np.trapz(u[0:j]- u_min[0:j])
    return v

def dv_upwind(v, j):
    # first order upwind finite difference
    if v[j] >= 0:
        dv = v[j] - v[j-1]
    else:
        dv = v[j+1] - v[j]
    return dv


# computational domain
nx = 15
ny = 11
h = 1/ny
dt = 0.01
t_end = 1

# starting and boundary conditions
Re = 5
u0 = np.ones(ny)
u0[[0, -1]] = 0
v0 = np.zeros(ny)
u_min = u0

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
                h, dt, nx, ny, Re))
        v[k].append(compute_v(u[k][timestep-1], u_min, h, nx, ny))

for i in range(nx):
    plt.plot(u[i][-1])
plt.show()