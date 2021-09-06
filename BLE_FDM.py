import numpy as np
import matplotlib.pyplot as plt
    
def momentum(u, u_min, v, h, dt, nx, ny, Re, flux):
    # computation of the pressure gradient (see reference)
    q = np.zeros(ny)
    Q = np.zeros(ny)
    R = np.zeros(ny)
    vdot = np.trapz(u_min)*h
    for j in range(1,ny-1):
        # q is rhs without pressure gradient dp/dx
        q[j] = \
            - u[j] * (u[j]-u_min[j])/h          \
            - v[j] * dudy_upwind(u, v, j)/h          \
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
        
def compute_v(u, u_min, h, nx, ny, flux):
    # computation of radial velocity using continuity equation
    v = np.zeros(ny)
    for j  in range(ny):
        v[j] = - np.trapz(u[0:j]- u_min[0:j])
    return v

def dudy_upwind(u, v, j):
    # first order upwind finite difference
    if v[j] >= 0:
        du = u[j] - u[j-1]
    else:
        du = u[j+1] - u[j]
    return du