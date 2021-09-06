import numpy as np
import matplotlib.pyplot as plt
    
def momentum(u, u_min, v, h, dt, nx, ny, Re, flux):
    # computation of the pressure gradient (see reference)
    q = np.zeros(ny)
    Q = np.zeros(ny)
    R = np.zeros(ny)
    vdot = np.trapz(u_min)*h
    for j in range(ny):
        # q is rhs without pressure gradient dp/dx
        if j == 0:
            q[j] = \
                - (u[j]**2     - u_min[j]**2)/h           \
                - (v[j]*u[j] + v[j+1]*u[j+1])/2/h         \
                + 1/Re * (u[j+1] -2*u[j] - u[j+1])/h**2
        elif j == ny-1:
            q[j] = \
                - (u[j]**2     - u_min[j]**2)/h           \
                + (v[j]*u[j] + v[j-1]*u[j-1])/2/h         \
                + 1/Re * (-u[j-1] -2*u[j] + u[j-1])/h**2
        else:
            jn, js = j_north_south(v, j)
            q[j] = \
                - (u[j]**2     - u_min[j]**2)/h           \
                - (v[jn]*u[jn] - v[js]*u[js])/h           \
                + 1/Re * (u[j+1] -2*u[j] + u[j-1])/h**2
        Q[j] = u[j] + dt * q[j]
        R[j] = - dt
    dpdx = (vdot+flux - np.trapz(Q*h)) / np.trapz(R*h)
    
    # computation of du/dt
    dudt = np.zeros(ny)
    for j in range(ny):
        dudt[j] = q[j] - dpdx
        
    # next step velocity
    return u + dt * dudt
        
def compute_v(u, u_min, h, nx, ny, flux):
    # computation of radial velocity using continuity equation
    v = np.zeros(ny)
    for j  in range(ny):
        if j > 0:
            v[j] = v[j-1] + u_min[j] - u[j]
        else:
            v[j] = flux + u_min[j] - u[j]
    return v

def j_north_south(v, j):
    # first order upwind finite difference
    if v[j] >= 0:
        north = j
        south = j-1
    else:
        north = j+1
        south = j
    return north, south