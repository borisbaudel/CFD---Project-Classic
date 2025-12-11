"""
Model of domain 
            +---------------------------+
            |                           |  inlet 
            |                           |
            |       +-----+             +
            |       | CO2 |             |  
            |       +-----+             | 
            |                           |  
            |                           |  
            |                           |
            +-------+-------------------+ 
              outlet
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from numba import jit   
import time 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import copy
from scipy.interpolate import RegularGridInterpolator


# ======= Problem parameters =================
CO2_LIMIT = 0.001 
time_interval = 24.0 #hrs
rho = 1.2             # density (kg/m^3) - incompressible baseline
nu = 15.6 * 1e-6             # kinematic viscosity (m^2/s)
D_co2 = nu         # molecular diffusivity for CO2 in air (m^2/s)

# ======= Domain parameters =================
L = 5.0            # length of the domain
h = 2.5 # height
yinj = 7*L/8  # inlet vertical location
linj = L / 4.0  # inlet vertical length

# Inlet velocity profile function
def u_inlet(y, Q):
    term = (y - (yinj - linj / 2)) / linj
    return -6 * Q / (linj * h) * term * (1 - term)

# ====== Time-stepping parameters ==================
t = time_interval  * 3600.0  # total time in seconds
Fo = 0.2 # conservative value
CFL = 0.25

# ======== Scalar Concentration field parameters =================
S0 = 3e-7  # s^-1 - source strength
xc = L * 3/8
yc = L * 5/8  
l_source = 1 # m

INLET_CO2 = 0.0004     # molar fraction at inlet (Dirichlet on inlet segment)
INIT_CO2 = 0.0004    # initial concentration in domain


def compute_source_term(c, S, x_source_start, x_source_end, y_source_start, y_source_end):
    S[x_source_start:x_source_end+1, y_source_start:y_source_end+1] = S0 * (1 - c[x_source_start:x_source_end+1, y_source_start:y_source_end+1])
    return S

# ---- Scalar boundary conditions (Accepts Indices) ----
def apply_scalar_bcs(c, y_inlet_idx, inlet_val):    
    """
    Enforce:
      - zero normal gradient (Neumann) on all walls
      - Dirichlet c = INLET_CO2 on inlet region: right boundary for j >= y_inlet_index
    """
    c[:, 0] = c[:, 1]       # Bottom
    c[:, -1] = c[:, -2]     # Top
    c[0, :] = c[1, :]       # Left
    
    c[-1, :y_inlet_idx] = c[-2, :y_inlet_idx] # Right Wall
    c[-1, y_inlet_idx:] = inlet_val           # Right Inlet
    return c

# --- Updated BC Function (Accepts indices as arguments) ---
def apply_velocity_bcs(u, v, x_extract_idx, y_inlet_idx, u_inlet_arr):
    """
    Applies BCs using dynamic indices calculated for the specific mesh size.
    u_inlet_arr: Pre-calculated array of inlet velocities to avoid re-computing
    """    
    # Bottom wall (y=0)
    # 1. Left of outlet
    u[:x_extract_idx+1, 0] = u[:x_extract_idx+1, 1]
    v[:x_extract_idx+1, 0] = v[:x_extract_idx+1, 1]
    #p[:x_extract_idx+1, 0] = 0.0 # Outlet pressure anchor
    
    # 2. Right of outlet (solid wall)
    u[x_extract_idx+1:, 0] = 0.0
    v[x_extract_idx+1:, 0] = 0.0
    #p[x_extract_idx+1:, 0] = p[x_extract_idx+1:, 1]

    # Top wall (y=L)
    u[:, -1] = 0.0
    v[:, -1] = 0.0
    #p[:, -1] = p[:, -2]

    # Left wall (x=0)
    u[0, :] = 0.0
    v[0, :] = 0.0
    #p[0, :] = p[1, :]

    # Right wall (x=L)
    #p[-1, :] = p[-2, :] # Zero gradient pressure
    
    # 1. Below inlet (Solid wall)
    u[-1, :y_inlet_idx] = 0.0
    v[-1, :] = 0.0 
    
    # 2. At Inlet (Dirichlet)
    # We assign the pre-calculated inlet profile
    # Ensure sizes match: u_inlet_arr should be length (Ny - y_inlet_idx)
    u[-1, y_inlet_idx:] = u_inlet_arr
    
    return u, v

# --- Updated BC Function (Accepts indices as arguments) ---
def apply_pressure_bcs(p, x_extract_idx):
    
    # Bottom wall (y=0)
    # 1. Left of outlet
    p[:x_extract_idx+1, 0] = 0.0 # Outlet pressure 
    # 2. Right of outlet (solid wall)
    p[x_extract_idx+1:, 0] = p[x_extract_idx+1:, 1]

    # Top wall (y=L)
    p[:, -1] = p[:, -2]

    # Left wall (x=0)
    p[0, :] = p[1, :]

    # Right wall (x=L)
    p[-1, :] = p[-2, :] # Zero gradient pressure
    
    return p

@jit(nopython=True)
def apply_pressure_bcs_loop(p, x_extract_idx):

    nx, ny = p.shape  # grid dimensions

    # --- Bottom wall (y = 0) ---
    # 1. Left of outlet
    for i in range(0, x_extract_idx + 1):
        p[i, 0] = 0.0

    # 2. Right of outlet (solid wall)
    for i in range(x_extract_idx + 1, nx):
        p[i, 0] = p[i, 1]

    # --- Top wall (y = L) ---
    for i in range(nx):
        p[i, ny - 1] = p[i, ny - 2]

    # --- Left wall (x = 0) ---
    for j in range(ny):
        p[0, j] = p[1, j]

    # --- Right wall (x = L) ---
    for j in range(ny):
        p[nx - 1, j] = p[nx - 2, j]

    return p

@jit(nopython=True)
def advect_second_order(u, v, dx, dy, dt):
    Nx, Ny = u.shape
    u_new = np.copy(u)
    v_new = np.copy(v)
    
    # We iterate through the internal grid points
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            
            # ----------------------------------------
            # 1. Advect U-Component
            # ----------------------------------------
            
            # --- dU/dx (Upwind biased) ---
            if u[i,j] > 0:
                if i > 1: # Check if we have 2 neighbors upstream
                    # 2nd Order Upwind
                    du_dx = (3*u[i,j] - 4*u[i-1,j] + u[i-2,j]) / (2*dx)
                else:
                    # Fallback to 1st Order (near boundary)
                    du_dx = (u[i,j] - u[i-1,j]) / dx
            else:
                if i < Nx-2:
                    # 2nd Order Upwind (looking right)
                    du_dx = (-3*u[i,j] + 4*u[i+1,j] - u[i+2,j]) / (2*dx)
                else:
                    du_dx = (u[i+1,j] - u[i,j]) / dx

            # --- dU/dy (Upwind biased) ---
            if v[i,j] > 0:
                if j > 1:
                    du_dy = (3*u[i,j] - 4*u[i,j-1] + u[i,j-2]) / (2*dy)
                else:
                    du_dy = (u[i,j] - u[i,j-1]) / dy
            else:
                if j < Ny-2:
                    du_dy = (-3*u[i,j] + 4*u[i,j+1] - u[i,j+2]) / (2*dy)
                else:
                    du_dy = (u[i,j+1] - u[i,j]) / dy

            # Update U
            u_new[i,j] = u[i,j] - dt * (u[i,j]*du_dx + v[i,j]*du_dy)

            # ----------------------------------------
            # 2. Advect V-Component (Same logic)
            # ----------------------------------------
            
            # --- dV/dx ---
            if u[i,j] > 0:
                if i > 1:
                    dv_dx = (3*v[i,j] - 4*v[i-1,j] + v[i-2,j]) / (2*dx)
                else:
                    dv_dx = (v[i,j] - v[i-1,j]) / dx
            else:
                if i < Nx-2:
                    dv_dx = (-3*v[i,j] + 4*v[i+1,j] - v[i+2,j]) / (2*dx)
                else:
                    dv_dx = (v[i+1,j] - v[i,j]) / dx

            # --- dV/dy ---
            if v[i,j] > 0:
                if j > 1:
                    dv_dy = (3*v[i,j] - 4*v[i,j-1] + v[i,j-2]) / (2*dy)
                else:
                    dv_dy = (v[i,j] - v[i,j-1]) / dy
            else:
                if j < Ny-2:
                    dv_dy = (-3*v[i,j] + 4*v[i,j+1] - v[i,j+2]) / (2*dy)
                else:
                    dv_dy = (v[i,j+1] - v[i,j]) / dy

            # Update V
            v_new[i,j] = v[i,j] - dt * (u[i,j]*dv_dx + v[i,j]*dv_dy)
    
    return u_new, v_new

@jit(nopython=True)
def advect(u, v, dx, dy, dt):
    Nx, Ny = u.shape
    u_new = np.copy(u)
    v_new = np.copy(v)
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # simple upwind for u
            if u[i,j] > 0:
                du_dx = (u[i,j] - u[i-1,j]) / dx
            else:
                du_dx = (u[i+1,j] - u[i,j]) / dx

            if v[i,j] > 0:
                du_dy = (u[i,j] - u[i,j-1]) / dy
            else:
                du_dy = (u[i,j+1] - u[i,j]) / dy

            # could calculate this using derivative squared velocity, as per formula , but this should work too
            u_new[i,j] = u[i,j] - dt * (u[i,j]*du_dx + v[i,j]*du_dy) 

            # same for v
            if u[i,j] > 0:
                dv_dx = (v[i,j] - v[i-1,j]) / dx
            else:
                dv_dx = (v[i+1,j] - v[i,j]) / dx

            if v[i,j] > 0:
                dv_dy = (v[i,j] - v[i,j-1]) / dy
            else:
                dv_dy = (v[i,j+1] - v[i,j]) / dy

            v_new[i,j] = v[i,j] - dt * (u[i,j]*dv_dx + v[i,j]*dv_dy)
    
    return u_new, v_new

@jit(nopython=True)
def diffuse(u, v, dx, dy, dt, nu):
    Nx, Ny = u.shape
    u_new = np.copy(u)
    v_new = np.copy(v)

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u_lap = (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx**2 + \
                    (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy**2
            v_lap = (v[i+1,j] - 2*v[i,j] + v[i-1,j]) / dx**2 + \
                    (v[i,j+1] - 2*v[i,j] + v[i,j-1]) / dy**2

            u_new[i,j] = u[i,j] + dt * nu * u_lap
            v_new[i,j] = v[i,j] + dt * nu * v_lap

    return u_new, v_new

@jit(nopython=True)
def pressure_projection_absolute(u, v, p, dx, dy, dt, rho, max_iter=500, tol=1e-6):
    Nx, Ny = u.shape
    b = np.zeros_like(p)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            b[i,j] = rho/dt * ((u[i+1,j]-u[i-1,j])/(2*dx) + 
                                (v[i,j+1]-v[i,j-1])/(2*dy))
    
    pn = np.copy(p)
    for it in range(max_iter):
        p_new = np.copy(pn)
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                p_new[i,j] = ((pn[i+1,j] + pn[i-1,j])*dy**2 + 
                              (pn[i,j+1] + pn[i,j-1])*dx**2 -
                              b[i,j]*dx**2*dy**2) / (2*(dx**2 + dy**2))
        if np.linalg.norm(p_new - pn, ord=np.inf) < tol:
            break
        pn = p_new.copy()
    p[:] = pn
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[i,j] -= dt/rho * (p[i+1,j]-p[i-1,j])/(2*dx)
            v[i,j] -= dt/rho * (p[i,j+1]-p[i,j-1])/(2*dy)
    
    return u, v, p

@jit(nopython=True)
def pressure_projection(u, v, p, dx, dy, dt, rho, x_extract_index, max_iter=100000, tol=1e-1, omega=1.35):
    """
    Solves the pressure Poisson equation using a SOR (Successive Over-Relaxation) solver.
    Convergence is checked using the L2 norm of the relative residual:
    ||A*p - b||  < tol * ||b||
    
    omega: Relaxation factor (1 < omega < 2). 
           If omega = 1, this becomes Gauss-Seidel.
    """

    Nx, Ny = u.shape
    b = np.zeros_like(p)
    
    # --- 1. Calculate RHS (b) and its L2 norm (norm_b) ---
    # b[i,j] = (rho/dt) * (divergence of u_intermediate)
    norm_b_sq = 0.0
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            b[i,j] = rho/dt * ((u[i+1,j] - u[i-1,j]) / (2*dx) + 
                               (v[i,j+1] - v[i,j-1]) / (2*dy))
            norm_b_sq += b[i,j]**2
    
    norm_b = np.sqrt(norm_b_sq)
    
    # --- 2. Set the convergence threshold ---
    if norm_b < 1e-12:
        check_tol = tol 
    else:
        check_tol = tol * norm_b

    # Pre-calculate constants
    dx2 = dx**2
    dy2 = dy**2
    denominator = 2.0 * (dx2 + dy2)
    
    num_iters = 0 

    for it in range(max_iter):
        num_iters = it + 1
        
        # --- 3. SOR iteration step ---
        # Note: SOR operates IN-PLACE. We read and write to 'p' directly.
        # This ensures we use the most recently updated neighbors (i-1, j-1).
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                # Calculate the Gauss-Seidel estimate
                # p[i-1,j] and p[i,j-1] are already the *new* values from this iteration
                # p[i+1,j] and p[i,j+1] are still the *old* values
                p_gs = ((p[i+1,j] + p[i-1,j]) * dy2 + 
                        (p[i,j+1] + p[i,j-1]) * dx2 -
                        b[i,j] * dx2 * dy2) / denominator
                
                # Apply SOR relaxation
                p[i,j] = (1.0 - omega) * p[i,j] + omega * p_gs
        
        p = apply_pressure_bcs_loop(p, x_extract_index)

        # --- 4. Convergence check: ||A*p - b|| < check_tol ---
        # Calculate the L2 norm of the residual: r = b - A*p
        norm_r_sq = 0.0
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                # Discrete Laplacian: A*p[i,j]
                lap_p = (p[i+1,j] - 2*p[i,j] + p[i-1,j]) / dx2 + \
                        (p[i,j+1] - 2*p[i,j] + p[i,j-1]) / dy2
                
                residual_ij = b[i,j] - lap_p
                norm_r_sq += residual_ij**2
                
        norm_r = np.sqrt(norm_r_sq)

        # --- 5. Check for convergence ---
        if norm_r < check_tol:
            print(f"SOR converged in {num_iters} iterations.")
            break 

    # --- 6. Update Velocity Fields ---
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[i,j] -= dt/rho * (p[i+1,j]-p[i-1,j])/(2*dx)
            v[i,j] -= dt/rho * (p[i,j+1]-p[i,j-1])/(2*dy)
    
    return u, v, p

@jit(nopython=True)
def diffuse_scalar(c, dx, dy, dt, D):
    Nx, Ny = c.shape
    c_new = np.copy(c)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            lap = (c[i+1, j] - 2.0*c[i, j] + c[i-1, j]) / dx**2 + \
                  (c[i, j+1] - 2.0*c[i, j] + c[i, j-1]) / dy**2
            c_new[i, j] = c[i, j] + dt * D * lap
    return c_new

@jit(nopython=True)
def advect_scalar_second_order(c, u, v, dx, dy, dt):
    """
    2nd Order Linear Upwind advection for scalar field C.
    Includes a safety clamp to prevent negative concentrations.
    """
    Nx, Ny = c.shape
    c_new = np.copy(c)

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            
            # ----------------------------------------
            # 1. Gradient in X (dc/dx)
            # ----------------------------------------
            if u[i,j] > 0:
                if i > 1:
                    # 2nd Order Upwind (Look back 2 points)
                    dc_dx = (3*c[i,j] - 4*c[i-1,j] + c[i-2,j]) / (2*dx)
                else:
                    # Fallback to 1st Order near boundary
                    dc_dx = (c[i,j] - c[i-1,j]) / dx
            else:
                if i < Nx-2:
                    # 2nd Order Upwind (Look forward 2 points)
                    dc_dx = (-3*c[i,j] + 4*c[i+1,j] - c[i+2,j]) / (2*dx)
                else:
                    dc_dx = (c[i+1,j] - c[i,j]) / dx

            # ----------------------------------------
            # 2. Gradient in Y (dc/dy)
            # ----------------------------------------
            if v[i,j] > 0:
                if j > 1:
                    dc_dy = (3*c[i,j] - 4*c[i,j-1] + c[i,j-2]) / (2*dy)
                else:
                    dc_dy = (c[i,j] - c[i,j-1]) / dy
            else:
                if j < Ny-2:
                    dc_dy = (-3*c[i,j] + 4*c[i,j+1] - c[i,j+2]) / (2*dy)
                else:
                    dc_dy = (c[i,j+1] - c[i,j]) / dy

            # ----------------------------------------
            # 3. Update Step
            # ----------------------------------------
            c_new[i,j] = c[i,j] - dt * (u[i,j]*dc_dx + v[i,j]*dc_dy)
            
            # --- SAFETY CLAMP ---
            # 2nd order schemes are not "Monotonic". They can cause artificial 
            # oscillations near sharp gradients (like the inlet). 
            # We must ensure concentration never drops below zero.
            if c_new[i,j] < 0.0:
                c_new[i,j] = 0.0

    return c_new

@jit(nopython=True)
def advect_scalar_lax_wendroff(c, u, v, dx, dy, dt):
    """
    2nd Order Lax-Wendroff Advection.
    Central difference in space + Time-series correction term.
    """
    Nx, Ny = c.shape
    c_new = np.copy(c)
    
    # Pre-calculate constants to save divisions
    dt2_2 = 0.5 * dt**2
    dx2 = dx**2
    dy2 = dy**2
    dx2dy = 4.0 * dx * dy # For cross derivative

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            
            # 1. First Derivatives (Central Difference)
            cx = (c[i+1,j] - c[i-1,j]) / (2*dx)
            cy = (c[i,j+1] - c[i,j-1]) / (2*dy)
            
            # 2. Second Derivatives (Curvature)
            cxx = (c[i+1,j] - 2*c[i,j] + c[i-1,j]) / dx2
            cyy = (c[i,j+1] - 2*c[i,j] + c[i,j-1]) / dy2
            
            # 3. Cross Derivative (The diagonal term)
            # d2c / dxdy
            cxy = (c[i+1,j+1] - c[i-1,j+1] - c[i+1,j-1] + c[i-1,j-1]) / dx2dy
            
            # 4. Velocities at the node
            ui = u[i,j]
            vi = v[i,j]
            
            # 5. The Update
            # New = Old - Advection + Correction
            c_new[i,j] = c[i,j] \
                         - dt * (ui * cx + vi * cy) \
                         + dt2_2 * ( (ui**2 * cxx) + (vi**2 * cyy) + (2 * ui * vi * cxy) )

            # Safety Clamp (Crucial for Scalars)
            # Lax-Wendroff creates "ripples" (dispersive errors) 
            # which can cause negative concentrations.
            if c_new[i,j] < 0.0:
                c_new[i,j] = 0.0
                
    return c_new

@jit(nopython=True)
def advect_scalar(c, u, v, dx, dy, dt):
    Nx, Ny = c.shape
    c_new = np.copy(c)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # upwind derivative in x
            if u[i, j] > 0.0:
                dc_dx = (c[i, j] - c[i-1, j]) / dx
            else:
                dc_dx = (c[i+1, j] - c[i, j]) / dx
            # upwind derivative in y
            if v[i, j] > 0.0:
                dc_dy = (c[i, j] - c[i, j-1]) / dy
            else:
                dc_dy = (c[i, j+1] - c[i, j]) / dy

            c_new[i, j] = c[i, j] - dt * (u[i, j] * dc_dx + v[i, j] * dc_dy)
    return c_new

def compute_pressure_coefficient(p, u, v, y_inlet_idx, x_extract_idx):
    """
    Computes a single scalar Cp representing the total pressure loss coefficient.
    
    Cp_global = (P_avg_inlet - P_avg_outlet) / (0.5 * rho * U_ref^2)
    
    Since P_outlet is set to 0 in BCs, this simplifies to P_avg_inlet / q_ref.
    """
    
    # Calculate Average Inlet Pressure ---
    p_inlet_vals = p[-1, y_inlet_idx:]
    p_avg_inlet = np.mean(p_inlet_vals)
    
    # Calculate Reference Velocity (U_ref)
    # u_out = u[:x_extract_idx+1, 0]
    v_out = v[:x_extract_idx+1, 0]
    
    # vel_mag_out = np.sqrt(u_out**2 + v_out**2)
    
    # U_ref = np.mean(vel_mag_out)
    U_ref = np.mean(v_out)

    if abs(U_ref) < 1e-9:
        return 0.0  # Avoid division by zero

    # Compute Dynamic Pressure ---
    q_ref = 0.5 * rho * U_ref**2
    
    # Compute Global Cp ---
    # P_outlet is 0.0
    Cp = p_avg_inlet / q_ref
    
    return Cp

def visualize_field(X, Y, u, v, p=None, step=None, show_pressure=False, time_hours = None, stride=5):
    """
    Visualize the 2D velocity and pressure fields.

    Parameters
    ----------
    X, Y : 2D arrays
        Meshgrid coordinates (from np.meshgrid with indexing='ij')
    u, v : 2D arrays
        Velocity components
    p : 2D array, optional
        Pressure field (if provided)
    step : int, optional
        Time-step number to display in title
    show_pressure : bool
        If True, overlay pressure contours
    stride : int
        Spacing for velocity arrows (higher = fewer arrows)
    """
    plt.clf()
    vel_mag = np.sqrt(u**2 + v**2)

    #plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, vel_mag, levels=50, cmap='viridis')
    plt.colorbar(label='Velocity magnitude |u|')

    # # Overlay velocity vectors
    # plt.quiver(
    #     X[::stride, ::stride],
    #     Y[::stride, ::stride],
    #     u[::stride, ::stride],
    #     v[::stride, ::stride],
    #     color='white',
    # )

    if show_pressure and p is not None:
        plt.contour(X, Y, p, levels=5, colors='k', linewidths=0.8, alpha=0.6)

    plt.xlabel('x')
    plt.ylabel('y')
    title = "Velocity Field"
    if step is not None:
        title += f" at step {step}"
    if time_hours is not None:
        title += f", time {time_hours:.2f} hrs"
    plt.title(title)

def visualize_CO2(X, Y, c, step=None, time_hours=None):

    plt.clf()
    plt.contourf(X, Y, c, levels=50)
    plt.colorbar(label='CO2 scalar (units depend on chosen S_scalar)')

    plt.xlabel('x')
    plt.ylabel('y')
    title = "CO2 Molar Fraction Field"
    if step is not None:
        title += f" at step {step}"
    if time_hours is not None:
        title += f", time {time_hours:.2f} hrs"
    plt.title(title)

def visualize_field_and_co2(ax_vel, ax_co2, X, Y, u, v, c, p=None, step=None, show_pressure=False, stride=5):
    ax_vel.clear()
    ax_co2.clear()

    # --- Velocity field ---
    vel_mag = np.sqrt(u**2 + v**2)
    cf1 = ax_vel.contourf(X, Y, vel_mag, levels=50, cmap='viridis')
    ax_vel.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        color='white'
    )

    if show_pressure and p is not None:
        ax_vel.contour(X, Y, p, levels=5, colors='k', linewidths=0.8, alpha=0.6)

    ax_vel.set_title(f"Velocity Field (step {step})")
    ax_vel.set_xlabel("x")
    ax_vel.set_ylabel("y")

    # --- CO2 ---
    cf2 = ax_co2.contourf(X, Y, c, levels=50)
    ax_co2.set_title(f"CO2 Scalar (step {step})")
    ax_co2.set_xlabel("x")
    ax_co2.set_ylabel("y")

def plot_CO2_time_evolution(Xavg, Xmax, dt):
    """
    Plot the time evolution of average and maximum CO2 concentration.

    Parameters
    ----------
    Xavg : list or array
        Average CO2 concentration at each time step.
    Xmax : list or array
        Maximum CO2 concentration at each time step.
    dt : float
        Time step size in seconds.
    """ 

    # Conversion en heures
    time_hours = np.arange(len(Xavg)) * dt / 3600.0

    plt.figure(figsize=(10,5))

    plt.plot(time_hours, Xavg, label="Concentration moyenne", linewidth=2, color="blue")
    plt.plot(time_hours, Xmax, label="Concentration maximale", linewidth=2, color="orange")

    # seuil réglementaire
    plt.axhline(CO2_LIMIT, color="red", linestyle="--", linewidth=2,
                label="Seuil réglementaire 0.001")

    # Séparation à 12h
    plt.axvline(time_interval / 2, color="black", linestyle=":", linewidth=2,
                label="Fin période avec source (12h)")

    plt.xlabel("Temps (heures)", fontsize=12)
    plt.ylabel("Fraction molaire de CO₂", fontsize=12)
    plt.title("Évolution temporelle de la concentration de CO₂ dans l’enceinte", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residual_history(history_arr, tol_val=1e-8):
    """
    Plots the convergence history of U, V, and P on a log scale.
    
    Args:
        history_arr: numpy array of shape (N_records, 4) -> [step, res_u, res_v, res_p]
        tol_val: float, the convergence threshold used in the simulation
    """
    if len(history_arr) == 0:
        print("No residual history to plot.")
        return

    steps = history_arr[:, 0]
    res_u = history_arr[:, 1]
    res_v = history_arr[:, 2]
    res_p = history_arr[:, 3]

    plt.figure(figsize=(10, 6))
    
    # Plot residuals
    plt.semilogy(steps, res_u, label='Velocity U Residual', linewidth=1.5)
    plt.semilogy(steps, res_v, label='Velocity V Residual', linewidth=1.5)
    plt.semilogy(steps, res_p, label='Pressure P Residual', linewidth=1.5, linestyle='--')
    
    # Plot Threshold Line
    plt.axhline(y=tol_val, color='r', linestyle=':', linewidth=2, label=f'Threshold ({tol_val})')

    plt.title("Convergence History (Relative Residuals)")
    plt.xlabel("Step")
    plt.ylabel("Relative Residual (L2 Norm)")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

@jit(nopython=True)
def compute_residuals(u_new, u_old, v_new, v_old, p_new, p_old):
    """
    Computes relative L2 norms of the change in fields.
    formula: sum((x_new - x_old)^2) / sum(x_new^2 + epsilon)
    """
    eps = 1e-15 # Safety to avoid division by zero
    
    # Calculate squared differences
    diff_u = (u_new - u_old)**2
    diff_v = (v_new - v_old)**2
    diff_p = (p_new - p_old)**2
    
    # Calculate squared magnitudes
    mag_u = u_new**2
    mag_v = v_new**2
    mag_p = p_new**2
    
    # Compute relative residuals (L2 Norm representation)
    # Sqrt(Sum(diff)) / Sqrt(Sum(mag)) is mathematically equivalent to relative Euclidean distance
    res_u = np.sqrt(np.sum(diff_u)) / (np.sqrt(np.sum(mag_u)) + eps)
    res_v = np.sqrt(np.sum(diff_v)) / (np.sqrt(np.sum(mag_v)) + eps)
    res_p = np.sqrt(np.sum(diff_p)) / (np.sqrt(np.sum(mag_p)) + eps)
    
    return res_u, res_v, res_p

def interpolate_velocity_field(u_fine, v_fine, N_fine, N_target, L):
    """
    Interpolates a velocity field from a fine grid (N_fine) to a target grid (N_target).
    """
    if N_fine == N_target:
        return u_fine, v_fine

    # 1. Define the coordinate systems
    x_fine = np.linspace(0, L, N_fine)
    y_fine = np.linspace(0, L, N_fine)
    
    x_target = np.linspace(0, L, N_target)
    y_target = np.linspace(0, L, N_target)
    
    # 2. Create Interpolator Functions
    # method='linear' is safer for velocity to avoid ringing artifacts from cubic splines
    # bounds_error=False/fill_value is a safeguard, though we shouldn't hit bounds
    interp_u = RegularGridInterpolator((x_fine, y_fine), u_fine, method='linear', bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((x_fine, y_fine), v_fine, method='linear', bounds_error=False, fill_value=None)

    # 3. Create the target meshgrid for query points
    # RegularGridInterpolator expects points in shape (n_points, n_dims)
    X_target, Y_target = np.meshgrid(x_target, y_target, indexing='ij')
    pts = np.array([X_target.flatten(), Y_target.flatten()]).T

    # 4. Interpolate and reshape back to (N, N)
    u_new = interp_u(pts).reshape((N_target, N_target))
    v_new = interp_v(pts).reshape((N_target, N_target))
    
    return u_new, v_new

def run_field_convergence_study(grid_sizes, Q_M3H, field_tol_value):
    """
    Runs the field simulation over multiple grid sizes to check for mesh independence
    based on the Pressure Coefficient (Cp).
    """
    print("="*60)
    print(f"STARTING FIELD MESH CONVERGENCE STUDY (Q={Q_M3H} m3/h)")
    print("="*60)

    results = [] # To store (Nx, Cp, Time, iterations)

    for N in grid_sizes:
        print(f"\n--- Running Grid: {N}x{N} ---")
        
        # We assume a square domain aspect ratio for simplicity, so Nx=Ny
        # Force viz_mode=None to speed up processing (no plotting overhead)
        u_s, v_s, y_idx, hist, cp_val = run_field_simulation(
            Nx=N, 
            Ny=N, 
            Q_M3H=Q_M3H, 
            tol_value=field_tol_value, 
            viz_mode=None, # IMPORTANT: Headless mode
        )
        
        # Get total steps taken from history
        steps_taken = hist[-1][0]
        results.append({
            'N': N,
            'Cp': cp_val,
            'steps': steps_taken
        })
        
        print(f"-> Result: Nx={N}, Cp={cp_val:.5f}")

    # --- Analysis & Plotting ---
    print("\n" + "="*60)
    print("CONVERGENCE RESULTS TABLE")
    print(f"{'Grid':<10} | {'Cp':<12} | {'Diff':<12} | {'Rel Error %':<15}")
    print("-" * 60)

    ns = [r['N'] for r in results]
    cps = [r['Cp'] for r in results]
    errors = []

    for i in range(len(results)):
        N = results[i]['N']
        cp = results[i]['Cp']
        
        if i == 0:
            diff = 0.0
            err = 0.0
            diff_str = "-"
            err_str = "-"
        else:
            prev_cp = results[i-1]['Cp']
            diff = cp - prev_cp
            # Relative Error: |(Fine - Coarse) / Fine|
            err = abs(diff / cp) * 100 
            diff_str = f"{diff:.2e}"
            err_str = f"{err:.4f}%"
        
        errors.append(err)
        print(f"{N}x{N:<6} | {cp:.6f}     | {diff_str:<12} | {err_str:<15}")

    print("="*60)

    # Check against a target threshold (e.g., < 1% change)
    if len(errors) > 1 and errors[-1] < 1.0:
        print(f"SUCCESS: Mesh converged within 1% error at {ns[-1]}x{ns[-1]}")
    elif len(errors) > 1:
        print(f"WARNING: Mesh not strictly converged yet (Error {errors[-1]:.2f}% > 1%)")

    # --- Plotting the Convergence Curve ---
    plt.figure(figsize=(10, 6))
    
    # Plot 1: Cp vs Grid Size
    plt.subplot(1, 2, 1)
    plt.plot(ns, cps, 'o-', linewidth=2, color='b')
    plt.xlabel('Grid Resolution ($N_x$)', fontsize=12)
    plt.ylabel('Pressure Coefficient ($C_p$)', fontsize=12)
    plt.title('Field Convergence ($C_p$)', fontsize=14)
    plt.grid(True, which='both', linestyle='--')

    # Plot 2: Relative Error vs Grid Size
    if len(ns) > 1:
        plt.subplot(1, 2, 2)
        plt.plot(ns[1:], errors[1:], 's--', linewidth=2, color='r')
        plt.xlabel('Grid Resolution ($N_x$)', fontsize=12)
        plt.ylabel('Relative Error (%)', fontsize=12)
        plt.title('Convergence Rate', fontsize=14)
        plt.grid(True, which='both', linestyle='--')
        plt.tight_layout()

    plt.show()

    return ns, cps, errors

def run_scalar_convergence_study(grid_sizes, fine_grid_N, Q_M3H, field_tol, schemes_dict):
    """
    Runs scalar convergence and plots both Absolute Cmax and Relative Error.
    """
    print("="*80)
    print(f"STARTING SCALAR CONVERGENCE STUDY (Reference Velocity Grid: {fine_grid_N}x{fine_grid_N})")
    print("="*80)

    # --- Step 1: Compute High-Fidelity Velocity Field (ONCE) ---
    print(f"Computing 'Frozen' Velocity Field on {fine_grid_N}x{fine_grid_N}...")
    u_fine, v_fine, y_inlet_idx_fine, _, _ = run_field_simulation(
        Nx=fine_grid_N, Ny=fine_grid_N, Q_M3H=Q_M3H, 
        tol_value=field_tol, viz_mode=None
    )
    print("High-fidelity velocity field ready.\n")

    # Storage
    results = {name: {'N': [], 'Cmax': [], 'Error': []} for name in schemes_dict.keys()}

    # --- Step 2: Loop through Grid Sizes ---
    for N in grid_sizes:
        print(f"--- Testing Grid Size: {N}x{N} ---")
        
        # Interpolate Velocity to current N
        u_curr, v_curr = interpolate_velocity_field(u_fine, v_fine, fine_grid_N, N, L=L)
        
        # Recalculate inlet index
        y_curr = np.linspace(0, L, N)
        y_mask = y_curr >= L * 0.75
        y_inlet_idx_curr = np.where(y_mask)[0][0]

        # Run both schemes
        for scheme_name, scheme_func in schemes_dict.items():
            print(f"   Running {scheme_name}...", end="")
            
            _, _, Xmax_hist = run_scalar_simulation(
                u_curr, v_curr, y_inlet_idx_curr, Nx=N, Ny=N, 
                viz_mode=None, 
                advect_scheme=scheme_func
            )
            
            final_cmax = np.max(Xmax_hist)
            results[scheme_name]['N'].append(N)
            results[scheme_name]['Cmax'].append(final_cmax)
            
            # Compute Relative Error (vs previous coarser grid)
            if len(results[scheme_name]['Cmax']) > 1:
                prev_cmax = results[scheme_name]['Cmax'][-2]
                rel_err = abs((final_cmax - prev_cmax) / final_cmax) * 100.0
                results[scheme_name]['Error'].append(rel_err)
            else:
                results[scheme_name]['Error'].append(np.nan) # No error for first grid

            print(f" Done. Cmax: {final_cmax:.4f}")

    # --- Step 3: Plotting ---
    plt.figure(figsize=(14, 6))
    
    styles = {'Upwind': 'o--', 'Lax-Wendroff': 's-'}
    colors = {'Upwind': 'blue', 'Lax-Wendroff': 'red'}

    # Plot 1: Absolute Cmax
    plt.subplot(1, 2, 1)
    for name in schemes_dict:
        plt.plot(results[name]['N'], results[name]['Cmax'], 
                 styles.get(name, 'o-'), color=colors.get(name, 'k'), 
                 linewidth=2, label=name)
    
    plt.xlabel('Grid Resolution ($N$)', fontsize=12)
    plt.ylabel('Peak Concentration ($C_{max}$)', fontsize=12)
    plt.title(f'Scalar Convergence: Magnitude\n(Fixed Velocity Field {fine_grid_N}x{fine_grid_N})', fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--')

    # Plot 2: Relative Error
    plt.subplot(1, 2, 2)
    for name in schemes_dict:
        # Skip the first point (NaN)
        ns = results[name]['N'][1:]
        errs = results[name]['Error'][1:]
        
        if len(ns) > 0:
            plt.plot(ns, errs, 
                     styles.get(name, 'o-'), color=colors.get(name, 'k'), 
                     linewidth=2, label=f"{name} Error")
    
    plt.xlabel('Grid Resolution ($N$)', fontsize=12)
    plt.ylabel('Relative Error vs Previous Grid (%)', fontsize=12)
    plt.title('Convergence Rate (Relative Error)', fontsize=14)
    # Log scale is often better for error plots to see the order of magnitude drop
    plt.yscale('log') 
    plt.grid(True, which='both', linestyle='--')
    plt.legend()

    plt.tight_layout()
    plt.show()

def analyze_flow_rate_sensitivity(Q_values, Nx, Ny, tol_value):
    """
    Runs simulations for a list of Q values, fits a curve C = A/Q + B,
    and calculates the exact flow rate required to meet the limit.
    """
    print(f"--- Starting Sensitivity Analysis (Inverse Fit) for {len(Q_values)} flow rates ---")
    
    peak_co2_values = []

    # 1. Iterate through Q values and simulate
    for i, q in enumerate(Q_values):
        print(f"\nTesting Q = {q} m^3/h ({i+1}/{len(Q_values)})...")
        
        # Run Field (Force viz_mode=None for speed)
        u_steady, v_steady, y_inlet_idx, _, _ = run_field_simulation(
            Nx, Ny, q, tol_value, viz_mode=None
        )
        
        # Run Scalar (Force viz_mode=None for speed)
        dt_s, _, Xmax_hist = run_scalar_simulation(
            u_steady, v_steady, y_inlet_idx, Nx, Ny, 
            viz_mode=None
        )
        
        global_max = np.max(Xmax_hist)
        peak_co2_values.append(global_max)
        print(f" -> Max CO2 for Q={q}: {global_max:.6f}")

    # Convert to arrays
    Q_data = np.array(Q_values)
    C_data = np.array(peak_co2_values)

    # 2. Fit the Curve: C = A * (1/Q) + B
    # We use polyfit on 1/Q to simulate an inverse fit
    inv_Q = 1.0 / Q_data
    coeffs = np.polyfit(inv_Q, C_data, 1) # Returns [A, B]
    A_fit, B_fit = coeffs
    
    print(f"\nFitted Model: CO2 = {A_fit:.5f}/Q + {B_fit:.5f}")

    # 3. Calculate Certified Q
    # Limit = A/Q + B  =>  Q = A / (Limit - B)
    
    certified_Q = None
    
    if np.all(C_data > CO2_LIMIT):
        print("WARNING: Even the highest Q tested did not meet the CO2 limit.")
    elif np.all(C_data < CO2_LIMIT):
        print("INFO: All tested Q values are safe. The lowest Q is sufficient.")
        certified_Q = np.min(Q_values)
    else:
        # Interpolate Q as a function of CO2
        # We need sorted arrays for interpolation. 
        # Since Q is increasing and CO2 is decreasing, we flip them for np.interp
        certified_Q = np.interp(CO2_LIMIT, C_data[::-1], Q_data[::-1])
        print(f"\n>>> CERTIFIED MINIMUM Q: {certified_Q:.4f} m^3/h <<<")

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Generate smooth curve for plotting
    Q_smooth = np.linspace(min(Q_data)*0.9, max(Q_data)*1.1, 100)
    C_smooth = (A_fit / Q_smooth) + B_fit
    
    # Plot Data and Fit
    plt.scatter(Q_data, C_data, color='blue', label='Simulated Data', zorder=5)
    plt.plot(Q_smooth, C_smooth, 'b--', label=f'Fit: $C = {A_fit:.4f}/Q + {B_fit:.4f}$')
    
    # Plot Threshold
    plt.axhline(y=CO2_LIMIT, color='r', linestyle='-', alpha=0.5, label=f'Limit ({CO2_LIMIT})')
    
    # Plot Intersection
    if certified_Q:
        plt.axvline(x=certified_Q, color='g', linestyle='-.', label=f'Required Q ({certified_Q:.2f})')
        plt.scatter([certified_Q], [CO2_LIMIT], color='green', s=100, zorder=10)

    plt.title(f'Sensitivity Analysis: Inverse Law Fit (Grid: {Nx}x{Ny})')
    plt.xlabel('Volumetric Flow Rate Q ($m^3/h$)')
    plt.ylabel('Maximum Mole Fraction $X_{CO2}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    return certified_Q

def run_field_simulation(Nx, Ny, Q_M3H, tol_value, viz_mode=None, plot_interval=100, vel_steady_max_steps=40000, stability_check_interval=20):
    """
    Args:
        Nx, Ny: Grid dimensions
        viz_mode: None (headless), 'field' (velocity), 'co2' (scalar), or 'both'
        plot_interval: Steps between plot updates
    Returns:
        KPIs: (u_center_final, c_max_final, total_time_taken)
    """
    start_time = time.time()
    
    # --- 1. Grid & Geometry Setup ---
    dx = L / Nx
    dy = L / Ny
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # We recalculate these based on the new dx/dy
    x_mask = x <= L / 4.0
    x_extract_index = np.where(x_mask)[0][-1]
    
    y_mask = y >= L * 0.75
    y_inlet_index = np.where(y_mask)[0][0]

    # Pre-calculate Inlet Profile (optimization)
    Q = Q_M3H / 3600  # m^3/s
    # Calculate y coordinates for the inlet section only
    y_inlet_coords = y[y_inlet_index:]
    u_inlet_arr = u_inlet(y_inlet_coords, Q) 

    # --- 2. Time Step & Stability ---
    # Estimate max velocity for CFL (conservative estimate)
    u_max = np.max(np.abs(u_inlet_arr))

    dt_Fo = dx**2 * Fo / nu
    dt_CFL = CFL * dx / u_max
    dt_vel = min(dt_Fo, dt_CFL)
    
    print(f"Grid: {Nx}x{Ny} | dt velocity field: {dt_vel:.5f}s")

    # Compute Reynolds 
    # Inlet Width (Best for checking jet stability/turbulence)
    L_jet = linj # other optoin could Domain Diagonal (measures global transport scale)
    Re_jet = (u_max * L_jet) / nu
    
    print(f"Re (Inlet based):   {Re_jet:.2f}")
    
    if Re_jet > 200: 
        print("WARNING: Re_jet is high (> 200). The physical flow may be unsteady (turbulent/oscillating). A steady-state convergence might be suboptimal.")
    else:
        print("Flow appears to be in the laminar regime.")
    

    # --- 3. Field Initialization ---
    u = np.zeros((Nx, Ny))
    v = np.zeros((Nx, Ny))
    p = np.zeros((Nx, Ny))

    # Apply initial BCs
    u, v = apply_velocity_bcs(u, v, x_extract_index, y_inlet_index, u_inlet_arr)
    p = apply_pressure_bcs(p, x_extract_index)

    # if viz_mode == 'both':
    #     fig, (ax_vel, ax_co2) = plt.subplots(1, 2, figsize=(12, 5))
    #     plt.ion()
    #     plt.show()

    vel_step = 0
    steady = False

    if viz_mode in ('field', 'both'):
        visualize_field(X, Y, u, v, p, step=vel_step, time_hours=0.0)
        #plt.savefig(f'C:/Users/hugog/Documents/CFD_CS/proj4/cfd_cs_proj4/field_profiles/field_{(vel_step)}.png', dpi=300)
        plt.pause(0.1)
    
    # -- Storage for Residual Plotting --
    history_residuals = [] # Will store (step, res_u, res_v, res_p)

    # --- 5. Main Loop ---
    while vel_step < vel_steady_max_steps:

        # Save old fields for residual check
        # Note: We only need to copy if we are about to check stability
        if vel_step % stability_check_interval == 0:
            u_old = u.copy()
            v_old = v.copy()
            p_old = p.copy()

        # -- Navier Stokes --
        u, v = advect(u, v, dx, dy, dt_vel)
        u, v = apply_velocity_bcs(u, v, x_extract_index, y_inlet_index, u_inlet_arr)
        
        u, v = diffuse(u, v, dx, dy, dt_vel, nu)
        u, v = apply_velocity_bcs(u, v, x_extract_index, y_inlet_index, u_inlet_arr)
        
        u, v, p = pressure_projection(u, v, p, dx, dy, dt_vel, rho, x_extract_index)
        # Pressure BCS are updated inside pressure projection  
        u, v = apply_velocity_bcs(u, v, x_extract_index, y_inlet_index, u_inlet_arr)
        
        vel_step += 1

        # -- Visualization --
        if viz_mode in ('field', 'both') and vel_step % plot_interval == 0:
            visualize_field(X, Y, u, v, p, step=vel_step, time_hours=vel_step*dt_vel/3600.0)
            #plt.savefig(f'C:/Users/hugog/Documents/CFD_CS/proj4/cfd_cs_proj4/field_profiles/field_{(vel_step)}.png', dpi=300)
            plt.pause(0.1)

        # -- Stability Check & Residual Recording --
        if vel_step % stability_check_interval == 0:
            res_u, res_v, res_p = compute_residuals(u, u_old, v, v_old, p, p_old)
            
            # Store for plotting
            history_residuals.append([vel_step, res_u, res_v, res_p])
            
            #print(f"Step {vel_step}: Res U={res_u:.1e} | Res V={res_v:.1e} | Res P={res_p:.1e}")

            # Convergence Condition: All fields must settle
            if res_u < tol_value and res_v < tol_value and res_p < tol_value:
                print(f"Steady state reached at step {vel_step}, i.e. {vel_step * dt_vel / 3600} hours")
                steady = True
                break


    if not steady:
        print(f"Warning: velocity did not reach steady state within {vel_steady_max_steps} steps, i.e. {vel_steady_max_steps * dt_vel / 3600} hours. Proceeding with last field.")

    # Freeze steady field (use u,v as steady)

    cp = compute_pressure_coefficient(p, u, v, y_inlet_index, x_extract_index)

    elapsed = time.time() - start_time
    print(f"{Nx}x{Ny} Field computed in {elapsed:.2f}s")
    
    return u.copy(), v.copy(), y_inlet_index, np.array(history_residuals), cp
    
def run_scalar_simulation(u_steady, v_steady, y_inlet_index, Nx, Ny, advect_scheme =  advect_scalar, viz_mode=None, plot_interval=100):

    start_time = time.time()
    
    # --- 1. Grid & Geometry Setup ---
    dx = L / Nx
    dy = L / Ny
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Source Indices
    x_source_start = np.where(xc - 0.5*l_source <= x)[0][0]
    x_source_end = np.where(x <= xc + 0.5*l_source)[0][-1]
    y_source_start = np.where(yc - 0.5*l_source <= y)[0][0]
    y_source_end = np.where(y <= yc + 0.5*l_source)[0][-1]

    # Re-evaluate dt for scalar using steady u_max and molecular diffusion D_co2
    u_max = np.max(np.sqrt(u_steady**2 + v_steady**2))

    dt_Fo_scalar = dx**2 * Fo / D_co2
    dt_CFL_scalar = CFL * dx / u_max
    dt_scalar = min(dt_Fo_scalar, dt_CFL_scalar)

    # Start scalar 24-hr clock now
    Nt_scalar = int(t / dt_scalar)
    print(f"Scalar dt re-evaluated using steady u_max={u_max:.3e} -> dt_scalar={dt_scalar:.5e}s")

    c = INIT_CO2 * np.ones((Nx, Ny))
    S = np.zeros((Nx,Ny))
    c = apply_scalar_bcs(c, y_inlet_index, INLET_CO2)
    c += dt_scalar * compute_source_term(c, S, x_source_start, x_source_end, y_source_start, y_source_end)

    Xavg = [np.mean(c)]
    Xmax = [np.max(c)]

    if viz_mode in ('co2', 'both'):
        visualize_CO2(X, Y, c, step=0, time_hours=0.0)
        #plt.savefig(f'C:/Users/hugog/Documents/CFD_CS/proj4/cfd_cs_proj4/scalar_profiles/scalar_{(0)}.png', dpi=300)
        plt.pause(0.1)

    for n in range(Nt_scalar):
        # # -- Scalar Transport --
        c = advect_scheme(c, u_steady, v_steady, dx, dy, dt_scalar)
        c = diffuse_scalar(c, dx, dy, dt_scalar, D_co2)
        
        # Source Term (Inline for speed)
        if n*dt_scalar < time_interval/2 * 3600:
            c += dt_scalar * compute_source_term(c, S, x_source_start, x_source_end, y_source_start, y_source_end)

        c = apply_scalar_bcs(c, y_inlet_index, INLET_CO2) 

        Xavg.append(np.mean(c))
        Xmax.append(np.max(c))

        if viz_mode in ('co2', 'both') and n % plot_interval == 0:
            visualize_CO2(X, Y, c, step=n, time_hours=n*dt_scalar/3600)
            #plt.savefig(f'C:/Users/hugog/Documents/CFD_CS/proj4/cfd_cs_proj4/scalar_profiles/scalar_{(n)}.png', dpi=300)
            plt.pause(0.1)

    elapsed = time.time() - start_time
    print(f"{Nx}x{Ny} Concentration field computed in {elapsed:.2f}s")

    return dt_scalar, Xavg, Xmax

if __name__ == "__main__":
    Nx, Ny = 151, 151
    Q_M3H = 3.0  # Inlet flow rate in m^3/h
    field_tol_value = 1e-5  # convergence tolerance for velocity and pressure fields 1e-6
    
    # ======== Evaluate SAMPLE CASE ==================================================== 
    viz_mode = 'both'  # Options: None, 'field', 'co2', 'both'

    u_steady, v_steady, y_inlet_index, history_res, cp = run_field_simulation(Nx, Ny, Q_M3H, field_tol_value, viz_mode=viz_mode)
    print(f"Cp for Q = {Q_M3H} m^3/h : {cp:.5f}")
    plot_residual_history(history_res, tol_val=field_tol_value)    # Plot the Residuals immediately to check quality

    dt_scalar, Xavg, Xmax = run_scalar_simulation(u_steady, v_steady, y_inlet_index, Nx, Ny, viz_mode=viz_mode)
    plt.show()
        
    max_X = np.max(Xmax)
    idx_max = np.argmax(Xmax)

    # Plot CO2 time evolution
    plot_CO2_time_evolution(Xavg, Xmax, dt_scalar)
    print(f"Maximum CO2 concentration reached: {max_X:.6f} at {idx_max * dt_scalar / 3600:.2f} hours.")

    # ======== Sensitivity Analysis, Optimal Q ========================================
    # Nx, Ny = 151, 151 # Slightly coarser grid for the sweep to save time
    # field_tol_value = 1e-5 
    
    # # Define the range of Q's to test (e.g., from 1 to 10 m3/h)
    # # More points = smoother curve but longer computation
    # Q_range = np.linspace(1.0, 10.0, 10) 
    # Q_range_specific = np.linspace(6.3, 7.0, 10) 

    # # --- Run Sensitivity Analysis ---
    # required_Q = analyze_flow_rate_sensitivity(Q_range_specific, Nx, Ny, field_tol_value)
    # plt.show()
    
    # # ===================== FIELD CONVERGENCE STUDY =================================
    # grid_resolutions = [21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261] 
    # # test_meshes = [21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261, 281] 

    # # 3. Run the Convergence Study
    # grids, cps, errs = run_field_convergence_study(grid_resolutions, Q_M3H, field_tol_value)

    # ===================== SCALAR CONVERGENCE STUDY ===================================
    # Configuration
    # Nxy_fine = 201 # The high-res velocity grid
    # grid_resolutions = [21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261] # The scalar grids to test
    
    # # 2. Define your available schemes
    # # Make sure you have these functions defined in your code scope!
    # schemes = {
    #     'Upwind': advect_scalar,               # Your standard function
    #     'Lax-Wendroff': advect_scalar_lax_wendroff # Your new implementation
    # }
    # # 3. Run the Comparative Study
    # run_scalar_convergence_study(grid_resolutions, Nxy_fine, Q_M3H, field_tol_value, schemes)