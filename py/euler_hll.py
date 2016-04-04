# Source:
# http://www.quarkgluon.com/a-one-dimensional-python-based-riemann-solver/
# http://www.quarkgluon.com/hll-riemann-solver-implementation-in-python/
import os.path
import matplotlib.pyplot as plt
import numpy as np

import sys
import pandas as pd

from math import sqrt
#from utils import total_energy_E, enthalpy_H, euler_flux, w2u
#from hll_flux import hll_flux
#from utils import w2u, u2w, internal_energy_e




def w2u(w, gamma):
    #"""
    #Convert the primitive to conservative variables.
    #"""
    u = np.zeros(3)
    u[0] = w[0]
    u[1] = w[0]*w[1]
    u[2] = w[2]/(gamma-1.0)+0.5*w[0]*w[1]*w[1]
    return u


def u2w(u, gamma):
    #"""
    #Convert the conservative to primitive variables.
    #"""
    w = np.zeros(3)
    w[0] = u[0]
    w[1] = u[1]/u[0]
    w[2] = (gamma-1.0)*( u[2] - 0.5*w[0]*w[1]*w[1] )
    return w


def internal_energy_e(w, gamma):
    #"""
    #Internal energy, e.
    #
    #See Toro, 2nd Ed., pg 3.
    #"""
    return w[2]/(w[0]*(gamma - 1.0))


def total_energy_E(w, gamma):
    #"""
    #Total energy per unit volume, E.
    #
    #See Toro, 2nd Ed., pg 3.
    #"""
    e = internal_energy_e(w, gamma)
    return w[0]*(0.5*w[1]*w[1] + e)


def enthalpy_H(w, gamma):
    #"""
    #Calculate enthalpy, H = (E+p)/rho
    #
    #See Toro, 2nd Ed., pg 324.
    #"""
    E = total_energy_E(w, gamma)
    return (E + w[2])/w[0]


def euler_flux(w, gamma):
    #"""
    #Calculate the conservative Euler fluxes.
    #"""
    rho = w[0]
    u = w[1]
    p = w[2]

    a2 = gamma*p/rho

    f_1 = rho*u
    f_2 = rho*u*u + p
    f_3 = rho*u*( a2/(gamma-1.0) + 0.5*u*u )

    return np.array([f_1, f_2, f_3])

def enthalpy_H_tilde(wL, wR, gamma):
    #"""
    #Calculate enthalpy estimate, H_tilde.
    #
    #See Toro, 2nd Ed., pg 324.
    #"""
    H_L = enthalpy_H(wL, gamma)
    H_R = enthalpy_H(wR, gamma)
    num = sqrt(wL[0])*H_L + sqrt(wR[0])*H_R
    denom = sqrt(wL[0]) + sqrt(wR[0])
    return num/denom


def calc_u_tilde(wL, wR):
    #"""
    #Roe-average particle wave-speed estimate.
    #
    #See Toro, 2nd Ed., pg 324.
    #"""
    num = sqrt(wL[0])*wL[1] + sqrt(wR[0])*wR[1]
    denom = sqrt(wL[0]) + sqrt(wR[0])
    return num/denom


def calc_a_tilde(wL, wR, gamma):
    #"""
    #Roe-average sound speed estimate.
    #
    #See Toro, 2nd Ed., pg 324.
    #"""
    u_tilde = calc_u_tilde(wL, wR)
    H_tilde = enthalpy_H_tilde(wL, wR, gamma)
    return sqrt( (gamma - 1.0) * (H_tilde - 0.5 * u_tilde**2) )


def calc_wave_speeds_tilde(wL, wR, gamma):
    #"""
    #Calculate wave-speed estimates, S_L and S_R.
    #
    #See Toro, 2nd Ed., pg 324.
    #"""
    u_tilde = calc_u_tilde(wL, wR)
    a_tilde = calc_a_tilde(wL, wR, gamma)
    S_L = u_tilde - a_tilde
    S_R = u_tilde + a_tilde
    return S_L, S_R


def calc_wave_speeds_direct(wL, wR, gamma):
    #"""
    #Calculate wave-speeds directly. Returns S_L and S_R.
    #
    #See Toro, 2nd Ed., pg 324.
    #"""
    S_L = wL[1] - sqrt(gamma*wL[2]/wL[0])
    S_R = wR[1] + sqrt(gamma*wR[2]/wR[0])
    return S_L, S_R    


def hll_flux(wL, wR, gamma):
    #"""
    #Use the HLL - Harten, Lax and van Leer solver to obtain
    #the Euler fluxes
    #
    #See Toro, 2nd Ed., pg 320.
    #"""
    S_L, S_R = calc_wave_speeds_direct(wL, wR, gamma)
    F_L = euler_flux(wL, gamma)
    F_R = euler_flux(wR, gamma)
    U_L = w2u(wL, gamma)
    U_R = w2u(wR, gamma)

    if 0.0 <= S_L:
        return F_L
    elif S_L <= 0.0 and 0.0 <= S_R:
        num = S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)
        denom = S_R - S_L
        return num/denom
    elif 0.0 >= S_R:
        return F_R

def test_case_sod(U, bcells, gamma):
    #"""
    #Populate the initial data with the Sod test case.
    #"""
    mid_point = int(bcells/10.0*3.0)
    sod_L = np.array([1.0, 0.75, 1.0])
    sod_R = np.array([0.125, 0.0, 0.1])

    for i in range(0, mid_point):
        U[i,:] = w2u(sod_L, gamma)
    for i in range(mid_point, bcells):
        U[i,:] = w2u(sod_R, gamma)


def calc_time_step(cfl, dx, gamma, bcells, U):
    #"""
    #Calculates the maximum wavespeeds and thus the timestep
    #via an enforced CFL condition.
    #"""
    max_speed = -1.0

    for i in range(1,bcells-1):
        w = u2w(U[i], gamma)
        u = w[1]
        c = np.sqrt(gamma*w[2]/w[0])
        max_speed = max(max_speed, abs(u)+c)

    dt = cfl*dx/max_speed  # CFL condition
    return dt


def update_solution(U, fluxes, dt, dx, gamma, bcells):
    #"""
    #Updates the solution of the equation 
    #via the Godunov procedure.
    #"""
    # Create fluxes
    for i in range(0, bcells-1):
        wL = u2w(U[i], gamma)
        wR = u2w(U[i+1], gamma)
        fluxes[i] = hll_flux(wL, wR, gamma)

    # Update solution
    for i in range(1, bcells-1):
        U[i] = U[i] + (dt/dx) * (fluxes[i-1]-fluxes[i])

    # BCs
    U[0] = U[1]
    U[bcells-1] = U[bcells-2]


if __name__ == "__main__":
    gamma = 1.4
    cells = 100
    bcells = cells + 2
    dx = 1.0/cells

    cfl = 0.8
    t = 0.0
    tf = 0.2
    nsteps = 0

    U = np.zeros((bcells,3))
    fluxes = np.zeros((bcells,3))
    test_case_sod(U, bcells, gamma)

    for n in range(1, 150):
        if (t==tf): break
        dt = calc_time_step(cfl, dx, gamma, bcells, U)
        if (t+dt > tf):
            dt = tf - t
        update_solution(U, fluxes, dt, dx, gamma, bcells)
        t += dt
        nsteps += 1
        print(t)

    filename = "d:\\GitHub\BGP\\output\\1d_hll_godunov_euler_sod.csv"          
    out_csv = open(filename, "w")
    for elem in U:
        new_elem = u2w(elem, gamma)
        #e = internal_energy_e(new_elem, gamma)
        #print(tuple(np.append(new_elem,e)))
        #out_csv.write("%s,%s,%s\n" % 
            #tuple(new_elem))
        print(tuple(new_elem))
        out_csv.write("%s,%s,%s\n" % tuple(new_elem))

        
    print('...wrote')
                      
##    #filename = sys.argv[1]
##    data = pd.io.parsers.read_csv(
##        filename, delimiter=",", header=None,
##        names=["rho", "u", "p", "e"],
##    )
##
##    # Plot three charts
##    fig = plt.figure()
##    fig.patch.set_facecolor('white')     # Set the outer colour to white
##    
##    # Plot the density
##    ax1 = fig.add_subplot(221, ylabel='Density', ylim=[0.0, 1.2])
##    data['rho'].plot(ax=ax1, marker='o', markersize=4, color="black", linestyle='')
##
##    # Plot the velocity
##    ax2 = fig.add_subplot(222, ylabel='Velocity', ylim=[0.0, 1.5])
##    data['u'].plot(ax=ax2, marker='o', markersize=4, color="black", linestyle='')
##
##    # Plot the pressure
##    ax3 = fig.add_subplot(223, ylabel='Pressure', ylim=[0.0, 1.2])
##    data['p'].plot(ax=ax3, marker='o', markersize=4, color="black", linestyle='')
##
##    # Plot the pressure
##    ax4 = fig.add_subplot(224, ylabel='Internal Energy', ylim=[1.8, 3.8])
##    data['e'].plot(ax=ax4, marker='o', markersize=4, color="black", linestyle='')  
##
##    # Plot the figure
##    plt.show()
