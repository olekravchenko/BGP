# Source:
# http://www.quarkgluon.com/a-one-dimensional-python-based-riemann-solver/
# http://www.quarkgluon.com/hll-riemann-solver-implementation-in-python/
import os.path
import matplotlib.pyplot as plt
import numpy as np

import sys
import pandas as pd



def plot(X, U, a, b):
    plt.plot(X, U, 'r-')
    plt.axis([a, b, -0.2, 1.2])
    plt.xlabel('x')
    plt.ylabel('U')
    # plt.show()

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

def roe_flux(wL, wR, gamma):
    #"""
    #Use the Roe approximate Riemann solver to calculate fluxes.
    #"""
    uL = w2u(wL, gamma)
    uR = w2u(wR, gamma)

    # Primitive and other variables.
    # Left state
    rhoL = wL[0]
    vL = wL[1]
    pL = wL[2]
    aL = np.sqrt(gamma*pL/rhoL)
    HL = ( uL[2] + pL ) / rhoL

    # Right state
    rhoR = wR[0]
    vR = wR[1]
    pR = wR[2]
    aR = np.sqrt(gamma*pR/rhoR)
    HR = ( uR[2] + pR ) / rhoR

    # First compute the Roe Averages
    RT = np.sqrt(rhoR/rhoL);
    rho = RT*rhoL
    v = (vL+RT*vR)/(1.0+RT)
    H = (HL+RT*HR)/(1.0+RT)
    a = np.sqrt( (gamma-1.0)*(H-0.5*v*v) )

    # Differences in primitive variables.
    drho = rhoR - rhoL
    du = vR - vL
    dP = pR - pL

    # Wave strength (Characteristic Variables).
    dV = np.array([0.0,0.0,0.0])
    dV[0] = 0.5*(dP-rho*a*du)/(a*a)
    dV[1] = -( dP/(a*a) - drho )
    dV[2] = 0.5*(dP+rho*a*du)/(a*a)

    # Absolute values of the wave speeds (Eigenvalues)
    ws = np.array([0.0,0.0,0.0])
    ws[0] = abs(v-a)
    ws[1] = abs(v)
    ws[2] = abs(v+a)

    # Modified wave speeds for nonlinear fields (the so-called entropy fix, which
    # is often implemented to remove non-physical expansion shocks).
    # There are various ways to implement the entropy fix. This is just one
    # example. Try turn this off. The solution may be more accurate.
    Da = max(0.0, 4.0*((vR-aR)-(vL-aL)) )
    if (ws[0] < 0.5*Da):
        ws[0] = ws[0]*ws[0]/Da + 0.25*Da
    Da = max(0.0, 4.0*((vR+aR)-(vL+aL)) )
    if (ws[2] < 0.5*Da):
        ws[2] = ws[2]*ws[2]/Da + 0.25*Da

    # Right eigenvectors
    R = np.zeros((3,3))

    R[0][0] = 1.0
    R[1][0] = v - a
    R[2][0] = H - v*a

    R[0][1] = 1.0
    R[1][1] = v
    R[2][1] = 0.5*v*v

    R[0][2] = 1.0
    R[1][2] = v + a
    R[2][2] = H + v*a

    # Compute the average flux.
    flux = 0.5*( euler_flux(wL, gamma) + euler_flux(wR, gamma) )

    # Add the matrix dissipation term to complete the Roe flux.
    for i in range(0,3):
        for j in range(0,3):
            flux[i] = flux[i] - 0.5*ws[j]*dV[j]*R[i][j]
    return flux

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
        fluxes[i] = roe_flux(wL, wR, gamma)

    # Update solution
    for i in range(1, bcells-1):
        U[i] = U[i] + (dt/dx) * (fluxes[i-1]-fluxes[i])

    # Boundary Conditions
    U[0] = U[1]
    U[bcells-1] = U[bcells-2]

if __name__ == "__main__":
    gamma = 1.4
    cells = 1000
    bcells = cells + 2
    dx = 1.0/cells

    cfl = 0.8
    t = 0.0
    tf = 0.2
    nsteps = 0

    U = np.zeros((bcells,3))
    #x = np.zeros(bcells)
    #for i in range(bcells):
    #    x[i] = (i-1) * dx
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
        print(n)

    filename = "d:/GitHub/BGP/output/1d_roe_godunov_euler_sod.csv"
    out_csv = open(filename, "w")
    for elem in U:
        new_elem = u2w(elem, gamma)
        #print(tuple(new_elem))
        out_csv.write("%s,%s,%s\n" % tuple(new_elem))

    #plot(X=x, U=new_elem, a=0, b=1)
    # plot(X=x, U=V, a=a, b=b)
    # plot(X=x, U=P, a=a, b=b)
    #plt.show()

    data = pd.io.parsers.read_csv(
        filename, delimiter=",", header=None,
        names=["rho", "u", "p", "e"],
    )
##
##    # Plot three charts
##    fig = plt.figure()
##    fig.patch.set_facecolor('white')     # Set the outer colour to white
##
##    clr = "red"
##    # Plot the density
##    ax1 = fig.add_subplot(221, ylabel='Density', ylim=[0.0, 1.2])
##    data['rho'].plot(ax=ax1, marker='o', markersize=4, color=clr, linestyle='')
##
##    # Plot the velocity
##    ax2 = fig.add_subplot(222, ylabel='Velocity', ylim=[0.0, 1.5])
##    data['u'].plot(ax=ax2, marker='o', markersize=4, color=clr, linestyle='')
##
##    # Plot the pressure
##    ax3 = fig.add_subplot(223, ylabel='Pressure', ylim=[0.0, 1.2])
##    data['p'].plot(ax=ax3, marker='o', markersize=4, color=clr, linestyle='')
##
##    # Plot the pressure
##    ax4 = fig.add_subplot(224, ylabel='Internal Energy', ylim=[1.8, 3.8])
##    data['e'].plot(ax=ax4, marker='o', markersize=4, color=clr, linestyle='')  
##
##    # Plot the figure
##    plt.show()
