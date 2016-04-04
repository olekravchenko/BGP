import matplotlib.pyplot as plt
import numpy as np


gamma = 1.4


def plot(X, U, a, b):
    plt.plot(X, U, 'r-')
    plt.axis([a, b, -0.2, 1.2])
    plt.xlabel('x')
    plt.ylabel('U')
    # plt.show()


def entry_condition(N, h):
    N += 3
    x = [0.0] * N
    rho = [0.0] * N
    V = [0.0] * N
    P = [0.0] * N
    for i in range(N):
        x[i] = (i-1) * h
        if x[i] < 0.5:
            rho[i] = 1.0
            V[i] = 0.0
            P[i] = 1.0
        else:
            rho[i] = 0.125
            V[i] = 0.0
            P[i] = 0.1
    return (x, rho, V, P)


def Coding(rho, V, P, h):
    N = len(rho)
    U = np.zeros((N, 3))
    for i in range(N):
        U[i][0] = rho[i]
        U[i][1] = rho[i] * V[i]
        U[i][2] = P[i] / (gamma - 1.0) + 0.5 * rho[i] * V[i] * V[i]
    Ux = np.zeros((N, 3))
    for i in range(1,N-1):
        Ux[i][0] = 0.5 * (rho[i+1] - rho[i-1]) / h
        Ux[i][1] = 0.5 * (U[i+1][1] - U[i-1][1]) / h
        Ux[i][2] = 0.5 * (U[i+1][2] - U[i-1][2]) / h
    return (U, Ux)


def Decoding(U):
    N = U.shape[0]
    rho = [0.0] * N
    V = [0.0] * N
    P = [0.0] * N
    for i in range(N):
        rho[i] = U[i][0]
        V[i] = U[i][1] / rho[i]
        P[i] = (gamma - 1.0) * (U[i][2] - 0.5 * rho[i] * V[i] * V[i])
    return rho, V, P


def solve(U, Ux, tau, h, i):
    N = U.shape[0]
    F = np.zeros((N, 3))
    Ft = np.zeros((N, 3))
    Fx = np.zeros((N, 3))
    A = np.zeros((3,3))
    for m in range(0, N, 1):
        W = U[m,:]
        A[0,0] = 0.0
        A[0,1] = 1.0
        A[0,2] = 0.0
        A[1,0] = 0.5 * (gamma - 3.0) * (W[1] / W[0]) ** 2.0
        A[1,1] = (3.0 - gamma) * W[1] / W[0]
        A[1,2] = gamma - 1.0
        A[2,0] = -gamma * W[1] * W[2] / W[0] ** 2.0 + (gamma - 1.0) * (W[1] / W[0]) ** 3.0
        A[2,1] = gamma * W[2] / W[0] - 1.5 * (gamma - 1.0) * (W[1] / W[0]) ** 2.0
        A[2,2] = gamma * W[1] / W[0]
        F[m,:] = A.dot(W)
        Ft[m,:] = -(A.dot(A)).dot(Ux[m,:])
        Fx[m,:] = A.dot(Ux[m,:])
    U1 = np.zeros((N, 3))
    Ux1 = np.zeros((N, 3))
    lam = tau / h
    for m in range(i+1, N-1, 2):
        U1[m,:] = 0.5 * (U[m-1,:] + U[m+1,:]) + 0.5 * lam * (F[m-1,:] - F[m+1,:]) + \
            0.25 * h * (Ux[m-1,:] - Ux[m+1,:]) + 0.25 * tau * lam * (Ft[m-1,:] - Ft[m+1,:])
        Ux1[m,:] = 0.5 * (Ux[m-1,:] + Ux[m+1,:]) + 0.5 * lam * (Fx[m-1,:] - Fx[m+1,:])
    for m in range(i, N-1, 2):
        U1[m,:] = U[m,:]
        Ux1[m,:] = Ux[m,:]
    U1[0,:] = U[0,:]
    U1[-1,:] = U[-1,:]
    Ux1[0,:] = Ux[0,:]
    Ux1[-1,:] = Ux[-1,:]
    return (U1, Ux1)


if __name__ == "__main__":
    a = 0.0
    b = 1.0
    N = 2**11
    h = (b - a) / N
    T = 0.25
    cfl = 0.45
    dt = h * cfl
    (x, rho, V, P) = entry_condition(N=N, h=h)
    (U, Ux) = Coding(rho=rho, V=V, P=P, h=h)
    for i in range(int(T/dt)):
        U, Ux = solve(U=U, Ux=Ux, tau=dt, h=h, i=i % 2)
        print(i*dt)
    rho, V, P = Decoding(U)
    plot(X=x, U=rho, a=a, b=b)
    # plot(X=x, U=V, a=a, b=b)
    # plot(X=x, U=P, a=a, b=b)
    plt.show()
