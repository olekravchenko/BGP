import matplotlib.pyplot as plt
from numpy import sign
from math import sqrt


def an_sol(a, b, h, t, N):
    L = b - a
    N += 1
    x = [0] * N
    u = [0] * N
    for i in range(N):
        x[i] = i * h
        if (a <= x[i] <= a + t) and (0 < t <= 2.0 * L):
            u[i] = (x[i] - a) / t
        elif (a + t <= x[i] <= b + 0.5 * t) and (0 < t <= 2.0 * L):
            u[i] = 1.0
        elif (x[i] < a) and (x[i] > b + 0.5 * t) and (0 < t <= 2.0 * L):
            u[i] = 0.0
        elif (a <= x[i] <= a + sqrt(2.0 * L * t)) and (t > 2.0 * L):
            u[i] = (x[i] - a) / t
        else:
            u[i] = 0.0
    # for i in range(N):
    #     x[i] = i * h
    #     if (x[i] <= b + 0.5*t):
    #         u[i] = 1.0
    #     else:
    #         u[i] = 0.0
    plt.plot(x, u, 'g-')
    # plt.show()


def plot(X, U, a, b):
    plt.plot(X, U, 'r-')
    plt.axis([a, b, -0.2, 1.2])
    plt.xlabel('x')
    plt.ylabel('U')
    plt.show()


def entry_condition(N, h):
    N += 3
    x = [0.0] * N
    u = [0.0] * N
    for i in range(N):
        x[i] = (i-1) * h
        if i in range(N / 3, 2 * N / 3):
            u[i] = 1.0
        else:
            u[i] = 0.0
        # if i in range(N / 2):
        #     u[i] = 1.0
        # else:
        #     u[i] = 0.0
    u_prim = [0] * N
    for i in range(1, N-1):
        u_prim[i] = 0.5 * (u[i+1] - u[i-1]) / h
    u_prim[-1] = u_prim[2]
    u_prim[0] = u_prim[-3]
    return (x, u, u_prim)


def solve(u, u_prim, c, tau, h):
    F = [0.0] * len(u)
    Ft = [0.0] * len(u)
    Fx = [0.0] * len(u)
    for m in range(len(u)):
        F[m] = 0.5*c[m]*u[m]
        Ft[m] = -c[m]**2 * u_prim[m]
        Fx[m] = c[m]*u_prim[m]
    u1 = [0.0] * len(u)
    u_prim1 = [0.0] * len(u)
    for m in range(1, len(u)-1):
        lam = tau / h
        u1[m] = 0.5 * (u[m-1] + u[m+1]) + 0.5 * lam * (F[m-1] - F[m+1]) + \
            0.25 * h * (u_prim[m-1] - u_prim[m+1]) + 0.25 * tau * lam * (Ft[m-1] - Ft[m+1])
        u_prim1[m] = 0.5 * (u_prim[m-1] + u_prim[m+1]) + 0.5 * lam * (Fx[m-1] - Fx[m+1])
        # u1[m] = u[m] - 0.5*lam*(F[m+1] - F[m-1]) + 0.5*lam*abs(c[m])*(u[m+1] - 2.0*u[m] + u[m-1]) + \
        # u_prim1[m] = u_prim[m] - lam * (Fx[m] - Fx[m-1])
        # u1[m] = u[m] - lam*(F[m] - F[m-1]) + \
             # 0.5 * (u_prim1[m] - u_prim[m]) / lam - 0.5 * lam *(Ft[m] - Ft[m-1])
        # u1[m] = 0.5*(u[m-1] + u[m+1]) - 0.5*lam*(F[m+1] - F[m-1])# + \
        # u_prim1[m] = 0.5 * (u_prim[m-1] + u_prim[m+1]) + 0.5 * tau * (Fx[m-1] - Fx[m+1]) / h
    for m in range(1, len(u)-1):
        u_prim[m] = minmod(0.5*(u_prim1[m-1] + u_prim1[m]), u_prim1[m], 0.5*(u_prim1[m] + u_prim1[m+1]))
        # u_prim[m] = minmod((u1[m] - u1[m-1])/h, u_prim1[m], (u1[m+1] - u1[m])/h)
    u1[-1] = u1[2]
    u1[0] = u1[-3]
    u_prim1[-1] = u_prim1[2]
    u_prim1[0] = u_prim1[-3]
    u_prim[-1] = u_prim[2]
    u_prim[0] = u_prim[-3]
    return (u1, u_prim1)


def minmod(a1, a2, a3):
    if ((sign(a1) == sign(a2)) and (sign(a1) == sign(a3))):
        return min(abs(a1), abs(a2), abs(a3))
    else:
        return 0.0


if __name__ == "__main__":
    a = 0.0
    b = 6.0
    N = 64
    h = (b - a) / N
    T = 1.5
    cfl = 0.9
    dt = h * cfl
    (x, u, u_prim) = entry_condition(N=N, h=h)
    c = [1.0] * (N + 3)
    for i in range(30):
        u, u_prim = solve(u=u, u_prim=u_prim, c=u, tau=dt, h=h)
    an_sol(a=2, b=4, h=h, t=30*dt, N=N)
    plot(X=x, U=u, a=a, b=b)