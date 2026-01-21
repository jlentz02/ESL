#Implentation of smoothing splines

#Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from General_methods import generate_sine_data



#Computes the value of B_i_k(x) for a list of knots t
#X is the vector of supports for the spline
#k is the degree of the spline
#i is the current index
#t is are the knots (need to add the extra knots for the ends, I think)
#Code is adapted from the scipy bspline method
def B(x, k, i, t):
    #Degree 1 spline
    if k == 0:
        ind = torch.logical_and(t[i] <= x, x <= t[i+1])
        ind = ind.type(torch.int)
        return ind
    #Checks tail knots
    if t[i + k] == t[i]:
        c1 = 0
    #First part of computation
    else:
        c1 = (x - t[i])/(t[i + k] - t[i])*B(x, k-1, i, t)
    #Checks tail knots
    if t[i + k + 1] == t[i + 1]:
        c2 = 0
    #Second part of computation
    else: 
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1])*B(x, k-1, i+1, t)
    #Returns combined sum
    return c1 + c2

#Computes the spline function
def bspline(t, k, c):
    n = len(t) - k - 1
    return lambda x: sum(c[i]*B(x, k, i, t) for i in range(n))

#Computes the derivate of B at a certain point
#We will apply this twice to compute the second derivative.
#Then, we will use numerical quadrature (D^TD*deltat) to compute omega
def B_deriv(x, k, i, t):
    if k == 0:
        return 0
    if t[i+k] == t[i]:
        c1 = 0
    else:
        c1 = B(x, k-1, i, t)/(t[i+k] - t[i])
    if t[i+k+1] == t[i+1]:
        c2 = 0
    else:
        c2 = B(x, k-1, i+1, t)/(t[i+k+1]-t[i+1])
    return k*(c1-c2)

#Computes second derivative by calling B_deriv inside the above expression
#This is probably redundant and these functions could be combined
def B_2deriv(x, k, i, t):
    if k == 0:
        return 0
    elif k == 1:
        return 0
    if t[i+k] == t[i]:
        c1 = 0
    else:
        c1 = B_deriv(x, k-1, i, t)/(t[i+k] - t[i])
    if t[i+k+1] == t[i+1]:
        c2 = 0
    else: 
        c2 = B_deriv(x, k-1, i+1, t)/(t[i+k+1]-t[i+1])
    return k*(c1 - c2)

#Creates the D matrix of second derivatives of the B-splines
def make_D(X, k, t):
    n = len(t) - k - 1
    D = torch.zeros([len(X), n])
    for l in range(n):
            D[:, l] = B_2deriv(X, k, l, t)
    return D

#Forms the n x n (the n as defined above) matrix D^TD*(deltat) which is our numerical 
#approximation of omega_ij = \int N''_j(t)N''_i(t)dt
def make_omega(X, k, t):
    D = make_D(X, k ,t)
    #Assuming your grid t is evenly spaced (it better be)
    dt = t[k+1]-t[k]
    omega = torch.transpose(D, 0, 1)@D*dt
    return omega

#Adds boundary knots to list of knots by duplicating the first and last knot k times
def add_boundary_knots(t, k):
    if k == 0:
        return t
    front = [t[0] for i in range(k)]
    back = [t[-1] for i in range(k)]
    new_t = front + t + back
    return new_t

#Makes X |-> B_mat by applying B_i to X for the i B-splines
def make_B(X, k, t):
    n = len(t) - k - 1
    B_mat = torch.zeros([len(X), n])
    for l in range(n):
        B_mat[:, l] = B(X, k, l, t)
    return B_mat

#Solve for gamma
#Gets the coefficients for gamma which will make our spline fit the data
#This is gamma = (B^TB + lam*omega)^-1B^Ty
#Note this is a ridge regression
def compute_gamma(B_mat, omega, lam, Y):
    BT = torch.transpose(B_mat, 0, 1)
    BTBlam = BT@B_mat + lam*omega
    inv = torch.linalg.inv(BTBlam)
    invBT = inv@BT
    return invBT@Y

#Generates t from X using a specified number of knots
#n knots
def make_t(X, n):
    low = torch.min(X)
    high = torch.max(X)
    t = [(low + i*(high-low)/(n)) for i in range(n+1)]
    return t

#Given X, Y, k, lam, and the number of internal knots generates the 
#spline function approximating the data
#PLEASE sort X and Y so that X is ascending
def generate_spline(X, Y, k, knots, lam):
    #Generate knots
    t = make_t(X, knots)
    t = add_boundary_knots(t, k)
    #Generate B and omega
    #TODO pass in omega to avoid recomputing
    omega = make_omega(X, k, t)
    B_mat = make_B(X, k, t)
    #Does the ridge regression to find gamma
    gamma = compute_gamma(B_mat, omega, lam, Y)
    #Generates spline function from gamma
    func = bspline(t, k, gamma)

    return func


#main
X, Y = generate_sine_data(100)

order = 3
knots = 50
func = generate_spline(X, Y, order, knots, 0.1)

Y_hat = func(X)

plt.scatter(X,Y)
plt.plot(X,Y_hat, color = "green")
plt.show()



