#Implentation of smoothing splines

#Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from General_methods import generate_sine_data


#main
X, Y = generate_sine_data()

#Computes the value of B_i_k(x) for a list of knots t
#x is the value the spline will be evaluated at
#k is the degree of the spline
#i is the current index
#t is are the knots (need to add the extra knots for the ends, I think)
#Code is adapted from the scipy bspline method
def B(x, k, i, t):
    #Degree 1 spline
    if k == 0:
        return 1 if t[i] <= x <= t[i+1] else 0
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
#This won't be used, the later function is more important
def bspline(t, k, c):
    n = len(t) - k - 1
    return lambda x: sum(c[i]*B(x, k, i, t) for i in range(n))

#Applies the B function to a whole vector X
def B_vec(X, k, i, t):
    output = torch.zeros(len(X))
    for j in range(len(X)):
        output[j] = B(X[j], k, i, t)
    return output

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
        for j in range(len(X)):
            D[j][l] = B_2deriv(X[j], k, l, t)
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

#Makes X |-> B by applying B_i to X for the i B-splines
def make_B(X, k, t):
    n = len(t) - k - 1
    B = torch.zeros([len(X), n])
    for l in range(n):
        B[:, l] = B_vec(X, k, l, t)
    return B


        
        
t = [0,0.5,1,1.5,2,2.5,3]
t = add_boundary_knots(t, 3)


X = [3*i/50 for i in range(51)]

omega = make_omega(X, 3, t)
B_mat = make_B(X, 3, t)




