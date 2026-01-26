#Implentation of algorithm 9.1 from ESL - "The Backfitting Algorithm for Additive Models"


#Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from General_methods import load_data, add_ones, tt_split, test, MSE, normalize_data
from Splines import generate_spline_t_omega, make_t, add_boundary_knots, make_omega, generate_spline

#Used in update loop to avoid rounding drift
def zero_function(func, X_j):
    n = len(X_j)
    func_zeroed = lambda x: func(x) - (1/n)*torch.sum(func(X_j))
    return func_zeroed

#Computes list of knots, t[j], given data X = {X_j}
def generate_knot_array(X, num_knots, k):
    n = len(X[0])
    t = [0 for i in range(n)]
    for i in range(n):
        t[i] = make_t(X[:,i], num_knots)
        t[i] = add_boundary_knots(t[i], k)
    return t

#Computes list of omegas, omega[j], given data X = {X_j} and knots t = {t_j}
def generate_omega_array(X, order, t):
    n = len(X[0])
    omega = [0 for i in range(n)]
    for i in range(n):
        omega[i] = make_omega(X[:,i], order, t[i])
    return omega

#backfitting algo
#For now, this is just for cubic splines
def backfit_cubic_spline(X, Y, order, num_knots, Lambda):
    #Fix alpha = avgY
    alpha = torch.mean(Y)
    n = len(X[0])
    #Fix list of functions f[j] = 0 for j in [0,...,len(X[0])]
    functions = [lambda X: 0 for i in range(n)]
    #precompute list of knots t[j] for j in [0,...,len(X[0])]
    t = generate_knot_array(X, num_knots, order)
    #precompute list of omega omega[j] for j in [0,...,len(X[0])]
    omega = generate_omega_array(X, order, t)

    #loop j = 1,..,p,1,...,p until converges (or probably just a fixed number of iterates)
        # f[j] = spline((y - alpha - sum_k \ne j f[k](X[:, k])), k, t, omega)
        # f[j] = zero_function(f[j], X[:, j])
    for i in range(n):
        j = i%n
    
        loop_input = Y - alpha - sum(functions[k](X[:, k]) for k in range(n) if k != j)
        functions[j] = generate_spline_t_omega(loop_input, Y, order, Lambda[j], t[j], omega[j])
        functions[j] = zero_function(functions[j], X[:,j])
        print(j)

    model = lambda X: alpha + sum(functions[j](X[:, j]) for j in range(n))
    return model



#main
data_X, data_Y, columns = load_data("UCI_Credit_Card.csv", ysplit = False)
trainval_X, trainval_Y, test_X, test_Y = tt_split(data_X, data_Y)
train_X, train_Y, val_X, val_Y = tt_split(trainval_X, trainval_Y, .75)

train_X, means, stds = normalize_data(train_X)


order = 3
num_knots = 5
#Need to figure out how to set Lambda
#See page 299 of ESL
Lambda = [0.1 for i in range(len(data_X[0]))]

model = backfit_cubic_spline(train_X, train_Y, order, num_knots, Lambda)

train_pred = model(train_X)
train_error = test(train_Y, train_pred)
print(f"Train error: {train_error}")