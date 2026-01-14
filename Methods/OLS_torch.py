#OLS_torch
#This file contains the implementation and execution of OLS using pytorch autograd.
#It will probably also include some feature engineering stuff at some point

#Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from general_methods import load_data, add_ones, tt_split, test, MSE


def OLS_grad(x,y, lr = 0.01):
    beta = torch.zeros((len(x[0]), len(y[0])), requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr = lr)
    for step in range(50):
        loss = MSE(x, y, beta)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return beta
    
#Main basically

#Raw data
data_X, data_Y, columns = load_data("UCI_Credit_Card.csv", pytorch = True)
data_X = add_ones(data_X)
train_X, train_Y, test_X, test_Y = tt_split(data_X, data_Y)

beta = OLS_grad(train_X, train_Y)
print(beta)

train_error = test(train_Y, train_X@beta)
print(f"Train error: {train_error}")
test_error = test(test_Y, test_X@beta)
print(f"Test error: {test_error}")










