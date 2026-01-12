#OLS
#This file contains the implementation and execution of OLS.
#It will probably also include some feature engineering stuff at some point

#Imports
import numpy as np
import matplotlib.pyplot as plt
from general_methods import load_data, add_ones, tt_split, test, MSE


#Basic multilinear regression using linear algebra
def OLS(x,y):
    xtx = np.matmul(x.T, x)
    xtx_inv = np.linalg.inv(xtx)
    xty = np.matmul(x.T, y)
    beta = np.matmul(xtx_inv, xty)

    #mse = round(MSE(x,y,beta),4)
    #print(f"MSE of standard linear regression is: {mse}")
    return beta

#Gradient descent method
#y intercept must be 0
def oneDGradDescent(x,y):
    beta_0 = 0
    beta = 1
    while abs(beta_0 - beta) > 0.001:
        #0.01 is learning rate
        grad = -2*np.sum((y - beta*x))*0.01
        beta_0 = beta
        beta = beta - grad
    mse = round(MSE(x,y, beta),4)
    print(f"MSE of 1D gradient descent is: {mse}")
    return beta


#Main basically
data_X, data_Y, columns = load_data("UCI_Credit_Card.csv")
data_X = add_ones(data_X)
train_X, train_Y, test_X, test_Y = tt_split(data_X, data_Y)


beta = OLS(train_X, train_Y)

train_error = test(train_Y, train_X@beta)
train_error = round(train_error, 4)
print(f"Train error: {train_error}")
test_error = test(test_Y, test_X@beta)
test_error = round(test_error, 4)
print(f"Test error: {test_error}")









