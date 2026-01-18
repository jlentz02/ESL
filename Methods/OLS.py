#OLS
#This file contains the implementation and execution of OLS.
#It will probably also include some feature engineering stuff at some point

#Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from general_methods import load_data, add_ones, tt_split, test, MSE


#Basic multilinear regression using linear algebra
def OLS(x,y):
    xtx = torch.matmul(x.T, x)
    xtx_inv = torch.linalg.inv(xtx)
    xty = torch.matmul(x.T, y)
    beta = torch.matmul(xtx_inv, xty)

    #mse = round(MSE(x,y,beta),4)
    #print(f"MSE of standard linear regression is: {mse}")
    return beta

#Gradient descent method
#y intercept must be 0
#I wrote this on the plane deriving the equations from scratch. 
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

#Raw data
data_X, data_Y, columns = load_data("UCI_Credit_Card.csv")
data_X = add_ones(data_X)
train_X, train_Y, test_X, test_Y = tt_split(data_X, data_Y)

beta = OLS(train_X, train_Y)

train_error = test(train_Y, train_X@beta)
print(f"Train error: {train_error}")
test_error = test(test_Y, test_X@beta)
print(f"Test error: {test_error}")

#Feature engineering / basis expansions
#This stuff is hard
monthly_repayment_ratio = torch.log(torch.abs(data_X[:, 12:18] + 1e-8) / torch.abs(data_X[:, 18:24] + 1e-8))
X_aug = torch.column_stack([data_X, monthly_repayment_ratio])

train_X_aug, train_Y, test_X_aug, test_Y = tt_split(X_aug, data_Y)

beta_aug = OLS(train_X_aug, train_Y)

train_error_aug = test(train_Y, train_X_aug@beta_aug)
print(f"Augmented train error: {train_error_aug}")
test_error_aug = test(test_Y, test_X_aug@beta_aug)
print(f"Augemented test error: {test_error_aug}")








