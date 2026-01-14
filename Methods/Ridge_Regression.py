#Ridge Regression

import numpy as np
from general_methods import *

#TODO Ridge regression
#Using k as tuning parameter
#Need to center data before doing this
def ridge_regression(x, y, k):
    #making data mean 0, unit variance
    x_mean = x.mean(axis = 0)
    x_std = x.std(axis = 0, correction = 1)
    x = (x - x_mean)/x_std
    y_mean = y.mean(axis = 0)
    y = y - y_mean
    

    xtx_kI = torch.matmul(x.T, x) + k*torch.eye(len(x[0]))
    xtx_kI_inv = torch.linalg.inv(xtx_kI)
    xty = torch.matmul(x.T, y)
    beta = torch.matmul(xtx_kI_inv, xty)

    return beta, x_mean, x_std, y_mean

data_X, data_Y, columns = load_data("UCI_Credit_Card.csv")
trainval_X, trainval_Y, test_X, test_Y = tt_split(data_X, data_Y)
train_X, train_Y, val_X, val_Y = tt_split(trainval_X, trainval_Y, .75)

beta, x_mean, x_std, y_mean = ridge_regression(train_X, train_Y, 8)

train_pred = ((train_X - x_mean)/x_std)@beta + y_mean
train_error = test(train_Y, train_pred)
print(f"Train error: {train_error}")
val_pred = ((val_X - x_mean)/x_std)@beta + y_mean
val_error = test(val_Y, val_pred)
print(f"Validation error: {val_error}")
test_pred = ((test_X - x_mean)/x_std)@beta + y_mean
test_error = test(test_Y, test_pred)
print(f"Validation error: {test_error}")


