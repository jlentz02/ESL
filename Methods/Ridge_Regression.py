#Ridge Regression

import numpy as np
from general_methods import *

#TODO Ridge regression
#Using k as tuning parameter
#Need to center data before doing this
def ridge_regression(x, y, k):
    #making data mean 0, unit variance
    x_mean = x.mean(axis = 0)
    x_std = x.std(axis = 0, ddof = 1)
    x = (x - x_mean)/x_std
    y_mean = y.mean(axis = 0)
    y = y - y_mean
    

    xtx_kI = np.matmul(x.T, x) + k*np.identity(len(x[0]))
    xtx_kI_inv = np.linalg.inv(xtx_kI)
    xty = np.matmul(x.T, y)
    beta = np.matmul(xtx_kI_inv, xty)

    return beta, x_mean, x_std, y_mean

data_X, data_Y, columns = load_data("UCI_Credit_Card.csv")
train_X, train_Y, test_X, test_Y = tt_split(data_X, data_Y)


beta, x_mean, x_std, y_mean = ridge_regression(train_X, train_Y, 1000)

prediction = ((train_X - x_mean)/x_std)@beta + y_mean
train_error = test(train_Y, prediction)
print(f"Train error: {train_error}")
