#Logistic Regression

import numpy as np
from general_methods import *
#Shouldn't need pytorch since it is binary

def sigmoid(X, beta):
    return 1/(1 + np.exp(-X@beta))

#Logistic regression (two classes) using linear algebra for Newton update
def logistic_regression(X, Y, threshold = 0.001):
    n = len(X[0])
    beta_old = np.ones(n)
    beta = np.zeros(n)

    while np.sum(np.abs(beta_old - beta)) > threshold:
        p = sigmoid(X, beta)
        W = np.diag(p*(1-p))
        xtwx = np.transpose(X)@W@X
        xtwx_inv = np.linalg.inv(xtwx)
        beta_old = beta
        beta = beta + (xtwx_inv@np.transpose(X))@(Y - p)

    return beta

#Logistic regression using pytorch autograd
def log_regression_grad(X, Y, threshold):
    print("hi")


data_X, data_Y, columns = load_data("UCI_Credit_Card.csv", ysplit = False)
#data_Y = np.array([0 if data_Y[i] == 1 else 1 for i in range(len(data_Y))])
data_X = add_ones(data_X)
train_X, train_Y, test_X, test_Y = tt_split(data_X, data_Y)

beta = logistic_regression(train_X, train_Y)

train_pred = sigmoid(train_X, beta)
train_error = test(train_Y, train_pred)
print(f"Train error: {train_error}")
test_pred = sigmoid(test_X, beta)
test_error = test(test_Y, test_pred)
print(f"Test error: {test_error}")

