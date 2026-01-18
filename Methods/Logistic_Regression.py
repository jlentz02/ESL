#Logistic Regression

import numpy as np
from general_methods import *
import torch

def sigmoid(X, beta):
    return 1/(1 + torch.exp(-X@beta))

#Logistic regression (two classes) using linear algebra for Newton update
def logistic_regression(X, Y, threshold = 0.001):
    n = len(X[0])
    beta_old = torch.ones(n)
    beta = torch.zeros(n)

    while torch.sum(torch.abs(beta_old - beta)) > threshold:
        p = sigmoid(X, beta)
        W = torch.diag(p*(1-p))
        xtwx = torch.transpose(X, 0, 1)@W@X
        xtwx_inv = torch.linalg.inv(xtwx)
        beta_old = beta
        beta = beta + (xtwx_inv@torch.transpose(X, 0, 1))@(Y - p)

    return beta

#Logistic regression using pytorch autograd
#TODO
def logistic_regression_grad(X, Y, lr = 0.001):
    beta = torch.zeros((len(X[0])), requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr = lr)
    for step in range(1000):
        #loss isn't correct
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(X@beta, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return beta



data_X, data_Y, columns = load_data("UCI_Credit_Card.csv", ysplit = False)
#data_Y = np.array([0 if data_Y[i] == 1 else 1 for i in range(len(data_Y))])
data_X = add_ones(data_X)
train_X, train_Y, test_X, test_Y = tt_split(data_X, data_Y)

beta = logistic_regression_grad(train_X, train_Y)


train_pred = sigmoid(train_X, beta)
train_error = test(train_Y, train_pred)
print(f"Train error: {train_error}")
test_pred = sigmoid(test_X, beta)
test_error = test(test_Y, test_pred)
print(f"Test error: {test_error}")

