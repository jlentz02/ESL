#RegressionTesting
#The purpose of this file is to implement the regression models presented in ESL, 
#in multiple ways if possible, for the purpose of advancing my own knowledge 
#of statistical models. If possible, I will try to implement linear algebra methods 
#and gradient descent approaches (and others if I find them). 

#Imports
import numpy as np
import matplotlib.pyplot as plt

#Functions
#Add ones: Adds a row of ones to the data set X -> n x p makes X -> n x (p+1)
def add_ones(x):
    if x.ndim == 1:
        x = x.reshape(-1 ,1)
    ones = np.ones((x.shape[0], 1))
    x_new = np.hstack((ones, x))
    return x_new

#Computes MSE
def MSE(x ,y, beta):
    try:
        len(beta)
        mse = np.sum((y - x@beta)**2)
    except:
        mse = np.sum((y - beta*x)**2)
    return mse

#Basic multilinear regression using linear algebra
def multRegLinAlg(x,y):
    x = add_ones(x)
    xtx = np.matmul(x.T, x)
    xtx_inv = np.linalg.inv(xtx)
    xty = np.matmul(x.T, y)
    beta = np.matmul(xtx_inv, xty)

    mse = round(MSE(x,y,beta),4)
    print(f"MSE of standard linear regression is: {mse}")
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
    
#Ridge regression?

#Plots 1D regression so I can see examples
def plotbeta(x ,y , beta):
    plt.scatter(x,y)
    plt.plot(x , beta[0] + beta[1]*x, color = "green")
    plt.show()

x = np.array([0,1,2,3,4,5,6,7])
y = np.array([0,2,5,6.6,8,10,11,14])
#x = np.array([[1,1], [2,4], [3,9]])
#y = np.array([2,6,12])

beta_grad = oneDGradDescent(x,y)
beta_reg = multRegLinAlg(x,y)





