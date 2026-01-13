#General
#This file will contain general functions that are used across multiple different
#statistical methods. Individual methods will get their own file in the "Methods" folder

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


#Plots 1D regression for simple examples
def plotbeta(x ,y , beta):
    plt.scatter(x,y)
    plt.plot(x , beta[0] + beta[1]*x, color = "green")
    plt.show()

#Loads real data from a csv file given its name
    #e.g. data.csv call load_data("data.csv")
#Returns data_X -> n x p matrix of predictors
#Returns data_y -> n x 1 matrix of targets
#Returns columns -> 1 x p list of column names for the predictors
def load_data(filename):
    data = np.loadtxt(filename, delimiter = ",", dtype = str)
    columns = data[0]
    n = len(data)
    data_X = np.array([data[i][1:-1].astype(float).astype(int) for i in range(1, n)])
    data_y = np.array([data[i][-1].astype(int) for i in range(1, n)])
    data_Y = convert_class(data_y)
    return data_X, data_Y, columns

#Converts binary data of 0 and 1's in a single column into two columns, one for yes
#and one for no. For example:
#[0 1 0] -> [[0, 1], [1, 0], [0, 1]]
def convert_class(data_y):
    data_Y = np.array([[0,1] if x == 0 else [1,0] for x in data_y])
    return data_Y

#Takes in data_X and prediction, computes the argmax of prediction,
#and computes the error of the prediction: 1 - (correct labels / total)
#TODO Add confusion matrix?
def test(data_Y, prediction):
    n = len(data_Y)
    acc = 0
    for i in range(n):
        index = np.argmax(prediction[i])
        if data_Y[i][index] == 1:
            acc +=1
    return round(1 - acc/n,4)

#splits X and Y into a training (80%) and test (20%) set
def tt_split(data_X, data_Y, split = 0.8):
    n = len(data_X)
    rng = np.random.default_rng(seed = 42)
    indices = rng.permutation(n)

    train_size = int(n*split)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    train_X, test_X = data_X[train_idx], data_X[test_idx]
    train_Y, test_Y = data_Y[train_idx], data_Y[test_idx]
    return train_X, train_Y, test_X, test_Y
    












