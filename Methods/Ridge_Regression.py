#Ridge Regression

import numpy as np
from general_methods import *

#TODO Ridge regression
#Using k as tuning parameter
#Need to center data before doing this
def ridge_regression(x, y, k):
    #Centering data
    x = x - np.mean(x, axis = 0)


    xtx_kI = np.matmul(x.T, x) - k*np.identity(len(x))
    xtx_kI_inv = np.linalg.inv(xtx_kI)
    xty = np.matmul(x.T, y)
    beta = np.matmul(xtx_kI_inv, xty)

    return beta