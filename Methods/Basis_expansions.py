#Basis expansions

#Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from General_methods import load_data, add_ones, tt_split, test, expand_basis
from OLS import OLS

#Main

#Raw data
data_X, data_Y, columns = load_data("UCI_Credit_Card.csv")
data_X = add_ones(data_X)
data_X_2nd = expand_basis(data_X)

train_X, train_Y, test_X, test_Y = tt_split(data_X_2nd, data_Y)

beta = OLS(train_X, train_Y)

train_error = test(train_Y, train_X@beta)
print(f"Train error: {train_error}")
test_error = test(test_Y, test_X@beta)
print(f"Test error: {test_error}")

