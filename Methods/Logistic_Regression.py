#Logistic Regression

import numpy as np
from general_methods import *
#Shouldn't need pytorch since it is binary


#TODO Logistic regresion using newton updates (see page 119)

data_X, data_Y, columns = load_data("UCI_Credit_Card.csv")
trainval_X, trainval_Y, test_X, test_Y = tt_split(data_X, data_Y)

