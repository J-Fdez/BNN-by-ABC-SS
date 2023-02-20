# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:28:06 2023

@author: jferna22
"""

import random # This one we donÂ´t really need yet but we include it in case we want to use it when defining X(inputs) and Y(expected outputs, or real outputs for the given X values)
import numpy as np
import math # This one we donÂ´t really need yet but we include it in case we want to use it when defining X(inputs) and Y(expected outputs, or real outputs for the given X values)




# sigmoid = (
#      lambda x: 1/(1+np.exp(-x)), # Funcion de activacion
#      lambda x: x*(1-x)           # Derivada de la funcion de activacion
# )

# tanh = (
#          lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)),
#          lambda x: 1 - ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))**2
# )

# linear = (
#      lambda x: x, # Funcion de activacion
#      lambda x: 1.0           # Derivada de la funcion de activacion
# )

def derivada_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

# relu = (
#    lambda x: x * (x > 0),
#    lambda x:derivada_relu(x)
#    )

def relu(x):
    return x * (x > 0), derivada_relu(x)

def linear(x):
    return x, 1.0

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)), 1 - ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))**2

def sigmoid(x):
    return 1/(1+np.exp(-x)), x*(1-x)
    
        

def mse(Yp, Yr):
     error = (np.array(Yp) - np.array(Yr)) **2
     error = np.mean(error)
     error_d = np.array(Yp) - np.array(Yr)
     return (error, error_d)

def rmse(Yp, Yr):
    error = (np.array(Yp) - np.array(Yr)) **2
    error = np.mean(error)
    return math.sqrt(error), 1.0

def max_e(Yp, Yr):
    return np.max(np.absolute(np.array(Yp) - np.array(Yr))), 1.0

actf_dic={'relu':relu, 'sigmoid':sigmoid, 'tanh':tanh, 'linear':linear}
metric_dic={'mse':mse, 'rmse':rmse, 'max_e':max_e}