"""
Custom functions for the logistic regression
"""
import numpy as np
import math


def sigm(z):
    return 1.0 / (1.0 + math.e**(-z))

def calcError(y, y_h):
    return (np.sum(np.abs(y-y_h)))/y.shape[0]

def calc_cost(h_t, y): #TODO FIX THIS SHIT ALWAYS NAN OR -INF, WHY!??! WHY !?!??
    term = np.log10(np.ones(y.shape) - h_t)
    term = np.nan_to_num(term)
    cost = -np.dot(y,np.log10(h_t)) - np.dot((np.ones(y.shape) - y),term)
    return np.nan_to_num(cost)