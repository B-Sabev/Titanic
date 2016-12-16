# IMPORT LIBRARIES
import numpy as np
import pandas as P
import matplotlib.pyplot as plt
import math
import data_cleaning as data
import functions as f


#TODO try cross-validation
x = np.array(data.input_data, float)
x = np.insert(x, [0], 1, axis = 1) # insert x_0 = 1
# classes
y = np.array(data.output_data, float)
y = y.reshape(y.shape[0],) # make it (n,) array
# init random parameters - theta
np.random.seed(1)  # seed random numbers 0 or 1
#theta = np.random.random(x.shape[1])
theta = np.ones(x.shape[1], dtype=float) / 2.0
#TODO Hard to generate good initial values - check why!!!




# learning rate - TODO TEST WITH MORE LEARNING RATES
alpha = 0.05
# parameter for regularization TODO experiment after non-linearity is added
reg_term = 50

iterations = 10000
# save all errors to output the best result
errors = np.zeros(iterations,dtype=float)
cost = np.zeros(iterations,dtype=float) #TODO calculate cost and plot it

for i in range(iterations):
    # generate predictions
    h_theta = f.sigm(np.dot(x, theta)) # hypothesis - chance to be alive or dead
    # make a concrete prediction: alive if more than 50% chance.
    y_h = np.zeros(h_theta.shape,dtype=float)
    y_h[h_theta >= 0.5] = 1
    cost[i] = f.calc_cost(h_theta,y)  #doesn't work, returns nan
    # save the error rate
    errors[i] = f.calcError(y, y_h)
    if (i % 500) == 0:
        print "Iteration {0:3d}\tError: {1:.3f}".format(i,errors[i])
    #update parameters
    theta -= alpha * (np.dot((h_theta - y), x) + reg_term / x.shape[0] * theta)

print "Best result with {0:.3f} error rate on training data for {1:3d} iterations".format(min(errors),iterations)

#print cost
# plotting the errors agains the iterations
plt.plot(range(len(errors)),errors)
plt.ylabel('Error Rates')
plt.show()
#TODO figure out why does it flactuate so much
#TODO plot cost function


#TODO write a submission to test with test.csv