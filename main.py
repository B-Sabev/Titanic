# IMPORT LIBRARIES
import numpy as np
import pandas as P
import matplotlib.pyplot as plt
import math
import data_cleaning as data
import functions as f
from sklearn.model_selection import train_test_split

# Converts the input and output data from pandas dataframe to nparray
x = np.array(data.input_data, float)
x = np.insert(x, [0], 1, axis = 1) # insert x_0 = 1
# classes
y = np.array(data.output_data, float)
y = y.reshape(y.shape[0],) # make it (n,) array

# Splits the training data further into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# init parameters TODO Experiment with parameters
theta = np.ones(X_train.shape[1], dtype=float) / 5.0  # weights
alphas = [0.005]    # learning rate
reg_terms = [0]   # parameter for regularization
iterations = 10000
# save all errors to output the best result
errors = np.zeros(iterations,dtype=float)
cost = np.zeros(iterations,dtype=float) #TODO calculate cost and plot it
y_h = np.zeros(X_train.shape[0],dtype=float)
thetas = np.zeros(iterations * X_train.shape[1])
thetas = thetas.reshape(iterations, X_train.shape[1])


for alpha in alphas:
    for reg_term in reg_terms:
        #Learning
        for i in range(iterations):
            # generate prediction hypothesis
            h_theta = f.sigm(np.dot(X_train, theta)) # hypothesis - chance to be alive or dead
            # make a concrete prediction: alive if more than 50% chance, death otherwise
            y_h[h_theta >= 0.5] = 1
            y_h[h_theta < 0.5] = 0

            # cost[i] = f.calc_cost(h_theta,y_train)  #doesn't work, returns nan

            # save the error rate
            errors[i] = f.calcError(y_train, y_h)
            if (i % 1000) == 0:
                print "Iteration {0:3d}\tError: {1:.3f}".format(i,errors[i])

            #save thetas for latter usage
            thetas[i,:] = theta
            #update parameters
            theta -= alpha * ((np.dot((h_theta - y_train), X_train) / X_train.shape[0]) + reg_term / X_train.shape[0] * theta)
        print "Alpha = {2:.3f}, reg_term = {3:3d} Best result with {0:.3f} error rate on training data for {1:3d} iterations".format(min(errors),iterations,alpha, reg_term)





# Cross-verification with test data
h_theta_test = f.sigm(np.dot(X_test, thetas[errors.argmin(), :]))
y_h_test = np.zeros(h_theta_test.shape)
y_h_test[h_theta_test >= 0.5] = 1
print "Cross-verify with test data {0:.3f} error rate".format(f.calcError(y_test, y_h_test))


#print cost
# plotting the errors agains the iterations
plt.plot(range(len(errors)),errors)
plt.ylabel('Error Rates')
plt.show()
#TODO plot cost function

#TODO write a submission to test with test.csv