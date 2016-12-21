# IMPORT LIBRARIES
import numpy as np
import pandas as P
import matplotlib.pyplot as plt
import math
import data_cleaning as data
import functions as f
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


import new_main

features = new_main.train_data.drop('Survived', axis=1)
labels = new_main.train_data['Survived']


# Converts the input and output data from pandas dataframe to nparray
x = np.array(features, float)
x = np.insert(x, [0], 1, axis = 1) # insert x_0 = 1
# classes
y = np.array(labels, float)
y = y.reshape(y.shape[0],) # make it (n,) array

# Splits the training data further into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

print random_forest.score(X_train, y_train)



"""
# init parameters TODO Experiment with parameters
theta = np.ones(X_train.shape[1], dtype=float) / 5.0  # weights
alphas = [0.005]    # learning rate
reg_terms = [0]   # parameter for regularization
iterations = 8000
# save all errors to output the best result
errors = np.zeros(iterations,dtype=float)
cost = np.zeros(iterations,dtype=float) #TODO calculate cost and plot it
y_h = np.zeros(X_train.shape[0],dtype=float)
thetas = np.zeros([iterations,X_train.shape[1]], dtype=float)


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
best_theta = thetas[errors.argmin(), :]
h_theta_test = f.sigm(np.dot(X_test, best_theta))
y_h_test = np.zeros(h_theta_test.shape)
y_h_test[h_theta_test >= 0.5] = 1
print "Cross-verify with test data {0:.3f} error rate".format(f.calcError(y_test, y_h_test))


#print cost
# plotting the errors agains the iterations
plt.plot(range(len(errors)),errors)
plt.ylabel('Error Rates')
plt.show()
#TODO plot cost function


"""

"""



#TODO write a submission to test with test.csv
import csv as csv
#load the test data
test_file = open('data/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

prediction_file = open("log_regression_model.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])

from data_cleaning_for_test import test_data

X_test_data = np.array(test_data.drop("PassengerId", axis=1))
X_test_data = np.insert(X_test_data, [0], 1, axis = 1)
h_theta_test = f.sigm(np.dot(X_test_data, best_theta))
y_h_test = np.zeros(h_theta_test.shape)
y_h_test[h_theta_test >= 0.5] = 1


print y_h_test.shape
for i in range(y_h_test.shape[0]):
    prediction_file_object.writerow([test_data.at[i,'PassengerId'], y_h_test[i]])

test_file.close()
prediction_file.close()

"""
