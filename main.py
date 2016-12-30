# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import data_cleaning as data
import functions as f
from sklearn.cross_validation import train_test_split

# Converts the input and output data from pandas dataframe to nparray
input_data = np.array(data.input_data, float)
input_data = np.insert(input_data, [0], 1, axis = 1) # add bias
# classes
output_data = np.array(data.output_data, float)
output_data = output_data.reshape(output_data.shape[0], ) # make it (n,) array

# Splits the training data further into train and test
input_train, input_test, output_train, output_test = train_test_split(input_data,
                                                                      output_data,
                                                                      test_size=0.2,
                                                                      random_state=42)

# init parameters TODO Experiment with parameters
weights = np.ones(input_train.shape[1], dtype=float) / 5.0  # weights
learning_rates = [0.005]    # learning rate
regularization_terms = [0]   # parameter for regularization
iterations = 100000
# save all errors to output the best result
errors = np.zeros(iterations,dtype=float)
cost = np.zeros(iterations,dtype=float) #TODO calculate cost and plot it
prediction = np.zeros(input_train.shape[0], dtype=float)


"""
With learning rate 0.0000005 it demonstrates proper learning. However it still doesn't achieve less than 0.19 error
With learning rate 0.05 it goes up and down like crazy, but it stumbles upon the similar error rate as the above
"""

for learning_rate in learning_rates:
    for regularization_term in regularization_terms:
        # Learning
        for iteration in range(iterations):
            chance_survival = f.sigm(np.dot(input_train, weights))
            prediction[chance_survival >= 0.5] = 1
            prediction[chance_survival < 0.5] = 0

            # cost[i] = f.calc_cost(h_theta,y_train)  #doesn't work, returns nan

            errors[iteration] = f.calcError(output_train, prediction)
            if (iteration % 10000) == 0:
                print ("Iteration {0:3d}\tError: {1:.3f}".format(iteration, errors[iteration]))

            gradient = 1 / input_train.shape[0] * (np.dot((output_train - chance_survival), input_train))
            weights += learning_rate * gradient

        print ("Alpha = {2:.3f}, reg_term = {3:3d} Best result with {0:.3f} error rate on training data for {1:3d} iterations".format(min(errors), iterations, learning_rate, regularization_term))

# print cost
# plotting the errors agains the iterations
# plt.plot(range(len(errors)),errors)


plt.plot(range(len(errors)), errors)
plt.ylabel('Error Rates')
plt.show()
#TODO figure out why does it flactuate so much
#TODO plot cost function


#TODO write a submission to test with test.csv