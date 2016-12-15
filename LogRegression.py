# IMPORT LIBRARIES
import numpy as np
import pandas as P
import matplotlib.pyplot as plt
import math

# GET DATA
# read file
df_train = P.read_csv("data/train.csv", header=0)
# make numbers from data
# Male and Female to 1 and 0
df_train['GenderNumber'] = 4
df_train['GenderNumber'] = df_train.Sex.map({'female': 0, 'male': 1}).astype(int)
# Location of embark to number
df_train['EmbarkedNumber'] = df_train.Embarked.map({'S': 0, 'C': 1, 'Q': 2}).fillna(0.0).astype(int)
# Fill out missing Ages by median of Pclass
median_ages = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df_train[(df_train.GenderNumber == i) & (df_train.Pclass == j+1)]['Age'].dropna().median()
df_train['AgeFill'] = df_train.Age
for i in range(0, 2):
    for j in range(0, 3):
        df_train.loc[(df_train.Age.isnull()) & (df_train.GenderNumber == i) & (df_train.Pclass == j+1), 'AgeFill'] = median_ages[i, j]
# drop values that are not relevant
input_data = df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Parch', 'Age', 'PassengerId', 'Survived'], axis=1)
output_data = df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Parch', 'Age', 'Pclass', 'SibSp', 'Fare',
                             'GenderNumber', 'EmbarkedNumber', 'AgeFill', 'PassengerId'], axis=1)

"""
so far the same as Johan neural network code,
so we can compare only the machine learning algorithm
"""

"""
Now we have the features:
  Pclass  SibSp   Fare  GenderNumber  EmbarkedNumber  AgeFill
For Pclass  SibSp   Fare AgeFill - multiply them with eachother to create non-linear features
"""

for column1 in ["Pclass", "SibSp", "Fare", "AgeFill"]:
    for column2 in ["Pclass", "SibSp", "Fare", "AgeFill"]:
        input_data[column1 + " * " + column2] = input_data[column1] * input_data[column2]




# TODO try cross-validation
x = np.array(input_data, float)
x = np.insert(x, [0], 1, axis = 1) # insert x_0 = 1
# classes
y = np.array(output_data, float)
y = y.reshape(y.shape[0],) # make it (n,) array
# init random parameters - theta
np.random.seed(1)  # seed random numbers 0 or 1
#theta = np.random.random(x.shape[1])
theta = np.ones(x.shape[1], dtype=float) / 2.0
#TODO Hard to generate good initial values - check why!!!




# some functions to log regression TODO put in seperate file, check if something else will be nicer abstracted

def sigm(z):
    return 1.0 / (1.0 + math.e**(-z))

def calcError(y, y_h):
    return (np.sum(np.abs(y-y_h)))/y.shape[0]

def calc_cost(h_t, y): #TODO FIX THIS SHIT ALWAYS NAN OR -INF, WHY!??! WHY !?!??
    term = np.log10(np.ones(y.shape) - h_t)
    term = np.nan_to_num(term)
    cost = -np.dot(y,np.log10(h_t)) - np.dot((np.ones(y.shape) - y),term)
    return np.nan_to_num(cost)



# learning rate - TODO TEST WITH MORE LEARNING RATES
alpha = 0.005
# parameter for regularization TODO experiment after non-linearity is added
reg_term = 2000

iterations = 10000
# save all errors to output the best result
errors = np.zeros(iterations,dtype=float)
cost = np.zeros(iterations,dtype=float) #TODO calculate cost and plot it

for i in range(iterations):
    # generate predictions
    h_theta = sigm(np.dot(x, theta)) # hypothesis - chance to be alive or dead
    # make a concrete prediction: alive if more than 50% chance.
    y_h = np.zeros(h_theta.shape,dtype=float)
    y_h[h_theta >= 0.5] = 1
    cost[i] = calc_cost(h_theta,y)  #doesn't work, returns nan
    # save the error rate
    errors[i] = calcError(y, y_h)
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


