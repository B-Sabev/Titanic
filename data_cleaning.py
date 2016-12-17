"""
The data is converted from csv file to pandas dataframe,
cleaned with only relevant for the analysis features.

more feature engineering in progress.....
"""

# IMPORT LIBRARIES
import pandas as P
import numpy as np

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
"""

for column1 in ["Pclass", "SibSp", "Fare", "AgeFill"]:
    for column2 in ["Pclass", "SibSp", "Fare", "AgeFill"]:
        input_data[column1 + " * " + column2] = input_data[column1] * input_data[column2]

"""

