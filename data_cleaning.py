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
df_train['GenderNumber'] = df_train.Sex.map({'female': 1, 'male': 0}).astype(float)
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



df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_train['Age*Class'] = df_train.AgeFill * df_train.Pclass / 10

# 0.188 - 0.184,
"""
'Pclass', 'AgeFill', 'FamilySize', 'Fare', 'GenderNumber'   0.188 - 0.184
added 'Age*Class'                                           0.206 - 0.168
removed 'Pclass', 'AgeFill' leads to worsening
female 1, male 0 leads to less error than the other way arround,
further changing the values in disastrous
'EmbarkedNumber' worsens the prediction
fideling with Pclass doesn't change anything
same for Fare and AgeFill
"""
input_data = df_train[['Pclass', 'AgeFill', 'FamilySize', 'Fare', 'GenderNumber', 'Age*Class']]
output_data = df_train.Survived



print input_data.head(10)
print input_data.dtypes

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

