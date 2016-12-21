#imports
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

#get data
train_data = pd.read_csv("data/train.csv", header=0)
test_data = pd.read_csv("data/test.csv", header=0)

"""
print train_data.info()
print test_data.info()

Cabin - a lot of NaN values, so probably it is useless
Embarked - 2 missing values, fill with the most common
Age - need to fill the missing values
Ticket - useless, doesn't tell anything about the survival
PassengerID is not needed in train
"""
train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].fillna('S')

# drop unnessesary columns
train_data = train_data.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1)
test_data = test_data.drop(['Cabin', 'Ticket'], axis=1)

"""
# Look at the relative importance of features
for column in ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']:
    print (train_data[[column, 'Survived']].groupby([column], as_index=False).mean())

Fare and Age need to be binned, SibSp and Parch show unclear results so should be engineered, Age still needs to be filled
Name is currently unusable, but information about the title can be extracted
Since the title is always followed by a dot, it can be easy taken out
"""

def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

train_data['Title'] = train_data['Name'].apply(get_title)
test_data['Title'] = test_data['Name'].apply(get_title)

# Put the rare categories into a common group
full_data = [train_data, test_data]
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


"""
print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

Now we have extracted the important feature of name and it can be dropped
"""
train_data = train_data.drop('Name', axis = 1)
test_data = test_data.drop('Name', axis = 1)

# Create a new feature - Family Size
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

"""
print (train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

So people with family size between 2-4 have better changes of survival. We can extract the other groups, too
"""
# Extracting new feature for people with family size between 2-4
train_data['MidSizedFamily'] = 1
train_data.loc[train_data['FamilySize'] < 2,'MidSizedFamily'] = 0
train_data.loc[train_data['FamilySize'] > 4,'MidSizedFamily'] = 0
test_data['MidSizedFamily'] = 1
test_data.loc[test_data['FamilySize'] < 2,'MidSizedFamily'] = 0
test_data.loc[test_data['FamilySize'] > 4,'MidSizedFamily'] = 0
"""
print (train_data[['MidSizedFamily', 'Survived']].groupby(['MidSizedFamily'], as_index=False).mean())

that one looks also nicely significant
"""

train_data['LargeFamily'] = 0
train_data.loc[train_data['FamilySize'] > 4, 'LargeFamily'] = 1
test_data['LargeFamily'] = 0
test_data.loc[test_data['FamilySize'] > 4, 'LargeFamily'] = 1
"""
print (train_data[['LargeFamily', 'Survived']].groupby(['LargeFamily'], as_index=False).mean())

Being in a large family also hurts your chances
"""


"""
Now that we have the MidSizeFamily and LargeFamily we can drop 'SibSp', 'Parch' and 'Family size'.
!!!!! OR TRY BINNING THE FAMILY SIZE
"""
train_data = train_data.drop(['SibSp', 'Parch','FamilySize'], axis = 1)
test_data = test_data.drop(['SibSp', 'Parch','FamilySize'], axis = 1)

"""
We have 3 objects - Sex, Embarked, Title, which need to be converted to numerical values
!!!!TRY DIFFERENT MAPPINGS
"""
train_data['Gender'] = train_data.Sex.map({'female': 1, 'male': 0}).astype(float)
train_data['EmbarkedNumber'] = train_data.Embarked.map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
train_data['TitleNumber'] = train_data.Title.map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master':3, 'Other':4 }).astype(int)
test_data['Gender'] = test_data.Sex.map({'female': 1, 'male': 0}).astype(float)
test_data['EmbarkedNumber'] = test_data.Embarked.map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
test_data['TitleNumber'] = test_data.Title.map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master':3, 'Other':4 }).astype(int)

train_data = train_data.drop(['Sex', 'Embarked', 'Title'], axis=1)
test_data = test_data.drop(['Sex', 'Embarked', 'Title'], axis=1)

"""
Now lets fill in the Age, by using the median age for Gender and Pclass and drop the old Age
!!!! TRY SMARTER WAY TO FILL AGES

"""
median_ages_test = np.zeros((2, 3))
median_ages_train = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages_train[i, j] = train_data[(train_data.Gender == i) & (train_data.Pclass == j+1)]['Age'].dropna().median()
        median_ages_test[i, j] = test_data[(test_data.Gender == i) & (test_data.Pclass == j + 1)]['Age'].dropna().median()
train_data['AgeFill'] = train_data.Age
test_data['AgeFill'] = test_data.Age
for i in range(0, 2):
    for j in range(0, 3):
        train_data.loc[(train_data.Age.isnull()) & (train_data.Gender == i) & (train_data.Pclass == j+1), 'AgeFill'] = median_ages_train[i, j]
        test_data.loc[(test_data.Age.isnull()) & (test_data.Gender == i) & (test_data.Pclass == j + 1), 'AgeFill'] = \
        median_ages_test[i, j]


train_data = train_data.drop('Age', axis=1)
test_data = test_data.drop('Age', axis=1)

"""
Now the only thing left is to bin the Age and the Fare

# Plotting the Ages
num_survivors = train_data[train_data.Survived==1].shape[0]
num_died = train_data[train_data.Survived==0].shape[0]
print num_survivors

data = [train_data.AgeFill[train_data.Survived==1] , train_data.AgeFill[train_data.Survived==0]]
n, bins, patches = plt.hist(data, 20, normed=1, alpha=0.75)
plt.show()

bins = 20
plt.hist(data[0], bins, alpha=0.5)
plt.hist(data[1], bins, alpha=0.5)
plt.show()
"""

# Mapping Fare
dataset = [train_data, test_data]

for dataset in dataset:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset.loc[ dataset['AgeFill'] <= 16, 'AgeFill'] 					       = 0
    dataset.loc[(dataset['AgeFill'] > 16) & (dataset['AgeFill'] <= 32), 'AgeFill'] = 1
    dataset.loc[(dataset['AgeFill'] > 32) & (dataset['AgeFill'] <= 48), 'AgeFill'] = 2
    dataset.loc[(dataset['AgeFill'] > 48) & (dataset['AgeFill'] <= 64), 'AgeFill'] = 3
    dataset.loc[ dataset['AgeFill'] > 64, 'AgeFill']                           = 4


"""
Now that Age and Fare are binned, we are ready.
!!!!!! TRY DIFFERENT BINNING
"""













