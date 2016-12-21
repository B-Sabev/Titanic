"""
Observations:
    'Pclass', 'AgeFill', 'FamilySize', 'Fare', 'GenderNumber'   0.188 - 0.184   0.2 test_data
    added 'Age*Class'                                           0.206 - 0.168   0.2 test_data
    removed 'Pclass', 'AgeFill' leads to worsening
    female 1, male 0 leads to less error than the other way arround,
    further changing the values in disastrous
    'EmbarkedNumber' worsens the prediction
    fideling with Pclass doesn't change anything
    same for Fare and AgeFill

    Added Title - Mr: 0, Mrs: 1, Master: 2, Miss: 3, other: 1.5
    worsenes the score
    with some changes to the values, it keeps the score the same
    Changed the values to
    Mr: 1, Mrs: 8, Master: 6, Miss: 7, other: 3 -  overfits     0.167  - 0.190   0.2 test_data
    However if you reduce the training data it gets better      0.173  - 0.159   0.33 test_data
"""
#Use this one to get the title
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""





def clean_data(path, train):



    import pandas as P
    import numpy as np

    # GET DATA
    # read file
    df_train = P.read_csv(path, header=0)
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
            median_ages[i, j] = df_train[(df_train.GenderNumber == i) & (df_train.Pclass == j + 1)]['Age'].dropna().median()
    df_train['AgeFill'] = df_train.Age
    for i in range(0, 2):
        for j in range(0, 3):
            df_train.loc[(df_train.Age.isnull()) & (df_train.GenderNumber == i) & (df_train.Pclass == j + 1), 'AgeFill'] = \
            median_ages[i, j]
    # drop values that are not relevant




    df_train['Title'] = 0.0

    df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
    df_train['Age*Class'] = df_train.AgeFill * df_train.Pclass / 10

    index_of_1st_class = df_train.Pclass == 1
    index_of_2nd_class = df_train.Pclass == 2
    index_of_3rd_class = df_train.Pclass == 3

    df_train.set_value(index_of_1st_class, 'Pclass', 4)
    df_train.set_value(index_of_2nd_class, 'Pclass', 2.5)
    df_train.set_value(index_of_3rd_class, 'Pclass', 3.5)
    df_train['Gender * Class'] = df_train['FamilySize'] * df_train['Pclass']

    import collections



    input_data = df_train[['Pclass', 'AgeFill', 'FamilySize', 'Fare', 'GenderNumber', 'Age*Class', 'Title']]
    output_data = df_train.Survived
    Point = collections.namedtuple('data', ['X', 'y'])
    data = Point(X = input_data, y = output_data)

    return data


p = clean_data("data/train.csv", True)

print p.X



"""
    import re

    txt = 'Mr.  Miss.   Master.   Mrs.'

    re_Mr = '(Mr\\.\\s+)'
    re_Mrs = '(Mrs\\.\\s+)'
    re_Master = '(Master\\.\\s+)'
    re_Miss = '(Miss\\.\\s+)'

    rg1 = re.compile(re_Mrs, re.IGNORECASE | re.DOTALL)
    rg2 = re.compile(re_Mr, re.IGNORECASE | re.DOTALL)
    rg3 = re.compile(re_Master, re.IGNORECASE | re.DOTALL)
    rg4 = re.compile(re_Miss, re.IGNORECASE | re.DOTALL)

    for index in range(len(df_train.Name)):
        m1 = rg1.search(df_train.Name[index])
        m2 = rg2.search(df_train.Name[index])
        m3 = rg3.search(df_train.Name[index])
        m4 = rg4.search(df_train.Name[index])
        if m1:  # Mrs
            word = m1.group(1)
            df_train.set_value(index, 'Title', 8)
            # df_train['Title'][index] = 1
        elif m2:  # Mr
            word = m2.group(1)
            df_train.set_value(index, 'Title', 1)
        elif m3:  # Master
            word = m3.group(1)
            df_train.set_value(index, 'Title', 6)
        elif m4:  # Miss
            word = m4.group(1)
            df_train.set_value(index, 'Title', 7)
        else:
            df_train.set_value(index, 'Title', 4)
"""