import numpy as np


def entropy(data):
    _, frequency = np.unique(data, return_counts=True)
    probability = frequency / len(data)
    entropy_value = 0
    for p in probability:
        entropy_value -= p * np.log2(p)
    return entropy_value


def information_gain(data_A, data_B):
    result = entropy(data_A)
    values, frequency = np.unique(data_B, return_counts=True)

    for p, v in zip(frequency, values):
        result -= p * entropy(data_A | data_B == v)

def split():
    return

def build_classification_tree(data):

