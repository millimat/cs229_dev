import numpy as np
import matplotlib.pyplot as plt
import sklearn
import csv

# Read data matrix from csv
# return (header, X, Y), where header[j] = name of feature j,
# X[i,j] is value of feature j for example i, and Y[i] is target value
# for feature i.
def fetch_data(datafile, train=True):
    header = None
    data = []

    with open(datafile) as datacsv:
        reader = csv.reader(datacsv)
        header = reader.next()
        for row in reader:
            data.append(row)

        data = np.array(data, dtype=np.float64)
        # train set has labels in column 1; test does not
        # both models have customer ids in column 0
        X = data[:,2:] if train else data[:,1:] 
        Y = data[:,1]
        return (header, X, Y)


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


def gini_scorer(estimator, X, y):
    return gini_normalized(y, estimator.predict(X))