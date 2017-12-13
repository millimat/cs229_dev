import numpy as np
from pandas import read_csv
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Imputer
import csv

# Read data matrix from csv
# return (header, X, Y), where header[j] = name of feature j,
# X[i,j] is value of feature j for example i, and Y[i] is target value
# for feature i.
#
# With impute activated, missing features will be set to the mean of
# the non-missiing features in that column.
def fetch_data(datafile, train=True, impute=False, missing=-1):
    data = read_csv(datafile, dtype=np.float64)
    header = data.columns.tolist()
    ids = data['id'].astype(int).tolist()
    data = data.values
    
    # train set has ids in col 0 and labels in col 1
    # test has ids in col 0 and no labels
    offset = (2 if train else 1)
    X = data[:,offset:]
    Y = data[:,offset-1] if train else None
    
    # Replace missing categorical values with mode and missing continuous values with mean.
    if impute:          
        imp_categorical = Imputer(missing_values=missing, strategy='most_frequent')
        imp_continuous = Imputer(missing_values=missing, strategy='mean')
        
        offset = 2 if train else 1 # Start column of data in raw csv
        
        cat_cols = []
        cont_cols = []
        
        for (i, feature_name) in enumerate(header[offset:]):
            if feature_name.endswith('cat') or feature_name.endswith('bin'):
                cat_cols.append(i)
            else:
                cont_cols.append(i)
         
        X[:,cat_cols] = imp_categorical.fit_transform(X)[:,cat_cols]
        X[:,cont_cols]  = imp_continuous.fit_transform(X)[:,cont_cols]            
            
    return (header, ids, X, Y)


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

# Use for regressors that automatically output continuous estimates
gini_scorer = make_scorer(gini_normalized)

# Use for classifiers that need predict_proba to output decision probabilities
def gini_proba_scorer(clf, X, y):
    return gini_normalized(y, clf.predict_proba(X)[:,1])


def learning_curves(model, Xtrain, Xtest, Ytrain, Ytest, scorer=gini_scorer, nsteps=20):
    train_results = []
    test_results = []
    
    trainsizes = [int(x) for x in np.linspace(0,Xtrain.shape[0], nsteps+1)][1:]
    
    for s in trainsizes:
        print('Evaluating model on training set size {}'.format(s))
        Xtr = Xtrain[:s,:]
        Ytr = Ytrain[:s]
        
        model.fit(Xtr, Ytr)
        
        train_results.append(scorer(model, Xtr, Ytr))
        test_results.append(scorer(model, Xtest, Ytest))
    
    return trainsizes, train_results, test_results


# Function of a classifier that returns the 1 (positive) column of its predict_proba.
# proba_method(clf) is a function that is called on a data matrix X to get
# predicted positive probabilities out of clf.
def proba_method(clf):
    def get_proba(X):
        return clf.predict_proba(X)[:,1]
    return get_proba


def make_prediction(estimator, testfile, outfile, predict_method=None):
    _, ids, X_test, _ = fetch_data(testfile, train=False, impute=True)
    ids = np.array(ids, dtype=np.int)
    
    Y_test = (estimator.predict(X_test) if predict_method == None
              else predict_method(X_test)) # e.g. model.predict_proba(...)[:,1]

    Y_test_normalized = Y_test - np.min(Y_test)
    Y_test_normalized /= np.max(Y_test_normalized)

    with open(outfile, 'w') as out:
        out.write('id,target\n')
        for i in range(Y_test_normalized.shape[0]):
            out.write('{},{}\n'.format(ids[i], Y_test_normalized[i]))
    
        
