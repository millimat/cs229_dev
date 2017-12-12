import numpy as np
from sklearn.metrics import make_scorer
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
        ids = np.array(data[:,0], dtype=np.int)
        X = data[:,2:] if train else data[:,1:] 
        Y = data[:,1] if train else None
            
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

gini_scorer = make_scorer(gini_normalized)


def learning_curves(model, Xtrain, Xtest, Ytrain, Ytest, nsteps=20, metric=gini_normalized):
    train_results = []
    test_results = []
    
    trainsizes = int(x) for x in np.linspace(0,Xtrain.shape[0], nsteps+1)[1:]
    
    for s in trainsizes:
        model.fit(Xtrain[:s,:], Ytrain[:s])
        train_results.append(metric(Ytrain[:s], model.predict(Xtrain[:s,:])))
        test_results.append(metric(Ytest, model.predict(Xtest)))
    
    return train_results, test_results


def make_prediction(estimator, testfile, outfile, predict_method=None, xgb_estimator=False):
    _, ids, X_test, _ = fetch_data(testfile,train=False)
    ids = np.array(ids, dtype=np.int)
    
    Y_test = (estimator.predict(X_test) if predict_method == None
              else predict_method(X_test)) # e.g. model.predict_proba(...)[:,1]

    Y_test_normalized = Y_test - np.min(Y_test)
    Y_test_normalized /= np.max(Y_test_normalized)

    with open(outfile, 'w') as out:
        out.write('id,target\n')
        for i in range(Y_test_normalized.shape[0]):
            out.write('{},{}\n'.format(ids[i], Y_test_normalized[i]))
    
        
