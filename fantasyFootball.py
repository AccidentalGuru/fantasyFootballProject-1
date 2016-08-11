import numpy as np
import matplotlib.pyplot as plt
import mlutils as ml
import neuralnetworks as nn
import scaledconjugategradient as scg
import itertools


def trainLinear(X, T, parameters):
    means = X.mean(0)
    stds = X.std(0)
    n, d = X.shape
    Xs1 = np.hstack((np.ones((n, 1)), (X - means) / stds))
    lambDiag = np.eye(d + 1) * parameters
    lambDiag[0, 0] = 0
    w = np.linalg.lstsq(np.dot(Xs1.T, Xs1) + lambDiag, np.dot(Xs1.T, T))[0]
    return {'w': w, 'means': means, 'stds': stds}


def evaluateLinear(model, X, T):
    columnOfOnes = np.ones((X.shape[0], 1))
    Xs1 = np.hstack((columnOfOnes, (X - model['means']) / model['stds']))
    A = np.dot(Xs1, model['w'])
    return np.sqrt(np.mean((A - T) ** 2))


def useLinear(model, X, T):
    columnOfOnes = np.ones((X.shape[0], 1))
    Xs1 = np.hstack((columnOfOnes, (X - model['means']) / model['stds']))
    A = np.dot(Xs1, model['w'])
    return A


def trainNN(X, T, parameters):
    net = nn.NeuralNetwork(X.shape[1], parameters[0], T.shape[1])
    train = net.train(X, T, nIterations=parameters[1])
    return train


def evaluateNN(model, X, T):
    use = model.use(X)
    error = np.sqrt(np.mean((use - T) ** 2))
    return error


def trainValidateTestKFolds(trainf, evaluatef, X, T, parameterSets, nFolds, shuffle=False, verbose=False):
    # Randomly arrange row indices
    rowIndices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(rowIndices)
    # Calculate number of samples in each of the nFolds folds
    nSamples = X.shape[0]
    nEach = int(nSamples / nFolds)
    if nEach == 0:
        raise ValueError("partitionKFolds: Number of samples in each fold is 0.")
    # Calculate the starting and stopping row index for each fold.
    # Store in startsStops as list of (start,stop) pairs
    starts = np.arange(0, nEach * nFolds, nEach)
    stops = starts + nEach
    stops[-1] = nSamples
    startsStops = list(zip(starts, stops))
    # Repeat with testFold taking each single fold, one at a time
    results = []

    # COMPLETE THIS FUNCTION BY IMPLEMENTING THE FOLLOWING STEPS.
    # For each test fold
    for testFold in range(nFolds):
        finalArray = []
        # For each set of parameter values, called parmSet
        for parmSet in parameterSets:
            newX = []
            newT = []
            # Find best set of parameter values
            bestError = []
            # For each validate fold (except when same as test fold)
            for validateFold in range(nFolds):
                if testFold == validateFold:
                    continue
                # trainFolds are all remaining folds, after selecting test and validate folds
                trainFolds = np.setdiff1d(range(nFolds), [testFold, validateFold])
                # Construct Xtrain and Ttrain by collecting rows for all trainFolds
                rows = []
                for tf in trainFolds:
                    a, b = startsStops[tf]
                    rows += rowIndices[a:b].tolist()
                Xtrain = X[rows, :]
                Ttrain = T[rows, :]
                # Construct Xvalidate and Tvalidate
                a, b = startsStops[validateFold]
                rows = rowIndices[a:b]
                Xvalidate = X[rows, :]
                Tvalidate = T[rows, :]
                # Construct Xtest and Ttest
                a, b = startsStops[testFold]
                rows = rowIndices[a:b]
                Xtest = X[rows, :]
                Ttest = T[rows, :]

                # Use trainf to fit model to training data using parmSet
                model = trainf(Xtrain, Ttrain, parmSet)

                # Calculate the error of this model by calling evaluatef with the model
                # and validation data
                error = evaluatef(model, Xvalidate, Tvalidate)
                bestError.append(error)

                # Make a new set of training data by concatenating the training and
                # validation data from previous step.
                newX = np.vstack((Xtrain, Xvalidate))
                newT = np.vstack((Ttrain, Tvalidate))

            # Calculate the mean of these errors.
            finalArray.append(np.array(bestError).mean())

        # If this error is less than the previously best error for parmSet,update best
        # parameter values and best error
        idx = (np.argmin(np.array(finalArray)))

        # Retrain, using trainf again, to fit a new model to this new training data.
        reTrain = trainf(newX, newT, parameterSets[idx])
        # Calculate error of this new model on the test data, and also on the new training data.
        trainError = evaluatef(reTrain, newX, newT)
        testError = evaluatef(reTrain, Xtest, Ttest)

        # Construct a list of the best parameter values with this training error, the
        # mean of the above valdiation errors, and the testing error

        iterResults = [parameterSets[idx], trainError, min(finalArray), testError]

        results.append(iterResults)
    # Print this list if verbose == True
    if verbose:
        print(result)

        # Append this list to a result list
        # Return this result list
    return np.array(results)



    # X = np.arange(20).reshape((-1,1))
    # T = np.abs(X -10) + X

    # result = trainValidateTestKFolds(trainLinear,evaluateLinear,X,T,
    #                                  range(0,101,10),nFolds=5,shuffle=False)

    # print('Linear Model\nlambda, train, validate, test RMSE')
    # for x in result:
    #     print('{:.2f}    {:.3f}   {:.3f}   {:.3f}'.format(*x))


    # parms = list(itertools.product([2,5,10, 20, [5,5], [10,2,10]], [10,20,100]))
    # result = trainValidateTestKFolds(trainNN,evaluateNN,X,T,
    #                                  parms,nFolds=5,shuffle=False)

    # print('NN Model\n(hidden units, iterations), train, validate, test RMSE')
    # for x in result:
    #     print('{:}  \t\t    {:.3f}   {:.3f}   {:.3f}'.format(*x))


def makeFootballData():
    TeamNames = []
    filenames = ['years_2011_opp_team_stats.csv', 'years_2012_opp_team_stats.csv',
                 'years_2013_opp_team_stats.csv', 'years_2014_opp_team_stats.csv',
                 'years_2015_opp_team_stats.csv']
    for filename in filenames:
        f = open(filename)
        lines = f.readlines()
        data = lines[3:35]
        for i in range(32):
            TeamNames.append(data[i].split(",")[1] + filename[8:10])
        data = np.loadtxt(data, delimiter=',', dtype=None, usecols=range(3, 28))
        #         print(data)
        if filename == 'years_2011_opp_team_stats.csv':
            X = data
            T = [1]
            for i in range(2, 33):
                T = np.vstack((T, i))
        else:
            X = np.vstack((X, data))
            for i in range(1, 33):
                T = np.vstack((T, i))

    Xnames = (lines[2].split(","))
    Xnames = Xnames[3:]
    Tname = "rank"
    return X, T, Xnames, Tname, TeamNames


# Prints the sorted order of ranks, lowest to highest.
def printRanks(NeuralNetAverages, TeamNames):
    NeuralNetAverages = sorted(NeuralNetAverages, key=lambda x: x[0])
    numPredicted = 0;
    for i in range(0, 32):
        print(NeuralNetAverages[i][1], "\tPredicted Rank:", i + 1, "\tActual Rank:",
              TeamNames.index(NeuralNetAverages[i][1]) - 127)
        if ((TeamNames.index(NeuralNetAverages[i][1]) - 127) <= i + 2 and (
            TeamNames.index(NeuralNetAverages[i][1]) - 127) >= i):
            numPredicted = numPredicted + 1

    print("\nNumber Correctly Predicted Within One Rank:", numPredicted, "\nPercent Correct:",
          float(numPredicted / 32) * 100, "%")