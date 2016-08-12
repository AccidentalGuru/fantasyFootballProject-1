import numpy as np
import NeuralNetworkStructure.NeuralNetworkUtilities.neuralnetworks as nn

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
    column_of_ones = np.ones((X.shape[0], 1))
    xs1 = np.hstack((column_of_ones, (X - model['means']) / model['stds']))
    a = np.dot(xs1, model['w'])
    return np.sqrt(np.mean((a - T) ** 2))


def useLinear(model, X, T):
    column_of_ones = np.ones((X.shape[0], 1))
    xs1 = np.hstack((column_of_ones, (X - model['means']) / model['stds']))
    a = np.dot(xs1, model['w'])
    return a


def trainNN(X, T, parameters):
    net = nn.NeuralNetwork(X.shape[1], parameters[0], T.shape[1])
    train = net.train(X, T, nIterations=parameters[1])
    return train


def evaluateNN(model, X, T):
    use = model.use(X)
    error = np.sqrt(np.mean((use - T) ** 2))
    return error