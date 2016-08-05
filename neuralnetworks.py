
import numpy as np
import scaledconjugategradient as scg
import mlutils as ml  # for draw()
from copy import copy

class NeuralNetwork:

    def __init__(self, ni,nhs,no):        
        try:
            nihs = [ni] + list(nhs)
        except:
            nihs = [ni] + [nhs]
            nhs = [nhs]
        self.Vs = [1/np.sqrt(nihs[i])*
                   np.random.uniform(-1,1,size=(1+nihs[i],nihs[i+1])) for i in range(len(nihs)-1)]
        self.W = 1/np.sqrt(nhs[-1])*np.random.uniform(-1,1,size=(1+nhs[-1],no))
        self.ni,self.nhs,self.no = ni,nhs,no
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self.trained = False
        self.reason = None
        self.errorTrace = None
        self.numberOfIterations = None

    def __repr__(self):
        str = 'NeuralNetwork({}, {}, {})'.format(self.ni,self.nhs,self.no)
        # str += '  Standardization parameters' + (' not' if self.Xmeans == None else '') + ' calculated.'
        if self.trained:
            str += '\n   Network was trained for {} iterations. Final error is {}.'.format(self.numberOfIterations,
                                                                                           self.errorTrace[-1])
        else:
            str += '  Network is not trained.'
        return str
            
    def standardizeX(self,X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:,self.Xconstant] = 0.0
        return result
    def unstandardizeX(self,Xs):
        return self.Xstds * Xs + self.Xmeans
    def standardizeT(self,T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:,self.Tconstant] = 0.0
        return result
    def unstandardizeT(self,Ts):
        return self.Tstds * Ts + self.Tmeans
   
    def train(self,X,T,nIterations=100,verbose=False,
              weightPrecision=0,errorPrecision=0):
        
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1
        X = self.standardizeX(X)

        if T.ndim == 1:
            T = T.reshape((-1,1))

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1
        T = self.standardizeT(T)

        # Local functions used by scg()

        def pack(Vs,W):
            return np.hstack([V.flat for V in Vs] + [W.flat])

        def unpack(w):
            first = 0
            numInThisLayer = self.ni
            for i in range(len(self.Vs)):
                self.Vs[i][:] = w[first:first+(numInThisLayer+1)*self.nhs[i]].reshape((numInThisLayer+1,self.nhs[i]))
                first += (numInThisLayer+1) * self.nhs[i]
                numInThisLayer = self.nhs[i]
            self.W[:] = w[first:].reshape((numInThisLayer+1,self.no))

        def objectiveF(w):
            unpack(w)
            Zprev = X
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])  # handling bias weight without adding column of 1's
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
            return np.mean((T-Y)**2)

        def gradF(w):
            unpack(w)
            Zprev = X
            Z = [Zprev]
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
                Z.append(Zprev)
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
            delta = -(T - Y) / (X.shape[0] * T.shape[1])
            dW = 2 * np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Z[-1].T, delta)))
            dVs = []
            delta = (1-Z[-1]**2) * np.dot( delta, self.W[1:,:].T)
            for Zi in range(len(self.nhs),0,-1):
                Vi = Zi - 1 # because X is first element of Z
                dV = 2 * np.vstack(( np.dot(np.ones((1,delta.shape[0])), delta),
                                 np.dot( Z[Zi-1].T, delta)))
                dVs.insert(0,dV)
                delta = np.dot( delta, self.Vs[Vi][1:,:].T) * (1-Z[Zi-1]**2)
            return pack(dVs,dW)

        scgresult = scg.scg(pack(self.Vs,self.W), objectiveF, gradF,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            verbose=verbose,
                            ftracep=True)

        unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = np.sqrt(scgresult['ftrace']) # * self.Tstds # to unstandardize the MSEs
        self.numberOfIterations = len(self.errorTrace)
        self.trained = True
        return self

    def use(self,X,allOutputs=False):
        Zprev = self.standardizeX(X)
        Z = [Zprev]
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
            Z.append(Zprev)
        Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
        Y = self.unstandardizeT(Y)
        return (Y,Z[1:]) if allOutputs else Y

    def getNumberOfIterations(self):
        return self.numberOfIterations
    
    def getErrors(self):
        return self.errorTrace
        
    def draw(self,inputNames = None, outputNames = None):
        ml.draw(self.Vs+[self.W], inputNames, outputNames)
