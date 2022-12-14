import numpy as np
from scipy.linalg import eigh
import matlab.engine
class StandardTRCA(object):
    def __init__(self, num_in, num_out, num_sample, matlab_engine_on=False) -> None:
        self.num_in = num_in
        self.num_out = num_out
        self.num_sample = num_sample
        self.num_class = 0
        self.W = np.zeros([num_in, num_out])
        self.averageM = []
        self.matlab_engine_on=matlab_engine_on
        if matlab_engine_on:
            self.eng = matlab.engine.start_matlab()
        
    def fit(self, X, y):
        # X: num_trial, num_channel(in), num_sample
        y = y.ravel()
        labels = np.unique(y)
        self.num_class = len(labels)
        S = np.zeros([self.num_in, self.num_in,self.num_class])
        Q = np.zeros([self.num_in, self.num_in,self.num_class])

        for cls in np.arange(self.num_class):
            x = X[y==labels[cls],:,:]
            s, q = self.trca(x)
            S[:,:,cls]=s
            Q[:,:,cls]=q

        S = np.mean(S, 2)
        Q = np.mean(Q, 2)

        if self.matlab_engine_on:
            [vec, val] = self.eng.eig(matlab.double(S.tolist()),matlab.double(Q.tolist()), 'qz',nargout=2)
            val = np.diag(val)
            vec = np.asarray(vec)
            
        else:
            [val, vec] = eigh(S, Q, driver='ev')

        argval = np.argsort(-val)
        self.W = vec[:, argval[:self.num_out]]
        self.W = self.W*np.sign(self.W[4,:]).reshape((1,self.num_out))
        
    
    def transform(self, X):
        # X: num_trial, num_channel(in), num_sample
        X = np.expand_dims(X, 3)
        W = np.expand_dims(self.W, (0,2))
        WX = (W*X).sum(1)
        WX = np.swapaxes(WX, 1, 2)
        return WX

    @staticmethod
    def trca(X):
        # X: num_trial, num_channel(in), num_sample
        n_trial, n_channel, n_sample = np.shape(X)
        qX = np.swapaxes(X, 1, 2).reshape((n_trial*n_sample, n_channel))
        qX = qX-np.mean(qX, 0).reshape(1,-1)
        Q = qX.T@qX/(n_trial*n_sample)

        sX = X-np.mean(X, 2).reshape(n_trial, n_channel,1)
        U = np.sum(sX, 0)
        V = np.zeros((n_channel, n_channel))
        for k in np.arange(n_trial):
            V = V+sX[k,:,:]@sX[k,:,:].T
        S = (U@U.T-V)/(n_sample*n_trial*(n_trial-1))
        return S, Q