'''
This file implement the generic basic matrix factorization method.

Implementation notes:
    - max iteration (maxIter) must be specified
    - tolerance, by default, is not specified
    
For basic MF, the generic gradient descent optimization method is used,
therefore, the learning rate (here it is lamb) needs to be specified.
'''
# Author: Jie Yang <J.Yang-3@tudelft.nl>

import numpy as np

class BasicMF():
    '''
        this model solves the following optimization problem:
            min ||X-UV^T||_F^2 + alpha*||U||_F^2 + beta*||V||_F^2
    '''
    
    def __init__(self, alpha, beta, lamb, K, maxIter, tol=-1):
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.maxIter = maxIter
        self.tol = tol
        
        self.lamb = lamb

    def decomp(self, X):
        m, n = X.shape
        
        # initialize U and V
        U = np.random.rand(m, self.K)
        V = np.random.rand(n, self.K)
        itr = 0

        # update low-rank matrices
        mae = float(1) / (m * n) * np.sum(np.absolute(X - np.dot(U, V.T)))
        while itr < self.maxIter and mae > self.tol:
            UU = np.dot(U.T, U)
            VV = np.dot(V.T, V)
            
            U = U - self.lamb*2*(np.dot(U, VV) - np.dot(X,V) + self.alpha * U)
            V = V - self.lamb*2*(np.dot(V, UU) - np.dot(X.T,U) + self.beta * V)
            mae = float(1) / (m * n) * np.sum(np.absolute(X - np.dot(U, V.T)))

            if itr % 100 == 0:
                obj = np.power(np.linalg.norm(X - np.dot(U, V.T)), 2) + self.alpha * np.power(np.linalg.norm(U), 2) + self.beta * np.power(np.linalg.norm(V), 2)
                print 'the objective value is ', obj, " at iteration ", itr
            
            itr += 1
        return U, V

class Sparse_BasicMF():
    '''
        this model solves the following optimization problem:
            min ||O*(X-UV^T)||_F^2 + alpha*||U||_F^2 + beta*||V||_F^2
        in which multiplication between matrices are Hadmard product
    '''
    
    def __init__(self, alpha, beta, lamb, K, maxIter, O, tol=-1):
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.maxIter = maxIter
        self.tol = tol
        
        self.lamb = lamb
        self.O = O

    def decomp(self, X):
        m, n = X.shape
        
        # initialize U and V
        U = np.random.rand(m, self.K)
        V = np.random.rand(n, self.K)
        itr = 0

        # update low-rank matrices
        mae = float(1) / (m * n) * np.sum(np.absolute(self.O*X - self.O*np.dot(U, V.T)))
        while itr < self.maxIter and mae > self.tol:
            UV = np.dot(U, V.T)
            VU = np.dot(V, U.T)
            
            U = U - self.lamb*2*(np.dot(self.O*UV, V) - np.dot(self.O*X,V) + self.alpha * U)
            V = V - self.lamb*2*(np.dot(self.O.T*VU, U) - np.dot(self.O.T*X.T,U) + self.beta * V)
            mae = float(1) / (m * n) * np.sum(np.absolute(self.O*X - self.O*np.dot(U, V.T)))

            if itr % 100 == 0:
                obj = np.power(np.linalg.norm(self.O*X - self.O*np.dot(U, V.T)), 2) + self.alpha * np.power(np.linalg.norm(U), 2) + self.beta * np.power(np.linalg.norm(V), 2)
                print 'the objective value is ', obj, " at iteration ", itr
            
            itr += 1
        return U, V
