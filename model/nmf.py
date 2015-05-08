'''
This file implement the generic non-negative matrix factorization method.

Implementation notes:
    - max iteration (maxIter) must be specified
    - tolerance, by default, is not specified

For NMF, we will use more sophisticated optimization method (see document) is used,
therefore, the learning rate does not need to be specified.
'''
# Author: Jie Yang <J.Yang-3@tudelft.nl>

import numpy as np


class NMF():

    '''
        this model solves the following optimization problem:
            min ||X-UV^T||_F^2 + alpha*||U||_F^2 + beta*||V||_F^2
            s.t. U>=0, V>=0
    '''

    def __init__(self, alpha, beta, K, maxIter, tol=-1):
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.maxIter = maxIter
        self.tol = tol

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

            U = U * np.sqrt(np.dot(X, V) / (np.dot(U, VV) + self.alpha * U))
            V = V * np.sqrt(np.dot(X.T, U) / (np.dot(V, UU) + self.beta * V))
            mae = float(1) / (m * n) * np.sum(np.absolute(X - np.dot(U, V.T)))

            if itr % 100 == 0:
                obj = np.power(np.linalg.norm(X - np.dot(U, V.T)), 2) + self.alpha * np.power(
                    np.linalg.norm(U), 2) + self.beta * np.power(np.linalg.norm(V), 2)
                print 'the objective value is ', obj, " at iteration ", itr
            itr += 1
        return U, V


class Sparse_NMF():

    '''
        this model solves the following optimization problem:
            min ||O*(X-UV^T)||_F^2 + alpha*||U||_F^2 + beta*||V||_F^2
            s.t. U>=0, V>=0
        in which multiplication between matrices are Hadmard product
    '''

    def __init__(self, alpha, beta, K, maxIter, O, tol=-1):
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.maxIter = maxIter
        self.tol = tol

        self.O = O

    def decomp(self, X):
        m, n = X.shape

        # initialize U and V
        U = np.random.rand(m, self.K)
        V = np.random.rand(n, self.K)
        itr = 0

        # update low-rank matrices
        mae = float(1) / (m * n) * \
            np.sum(np.absolute(self.O * X - self.O * np.dot(U, V.T)))
        while itr < self.maxIter and mae > self.tol:
            UV = np.dot(U, V.T)
            VU = np.dot(V, U.T)

            U = U * \
                np.sqrt(
                    np.dot(self.O * X, V) / (np.dot(self.O * UV, V) + self.alpha * U))
            V = V * np.sqrt(np.dot(self.O.T * X.T, U) /
                            (np.dot(self.O.T * VU, U) + self.beta * V))
            mae = float(
                1) / (m * n) * np.sum(np.absolute(self.O * X - self.O * np.dot(U, V.T)))

            if itr % 100 == 0:
                obj = np.power(np.linalg.norm(self.O * X - self.O * np.dot(U, V.T)), 2) + self.alpha * \
                    np.power(np.linalg.norm(U), 2) + self.beta * \
                    np.power(np.linalg.norm(V), 2)
                print 'the objective value is ', obj, " at iteration ", itr
            itr += 1
        return U, V
