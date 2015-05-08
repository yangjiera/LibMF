'''
This file implement the social regularized non-negative matrix factorization method.

Implementation notes:
    - max iteration (maxIter) must be specified
    - tolerance, by default, is not specified
    
we will use more sophisticated optimization method (see document) is used,
therefore, the learning rate does not need to be specified.
'''
# Author: Jie Yang <J.Yang-3@tudelft.nl>

import numpy as np
from sklearn import linear_model


class Social_Sparse_NMF():

    '''
        this model solves the following optimization problem:
            min ||O*(X-UV^T)||_F^2 + lambda*Tr(U^TLU)
                + alpha*||U||_F^2 + beta*||V||_F^2
            s.t. U>=0, V>=0
        in which multiplication between matrices are Hadmard product, and L is the Laplacian
        matrix of Z (the user relation matrix).

        Note: lambda here is for regularizing Tr(U^TLU), not used as learning rate.
    '''

    def __init__(self, alpha, beta, lamb, K, maxIter, O, tol=-1):
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.maxIter = maxIter
        self.tol = tol

        self.lamb = lamb
        self.O = O

    def decomp(self, X, Z):
        m, n = X.shape
        D = get_degree_matrix(Z)

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

            U = U * np.sqrt((np.dot(self.O * X, V) + self.lamb * np.dot(Z, U)) /
                            (np.dot(self.O * UV, V) + self.alpha * U + self.lamb * np.dot(D, U)))
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


def get_degree_matrix(Z):
    m, n = Z.shape
    assert m == n
    D = np.zeros((m, n))

    for i in xrange(m):
        D[i, i] = sum(Z[i, :])

    return D
