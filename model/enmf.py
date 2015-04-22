'''
This file implement the enhanced non-negative matrix factorization method.

Implementation notes:
    - max iteration (maxIter) must be specified
    - tolerance, by default, is not specified
    
For OneSided_Enhanced_Sparse_NMF, we will use more sophisticated optimization method (see document) is used,
therefore, the learning rate does not need to be specified.

For TwoSided_Enhanced_Sparse_NMF, we will use the basic gradient descent method, therefore, the learning
rate needs to be specified. 

We require lasso in TwoSided_Enhanced_Sparse_NMF
'''
# Author: Jie Yang <J.Yang-3@tudelft.nl>

import numpy as np
from sklearn import linear_model

class USided_Enhanced_Sparse_NMF():
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
        mae = float(1) / (m * n) * np.sum(np.absolute(self.O*X - self.O*np.dot(U, V.T)))
        while itr < self.maxIter and mae > self.tol:
            UV = np.dot(U, V.T)
            VU = np.dot(V, U.T)
            
            U = U * np.sqrt((np.dot(self.O*X, V) + self.lamb * np.dot(Z, U))/ (np.dot(self.O*UV, V) + self.alpha * U + self.lamb * np.dot(D, U)))
            V = V * np.sqrt(np.dot(self.O.T*X.T,U) / (np.dot(self.O.T*VU, U) + self.beta * V))
            mae = float(1) / (m * n) * np.sum(np.absolute(self.O*X - self.O*np.dot(U, V.T)))
        
            if itr % 100 == 0:
                obj = np.power(np.linalg.norm(self.O*X - self.O*np.dot(U, V.T)), 2) + self.alpha * np.power(np.linalg.norm(U), 2) + self.beta * np.power(np.linalg.norm(V), 2)
                print 'the objective value is ', obj, " at iteration ", itr
            itr += 1
        return U, V
    
def get_degree_matrix(Z):
    m, n = Z.shape
    assert m == n
    D = np.zeros((m,n))
    
    for i in xrange(m):
        D[i,i] = sum(Z[i,:])
    
    return D


class TwoSided_Enhanced_Sparse_NMF():
    '''
        this model solves the following optimization problem:
            min ||O*(X-UHV^T)||_F^2 
                + lambda_a*||A-UG^T||_F^2 + lambda_b||B-V(G-D)^T||_F^2 + delta||D||_1
                + alpha*(||U||_F^2 + ||V||_F^2 + ||H||_F^2 + ||G||_F^2)
            s.t. U>=0, V>=0, H>=0
        in which multiplication between matrices are Hadmard product, and L is the Laplacian
        matrix of Z (the user relation matrix).
        
        Note: here we use one alpha to regularize all the low-rank matrices;
              additionally, lambda_a, lambda_b and delta are used for regularizing the 
              extra matrices (factorizations).
    '''
    
    def __init__(self, alpha, lamb_a, lamb_b, delta, K, maxIter, O, tol=-1):
        self.alpha = alpha
        self.K = K
        self.maxIter = maxIter
        self.tol = tol
        
        self.lamb_a = lamb_a
        self.lamb_b = lamb_b
        self.delta = delta
        self.O = O

    def decomp(self, X, A, B):
        m, n = X.shape
        m_a, n_a = A.shape
        assert m_a == m
        m_b, n_b = B.shape
        assert m_b == n
        assert n_a == n_b
        
        # initialize U, H, V, and G
        U = np.random.rand(m, self.K)
        H = np.random.rand(self.K, self.K)
        V = np.random.rand(n, self.K)
        
        G = np.random.rand(n_a, self.K)
        D = np.random.rand(n_a, self.K)
        
        itr = 0

        lasso = linear_model.Lasso(self.delta)
        mae = float(1) / (m * n) * np.sum(np.absolute(self.O*X - self.O*np.dot(np.dot(U, H), V.T)))
        while itr < self.maxIter and mae > self.tol:
            VH = np.dot(V, H.T)
            UHV = np.dot(U, np.dot(H, V.T))
            UH = np.dot(U, H)
            
            UU = np.dot(U.T, U)
            VV = np.dot(V.T, V)
            # TODO: set up gamma's
            gamma_u = 0.001
            gamma_h = 0.001
            gamma_v = 0.001
            gamma_g = 0.001
            
            # update U, H, V, G
            U = U - gamma_u*2*(np.dot(self.O*UHV, VH) - np.dot(self.O*X, VH) - self.lamb_a*np.dot(A, G) - self.lamb_a*np.dot(U, np.dot(G.T, G)) + self.alpha*U)
            U = project_to_non_negative(U)
            H = H - gamma_h*2*(np.dot(U.T, np.dot(self.O*UHV, V)) - np.dot(U.T, np.dot(self.O*X, V)) + self.alpha*H)
            H = project_to_non_negative(H)
            V = V - gamma_v*2*(np.dot(self.O.T*UHV.T, UH) - np.dot(self.O.T*X.T, UH) - self.lamb_b*np.dot(B-np.dot(V, G.T)+np.dot(V, D.T), G-D) +self.alpha*V)
            V = project_to_non_negative(V)
            G = G - gamma_g*2*(self.lamb_a*np.dot(G, UU) - self.lamb_a*np.dot(A.T, U) + self.lamb_b*np.dot(G-D, VV) - self.lamb_b*np.dot(B.T, V) + self.alpha*G)
            # update D
            for j in xrange(self.K):
                lasso.fit(V, np.dot(V,G.T)-B)
                D[:,j] = lasso.coef_
            
            mae = float(1) / (m * n) * np.sum(np.absolute(self.O*X - self.O*np.dot(np.dot(U, H), V.T)))
        
            if itr % 100 == 0:
                obj = np.power(np.linalg.norm(self.O*X - self.O*np.dot(np.dot(U, H), V.T)), 2) + \
                self.lamb_a*(np.power(np.linalg.norm(A - np.dot(U, G.T)), 2)) + self.lamb_b*(np.power(np.linalg.norm(B - np.dot(V, (G-D).T)), 2)) + self.delta*np.linalg.norm(D,1) + \
                self.alpha * (np.power(np.linalg.norm(U), 2) + np.power(np.linalg.norm(V), 2) + np.power(np.linalg.norm(H), 2) + np.power(np.linalg.norm(G), 2))
                print 'the objective value is ', obj, " at iteration ", itr
            itr += 1
        return U, H, V
    
def project_to_non_negative(M):
    m, n = M.shape
    for i in xrange(m):
        for j in xrange(n):
            if M[i,j]<0:
                M[i,j] = 0
    return M