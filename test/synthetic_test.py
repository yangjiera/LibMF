'''
Test files for the modules MF&NMF
'''
# Author: Jie Yang <J.Yang-3@tudelft.nl>

import numpy as np
import sys
sys.path.append(sys.path[0][0:-5]+'/model/')
from scipy.sparse import rand

from mf import *
from nmf import *

def generate_non_negative_full_matrix(n_users, n_items, n_factors):
    while True:
        U = np.random.rand(n_users, n_factors)
        V = np.random.rand(n_items, n_factors)
        X = np.dot(U,V.T)
        
        noise = np.random.randn(n_users, n_items)*0.1
        X += .1*noise
        
        is_non_negative = True
        for i in xrange(n_users):
            for j in xrange(n_items):
                if X[i,j]<0:
                    is_non_negative = False
                    break
            if is_non_negative == False:
                break
        if is_non_negative:
            return X, U, V
        
def get_random_indicator_matrix(n_users, n_items, sparsity):
    while True:
        non_missing = 0
        O = rand(n_users, n_items, sparsity).toarray()
        is_ill = True
        for i in xrange(n_users):
            is_ill = True
            for j in xrange(n_items):
                if O[i,j] != 0:
                    O[i,j] = 1
                    is_ill = False
                    non_missing += 1
            if is_ill:
                break
        if not is_ill:
            #for i in xrange(n_users):
                #print O[i,:]
            return O, non_missing
        
if __name__ == '__main__':
    '''
    Set basic parameters
    '''
    n_users = 100
    n_items = 100
    n_factors = 10
    alpha = 0.01
    beta = 0.01
    
    # set learning rate: only used in Basic MF
    lamb = 0.001
    
    # generate ground truth matrix
    X, U, V = generate_non_negative_full_matrix(n_users, n_items, n_factors)
    O, non_missing = get_random_indicator_matrix(n_users, n_items, 0.1)
    sparse_X = X * O
    
    '''
    Test Basic MF
    '''
    #mf = BasicMF(alpha, beta, lamb, 10, sys.maxint, 5e-2)
    mf = BasicMF(alpha, beta, lamb, 10, 1000)
    U_dcp, V_dcp = mf.decomp(X)
    #print X[-1]
    #print np.dot(U_dcp, V_dcp.T)[-1]
    print 'Full version ... '
    #print 'MAE U:', float(1)/(n_users*n_factors) * np.sum(np.absolute(U_dcp - U))
    print 'MAE X: ', float(1)/(n_users*n_items) * np.sum(np.absolute(X-np.dot(U_dcp, V_dcp.T)))
    #print '#elements < 0: ', (np.dot(U_dcp, V_dcp.T)<0).sum()
    
    smf = Sparse_BasicMF(alpha, beta, lamb, 10, 2000, O)
    U_dcp, V_dcp = smf.decomp(sparse_X)
    print 'Sparse version ... '
    #print 'MAE U:', float(1)/(n_users*n_factors) * np.sum(np.absolute(U_dcp - U))
    print 'MAE X: ', float(1)/(n_users*n_items) * np.sum(np.absolute(X-np.dot(U_dcp, V_dcp.T)))
    print '[train] MAE O*X: ', float(1)/non_missing * np.sum(np.absolute(O*X-O*np.dot(U_dcp, V_dcp.T)))
    print '[test] MAE (1-O)*X: ', float(1)/(n_users*n_items-non_missing) * np.sum(np.absolute((1-O)*X-(1-O)*np.dot(U_dcp, V_dcp.T)))
    #print '#elements < 0: ', (np.dot(U_dcp, V_dcp.T)<0).sum()
    
    '''
    Test Non-negative MF
    '''
    #mf = NMF(alpha, beta, 10, sys.maxint, 5e-2)
    mf = NMF(alpha, beta, 10, 1000)
    U_dcp, V_dcp = mf.decomp(X)
    print 'Full version ... '
    #print 'MAE U: ', float(1)/(n_users*n_factors) * np.sum(np.absolute(U_dcp - U))
    print 'MAE X: ', float(1)/(n_users*n_items) * np.sum(np.absolute(X-np.dot(U_dcp, V_dcp.T)))
    
    snmf = Sparse_NMF(alpha, beta, 10, 2000, O)
    U_dcp, V_dcp = smf.decomp(sparse_X)
    print 'Sparse version ... '
    #print 'MAE U:', float(1)/(n_users*n_factors) * np.sum(np.absolute(U_dcp - U))
    print 'MAE X: ', float(1)/(n_users*n_items) * np.sum(np.absolute(X-np.dot(U_dcp, V_dcp.T)))
    print '[train] MAE O*X: ', float(1)/non_missing * np.sum(np.absolute(O*X-O*np.dot(U_dcp, V_dcp.T)))
    print '[test] MAE (1-O)*X: ', float(1)/(n_users*n_items-non_missing) * np.sum(np.absolute((1-O)*X-(1-O)*np.dot(U_dcp, V_dcp.T)))
    
    