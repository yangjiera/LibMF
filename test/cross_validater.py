import sys
sys.path.append(sys.path[0][0:-5]+'/model/')
import numpy as np

from dataset_loader import *
from mf import *
from evaluater import *

def cross_validate(dataset, cv=True):
    print 'Loading data...\n' + \
            '[warning]: users and items in test set much appear in training set, ' + \
            'otherwise the corresponding UI interaction records are removed.'
    for i in xrange(5):
        UI_matrix_train, UI_test, u_mapper, i_mapper = load_dataset(dataset, cv, i+1)
        O = get_mask_matrix(UI_matrix_train)
        
        alpha = 0.01 
        beta = 0.01
        lamb = 0.001
        mf = Sparse_BasicMF(alpha, beta, lamb, 10, 2000, O)
        
        print 'Decomposing...'
        U_dcp, V_dcp = mf.decomp(UI_matrix_train)
        
        evaluate(U_dcp, V_dcp, UI_test)
        
def get_mask_matrix(X):
    m,n = X.shape
    O = np.zeros((m, n))
    for i in xrange(m):
    	# DEPRECATED: this  loop takes time since it iterates over all entries of the data matrix
        '''for j in xrange(n):
            if X[i,j] != 0:
                O[i,j] = 1'''
        # IMPROVED:
        non_zero_indices = np.where(X[i] != 0)[0]
        for j in non_zero_indices:
        	O[i,j] = 1

    return O

if __name__ == '__main__':
    cross_validate('FourCity', True)