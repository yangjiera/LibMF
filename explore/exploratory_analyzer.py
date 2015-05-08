import numpy as np
from collections import Counter
import sys
sys.path.append(sys.path[0][0:-8] + '/tool/')
from dataset_loader import *


def write_item_per_user(UI_matrix):
    f = open('item_per_user', 'w')
    m, n = UI_matrix.shape
    no_interacted_list = []
    for i in xrange(m):
        no_interacted = np.count_nonzero(UI_matrix[i, :])
        f.write(str(no_interacted) + '\n')
        no_interacted_list.append(no_interacted)
    f.close()
    print 'no.ratings per user: ', np.mean(no_interacted_list), '/', \
        np.median(no_interacted_list), '+/-', np.std(no_interacted_list)


if __name__ == '__main__':
    dataset =  'Movielens/ml-100k'#'FourCity5'
    #UI_matrix, u_mapper, i_mapper = load_dataset(
        #dataset, False, 0, "london")
    UI_matrix, u_mapper, i_mapper = load_dataset(dataset, False, 0)
    u_info = read_side_info(dataset, u_mapper, 'u')
    v_info = read_side_info(dataset, i_mapper, 'v')
    m, n = UI_matrix.shape
    print '#users: ', m, '; #items: ', n
    print '#ratings: ', np.count_nonzero(UI_matrix), '; sparsity: ', \
        float(np.count_nonzero(UI_matrix)) / (m * n)
    write_item_per_user(UI_matrix)
