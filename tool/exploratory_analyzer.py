import numpy as np
from collections import Counter
from dataset_loader import *


if __name__ == '__main__':
    UI_matrix, u_mapper, i_mapper = load_dataset('FourCity', False, 0, "amsterdam")
    u_info = read_side_info('FourCity', u_mapper, 'u')
    v_info = read_side_info('FourCity', i_mapper, 'v')
    #print UI_matrix.shape
    