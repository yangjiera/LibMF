import numpy as np
import csv
import sys


def load_dataset(name, cv=False, iteration=0, filter=""):
    if not cv:
        UI_matrix, u_mapper, i_mapper = read_UI_matrix(name, 0, filter)
        return UI_matrix, u_mapper, i_mapper

    else:
        UI_matrix_train, u_mapper, i_mapper = read_UI_matrix(
            name, iteration, filter)
        # print '#users in training set: ', len(u_mapper)
        UI_test = dict([])
        with open("../dataset/" + name + '/u' + str(iteration) + '.test', 'r') as csvfile:
            contentreader = csv.reader(csvfile, delimiter='\t')
            for line in contentreader:
                uid = line[0]
                iid = line[1]
                rating = line[2]
                if uid not in u_mapper or iid not in i_mapper:
                    continue

                if u_mapper[uid] in UI_test:
                    UI_test[u_mapper[uid]].append([i_mapper[iid], rating])
                else:
                    UI_test[u_mapper[uid]] = [[i_mapper[iid], rating]]

        for u in UI_test:
            i_rank = UI_test[u]
            UI_test[u] = sorted(
                i_rank, key=lambda i_rank: i_rank[1], reverse=True)
        # print '#users in test set: ', len(UI_test)
        return UI_matrix_train, UI_test, u_mapper, i_mapper


def read_UI_matrix(name, iteration=0, filter=""):
    u_mapper = dict([])
    i_mapper = dict([])
    u_index = 0
    i_index = 0
    rating_tuples = []

    if iteration == 0:
        fname = "../dataset/" + name + '/u.data'
    else:
        fname = "../dataset/" + name + '/u' + str(iteration) + '.base'

    with open(fname, 'r') as csvfile:
        contentreader = csv.reader(csvfile, delimiter='\t')
        for line in contentreader:
            uid = line[0]
            iid = line[1]
            rating = line[2]

            if filter != "":
                if line[4] != filter:
                    continue

            if uid not in u_mapper:
                u_mapper[uid] = u_index
                u_index += 1
            if iid not in i_mapper:
                i_mapper[iid] = i_index
                i_index += 1

            rating_tuples.append([u_mapper[uid], i_mapper[iid], rating])

    UI_matrix = np.zeros((u_index, i_index))
    for rt in rating_tuples:
        UI_matrix[rt[0], rt[1]] = rt[2]

    return UI_matrix, u_mapper, i_mapper


def read_side_info(name, mapper, object):
    if object == 'u':
        fname = "../dataset/" + name + '/u.user'
    elif object == 'v':
        fname = "../dataset/" + name + '/u.item'
    else:
        print 'Error: please select the right object for reading side information'
        sys.exit(0)
    info_dict = dict([])
    with open(fname, 'r') as csvfile:
        contentreader = csv.reader(csvfile, delimiter='|')
        for info_list in contentreader:
            oid = info_list[0]
            if oid in mapper:
                info_dict[mapper[oid]] = info_list[1:]
    return info_dict

if __name__ == '__main__':
    for i in xrange(5):
        UI_matrix_train, UI_test, u_mapper, i_mapper = load_dataset(
            'FourCity', True, i + 1, "")
    u_info = read_side_info('FourCity', u_mapper, 'u')
    print len(u_info), u_mapper['5976'], u_info[0]
