import numpy as np

def evaluate(U, V, UI_test):
    performance_at_1 = [[],[]]
    performance_at_5 = [[],[]]
    performance_at_10 = [[],[]]
    
    for i in UI_test:
        rating_predicted = []
        m,n = V.shape
        for j in xrange(m):
            rating_predicted.append([j, np.dot(U[i,:],V[j,:])])
        rating_predicted = sorted(rating_predicted, key=lambda rating_predicted : rating_predicted[1], reverse=True)
        performance_at_1 = append_precision_recall(performance_at_1, rating_predicted[0:1], UI_test[i])
        performance_at_5 = append_precision_recall(performance_at_5, rating_predicted[0:5], UI_test[i])
        performance_at_10 = append_precision_recall(performance_at_10, rating_predicted[0:10], UI_test[i])
        
    print '*****************************************************' +'\n' +\
          '  N    Precision    Recall\n' +\
          '  1    '+str(np.mean(performance_at_1[0]))+'    '+str(np.mean(performance_at_1[1]))+'\n' +\
          '  5    '+str(np.mean(performance_at_5[0]))+'    '+str(np.mean(performance_at_5[1]))+'\n' +\
          ' 10    '+str(np.mean(performance_at_10[0]))+'    '+str(np.mean(performance_at_10[1]))+ '\n' +\
          '*****************************************************'
        
def append_precision_recall(performance, rating_predicted, rating_gt):
    item_predicted = set([x[0] for x in rating_predicted])
    item_gt = set([x[0] for x in rating_gt])
    
    correctly_predicted = len(item_predicted.intersection(item_gt))
    precision = float(correctly_predicted)/len(item_predicted)
    recall = float(correctly_predicted)/len(item_gt)
    '''if correctly_predicted!= 0:
        print correctly_predicted, len(item_predicted), len(item_gt), precision, recall '''
    
    performance[0].append(precision)
    performance[1].append(recall)
    return performance