from sklearn import metrics
from munkres import Munkres, print_matrix
import numpy as np

# def cal_clustering_acc(true_label, pred_label):
#     l1 = list(set(true_label))
#     numclass1 = len(l1)

#     l2 = list(set(pred_label))
#     numclass2 = len(l2)
    
#     cost = np.zeros((numclass1, numclass2), dtype=int)
#     for i, c1 in enumerate(l1):
#         mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
#         for j, c2 in enumerate(l2):
#             mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
#             cost[i][j] = len(mps_d)

#     # match two clustering results by Munkres algorithm # Kuhn-Munkres or Hungarian Algorithm
#     m = Munkres()
#     cost = cost.__neg__().tolist()

#     indexes = m.compute(cost)

#     # get the match results
#     new_predict = np.zeros(len(pred_label))
#     for i, c in enumerate(l1):
#         # correponding label in l2:
#         c2 = l2[indexes[i][1]]

#         # ai is the index with label==c2 in the pred_label list
#         ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
#         new_predict[ai] = c

#     acc = metrics.accuracy_score(true_label, new_predict)
#     return acc

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def accuracy(truth, prediction):
    confusion_m = confusion_matrix(truth, prediction)
    _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    acc = np.trace(confusion_m[:, col_idx]) / confusion_m.sum()
    
    return acc


def cal_clustering_metric(truth, prediction):
    nmi = metrics.normalized_mutual_info_score(truth, prediction) #official
    ari = metrics.adjusted_rand_score(truth,prediction) #official
    #acc = cal_clustering_acc(truth,prediction)
    acc = accuracy(truth,prediction)
    ri  = metrics.rand_score(truth,prediction) #official
  
    return  nmi, ari, acc,ri
    