import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from munkres import Munkres
from sklearn import metrics
from .kmeans_gpu import *
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


def clustering(feature, true_labels, cluster_num):
    predict_labels, _ = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda")
    acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)
    return round(100 * acc, 2), round(100 * nmi, 2), round(100 * ari, 2), round(100 * f1, 2), predict_labels.numpy()


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, show_details=True):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1

# def kmeans(X, y, n_clusters, repeat=10):
#     nmi_list = []
#     ari_list = []
#     acc_list = []
#     f1_list = []
#
#     for i in range(repeat):
#         seed = i
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         kmeans = KMeans(n_clusters=n_clusters, random_state=i)
#         y_pred = kmeans.fit_predict(X.cpu().numpy())
#
#         y_pred = torch.tensor(y_pred)  # 将y_pred转换为张量以便后续计算
#         nmi_score = normalized_mutual_info_score(y.cpu().numpy(), y_pred.numpy(), average_method='arithmetic')
#         ari_score = adjusted_rand_score(y.cpu().numpy(), y_pred.numpy())
#         acc_score, f1_score_macro = cluster_acc(y.cpu().numpy(), y_pred.numpy())
#
#         nmi_list.append(nmi_score)
#         ari_list.append(ari_score)
#         acc_list.append(acc_score)
#         f1_list.append(f1_score_macro)
#
#     nmi_mean = np.mean(nmi_list)
#     ari_mean = np.mean(ari_list)
#     acc_mean = np.mean(acc_list)
#     f1_mean = np.mean(f1_list)
#
#     print('\t[Clustering] ACC: {:.4f}'.format(acc_mean))
#     print('\t[Clustering] NMI: {:.4f}'.format(nmi_mean))
#     print('\t[Clustering] ARI: {:.4f}'.format(ari_mean))
#     print('\t[Clustering] F1: {:.4f}'.format(f1_mean))
#
#     return {
#         'ACC': acc_mean,
#         'NMI': nmi_mean,
#         'ARI': ari_mean,
#         'F1': f1_mean
#     }
#
#
# def cluster_acc(y_true, y_pred):
#
#     y_true = y_true - np.min(y_true)
#     l1 = list(set(y_true))
#     num_class1 = len(l1)
#     l2 = list(set(y_pred))
#     num_class2 = len(l2)
#     ind = 0
#     if num_class1 != num_class2:
#         for i in l1:
#             if i in l2:
#                 pass
#             else:
#                 y_pred[ind] = i
#                 ind += 1
#     l2 = list(set(y_pred))
#     numclass2 = len(l2)
#     if num_class1 != numclass2:
#         print('error')
#         return
#     cost = np.zeros((num_class1, numclass2), dtype=int)
#     for i, c1 in enumerate(l1):
#         mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
#         for j, c2 in enumerate(l2):
#             mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
#             cost[i][j] = len(mps_d)
#     m = Munkres()
#     cost = cost.__neg__().tolist()
#     indexes = m.compute(cost)
#     new_predict = np.zeros(len(y_pred))
#     for i, c in enumerate(l1):
#         c2 = l2[indexes[i][1]]
#         ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
#         new_predict[ai] = c
#     acc = metrics.accuracy_score(y_true, new_predict)
#     f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
#     return acc, f1_macro

