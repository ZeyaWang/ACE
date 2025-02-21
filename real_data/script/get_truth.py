import numpy as np
import sklearn
from sklearn import metrics
import os, h5py
import pickle as pk
from collections import defaultdict
import argparse
from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score

def clustering_accuracy(gtlabels, labels):
    #print(gtlabels, labels)
    gtlabels = np.array(gtlabels, dtype='int64')
    labels = np.array(labels, dtype='int64')
    cost_matrix = []
    categories = np.unique(gtlabels)
    nr = np.amax(labels) + 1
    for i in np.arange(len(categories)):
      cost_matrix.append(np.bincount(labels[gtlabels == categories[i]], minlength=nr))
    cost_matrix = np.asarray(cost_matrix).T
    row_ind, col_ind = linear_sum_assignment(np.max(cost_matrix) - cost_matrix)
    return float(cost_matrix[row_ind, col_ind].sum()) / len(gtlabels)


def calinski_harabasz_score(x,y):
    if len(np.unique(y)) > 1:
        return metrics.calinski_harabasz_score(x,y)
    else:
        return None

def davies_bouldin_score(x,y):
    if len(np.unique(y)) > 1:
        return - metrics.davies_bouldin_score(x,y)
    else:
        return None

def silhouette_score(x,y, metric):
    if len(np.unique(y)) > 1:
        return metrics.silhouette_score(x, y, metric=metric)
    else:
        return None


def clustering_score(x,y, metric):
    if metric == 'dav':
        return davies_bouldin_score(x,y)
    elif metric == 'ch':
        return calinski_harabasz_score(x,y)
    else:
        return silhouette_score(x, y, metric=metric)


def gen_value(feature, eig=True): 
    nn, pp = feature.shape
    TT = feature.T @ feature 
    if eig:
        eigenValues, _ = LA.eig(TT/(nn-1))
        ss = np.sqrt(eigenValues)
        ss[ss==0] = 1
        vv  = np.prod(ss)
    jeu = np.array(feature)
    md = metrics.pairwise_distances(jeu, metric='euclidean')
    cmd = metrics.pairwise_distances(jeu, metric='cosine')
    if eig:
        return jeu, TT, ss, vv, md, cmd
    else:
        return jeu, TT, md, cmd


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='COIL-20')
    parser.add_argument('--task', default='JULE_hyper')
    args = parser.parse_args()
    eval_data = args.dataset
    task = args.task
    tpath = 'true_{}.pkl'.format(eval_data)

    os.makedirs(os.path.join(task, 'external_metric'), exist_ok=True)
    tpath = os.path.join('external_metric', tpath)

    with open(os.path.join('file_list', task, "{}.txt".format(eval_data)), "r") as file:
        modelFiles = [line.strip() for line in file.readlines()]

    labels = {}
    tfname = 'datasets/{}/data4torch.h5'.format(eval_data)
    truth= np.squeeze(np.array(h5py.File(tfname, 'r')['labels']))

    if 'JULE' in task:
        for m in modelFiles:
            lfname = os.path.join(task, 'label{}.h5'.format(m))
            labels[m] = np.squeeze(np.array(h5py.File(lfname, 'r')['label']))
    else:
        for m in modelFiles:
            files = np.load(os.path.join(task,m))
            labels[m] = np.squeeze(np.array(files['y_pred']))

    nmv = {}
    acv = {}
    for key in modelFiles:
        y = labels[key]
        nmv[key] = normalized_mutual_info_score(truth, y)
        acv[key] = clustering_accuracy(truth, y)

    nmv = dict(sorted(nmv.items(), key=lambda item: item[0]))
    acv = dict(sorted(acv.items(), key=lambda item: item[0]))

    with open(tpath, 'wb') as ff:
        pk.dump([nmv, acv], ff)
