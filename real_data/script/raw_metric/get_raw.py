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
    parser.add_argument('--metric', default='euclidean')
    parser.add_argument('--task', default='jule')

    args = parser.parse_args()
    metric = args.metric
    eval_data = args.dataset
    task = args.task

    if not os.path.isdir(task):
        os.mkdir(task)
    tpath = os.path.join(task, 'raw_metric')
    if not os.path.isdir(tpath):
        os.mkdir(tpath)
    tmppath = os.path.join(task, 'raw_tmp')
    if not os.path.isdir(tmppath):
        os.mkdir(tmppath)

    with open(os.path.join('file_list', task, "{}.txt".format(eval_data)), "r") as file:
        modelFiles = [line.strip() for line in file.readlines()]

    labels = {}
    scored = defaultdict(dict)

    tfname = 'datasets/{}/data4torch.h5'.format(eval_data)
    truth= np.squeeze(np.array(h5py.File(tfname, 'r')['labels']))
    data=np.array(h5py.File(tfname, 'r')['data'])
    data = [data[i].flatten() for i in range(data.shape[0])]
    data = np.stack(data, axis=0)

    if 'jule' in task:
        for m in modelFiles:
            lfname = os.path.join(task, 'label{}.h5'.format(m))
            labels[m] = np.squeeze(np.array(h5py.File(lfname, 'r')['label']))
    else:
        for m in modelFiles:
            files = np.load(os.path.join(task,m))
            labels[m] = np.squeeze(np.array(files['y_pred']))

    print(task, eval_data, data.shape)

    if eval_data in ['COIL-100']:
        jeu0, TT0, md0, cmd0 = gen_value(data, False)
        vv0 = 0 
        ss0 = 0
    else:
        jeu0, TT0, ss0, vv0, md0, cmd0 = gen_value(data)
    
    np.savez(os.path.join(tmppath, 'data_{}.npz'.format(eval_data)), jeu=jeu0, TT=TT0, ss=ss0, vv=vv0, md=md0, cmd=cmd0)


    for key in modelFiles:
        outf = os.path.join(tmppath, 'key_{}.npz'.format(key))
        y = labels[key]
        scored[metric][key] = clustering_score(data, y, metric=metric)
        scored['nmi'][key] = normalized_mutual_info_score(truth, y)
        scored['acc'][key] = clustering_accuracy(truth, y)
        np.savez(outf, labelset=np.array(y), jeu=jeu0, TT=TT0, ss=ss0, vv=vv0, md=md0, cmd=cmd0)


    with open(os.path.join(tpath,'merge_{}_{}_score.pkl'.format(eval_data, metric)), 'wb') as file:
        pk.dump(scored, file)
