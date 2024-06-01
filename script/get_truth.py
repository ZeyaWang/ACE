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


def get_files_with_substring_and_suffix(directory, substring, suffix):
    files = []
    # Use os.listdir to get a list of all files in the directory
    all_files = os.listdir(directory)
    
    # Use a list comprehension to filter files based on the substring and suffix
    files = [file for file in all_files if substring in file and file.endswith(suffix)]
    
    return files



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


    modelpath = {
        'jule': '/home/zwa281/unsupeval/jule',
        'julenum': '/home/zwa281/unsupeval_num/julenum',
        'DEPICT': '/home/zwa281/DEPICT',
        'DEPICTnum': '/home/zwa281/DEPICTnum',
    }
    rootpath = {
        'jule': '/home/zwa281/unsupeval',
        'julenum': '/home/zwa281/unsupeval_num',
        'DEPICT': '/home/zwa281/DEPICT',
        'DEPICTnum': '/home/zwa281/DEPICTnum',
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='COIL-20')
    parser.add_argument('--task', default='jule')
    args = parser.parse_args()
    eval_data = args.dataset
    task = args.task
    tpath = 'true_{}.pkl'.format(eval_data)

    if 'jule' in task:
        modelFiles = get_files_with_substring_and_suffix(modelpath[task], 'feature'+eval_data, 'h5')
        modelFiles = [m[7:-3] for m in modelFiles]
    else:
        modelFiles = get_files_with_substring_and_suffix(modelpath[task], 'output'+eval_data, 'npz')

    labels = {}

    if 'jule' in task:
        tfname = '/home/zwa281/unsupeval/jule/datasets/{}/data4torch.h5'.format(eval_data)
        truth= np.squeeze(np.array(h5py.File(tfname, 'r')['labels']))
        for m in modelFiles:
            lfname = os.path.join(modelpath[task], 'label{}.h5'.format(m))
            labels[m] = np.squeeze(np.array(h5py.File(lfname, 'r')['label']))
    else:
        tfname = '/home/zwa281/DEPICT/datasets/{}/data.h5'.format(eval_data)
        truth=np.squeeze(np.array(h5py.File(tfname, 'r')['labels']))
        for m in modelFiles:
            files = np.load(os.path.join(modelpath[task],m))
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
