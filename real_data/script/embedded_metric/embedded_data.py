import numpy as np
import sklearn
from sklearn import metrics
import os, h5py
import pickle as pk
from collections import defaultdict
import argparse
from numpy import linalg as LA


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


    modelpath = {
        'jule': 'JULE_hyper',
        'julenum': 'JULE_num',
        'DEPICT': 'DEPICT_hyper',
        'DEPICTnum': 'DEPICT_num',
    }
    rootpath = {
        'jule': 'JULE_hyper',
        'julenum': 'JULE_num',
        'DEPICT': 'DEPICT_hyper',
        'DEPICTnum': 'DEPICT_num',
    }


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
    tpath = os.path.join(task, 'embedded_data')
    if not os.path.isdir(tpath):
        os.mkdir(tpath)
    tmppath = os.path.join(task, 'tmp')
    if not os.path.isdir(tmppath):
        os.mkdir(tmppath)

    with open(os.path.join('file_list', task, "{}.txt".format(eval_data)), "r") as file:
        modelFiles = [line.strip() for line in file.readlines()]
    
    features={}
    labels={}
    scored = defaultdict(dict)

    if 'jule' in task:
        for m in modelFiles:
            ffname = os.path.join(modelpath[task], 'feature{}.h5'.format(m))
            lfname = os.path.join(modelpath[task], 'label{}.h5'.format(m))
            features[m] = np.array(h5py.File(ffname, 'r')['feature'])
            labels[m] = np.squeeze(np.array(h5py.File(lfname, 'r')['label']))
    else:
        for m in modelFiles:
            files = np.load(os.path.join(modelpath[task],m))
            features[m] = np.array(files['y_features'])
            labels[m] = np.squeeze(np.array(files['y_pred']))
    
    for m in modelFiles:
        x = features[m]
        jeu0, TT0, ss0, vv0, md0, cmd0 = gen_value(x)
        np.savez(os.path.join(tmppath, '{}.npz'.format(m)), jeu=jeu0, TT=TT0, ss=ss0, vv=vv0, md=md0, cmd=cmd0)
        for key in labels.keys():
            y = labels[key]
            score_local = clustering_score(x, y, metric=metric)
            scored[m][key] = score_local
            labelset = np.array(y)
            np.savez(os.path.join(tmppath, '{}_{}.npz'.format(m, key)), labelset=labelset)


    with open(os.path.join(tpath,'merge_{}_{}_score.pkl'.format(eval_data, metric)), 'wb') as file:
        pk.dump(scored, file)
