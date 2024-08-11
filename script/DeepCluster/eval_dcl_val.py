import numpy as np
import os, sys
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import scipy.stats as stats
import glob
from scipy.optimize import linear_sum_assignment
import pickle
import argparse
from joblib import parallel_backend


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


def clustering_accuracy(gtlabels, labels):
    cost_matrix = []
    categories = np.unique(gtlabels)
    nr = np.amax(labels) + 1
    for i in np.arange(len(categories)):
      cost_matrix.append(np.bincount(labels[gtlabels == categories[i]], minlength=nr))
    cost_matrix = np.asarray(cost_matrix).T
    row_ind, col_ind = linear_sum_assignment(np.max(cost_matrix) - cost_matrix)
    return float(cost_matrix[row_ind, col_ind].sum()) / len(gtlabels)


parser = argparse.ArgumentParser()
parser.add_argument('--id', default='4', help="the index of checkpiont to be saved")
parser.add_argument('--metric', default='euclidean', help="metrics to be saved; have to be in [cosine, euclidean, dav, ch]")
args = parser.parse_args()


estimates = {}
#for file1 in os.listdir('npfiles_val'):
#    if file1.endswith('npz') and file1.startswith('pro'):
        #modelFiles.append()

for ii in range(4, 100, 5):
    file2 = os.path.join('npfiles_val', 'pro_output_{}.npz'.format(ii))
    print(file2)
    data = np.load(file2)
    estimates[str(ii)] = np.squeeze(data['estimates'])

print('reading')
run_file='pro_output_{}.npz'.format(args.id)
print(run_file)
run_path = os.path.join('npfiles_val', run_file)
data = np.load(run_path)
features = data['pro_features']
labels = data['labels']
estimate = data['estimates']

true_nmi = nmi(labels, estimate)
true_acc = clustering_accuracy(labels, estimate)


print(features.shape)
metric = args.metric#'euclidean' # 'cosine'
sil_eu = {}
for key, value in estimates.items():
    print(key)
    sil_eu[key] = clustering_score(features, value, metric=metric)


saved_objects = [sil_eu, true_nmi, true_acc]


with open(os.path.join('DeepCluster', 'saved_{}_{}.pkl'.format(args.id, args.metric)), 'wb') as f:
    pickle.dump(saved_objects, f)





