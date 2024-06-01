import h5py
import numpy as np
import os, sys
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import scipy.stats as stats
import pandas as pd
import glob
import dill
from utility import *
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle as pk

critd = {'tau': '${}tau_B$'.format(chr(92)), 'cor': '$r_s$'}


def kendalltau(score1, score2):
    stat, pval = stats.kendalltau(score1, score2)
    return np.round(stat, 3), pval

def spearmanr(score1, score2):
    stat, pval = stats.spearmanr(score1, score2)
    return np.round(stat, 3), pval

def transpose(df):
    transposed_df = df.T
    transposed_df.columns = transposed_df.iloc[0]
    # Drop the first row (original column headers)
    transposed_df = transposed_df.drop(transposed_df.index[0])
    return transposed_df

def create_subcol(colnames, vertical):
    if vertical:
        prefix = [(c.split('=')[1], critd[c.split('=')[0]]) for c in colnames if c != 'metric']
    else:
        prefix = [('metric','')]+[(c.split('=')[1], critd[c.split('=')[0]]) for c in colnames if c != 'metric']
    return prefix


def sileval(ffname, lfname, tfname, metric='cosine'):
    f1=np.array(h5py.File(ffname, 'r')['feature'])
    f2=np.array(h5py.File(lfname, 'r')['label'])
    f3=np.array(h5py.File(tfname, 'r')['labels'])
    d=np.array(h5py.File(tfname, 'r')['data'])
    d = [d[i].flatten() for i in range(d.shape[0])]
    #print(d[0].shape)
    d = np.stack(d, axis=0)
    #print(d.shape, f1.shape)
    sil = metrics.silhouette_score(f1,np.squeeze(f2),metric=metric)
    sil0 = metrics.silhouette_score(d,np.squeeze(f2),metric=metric)
    n = nmi(f3, np.squeeze(f2))
    return round(sil,3), round(n,3), round(sil0,3)


def eval_raw_jule(modelFiles, dataset='UCOIL-20', metric='cosine'):
    tfname = 'jule/datasets/{}/data4torch.h5'.format(dataset)
    result = {}
    for m in modelFiles:
        ffname = 'jule/feature{}.h5'.format(m)
        lfname = 'jule/label{}.h5'.format(m)
        sl, nv, sr = sileval(ffname,lfname,tfname, metric=metric)
        #print(sl, nv, sr)
        result[m] = [sl, nv, sr]
    return result

def eval_vote0(modelFiles, metric='cosine'):
    features = {}
    labels = {}
    for m in modelFiles:
        ffname = 'jule/feature{}.h5'.format(m)
        lfname = 'jule/label{}.h5'.format(m)
        f1=h5py.File(ffname, 'r')['feature']
        f2=np.squeeze(h5py.File(lfname, 'r')['label'])
        features[m] = f1
        labels[m] = f2
    # eval feature 
    feat_score = {}
    for m in modelFiles:
        feature = features[m]
        sil = 0
        for n in modelFiles:
            label = labels[n]
            sil += metrics.silhouette_score(feature,label,metric=metric)/len(modelFiles)
        feat_score[m] = sil
    # eval label 
    label_score = {}
    for m in modelFiles:
        label = labels[m]
        sil = 0
        for n in modelFiles:
            feature = features[n]
            sil += metrics.silhouette_score(feature,label,metric=metric)/len(modelFiles)
        label_score[m] = sil
    return feat_score, label_score

def eval_vote_final(modelFiles, metric='cosine'):
    features = {}
    labels = {}
    for m in modelFiles:
        ffname = 'jule/feature{}.h5'.format(m)
        lfname = 'jule/label{}.h5'.format(m)
        f1=h5py.File(ffname, 'r')['feature']
        f2=np.squeeze(h5py.File(lfname, 'r')['label'])
        features[m] = f1
        labels[m] = f2
    # eval feature 
    feat_score = {}
    for m in modelFiles:
        feature = features[m]
        sil = []
        for n in modelFiles:
            label = labels[n]
            sil.append(metrics.silhouette_score(feature,label,metric=metric))
        sil = np.array(sil)
        feat_score[m] = sil.max()
    # eval label 
    label_score = {}
    for m in modelFiles:
        label = labels[m]
        sil = []
        for n in modelFiles:
            feature = features[n]
            sil.append(metrics.silhouette_score(feature,label,metric=metric))
        sil = np.array(sil)
        label_score[m] = sil.max()
    return feat_score, label_score



def eval_vote_eff(modelFiles, metric='cosine'):
    features = {}
    labels = {}
    for m in modelFiles:
        files = np.load(m)
        f1=np.array(files['y_features'])
        f2=np.squeeze(np.array(files['y_pred']))
        features[m] = f1
        labels[m] = f2

    scores = []
    # get score
    for m in modelFiles:
        feature = features[m]
        scores_row = []
        for n in modelFiles:
            label = labels[n]
            #print(m,n)
            scores_row.append(metrics.silhouette_score(feature,label,metric=metric))
        scores.append(scores_row)
    scores = np.array(scores)
    # eval feature 
    feat_score = {}
    for i, m in enumerate(modelFiles):
        feat_score[m] = scores[i,:].max()
    # eval label 
    label_score = {}
    for i, m in enumerate(modelFiles):
        label_score[m] = scores[:,i].max()
    return feat_score, label_score, scores


def eval_vote_again(modelFiles, scores):
    # eval label 
    label_score = {}
    for i, m in enumerate(modelFiles):
        label_score[m] = scores[:,i].mean()
    return label_score

def get_files_with_substring_and_suffix(directory, substring, suffix):
    files = []
    # Use os.listdir to get a list of all files in the directory
    all_files = os.listdir(directory)
    
    # Use a list comprehension to filter files based on the substring and suffix
    files = [file for file in all_files if substring in file and file.endswith(suffix)]
    
    return files


def eval_vote_solve2(modelFiles, scores, remove_outlier=True, partition=True):
    # eval feature
    affinity, pvals, index = affinity_matrix(scores)
    #affinity1 = correct_matrix(affinity, pvals, index, alpha=0.05)
    #threshold = 1-affinity1[np.nonzero(affinity1)].min()
    #print('++++++++++++', threshold, affinity1)
    affinity[affinity < 0] = 0
    #print(affinity.shape)
    affinity = correct_matrix(affinity, pvals, index, alpha=0.05)
    # cores, _, _ = dbscan_graph(affinity, min_samples=3)
    cores, outliers, labels = dbscan_graph(affinity, eps=0.2, min_samples=5)
    #print('=========', labels, outliers)
    outliers, labels = hdbscan_graph(affinity, min_samples=3)
    #print('-------', labels, outliers)
    val_map = {}
    for c in cores:
        val_map[c] = 1
    for o in outliers:
        val_map[o] = 0
    graph = nx.Graph(affinity)
    #nx.draw(G)
    #plt.savefig("graph.png")
    #bisec = graph_bipartition(affinity)
    #print(cores, outliers, affinity.shape)
    values = [val_map.get(node, 0.25) for node in graph.nodes()]
    if partition:
        npart = np.unique(labels)
        if len(npart) > 2: # multiple paritions
            result = {}
            nparts = {}
            parts = {}
            coredots = {}
            prvs = {} # page rank value
            for n in npart:
                if n != -1:
                    part = np.where(labels == n)[0]
                    aff = affinity[part, :]
                    aff = aff[:, part]
                    sc = scores[part, :]
                    models  = [m for i,m in enumerate(modelFiles) if i in part]
                    pr = page_rank(aff)
                    prv = dict(zip(models, pr))
                    solve_score = {}
                    for i, m in enumerate(modelFiles):
                        solve_score[m] = np.sum(sc[:, i] * pr)/np.sum(pr)
                    sum_score = sum(solve_score.values())
                    result[sum_score] = solve_score
                    parts[sum_score] = part
                    coredots[sum_score] = np.intersect1d(cores, part)
                    nparts[sum_score] = n
                    prvs[sum_score] = prv
            best_part = max(result.keys())
            solve_score = result[best_part]
            cores = coredots.pop(best_part)
            best_n = nparts[best_part]
            prvv = prvs[best_part]
            return solve_score, graph, values, cores, outliers, coredots, labels, best_n, prvv
        elif len(npart) == 2: # should be same as remove outlier
            part = np.where(labels == 0)[0]
            affinity = affinity[part,:]
            affinity = affinity[:,part]
            #print(part, affinity.shape)
            scores = scores[part,:]
            models = [m for i,m in enumerate(modelFiles) if i in part]
        else:
            models = [m for i,m in enumerate(modelFiles)]
    else:
        if remove_outlier:
            affinity = np.delete(affinity, outliers, axis=0)
            affinity = np.delete(affinity, outliers, axis=1)
            scores = np.delete(scores, outliers, axis=0)
            models = [m for i,m in enumerate(modelFiles) if i not in outliers]
    pr = page_rank(affinity)
    prv = dict(zip(models, pr))
    solve_score = {}
    for i, m in enumerate(modelFiles):
        solve_score[m] = np.sum(scores[:,i]*pr)/np.sum(pr)
    return solve_score, graph, values, cores, outliers, None, labels, 0, prv


def eval_final(modelFiles, scores, remove_outlier=True, partition=True):
    # eval feature
    affinity, pvals, index = affinity_matrix(scores)
    affinity[affinity < 0] = 0
    print(affinity.shape)
    affinity = correct_matrix(affinity, pvals, index, alpha=0.01)
    # cores, _, _ = dbscan_graph(affinity, min_samples=3)
    cores, outliers, labels = dbscan_graph(affinity, eps=0.1, min_samples=10)
    #print('=========', labels, outliers)
    #outliers, labels = hdbscan_graph(affinity, min_samples=3)
    #print('-------', labels, outliers)
    graph = nx.Graph(affinity)
    color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'orange', 5: 'yellow'}
    values = [color_map[v] if v >= 0 else 'grey' for i, v in enumerate(labels)]
    #print(affinity)
    if partition:
        npart = np.unique(labels[labels >= 0])
        if len(npart) > 1: # multiple paritions
            result, nparts, prvs = {}, {}, {} # page rank value
            for n in npart:
                part = np.where(labels == n)[0]
                aff = affinity[part, :]
                aff = aff[:, part]
                sc = scores[part, :]
                models  = [m for i,m in enumerate(modelFiles) if i in part]
                pr = page_rank(aff)
                solve_score = {}
                for i, m in enumerate(modelFiles):
                    solve_score[m] = np.sum(sc[:, i] * pr)/np.sum(pr)
                sum_score = sum(solve_score.values())
                nparts[sum_score] = n
                result[n] = solve_score
                prvs[n] = dict(zip(models, pr))
            best_part = max(nparts.keys())
            best_n = nparts[best_part]
            solve_score = result[best_n]
            prv = prvs[best_n]
            return solve_score, graph, outliers, labels, best_n, prv, values
        elif len(npart) == 1: # one partition; should be same as remove outlier
            part = np.where(labels == 0)[0]
            affinity = affinity[part,:]
            affinity = affinity[:,part]
            scores = scores[part,:]
            models = [m for i,m in enumerate(modelFiles) if i in part]
            best_n = 0
        else: # all outliers
            models = [m for i,m in enumerate(modelFiles)]
            best_n = -1
    else:
        if remove_outlier:
            affinity = np.delete(affinity, outliers, axis=0)
            affinity = np.delete(affinity, outliers, axis=1)
            scores = np.delete(scores, outliers, axis=0)
            models = [m for i,m in enumerate(modelFiles) if i not in outliers]
            best_n = None
    pr = page_rank(affinity)
    prv = dict(zip(models, pr))
    solve_score = {}
    for i, m in enumerate(modelFiles):
        solve_score[m] = np.sum(scores[:,i]*pr)/np.sum(pr)
    return solve_score, graph, outliers, labels, best_n, prv, values

def filter_out(pvalues, models, method='holm',alpha=0.1):
    r, _, _, _ = multipletests(pvalues, method=method, alpha=alpha)
    keep_models = [m for i, m in enumerate(models) if r[i]]
    return keep_models

# # def eval_final_v3(modelFiles, scores, eps=0.05, alpha=0.01, cl_method='dbscan', rank_method='pr'):
#     # rows_to_remove = np.all(scores[:, 1:] == scores[:, :-1], axis=1)
#     # scores = scores[~rows_to_remove]
#     affinity, pvals, index = affinity_matrix(scores)
#     print(affinity.shape)
#     #print(affinity)
#     min_samples = 2
#     if cl_method == 'dbscan':
#         cores, outliers, labels = dbscan_graph(affinity, eps=eps, min_samples=min_samples)
#     else:
#         outliers, labels = hdbscan_graph(affinity, min_samples=min_samples)
#     affinity = correct_matrix(affinity, pvals, index, alpha=alpha)
#     graph = nx.Graph(affinity)


#     npart = np.unique(labels[labels >= 0])
#     labels_updated = labels.copy()
#     if len(npart) > 0:
#         for n in npart:
#             part = np.where(labels == n)[0]
#             sc = scores[part, :]
#             if len(sc) > 5:
#                 # for each partition we cluster based on the scale
#                 dist = metrics.pairwise_distances(sc, metric='euclidean').astype('double')
#                 outliers_d, labels_d = hdbscan_graph(1-dist, min_samples=min_samples)
#                 for l, lb in enumerate(labels_d):
#                     if lb == -1:
#                         labels_updated[part[l]] = -1
#                     else:
#                         labels_updated[part[l]] = (n+1)*1000+lb
#     #print(labels_updated)
#     # calculate outlier
#     part_out = np.where(labels_updated == -1)[0]
#     if len(part_out) > 0:
#         result_out, parts_out = {}, {}
#         for n in part_out:
#             sc = scores[n, :]
#             solve_score = {}
#             for i, m in enumerate(modelFiles):
#                 solve_score[m] = sc[i]
#             sum_score = sum(solve_score.values())
#             parts_out[sum_score] = n
#             result_out[n] = solve_score
#         best_out_score = max(parts_out.keys())
#         best_out = parts_out[best_out_score]
#         solve_out_score = result_out[best_out]
#         #print('best outlier', best_out, best_out_score)

#     npart = np.unique(labels_updated[labels_updated >= 0])
#     if len(npart) > 0:
#         result, nparts, prvs, nparts_id = {}, {}, {}, {}  # page rank value
#         for n in npart:
#             part = np.where(labels_updated == n)[0]
#             aff = affinity[part, :]
#             aff = aff[:, part]
#             sc = scores[part, :]
#             models  = [m for i,m in enumerate(modelFiles) if i in part]
#             pr = rank(aff, method=rank_method)
#             solve_score = {}
#             for i, m in enumerate(modelFiles):
#                 solve_score[m] = np.sum(sc[:, i] * pr)/np.sum(pr)
#             sum_score = sum(solve_score.values())
#             nparts[sum_score] = n
#             result[n] = solve_score
#             nparts_id[n] = part
#             prvs[n] = dict(zip(models, pr))
#         best_part_score = max(nparts.keys())
#         best_n = nparts[best_part_score]
#         solve_score = result[best_n]
#         prv = prvs[best_n]
#         #print('best n', best_n, nparts, best_part_score, nparts_id)

#         if len(part_out) == 0:
#             return solve_score, graph, outliers, labels_updated, best_n, prv, labels
#         else:
#             if best_part_score > best_out_score:
#                 return solve_score, graph, outliers, labels_updated, best_n, prv, labels
#             else:
#                 solve_value = dict(sorted(solve_score.items(), key=lambda item: item[0]))
#                 solve_out_value = dict(sorted(solve_out_score.items(), key=lambda item: item[0]))
#                 solve_value = list(solve_value.values())
#                 solve_out_value = list(solve_out_value.values())
#                 _, pval = stats.ttest_rel(solve_out_value, solve_value, alternative='greater')
#                 if pval < 0.05:
#                     return solve_out_score, graph, outliers, labels_updated, best_out, modelFiles[best_out], labels
#                 else:
#                     return solve_score, graph, outliers, labels_updated, best_n, prv, labels
#     else:
#         return solve_out_score, graph, outliers, labels_updated, best_out, modelFiles[best_out], labels



# # def eval_final_v5(modelFiles, scores, spaceFiles, eps=0.05, alpha=0.01, cl_method='dbscan', rank_method='pr', remove_outlier=True):
#     rows_to_remove = np.all(scores[:, 1:] == scores[:, :-1], axis=1)
#     spaceFiles = np.array(spaceFiles)
#     spaceFiles = spaceFiles[~rows_to_remove]
#     scores = scores[~rows_to_remove]
#     affinity, pvals, index = affinity_matrix(scores)
#     print(affinity.shape)
#     #print(affinity)
#     min_samples = 2
#     if cl_method == 'dbscan':
#         cores, outliers, labels = dbscan_graph(affinity, eps=eps, min_samples=min_samples)
#     else:
#         outliers, labels = hdbscan_graph(affinity, min_samples=min_samples)
#     affinity = correct_matrix(affinity, pvals, index, alpha=alpha)
#     graph = nx.Graph(affinity)
#     labels_updated = labels.copy()
#     # if remove_outlier:
#     #     labels_updated[labels_updated == -1] = -2
#     npart = np.unique(labels[labels >= 0])
#     if len(npart) > 0:
#         for n in npart:
#             part = np.where(labels == n)[0]
#             sc = scores[part, :]
#             if len(sc) > 5:
#                 # for each partition we cluster based on the scale
#                 dist = metrics.pairwise_distances(sc, metric='euclidean').astype('double')
#                 outliers_d, labels_d = hdbscan_graph(1-dist, min_samples=min_samples)
#                 out_idx = 1 # for outlier
#                 for l, lb in enumerate(labels_d):
#                     if lb == -1:
#                         labels_updated[part[l]] = (n+1)*100000 + out_idx
#                         out_idx = out_idx + 1
#                     else:
#                         labels_updated[part[l]] = (n+1)*1000+lb
#     #print(labels_updated)
#     # calculate outlier
#     dlabels = dict(zip(spaceFiles, labels))
#     dlabels_updated = dict(zip(spaceFiles, labels_updated))
#     part_out = np.where(labels_updated == -1)[0]
#     if len(part_out) > 0:
#         result_out, parts_out = {}, {}
#         for n in part_out:
#             sc = scores[n, :]
#             solve_score = {}
#             for i, m in enumerate(modelFiles):
#                 solve_score[m] = sc[i]
#             sum_score = sum(solve_score.values())
#             parts_out[sum_score] = n
#             result_out[n] = solve_score
#         best_out_score = max(parts_out.keys())
#         best_out = parts_out[best_out_score]
#         solve_out_score = result_out[best_out]
#         #print('best outlier', best_out, best_out_score)

#     npart = np.unique(labels_updated[labels_updated >= 0])
#     if len(npart) > 0:
#         result, nparts, prvs, nparts_id = {}, {}, {}, {}  # page rank value
#         for n in npart:
#             part = np.where(labels_updated == n)[0]
#             solve_score = {}
#             if len(part) > 1:
#                 aff = affinity[part, :]
#                 aff = aff[:, part]
#                 sc = scores[part, :]
#                 pr = rank(aff, method=rank_method)
#                 for i, m in enumerate(modelFiles):
#                     solve_score[m] = np.sum(sc[:, i] * pr)/np.sum(pr)
#             else:
#                 sc = scores[part[0], :]
#                 for i, m in enumerate(modelFiles):
#                     solve_score[m] = sc[i]
#                 pr = [1]
#             sum_score = sum(solve_score.values())
#             nparts[sum_score] = n
#             result[n] = solve_score
#             nparts_id[n] = part
#             prvs[n] = dict(zip(part, pr))
#         best_part_score = max(nparts.keys())
#         best_n = nparts[best_part_score]
#         solve_score = result[best_n]
#         prv = prvs[best_n]
#         #print('best n', best_n, nparts, best_part_score, nparts_id)

#         if len(part_out) == 0:
#             return solve_score, graph, outliers, dlabels_updated, best_n, prv, dlabels
#         else:
#             if (best_part_score > best_out_score) or remove_outlier:
#                 return solve_score, graph, outliers, dlabels_updated, best_n, prv, dlabels
#             else:
#                 solve_value = dict(sorted(solve_score.items(), key=lambda item: item[0]))
#                 solve_out_value = dict(sorted(solve_out_score.items(), key=lambda item: item[0]))
#                 solve_value = list(solve_value.values())
#                 solve_out_value = list(solve_out_value.values())
#                 _, pval = stats.ttest_rel(solve_out_value, solve_value, alternative='greater')
#                 if pval < 0.05:
#                     return solve_out_score, graph, outliers, dlabels_updated, best_out, spaceFiles[best_out], dlabels
#                 else:
#                     return solve_score, graph, outliers, dlabels_updated, best_n, prv, dlabels
#     else:
#         return solve_out_score, graph, outliers, dlabels_updated, best_out, spaceFiles[best_out], dlabels



def sort_and_match(obj, modelFiles):
    new_obj = {key: value for key, value in obj.items() if key in modelFiles}
    new_obj = dict(sorted(new_obj.items(), key=lambda item: item[0]))
    key_obj = list(new_obj.keys())
    val_obj = np.squeeze(np.array(list(new_obj.values())))
    return new_obj, key_obj, val_obj

if __name__ == '__main__':
    l = 0
    num = 20

    nmi_tau, acc_tau, nmi_cor, acc_cor = [], [], [], [] 

    for metric in ['cosine', 'euclidean', 'dav', 'ch']:
        l = l + 1
        nmv = []
        acv = []
        modelFiles = []
        pair_scores = np.zeros((num, num))
        for i in range(num):
            fname = 'saved_{}_{}.pkl'.format(i*5+4, metric)
            with open(fname, 'rb') as f:
                silds, nm, ac = pk.load(f)
            for j in range(num):
                #kname = 'pro_output_{}.npz'.format(j*5+4)
                kname = '{}'.format(j*5+4)
                #print(fname, kname)
                pair_scores[i,j] = silds[kname]
            nmv.append(np.round(nm,3))
            acv.append(np.round(ac,3))
            modelFiles.append('model{}'.format(i))
            #print(fname)
        #print(pair_scores)
        #print(nmv)
        #print(acv)
        print(modelFiles)
        nmv = dict(zip(modelFiles, nmv))
        acv = dict(zip(modelFiles, acv))

        sv = np.diag(pair_scores)
        sv = dict(zip(modelFiles, sv))

        nmvd, _, nmv = sort_and_match(nmv, modelFiles)
        acvd, _, acv = sort_and_match(acv, modelFiles)
        svd, _, sv = sort_and_match(sv, modelFiles)
        #print(nmv, acv,sv)

        label_score = eval_vote(modelFiles, pair_scores)

        # label_score = dict(sorted(label_score.items(), key=lambda item: item[0]))
        # label_score = list(label_score.values())

        lsd, _, label_score = sort_and_match(label_score, modelFiles)
        #st_score, graph, outliers, labels, best_n, prv, values = eval_final(modelFiles, pair_scores)
        st_score, graph, outliers, labels, best_n, prv, labels_initial = eval_final_v5(modelFiles, pair_scores, modelFiles, 0.05,
                                                                        0.1, 'hdbscan', 'pr')

        print('--------------------------------------------------')

        # st_score = dict(sorted(st_score.items(), key=lambda item: item[0]))
        # st_score = list(st_score.values())

        std, _, st_score = sort_and_match(st_score, modelFiles)


        print(nmvd)
        print(acvd)
        print(svd)
        print(lsd)
        print(std)
        
        tau_label, _ = kendalltau(label_score, nmv)
        tau_st, _ = kendalltau(st_score, nmv)
        tau_sv, _ = kendalltau(sv, nmv)
        cor_label, _ = spearmanr(label_score, nmv)
        cor_st, _ = spearmanr(st_score, nmv)
        cor_sv, _ = spearmanr(sv, nmv)
        # tau1 = ['nmi', tau_st,tau_label, tau_sv]
        # corr1 = ['nmi', cor_st, cor_label, cor_sv]
        nmi_tau.append([metric, tau_st,tau_label, tau_sv])
        nmi_cor.append([metric, cor_st, cor_label, cor_sv])

        tau_label, _ = kendalltau(label_score, acv)
        tau_st, _ = kendalltau(st_score, acv)
        tau_sv, _ = kendalltau(sv, acv)
        cor_label, _ = spearmanr(label_score, acv)
        cor_st, _ = spearmanr(st_score, acv)
        cor_sv, _ = spearmanr(sv, acv)

        acc_tau.append([metric, tau_st,tau_label, tau_sv])
        acc_cor.append([metric, cor_st, cor_label, cor_sv])

        # tau2 = ['acc', tau_st,tau_label, tau_sv]
        # corr2 = ['acc', cor_st, cor_label, cor_sv]
        # taus = [tau1, tau2]
        # corrs = [corr1, corr2]
        # tdf = pd.DataFrame(columns=['metric',  'st_score', 'label_score', 'subspace_score'], data=taus)
        # cdf = pd.DataFrame(columns=['metric',  'st_score', 'label_score', 'subspace_score'], data=corrs)
        # tdf.to_csv('res/{}_tau.csv'.format(metric), index=False)
        # cdf.to_csv('res/{}_cor.csv'.format(metric), index=False)

    vertical = True

    
    lname = ['metric',  'proposed score', 'label score', 'space score']
    mname = ['cosine', 'euclidean', 'dav', 'ch']
    if vertical:
        cols = ['cosine', 'euclidean', 'dav', 'ch']
    else:
        cols = ['proposed score', 'label score', 'space score']

    df_nmi_tau = pd.DataFrame(columns=lname, data=nmi_tau).round(2)
    df_nmi_cor = pd.DataFrame(columns=lname, data=nmi_cor).round(2)
    df_acc_tau = pd.DataFrame(columns=lname, data=acc_tau).round(2)
    df_acc_cor = pd.DataFrame(columns=lname, data=acc_cor).round(2)

    #print(df_nmi_cor)

    if vertical:
        df_nmi_tau = transpose(df_nmi_tau)
        df_nmi_cor = transpose(df_nmi_cor)
        df_acc_tau = transpose(df_acc_tau)
        df_acc_cor = transpose(df_acc_cor)

    #print(df_nmi_cor)

    df_nmi_tau = df_nmi_tau.rename(columns={c: 'tau={}'.format(c) for c in df_nmi_tau.columns if c in cols})
    df_nmi_cor = df_nmi_cor.rename(columns={c: 'cor={}'.format(c) for c in df_nmi_cor.columns if c in cols})
    df_acc_tau = df_acc_tau.rename(columns={c: 'tau={}'.format(c) for c in df_acc_tau.columns if c in cols})
    df_acc_cor = df_acc_cor.rename(columns={c: 'cor={}'.format(c) for c in df_acc_cor.columns if c in cols})
    #print(df_nmi_cor)

    if not vertical:
        df_nmi = pd.merge(df_nmi_cor, df_nmi_tau, on='metric', how='inner')
        df_acc = pd.merge(df_acc_cor, df_acc_tau, on='metric', how='inner')
    else:
        df_nmi = pd.merge(df_nmi_cor, df_nmi_tau, left_index=True, right_index=True)
        df_acc = pd.merge(df_acc_cor, df_acc_tau, left_index=True, right_index=True)



    subcols = create_subcol(df_nmi.columns.tolist(), vertical)
    df_nmi.columns = pd.MultiIndex.from_tuples(subcols)

    subcols = create_subcol(df_acc.columns.tolist(), vertical)
    df_acc.columns = pd.MultiIndex.from_tuples(subcols)

    df_nmi = df_nmi.sort_index(axis=1, level=0, ascending=False)
    df_acc = df_acc.sort_index(axis=1, level=0, ascending=False)

    #print(df_nmi)

    if vertical:
        df_nmi = df_nmi.reindex(lname[1:][::-1])
        df_acc = df_acc.reindex(lname[1:][::-1])
        df_nmi = df_nmi.reindex(columns=mname, level=0)
        df_acc = df_acc.reindex(columns=mname, level=0)
    else:
        df_nmi = df_nmi.reindex(columns=lname, level=0)
        df_acc = df_acc.reindex(columns=lname, level=0)
    #print(df_nmi)

    # with pd.option_context('display.max_rows', None,
    #                     'display.max_columns', None,
    #                     'display.precision', 3,
    #                     ):
    print(df_nmi)
    print(df_acc)

    # df_nmi = df_nmi.round(3)
    # df_acc = df_acc.round(3)
    # print(df_nmi)
    # print(df_acc)

    if vertical:
        print(df_nmi.to_latex(index=True,
                        escape = False,
                        formatters={"name": str.upper},
                         float_format="{:.2f}".format).replace('space score', 'Paired score').replace('label score', 'Pooled score').replace('proposed score', '\\textbf{ACE}')
        )
        print(df_acc.to_latex(index=True,
                        escape = False,
                        formatters={"name": str.upper},
                         float_format="{:.2f}".format).replace('space score', 'Paired score').replace('label score', 'Pooled score').replace('proposed score', '\\textbf{ACE}')
        )
    else:
        print(df_nmi.to_latex(index=False,
                        escape = False,
                        formatters={"name": str.upper},
                         float_format="{:.2f}".format).replace('space score', 'Paired score').replace('label score', 'Pooled score').replace('proposed score', '\\textbf{ACE}')
        )
        print(df_acc.to_latex(index=False,
                        escape = False,
                        formatters={"name": str.upper},
                         float_format="{:.2f}".format).replace('space score', 'Paired score').replace('label score', 'Pooled score').replace('proposed score', '\\textbf{ACE}')
        )        