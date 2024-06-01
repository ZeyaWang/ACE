import os, sys
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.mixture import BayesianGaussianMixture
import dill
from sklearn.cluster import KMeans
import networkx as nx
from statsmodels.stats.multitest import multipletests
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.manifold import TSNE
import numpy as np
import plotly.express as px
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import hdbscan
import pickle as pk

def clustering_accuracy(gtlabels, labels):
    cost_matrix = []
    categories = np.unique(gtlabels)
    nr = np.amax(labels) + 1
    for i in np.arange(len(categories)):
      cost_matrix.append(np.bincount(labels[gtlabels == categories[i]], minlength=nr))
    cost_matrix = np.asarray(cost_matrix).T
    row_ind, col_ind = linear_sum_assignment(np.max(cost_matrix) - cost_matrix)
    #print(cost_matrix[row_ind, col_ind])
    return float(cost_matrix[row_ind, col_ind].sum()) / len(gtlabels)

def plot_tsne(X, y, title):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    #print(tsne.kl_divergence_)
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y)# symbol=m, size=s)
    fig.update_layout(
        #title="t-SNE visualization of {}".format(title),
        title=title,
        #legend_title="truth",
        #legend_title_text="truth",
        coloraxis_colorbar_title="Truth",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2"
    )
    return fig

def similarity(x, y, metric):
    if metric == 'tau':
        stat, pvalue = stats.kendalltau(x, y)
    else:
        stat, pvalue = stats.spearmanr(x, y, alternative='greater')
        #print('=====',x,y,stat)
    return stat, pvalue


def affinity_matrix(scores, metric = 'spearman'):
    '''
    scores is nfeat x nlabel matrix
    metric: spearman or tau
    '''
    # we first get a nfeat x nfeat rank similairty matrix
    #print(scores)
    nfeat, _ = scores.shape
    sim = np.zeros((nfeat, nfeat))
    pvals = []
    index = []
    for i in range(nfeat):
        feat_i = scores[i,:]
        for j in range(i+1, nfeat):
            feat_j = scores[j,:]
            val, pval = similarity(feat_i, feat_j, metric)
            pvals.append(pval)
            index.append([i,j])
            sim[i,j] = val
            sim[j,i] = val
    #sim[sim < 0] = 0
    return sim, pvals, index


def correct_matrix(affinity, pvals, index, method='holm',alpha=0.1):
    r, _, _, _ = multipletests(pvals, method=method, alpha=alpha)
    corrected = affinity.copy()
    #affinity[affinity < 0] = 0
    corrected[corrected < 0] = 0
    #print('========rrrrrrrrrrrrrrrrrrrrrrrr============', r)
    for idx, ind in enumerate(index):
        if r[idx] == False:
            i = ind[0]
            j = ind[1]
            corrected[i,j] = 0
            corrected[j,i] = 0
    return corrected

def dbscan_graph(affinity, eps=0.2, min_samples=5):
    X = 1 - affinity
    #print(X)
    np.fill_diagonal(X, 0)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(X)
    labels = clustering.labels_
    #print(labels)
    cores = clustering.core_sample_indices_
    outliers = np.where(labels == -1)[0]
    return cores, outliers, labels

def hdbscan_graph(affinity, min_samples=5):
    X = 1 - affinity
    np.fill_diagonal(X, 0)
    #print(X)
    #clustering = HDBSCAN(min_cluster_size=min_samples, metric='precomputed').fit(X)
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_samples, metric='precomputed').fit(X)

    labels = clustering.labels_
    #print(labels)
    outliers = np.where(labels == -1)[0]
    if len(outliers) == len(labels):
        outlier_score = clustering.outlier_scores_
        outlier_score = np.nan_to_num(outlier_score, nan=0)
        q3 = np.quantile(outlier_score, 0.75)
        q1 = np.quantile(outlier_score, 0.25)
        iqr = q3 - q1
        threshold = q3 + 1.5*iqr
        #print(outlier_score, threshold)
        outliers = np.where(outlier_score > threshold)[0]
        normals = np.where(outlier_score <= threshold)[0]
        labels[normals] = 0
    #return outliers, labels, outlier_score
    return outliers, labels


def page_rank(affinity):
    n = len(affinity)
    try:
        G = nx.DiGraph(affinity)
        pr = nx.pagerank(G, alpha=0.9)
        pr=dict(sorted(pr.items(), key=lambda item: item[0]))
        pr = list(pr.values())
        return np.array(pr)
    except:
        pr = [1/n for _ in range(n)]
        return np.array(pr)


def hits(affinity):
    n = len(affinity)
    try:
        G= nx.DiGraph(affinity)
        pr, pra = nx.hits(G)
        pr = dict(sorted(pra.items(), key=lambda item: item[0]))
        pr = list(pr.values())
        return np.array(pr)
    except:
        pr = [1/n for _ in range(n)]
        return np.array(pr)

def rank(affinity, method='pr'):
    if method=='pr':
        return page_rank(affinity)
    else:
        return hits(affinity)


def get_files_with_substring_and_suffix(directory, substring, suffix):
    files = []
    # Use os.listdir to get a list of all files in the directory
    all_files = os.listdir(directory)

    # Use a list comprehension to filter files based on the substring and suffix
    files = [file for file in all_files if substring in file and file.endswith(suffix)]

    return files

def filter_merge_files(eval_data, threshold=0.5, metric='cosine', root='metric_result'):
    '''

    Parameters
    ----------
    metric: cosine/euclidean for silhouette score; metrics = ['ccc', 'dunn', 'cind', 'dunn2', 'cind2', 'db', 'sdbw']
    Returns
    -------
    '''
    suffix = '{}_score.pkl'.format(metric)
    result_file = get_files_with_substring_and_suffix(root, '_'+eval_data, suffix)
    if len(result_file) > 1:
        print('!!!!!!!!!!!!!!!!!!!!!!!!! multiple metrics files', result_file)
    result_file = os.path.join(root, result_file[0])
    all_scores, sc = pk.load(open(result_file, 'rb'))
    scores = np.array([s[0] for _, s in sc.items()])
    if len(scores[scores < threshold]) > 1:
        keep_models = [k for k, s in sc.items() if s[0] < threshold]
    else:
        keep_models = list(sc.keys())
    return keep_models, all_scores

def filter_merge_files_rep(eval_data, metric='cosine', root='metric_result2'):
    '''

    Parameters
    ----------
    metric: cosine/euclidean for silhouette score; metrics = ['ccc', 'dunn', 'cind', 'dunn2', 'cind2', 'db', 'sdbw']
    Returns
    -------
    '''
    suffix = '{}_score.pkl'.format(metric)
    result_file = get_files_with_substring_and_suffix(root, '_'+eval_data, suffix)
    if len(result_file) > 1:
        print('!!!!!!!!!!!!!!!!!!!!!!!!! multiple metrics files', result_file)
    result_file = os.path.join(root, result_file[0])
    scores = pk.load(open(result_file, 'rb'))

    return scores


def collect_raw_files(eval_data, root='raw_metric'):
    '''

    Parameters
    ----------
    metric: cosine/euclidean for silhouette score; metrics = ['ccc', 'dunn', 'cind', 'dunn2', 'cind2', 'db', 'sdbw']
    Returns
    -------
    '''
    result_file = os.path.join(root, 'merge_all_{}_score.pkl'.format(eval_data))
    scores = pk.load(open(result_file, 'rb'))
    return scores

def sort_and_match(obj, modelFiles):
    new_obj = {key: value for key, value in obj.items() if key in modelFiles}
    new_obj = dict(sorted(new_obj.items(), key=lambda item: item[0]))
    key_obj = list(new_obj.keys())
    val_obj = np.squeeze(np.array(list(new_obj.values())))
    return new_obj, key_obj, val_obj

def process_raw(raw):
    raw_output = {}
    for k, v in raw.items():
        if isinstance(v, np.ndarray):
            v = v.item()
        if v == None:
            v = np.nan
        raw_output[k] = v
    return raw_output

def eval_vote(modelFiles, scores):
    # eval label
    label_score = {}
    for i, m in enumerate(modelFiles):
        label_score[m] = scores[:,i].mean()
    return label_score




def filter_out(pvalues, models, method='holm',alpha=0.1):
    r, _, _, _ = multipletests(pvalues, method=method, alpha=alpha)
    keep_models = [m for i, m in enumerate(models) if r[i]]
    return keep_models


def generate_pair_scores(all_scores, modelFiles):
    scores = []
    # get score
    diag_score = {}
    for m in modelFiles:
        scores_row = []
        #print(len(all_scores[m]))
        for n in modelFiles:
            #print(m,n)
            if n not in all_scores[m].keys():
                s = 0
            else:
                svec = all_scores[m][n]
                if isinstance(svec, np.ndarray):
                    svec = svec.tolist()
                if isinstance(svec, list):
                    s = svec[0]
                else:
                    s = svec
            # if (np.isinf(s) & (s < 0)) or np.isnan(s):
            #     s = -100000
            scores_row.append(s)
            if n == m:
                diag_score[m] = s
        scores.append(scores_row)
        #print(scores)
    return np.array(scores), diag_score


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def kendalltau(score1, score2):
    stat, pval = stats.kendalltau(score1, score2)
    return np.round(stat, 3), pval

def spearmanr(score1, score2):
    stat, pval = stats.spearmanr(score1, score2)
    return np.round(stat, 3), pval

def get_max(score):
    if np.max(score) == np.min(score):
        return None
    else:
        return np.where(score == np.max(score))[0].tolist()

def max_number(score, models, eval_data):
    imax = get_max(score)
    if imax == None:
        return 'None'
    else:
        ms = []
        for i in imax:
            m = models[i]
            m = m.replace(eval_data, "")
            if '_' in m:
                m = m.replace("_", "")
            if '.npz' in m:
                m = m.replace(".npz", "")
            if 'output' in m:
                m = m.replace("output", "")
            ms.append(m)
        return ','.join(ms)


def eval_final(modelFiles, scores, spaceFiles, eps=0.05, alpha=0.01, cl_method='dbscan', rank_method='pr', remove_outlier=True):
    rows_to_remove = np.all(scores[:, 1:] == scores[:, :-1], axis=1)
    spaceFiles = np.array(spaceFiles)
    spaceFiles = spaceFiles[~rows_to_remove]
    scores = scores[~rows_to_remove]
    affinity, pvals, index = affinity_matrix(scores)
    print(affinity.shape)
    #print(affinity)
    min_samples = 2
    if cl_method == 'dbscan':
        cores, outliers, labels = dbscan_graph(affinity, eps=eps, min_samples=min_samples)
    else:
        outliers, labels = hdbscan_graph(affinity, min_samples=min_samples)
    affinity = correct_matrix(affinity, pvals, index, alpha=alpha)
    graph = nx.Graph(affinity)
    labels_updated = labels.copy()
    # if remove_outlier:
    #     labels_updated[labels_updated == -1] = -2
    npart = np.unique(labels[labels >= 0])
    if len(npart) > 0:
        for n in npart:
            part = np.where(labels == n)[0]
            sc = scores[part, :]
            if len(sc) > 5:
                # for each partition we cluster based on the scale
                dist = metrics.pairwise_distances(sc, metric='euclidean').astype('double')
                outliers_d, labels_d = hdbscan_graph(1-dist, min_samples=min_samples)
                out_idx = 1 # for outlier
                for l, lb in enumerate(labels_d):
                    if lb == -1:
                        labels_updated[part[l]] = (n+1)*100000 + out_idx
                        out_idx = out_idx + 1
                    else:
                        labels_updated[part[l]] = (n+1)*1000+lb
    #print(labels_updated)
    # calculate outlier
    dlabels = dict(zip(spaceFiles, labels))
    dlabels_updated = dict(zip(spaceFiles, labels_updated))
    part_out = np.where(labels_updated == -1)[0]
    if len(part_out) > 0:
        result_out, parts_out = {}, {}
        for n in part_out:
            sc = scores[n, :]
            solve_score = {}
            for i, m in enumerate(modelFiles):
                solve_score[m] = sc[i]
            sum_score = sum(solve_score.values())
            parts_out[sum_score] = n
            result_out[n] = solve_score
        best_out_score = max(parts_out.keys())
        best_out = parts_out[best_out_score]
        solve_out_score = result_out[best_out]
        #print('best outlier', best_out, best_out_score)

    npart = np.unique(labels_updated[labels_updated >= 0])
    if len(npart) > 0:
        result, nparts, prvs, nparts_id = {}, {}, {}, {}  # page rank value
        for n in npart:
            part = np.where(labels_updated == n)[0]
            solve_score = {}
            if len(part) > 1:
                aff = affinity[part, :]
                aff = aff[:, part]
                sc = scores[part, :]
                pr = rank(aff, method=rank_method)
                for i, m in enumerate(modelFiles):
                    solve_score[m] = np.sum(sc[:, i] * pr)/np.sum(pr)
            else:
                sc = scores[part[0], :]
                for i, m in enumerate(modelFiles):
                    solve_score[m] = sc[i]
                pr = [1]
            sum_score = sum(solve_score.values())
            nparts[sum_score] = n
            result[n] = solve_score
            nparts_id[n] = part
            prvs[n] = dict(zip(part, pr))
        best_part_score = max(nparts.keys())
        best_n = nparts[best_part_score]
        solve_score = result[best_n]
        prv = prvs[best_n]
        #print('best n', best_n, nparts, best_part_score, nparts_id)

        if len(part_out) == 0:
            return solve_score, graph, outliers, dlabels_updated, best_n, prv, dlabels
        else:
            if (best_part_score > best_out_score) or remove_outlier:
                return solve_score, graph, outliers, dlabels_updated, best_n, prv, dlabels
            else:
                solve_value = dict(sorted(solve_score.items(), key=lambda item: item[0]))
                solve_out_value = dict(sorted(solve_out_score.items(), key=lambda item: item[0]))
                solve_value = list(solve_value.values())
                solve_out_value = list(solve_out_value.values())
                _, pval = stats.ttest_rel(solve_out_value, solve_value, alternative='greater')
                if pval < 0.05:
                    return solve_out_score, graph, outliers, dlabels_updated, best_out, spaceFiles[best_out], dlabels
                else:
                    return solve_score, graph, outliers, dlabels_updated, best_n, prv, dlabels
    else:
        return solve_out_score, graph, outliers, dlabels_updated, best_out, spaceFiles[best_out], dlabels


def label_map_color2(labels):
    color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'orange', 5: 'purple', 6: 'pink', 7: 'brown', 8:'violet', 9:'cyan', 10:'maroon', 11: 'teal', 12: 'indigo', 13: 'tan', 14: 'magenta', 15: 'salmon'}
    ulabels = np.unique(labels)
    color_map1 = {}
    i = 0
    for _, ul in enumerate(ulabels):
        if ul >= 100000:
            color_map1[ul] = 'olive'
        else:
            color_map1[ul] = color_map[i]
            i = i + 1
    color_map1[-1] = 'grey'
    values = [color_map1[v] for v in labels]
    return values



if __name__ == '__main__':
    exit()