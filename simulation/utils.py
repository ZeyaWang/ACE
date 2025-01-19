import os, sys
import scipy.stats as stats
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
import matplotlib.pyplot as plt
import h5py

def clustering_accuracy(gtlabels, labels):
    '''
    return clustering accuracy
    '''
    cost_matrix = []
    categories = np.unique(gtlabels)
    nr = np.amax(labels) + 1
    for i in np.arange(len(categories)):
      cost_matrix.append(np.bincount(labels[gtlabels == categories[i]], minlength=nr))
    cost_matrix = np.asarray(cost_matrix).T
    row_ind, col_ind = linear_sum_assignment(np.max(cost_matrix) - cost_matrix)
    return float(cost_matrix[row_ind, col_ind].sum()) / len(gtlabels)

def plot_tsne(X, y, title):
    '''
    make tsne plot
    '''
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(tsne.kl_divergence_)
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y)# symbol=m, size=s)
    fig.update_layout(
        title=title,
        coloraxis_colorbar_title="Truth",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2"
    )
    return fig

def similarity(x, y, metric):
    '''
    get the similarity between two scores based on rank correlation
    '''
    if metric == 'tau':
        stat, pvalue = stats.kendalltau(x, y)
    else:
        stat, pvalue = stats.spearmanr(x, y, alternative='greater')
    return stat, pvalue


def affinity_matrix(scores, metric = 'spearman'):
    '''
    get the affinity matrix of rank correlation from all the scores for the graph
    scores: is nfeat x nlabel matrix
    metric: spearman or tau
    '''
    nfeat, _ = scores.shape
    #print(nfeat)
    sim = np.zeros((nfeat, nfeat))
    pvals, index = [], []
    for i in range(nfeat):
        feat_i = scores[i,:]
        for j in range(i+1, nfeat):
            feat_j = scores[j,:]
            val, pval = similarity(feat_i, feat_j, metric)
            pvals.append(pval)
            index.append([i,j])
            sim[i,j] = val
            sim[j,i] = val
    return sim, pvals, index


def correct_matrix(affinity, pvals, index, method='holm',alpha=0.1):
    '''
    perform multiple testing to remove non-significant edges in the graph
    '''
    r, _, _, _ = multipletests(pvals, method=method, alpha=alpha)
    corrected = affinity.copy()
    corrected[corrected < 0] = 0
    for idx, ind in enumerate(index):
        if r[idx] == False:
            i = ind[0]
            j = ind[1]
            corrected[i,j] = 0
            corrected[j,i] = 0
    return corrected

def dbscan_graph(affinity, eps=0.2, min_samples=5):
    '''
    peform dbscan clustering based on the affinity matrix
    '''
    X = 1 - affinity
    np.fill_diagonal(X, 0)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(X)
    labels = clustering.labels_
    cores = clustering.core_sample_indices_
    outliers = np.where(labels == -1)[0]
    return cores, outliers, labels

def hdbscan_graph(affinity, min_samples=5):
    '''
    peform hdbscan clustering based on the affinity matrix
    '''
    X = 1 - affinity
    np.fill_diagonal(X, 0)
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_samples, metric='precomputed').fit(X)
    labels = clustering.labels_
    outliers = np.where(labels == -1)[0]
    if len(outliers) == len(labels): # in case all the spaces are identified as outliers
        outlier_score = clustering.outlier_scores_
        outlier_score = np.nan_to_num(outlier_score, nan=0)
        q3 = np.quantile(outlier_score, 0.75)
        q1 = np.quantile(outlier_score, 0.25)
        iqr = q3 - q1
        threshold = q3 + 1.5*iqr
        outliers = np.where(outlier_score > threshold)[0]
        normals = np.where(outlier_score <= threshold)[0]
        labels[normals] = 0
    return outliers, labels

def page_rank(affinity):
    '''
    get rates based on page rank
    '''
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
    '''
    get rates based on HITS algorithm
    '''
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
    '''
    get rates for all the kept spaces
    '''
    if method=='pr':
        return page_rank(affinity)
    else:
        return hits(affinity)

def get_files_with_substring_and_suffix(directory, substring, suffix):
    '''
    get files with certain substring and suffix
    '''
    all_files = os.listdir(directory)
    files = [file for file in all_files if substring in file and file.endswith(suffix)]
    return files


def collect_score_files(metric, root='raw_metric'):
    '''
    collect all raw scores from the score file
    '''
    result_file = os.path.join(root, 'merge_{}_score.pkl'.format(metric))
    scores = pk.load(open(result_file, 'rb'))
    return scores

def sort_and_match(obj, modelFiles):
    '''
    sort the object according to the order in modelFiles
    '''
    new_obj = {key: value for key, value in obj.items() if key in modelFiles}
    new_obj = dict(sorted(new_obj.items(), key=lambda item: item[0]))
    key_obj = list(new_obj.keys())
    val_obj = np.squeeze(np.array(list(new_obj.values())))
    return new_obj, key_obj, val_obj

def process_raw(raw):
    '''
    process raw scores
    '''
    raw_output = {}
    for k, v in raw.items():
        if isinstance(v, np.ndarray):
            v = v.item()
        if v == None:
            v = np.nan
        raw_output[k] = v
    return raw_output

def filter_out(pvalues, models, method='holm',alpha=0.1):
    '''
    filter out spaces based on the p-values from the dip test
    '''
    r, _, _, _ = multipletests(pvalues, method=method, alpha=alpha)
    keep_models = [m for i, m in enumerate(models) if r[i]]
    return keep_models

def community_layout(g, partition):
    '''
    get the layout for plotting the graph
    '''
    pos_communities = _position_communities(g, partition, scale=3.)
    pos_nodes = _position_nodes(g, partition, scale=1.)
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]
    return pos

def _position_communities(g, partition, **kwargs):
    '''
    get the layout for plotting the graph
    '''
    between_community_edges = _find_between_community_edges(g, partition)
    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))
    pos_communities = nx.spring_layout(hypergraph, **kwargs)
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]
    return pos

def _find_between_community_edges(g, partition):
    '''
    get the layout for plotting the graph
    '''
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
    '''
    get the layout for plotting the graph
    '''
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
    '''
    calculate kendal tau
    '''
    stat, pval = stats.kendalltau(score1, score2)
    return np.round(stat, 3), pval

def spearmanr(score1, score2):
    '''
    calculate spearman correlation
    '''
    stat, pval = stats.spearmanr(score1, score2)
    return np.round(stat, 3), pval

def get_max(score):
    '''
    get the index of the max score
    '''
    if np.max(score) == np.min(score):
        return None
    else:
        return np.where(score == np.max(score))[0].tolist()

def max_number(score, models, eval_data):
    '''
    get the index of the max score
    '''
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

def collect_all_scores(all_scores, modelFiles):
    '''
    collect scores from all the possible pairs of partition result and embedded data
    also return the paired score
    '''
    scores = []
    diag_score = {}
    for m in modelFiles:
        scores_row = []
        for n in modelFiles:
            if n not in all_scores[m].keys():
                s = 0
            else:
                s = all_scores[m][n]
                if isinstance(s, np.ndarray):
                    s = s.item()
            if s == None:
                s = np.nan
            scores_row.append(s)
            if n == m:
                diag_score[m] = s
        scores.append(scores_row)
    return np.array(scores), diag_score

def eval_pool(modelFiles, scores):
    '''
    get pool score
    '''
    label_score = {}
    for i, m in enumerate(modelFiles):
        label_score[m] = scores[:,i].mean()
    return label_score
    
def eval_ace(modelFiles, scores, spaceFiles, eps=0.05, alpha=0.01, cl_method='dbscan', rank_method='pr', remove_outlier=True):
    '''
    implementation of ACE to get ACE scores
    Input:
    modelFiles: list of files names for results (M)
    scores: numpy array of L x M with rows corresponding to L spaces and columns corresponding to M results
    spaceFiles: list of files names for spaces (L)
    '''
    # remove all the spaces with costant scores
    rows_to_remove = np.all(scores[:, 1:] == scores[:, :-1], axis=1)
    spaceFiles = spaceFiles[~rows_to_remove]
    scores = scores[~rows_to_remove]
    # get the affinity matrix based on rank correlation from the score array
    affinity, pvals, index = affinity_matrix(scores)

    # phase 1 clustering
    if cl_method == 'dbscan':
        cores, outliers, labels = dbscan_graph(affinity, eps=eps, min_samples=2)
    else:
        outliers, labels = hdbscan_graph(affinity, min_samples=2)
    # remove non-significant edges
    affinity = correct_matrix(affinity, pvals, index, alpha=alpha)
    # build the graph
    graph = nx.Graph(affinity)
    labels_updated = labels.copy()
    # get the number of groups except for outliers
    npart = np.unique(labels[labels >= 0])
    if len(npart) > 0:
        for n in npart:
            # for each group we obtain from phase 1 clustering, we create subgroups
            part = np.where(labels == n)[0]
            sc = scores[part, :]
            if len(sc) > 5:
                # for each partition we cluster based on the scale
                dist = metrics.pairwise_distances(sc, metric='euclidean').astype('double')
                outliers_d, labels_d = hdbscan_graph(1-dist, min_samples=2)
                out_idx = 1 # for outlier spaces we consider them as singleton subgroups
                # create index for the subgroups to make all the subgroup indices are non-overlapping
                for l, lb in enumerate(labels_d):
                    if lb == -1:
                        labels_updated[part[l]] = (n+1)*100000 + out_idx
                        out_idx = out_idx + 1
                    else:
                        labels_updated[part[l]] = (n+1)*1000+lb
    # create map between spaces and groups for phase 1 grouping
    dlabels = dict(zip(spaceFiles, labels))
    # create map between spaces and subgroups for phase 2 grouping
    dlabels_updated = dict(zip(spaceFiles, labels_updated))
    # get outlier spaces from phase 1 grouping
    part_out = np.where(labels_updated == -1)[0]

    # handle outlier space; get the outlier space with the largest sum score
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

    # handle subgroups (excluding outlier spaces)
    npart = np.unique(labels_updated[labels_updated >= 0])
    if len(npart) > 0:
        result, nparts, prvs, nparts_id = {}, {}, {}, {}
        # for each subgroup, use link analysis to get rates of spaces in the subgroup
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

        if len(part_out) == 0:
            return solve_score, graph, outliers, dlabels_updated, best_n, prv, dlabels
        else:
            if (best_part_score > best_out_score) or remove_outlier:
                return solve_score, graph, outliers, dlabels_updated, best_n, prv, dlabels
            else:
                # this case is discussed in our supplementary
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

def label_map_color(labels):
    '''
    map color to label in the graph plot
    '''
    color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'orange', 5: 'purple', 6: 'pink',
                 7: 'brown', 8:'violet', 9:'cyan', 10:'maroon', 11: 'teal', 12: 'indigo',
                 13: 'tan', 14: 'magenta', 15: 'salmon'}
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

def graph_plot(l, graph, dlabels, save_path, eval_data, metric):
    '''
    make the plot for the graph of all the spaces
    '''
    values = label_map_color(list(dlabels.values()))
    plt.figure(l)
    partitions = {}
    for i, v in enumerate(list(dlabels.values())):
        partitions[i] = v
    nx.draw(graph, pos=community_layout(graph, partitions), node_color=values, with_labels=True)
    plt.savefig(os.path.join(save_path, "{}_{}.png".format(eval_data, metric)))
    l = l+1
    return l



def make_data_plot(dlabels, selected_group, eval_data, task, metric, save_path):
    '''
    make tsne plot
    '''
    tasks = {'jule_hyper': './JULE_hyper',
             'jule_num': './JULE_num',
             'DEPICT': './DEPICT_hyper',
             'DEPICTnum': './DEPICT_num'
             }
    spaceFiles = np.array(list(dlabels.keys()))
    labels = np.array(list(dlabels.values()))
    save_path = os.path.join(save_path, 'tnse')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, eval_data)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, metric)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # prepare truth label
    rpath = tasks[task]
    if 'jule' in task:
        rpath = os.path.join(rpath, 'jule') # root path
    dpath = os.path.join(rpath, 'datasets') # data path
    dpath = os.path.join(dpath, eval_data)
    if 'jule' in task:
        dpath = os.path.join(dpath, 'data4torch.h5')
    else:
        dpath = os.path.join(dpath, 'data.h5')
    y = np.squeeze(np.array(h5py.File(dpath, 'r')['labels']))
    _, y = np.unique(y, return_inverse=True)
    labels_unique = np.unique(labels)
    best_space = None
    if selected_group not in labels_unique:
        best_space = spaceFiles[selected_group]
    # make tsne plot for each space in each group
    for ul in labels_unique:
        if ul == selected_group:
            uu = '{}_selected'.format(ul)
        else:
            uu= str(ul)
        spath = os.path.join(save_path, uu)
        if not os.path.isdir(spath):
            os.mkdir(spath)
        spaces = spaceFiles[labels==ul].tolist()
        for sp in spaces:
            if 'jule' in task:
                fpath = os.path.join(rpath, 'feature{}.h5'.format(sp))
                X = np.array(h5py.File(fpath, 'r')['feature'])
            else:
                fpath = os.path.join(rpath, sp)
                X=np.array(np.load(fpath)['y_features'])
            fig = plot_tsne(X, y, eval_data)
            if (best_space != None) and (best_space == sp):
                wpath = os.path.join(spath,"{}_selected.png".format(sp))
            else:
                wpath = os.path.join(spath,"{}.png".format(sp))
            fig.write_image(wpath)

if __name__ == '__main__':
    exit()