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
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import plotly.express as px

def clustering_accuracy(gtlabels, labels):
    cost_matrix = []
    categories = np.unique(gtlabels)
    nr = np.amax(labels) + 1
    for i in np.arange(len(categories)):
      cost_matrix.append(np.bincount(labels[gtlabels == categories[i]], minlength=nr))
    cost_matrix = np.asarray(cost_matrix).T
    row_ind, col_ind = linear_sum_assignment(np.max(cost_matrix) - cost_matrix)
    print(cost_matrix[row_ind, col_ind])
    return float(cost_matrix[row_ind, col_ind].sum()) / len(gtlabels)

def paired_test(x, y, alternative='two-sided'):
    stat, pvalue = stats.ttest_rel(x, y, alternative=alternative)
    return stat, pvalue

def plot_tsne(X, y):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(tsne.kl_divergence_)
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y)# symbol=m, size=s)
    fig.update_layout(
        title="t-SNE visualization of Custom Classification dataset",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
    )
    return fig

def similarity(x, y, metric):
    if metric == 'tau':
        stat, pvalue = stats.kendalltau(x, y)
    else:
        stat, pvalue = stats.spearmanr(x, y, alternative='greater')
    return stat, pvalue

def graph_bipartition(affinity):
    G = nx.Graph(affinity)
    return nx.kernighan_lin_bisection(G)
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
            #print(feat_i, feat_j, val)
            sim[i,j] = val
            sim[j,i] = val
    sim[sim < 0] = 0
    return sim, pvals, index

def affinity_matrix_ttest(scores):
    nfeat, _ = scores.shape
    sim = np.zeros((nfeat, nfeat))
    pvals = []
    index = []
    for i in range(nfeat):
        feat_i = scores[i,:]
        for j in range(i+1, nfeat):
            feat_j = scores[j,:]
            val, pval = paired_test(feat_i, feat_j)
            pvals.append(pval)
            index.append([i,j])
            #print(feat_i, feat_j, val)
            sim[i,j] = val
            sim[j,i] = val
    sim[sim < 0] = 0
    return sim, pvals, index



# def affinity_combine(scores, metric = 'spearman'):
#     '''
#     '''
#     sim1, pvals1, index = affinity_matrix(scores, metric)
#     sim2, pvals2, _ = affinity_matrix_ttest(scores)
#     return sim1, sim2, pvals1, pvals2, index

def correct_matrix_combine(affinity, pvals1, pvals2, index, method='holm',alpha1=0.1,alpha2=0.1):
    r1, _, _, _ = multipletests(pvals1, method=method, alpha=alpha1) # null: cor is negative and zero;
    r2, _, _, _ = multipletests(pvals2, method=method, alpha=alpha2) # null: equal mean of two spaces

    for idx, ind in enumerate(index):
        if (r1[idx] == False) or (r2[idx] == True): # true to be rejected and false not to be rejected (accepted)
            i = ind[0]
            j = ind[1]
            affinity[i,j] = 0
            affinity[j,i] = 0
    return affinity
def correct_matrix(affinity, pvals, index, method='holm',alpha=0.1):
    r, _, _, _ = multipletests(pvals, method=method, alpha=alpha)
    for idx, ind in enumerate(index):
        if r[idx] == False:
            i = ind[0]
            j = ind[1]
            affinity[i,j] = 0
            affinity[j,i] = 0
    return affinity

def dbscan_graph(affinity, eps=0.2, min_samples=5):
    X = 1 - affinity
    #print(X)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(X)
    labels = clustering.labels_
    #print(labels)
    cores = clustering.core_sample_indices_
    outliers = np.where(labels == -1)[0]
    return cores, outliers, labels

def hdbscan_graph(affinity, min_samples=5):
    X = 1 - affinity
    #print(X)
    clustering = HDBSCAN(min_cluster_size=min_samples, metric='precomputed').fit(X)
    labels = clustering.labels_
    #print(labels)
    outliers = np.where(labels == -1)[0]
    return outliers, labels
def graph_embedding(affinity_matrix, n_components=None, eigen_solver=None, eigen_tol="auto", random_state=None):
    if n_components == None:
        n_components = np.min(affinity_matrix.shape)
    return sklearn.manifold.spectral_embedding(
            affinity_matrix,
            n_components=n_components,
            eigen_solver=eigen_solver,
            eigen_tol=eigen_tol,
            random_state=random_state,
        )



def emebdding_clustering(embedding, alpha=1.0):
    nsp, dim = embedding.shape
    n_components = min(nsp, 100)
    dpgmm = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior=alpha/n_components,
        weight_concentration_prior_type='dirichlet_process',
        covariance_prior=dim*np.identity(dim),
        covariance_type='full').fit(embedding)
    preds = dpgmm.predict(embedding)
    print(preds)
    return preds


def page_rank_1(affinity, outliers = []):
    length = len(affinity)
    if len(outliers) > 0:
        affinity = np.delete(affinity, outliers, 0)
        affinity = np.delete(affinity, outliers, 1)
        # get a mapping between matrix id between before and after removing
        mapping = {}
        cnt = 0
        for i in range(length):
            if i not in outliers:
                mapping[i] = cnt
                cnt = cnt + 1
    G= nx.DiGraph(affinity)
    pr = nx.pagerank(G, alpha=0.9)
    if len(outliers) > 0:
        pr_new = {}
        for i in range(length):
            if i in outliers:
                pr_new[i] = 0
            else:
                pr_new[i] =pr[mapping[i]]
        pr = pr_new
    pr=dict(sorted(pr.items(), key=lambda item: item[0]))
    pr = list(pr.values())
    return np.array(pr)

def page_rank(affinity):
    G= nx.DiGraph(affinity)
    pr = nx.pagerank(G, alpha=0.9)
    pr=dict(sorted(pr.items(), key=lambda item: item[0]))
    pr = list(pr.values())
    return np.array(pr)

def hits(affinity):
    G= nx.DiGraph(affinity)
    pr, pra = nx.hits(G)
    pr = dict(sorted(pr.items(), key=lambda item: item[0]))
    pr = list(pr.values())
    return np.array(pr)


def rank(affinity):
    try:
        return page_rank(affinity)
    except:
        return hits(affinity)

def build_graph(scores, nmi_model, method='holm',alpha1=0.01, alpha2=0.1):
    nfeat, nlabel = scores.shape
    affinity = np.zeros((nlabel, nlabel))
    pvals = []
    index = []
    feats = []
    for i in range(nlabel):
        feat_i = scores[:,i]
        for j in range(nlabel):
            if j!=i:
                feat_j = scores[:,j]
                val, pval = paired_test(feat_i, feat_j, 'greater') # feat_i > feat_j
                pvals.append(pval) # null: less; reject means greater
                feats.append([feat_i, feat_j])
                index.append([i,j])
    r, _, _, _ = multipletests(pvals, method=method, alpha=alpha1)
    # for idx, ind in enumerate(index):
    #     if r[idx] == True: # true to be rejected
    #         i = ind[0]
    #         j = ind[1]
    #         affinity[j,i] += 1
    #         print(i,j)
    #         print(nmi_model[i], nmi_model[j])
    #         print(feats[idx][0]-feats[idx][1],feats[idx][0],feats[idx][1])
    pvals1 = []
    pvals2 = []
    index = []
    for i in range(nfeat):
        feat_i = scores[i,:]
        for j in range(nfeat):
            feat_j = scores[j,:]
            val1, pval1 = paired_test(feat_i, feat_j, 'greater') # feat_i > feat_j
            pvals1.append(pval1) # null: less; reject means greater
            val2, pval2 = paired_test(feat_i, feat_j, 'less') # feat_i < feat_j
            pvals2.append(pval2) # null: less; reject means greater
            index.append([i,j])
    r1, _, _, _ = multipletests(pvals1, method=method, alpha=alpha2)
    r2, _, _, _ = multipletests(pvals2, method=method, alpha=alpha2)
    for idx, ind in enumerate(index):
        if r1[idx] == True: # true to be rejected; feat_i > feat_j
            i = ind[0]
            j = ind[1]
            if scores[i,i] > scores[i,j]:
                # if affinity[j,i] == 0:
                #     affinity[j, i] = 1
                affinity[j,i] += 1
        if r2[idx] == True: # true to be rejected; feat_i < feat_j
            i = ind[0]
            j = ind[1]
            if scores[j,j] > scores[j,i]:
                # if affinity[i,j] == 0:
                #     affinity[i,j] = 1
                affinity[i,j] += 1
    return affinity

def whitening(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None

    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method == 'pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)

def remove_outlier(x, q=0.9):
    center = np.mean(x, axis=0)
    dist = np.linalg.norm(x - center, axis=1)
    thresh = np.quantile(dist,q=q)
    return x[dist<=thresh,:]

def r2_adj(x_list):
    """
    Parameters
    ----------
    x_list: a list of X belonging to different clusters
    Returns
    -------
    1 - R2_adj
    """
    z_list = []
    K = len(x_list)
    for i, x in enumerate(x_list):
        z = np.zeros((len(x), K))
        z[:, i] = 1
        z_list.append(z)
    X = np.concatenate(x_list, axis=0)
    #X = whitening(X)
    X = X - np.mean(X, axis=0)
    Z = np.concatenate(z_list, axis=0)
    X_bar = np.linalg.inv(Z.T@Z)@Z.T@X
    T = X.T@X
    B = X_bar.T@Z.T@Z@X_bar
    W = T-B
    N = len(X)
    return (np.trace(W)/(N-K))/(np.trace(T)/(N-1)) # 1 - R2
    #return np.trace(W)/np.trace(T) # 1 - R2
    #return np.linalg.det(W)/np.linalg.det(T)
    #return np.trace(W@np.linalg.inv(T))
def r2_merge_near2(X, y, metric='euclidean'):
    """
    guarantee y is contiguous (e.g., 0,1,2,3...)
    return r2 for merging two clusters
    """
    N, p = X.shape
    cl = np.unique(y)
    # calculate centroid for each cluster
    cluster_means = np.zeros((len(cl), p))
    for cluster_mean_index, _ in enumerate(cluster_means):
        cl_ind = cl[cluster_mean_index]
        cluster_elements = X[y == cl_ind,:]
        if len(cluster_elements):
            cluster_means[cluster_mean_index, :] = cluster_elements.mean(axis=0)
    # calculate the pair-wise distance
    distances = metrics.pairwise_distances(cluster_means, metric=metric)
    np.fill_diagonal(distances, 10**7)
    clpair = zip(range(len(cl)), np.argmin(distances, axis=1).tolist())
    clpair = [{elem1, elem2} for elem1, elem2 in clpair]
    clpair = list({frozenset(s) for s in clpair})
    clpair = [list(s) for s in clpair]
    merged_r2 = {}
    for pair in clpair:
        p0=cl[pair[0]]
        p1=cl[pair[1]]
        #print(p0,p1)
        x0 = X[y == p0, :]
        x0 = remove_outlier(x0)
        x1 = X[y == p1, :]
        x1 = remove_outlier(x1)
        #print(x0,x1)
        merged_r2[tuple([p0,p1])] = r2_adj([x0, x1])
    return merged_r2

def r2_split_two(X, y, metric='euclidean'):
    cl = np.unique(y)
    split_r2 = {}
    for c in cl:
        xx = X[y == c,:]
        xx = remove_outlier(xx)
        kmeans = KMeans(n_clusters=2, n_init='auto').fit(xx)
        yc = kmeans.labels_
        x0 = xx[np.where(yc == 0)[0], :]
        x1 = xx[np.where(yc == 1)[0], :]
        #print(c,x0,x1)
        if ((len(x0)==0) or (len(x1)==0)):
            split_r2[c] = 1.0
        else:
            split_r2[c] = r2_adj([x0, x1])
    return split_r2

def r2_ratio(X, y, metric='euclidean'):
    cc, cnt = np.unique(y,return_counts=True)
    y_in = cc[cnt > 10]
    X = X[np.isin(y,y_in),:]
    y = y[np.isin(y,y_in)]
    merged_r2 = r2_merge_near2(X, y, metric)
    split_r2 = r2_split_two(X,y, metric)
    cl = np.unique(y).tolist()
    res = []
    for k, v in merged_r2.items():
        cl_r2 = [split_r2[c] for c in split_r2.keys() if c not in k]
        cl_r2 = min(cl_r2)
        res.append(np.log(v/cl_r2))
    #print(res)
    return res, merged_r2, split_r2

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

if __name__ == '__main__':
    import argparse,os
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='1')
    args = parser.parse_args()
    npfiles='npfiles_val'
    imgdir = os.path.join(npfiles, "app")
    if not os.path.isdir(imgdir):
        os.mkdir(imgdir)
    
    estimates={}
    features={}
    labels={}
    for ii in range(4, 100, 5):
        file1='pro_output_{}.npz'.format(ii)
        file2 = os.path.join(npfiles, file1)
        data = np.load(file2)
        estimates[file1] = np.squeeze(data['estimates'])
        features[file1] = data['pro_features']
        labels[file1] = np.squeeze(data['labels'])

    pvalues = np.load('dip_dcl.npz')['pvalues1']
    models = np.load('dip_dcl.npz')['models'][1:]
    pvals = dict(zip(models,pvalues))
    ii = args.id
    folder = os.path.join(imgdir, ii)
    file1='pro_output_{}.npz'.format(ii)

    pval = np.round(pvals[os.path.join(npfiles, file1)],3)

    X=features[file1]
    y=labels[file1]
    print(len(y))
    print(np.unique(y, return_counts=True))
    tsne = TSNE(n_components=2, random_state=1)
    X_tsne = tsne.fit_transform(X)
    print(tsne.kl_divergence_)


    X_with_y = np.column_stack((X_tsne, y.reshape(-1, 1)))
    np.random.shuffle(X_with_y)
    X_tsne = X_with_y[:, :-1]
    y = X_with_y[:, -1]
    #y = y+1
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y)# symbol=m, size=s)
    fig.add_annotation(
        text="p-value: {}".format(pval),
        xref="paper", yref="paper",
        x=0, y=1,  # Set x and y to place the annotation in the left corner
        showarrow=False,  # Do not show arrow
        font=dict(
            family="Arial",  # Set the font family
            size=25,         # Set the font size # previously 22
            color="blue"      # Set the font color
        )
    )
    fig.update_layout(
        #title="t-SNE visualization of Custom Classification dataset",
        coloraxis_colorbar_title="Class labels",
        coloraxis_colorbar=dict(tickvals=[200,400,600,800,1000]),
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        margin=dict(l=0, r=0, t=0, b=0)  # Set the left, right, top, and bottom margins to 0
    )
 
    #fig.write_image(os.path.join(folder, 'true.png'))
    fig.write_image(folder+'_true.png')





