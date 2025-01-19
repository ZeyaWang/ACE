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





