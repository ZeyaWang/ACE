from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import argparse
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()
parser.add_argument('--option', default='dense', help='dense or sparse')
parser.add_argument('--seed', type=int, default=10, help='seed value')
parser.add_argument('--cluster_std', type=float, default=1, help='sample standard deviation')
args = parser.parse_args()
option = args.option
seed_value = args.seed
np.random.seed(seed_value)

# Parameters for high-dimensional data
n_total_features = 50000
n_embedding_features = 100
n_samples = 1000     # Number of samples
n_latent_features = 50      # Number of features (dimensions)
n_clusters = 20     # Number of clusters
cluster_std = args.cluster_std 

if option == 'dense':
    n_dense_features = 10000
    n_noise_features = n_total_features - n_dense_features - n_latent_features  # Number of additional noisy dimensions
else:
    n_noise_features = n_total_features - n_latent_features  # Number of additional noisy dimensions


latent_cov_matrix = (cluster_std ** 2) * np.eye(n_latent_features)
if option == 'dense':
    dense_cov_matrix = (cluster_std ** 2) * np.eye(n_dense_features)

# Generate high-dimensional data
std = 1
mean_clusters = np.random.multivariate_normal(np.zeros(n_latent_features), (std ** 2) * np.eye(n_latent_features), size=n_clusters)
print('mean_clusters', mean_clusters.shape)
if option == 'dense':
    std_dense = 0.1*std
    mean_clusters_dense = np.random.multivariate_normal(np.zeros(n_dense_features), (std_dense ** 2) * np.eye(n_dense_features), size=n_clusters)
    print('mean_clusters_dense', mean_clusters_dense.shape)

# Initialize data array
X_latent = np.zeros((n_samples, n_latent_features))  
if option == 'dense':
    X_dense = np.zeros((n_samples, n_dense_features)) 

y = np.zeros(n_samples, dtype=int)
n_samples_per_cluster =  int(n_samples / n_clusters)

stop = 0
for i in range(n_clusters):
    start, stop = stop, stop + n_samples_per_cluster
    y[start:stop] = i
    # Generate data for the current cluster
    cluster_data = np.random.multivariate_normal(
        mean=mean_clusters[i], cov=latent_cov_matrix, size=n_samples_per_cluster
    )
    # Assign generated data to the corresponding indices in X
    X_latent[start:stop] = cluster_data
    if option == 'dense':
        cluster_data_dense = np.random.multivariate_normal(
            mean=mean_clusters_dense[i], cov=dense_cov_matrix, size=n_samples_per_cluster
        )
        X_dense[start:stop] = cluster_data_dense

if option == 'dense':
    print(f"X_dense shape: {X_dense.shape}")
    X = np.hstack((X_latent, X_dense))
else:
    X = X_latent

X_noise = np.random.normal(0, 1, size=(n_samples, n_noise_features))  # Random Gaussian noise
X_expanded = np.hstack((X, X_noise))
print(f"X_expanded shape: {X_expanded.shape}, X shape: {X.shape}, X_latent shape: {X_latent.shape},  y shape: {y.shape}")


simtype = 'sim_{}_{}'.format(option, args.cluster_std)
eval_dir_root = '{}'.format(simtype)
if not os.path.isdir(eval_dir_root):
    os.mkdir(eval_dir_root)
eval_dir = os.path.join(eval_dir_root, str(seed_value))
if not os.path.isdir(eval_dir):
    os.mkdir(eval_dir)

scaler = StandardScaler()
X_expanded = scaler.fit_transform(X_expanded)
if option == 'dense':
    X_dense_noise = np.hstack((X_dense, X_noise))
else:
    X_dense_noise = X_noise

if args.cluster_std <= 1.5:
    clist = [10, 15, 20, 25]
else:
    clist = [10, 25, 40, 50]
for n_inf in clist:
    n_ninf = n_embedding_features - n_inf
    inf_columns = np.random.choice(X_latent.shape[1], n_inf, replace=False)
    inf_Z = X_latent[:, inf_columns]
    ninf_columns = np.random.choice(X_dense_noise.shape[1], n_ninf, replace=False)
    ninf_Z = X_dense_noise[:, ninf_columns]
    Z_raw = np.hstack((inf_Z, ninf_Z))
    scaler = StandardScaler()
    y_features = scaler.fit_transform(Z_raw)    
    print(y_features.shape)
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    kmeans.fit(y_features)
    y_pred = kmeans.labels_
    nmi = normalized_mutual_info_score(y_pred, y)
    np.savez(os.path.join(eval_dir, 'output{}_{}_{}_{}_{}.npz'.format(n_inf, n_ninf, 'None', 'None', nmi)),
        y_features=y_features, y_pred=y_pred, truth=y, Z_raw=Z_raw)
    for n_neighbors in [5, 20, 50]:
        for min_dist in [0.1, 0.5, 0.99]:
            projector = umap.UMAP(n_components=n_embedding_features, random_state=1, metric='euclidean', n_neighbors=n_neighbors, min_dist=min_dist)
            Z_map = projector.fit_transform(Z_raw)
            scaler = StandardScaler()
            y_features = scaler.fit_transform(Z_map)           
            print(y_features.shape)
            kmeans = KMeans(n_clusters=n_clusters, random_state=1)
            kmeans.fit(y_features)
            y_pred = kmeans.labels_
            nmi = normalized_mutual_info_score(y_pred, y)
            np.savez(os.path.join(eval_dir, 'output{}_{}_{}_{}_{}.npz'.format(n_inf, n_ninf, n_neighbors, min_dist, nmi)),
                y_features=y_features, y_pred=y_pred, truth=y, Z_raw=Z_raw, Z_map=Z_map)

np.savez(os.path.join(eval_dir,'sim.npz'),
         X=X_expanded, y=y)



