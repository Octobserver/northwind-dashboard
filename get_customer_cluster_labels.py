#!/usr/bin/env python3
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
from datetime import datetime

def preprocess_and_cluster(data, pca_params, kmeans_params, umap_params, use_constrained=False):
    reducer = PCA(**pca_params)
    embeddings = reducer.fit_transform(data)
    
    clusterer = KMeansConstrained(**kmeans_params) if use_constrained else KMeans(**kmeans_params)
    labels = clusterer.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    
    cluster_counts = dict(zip(*np.unique(labels, return_counts=True)))
    
    umap_2d = umap.UMAP(**umap_params)
    umap_embeddings = umap_2d.fit_transform(data)
    
    #df_plot = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
    #df_plot['Cluster'] = labels
    
    #sns.set_theme(style="darkgrid", palette="dark")
    #plt.figure(figsize=(10, 6))
    #sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster', data=df_plot, palette="deep")
    #plt.title(f'Silhouette Score: {score:.2f}, Cluster Counts: {cluster_counts}')
    #plt.show()
    
    return score, labels, cluster_counts

X = load('serialized_data.joblib')
pca_params = {
    'n_components': 5,
    'whiten': False,
    'svd_solver': 'randomized',
    'tol': 8,
    'n_oversamples': 4,
    'power_iteration_normalizer': 'none',
    'random_state': 99
}

kmeans_params = {
    'n_clusters': 7,
    'init': "k-means++",
    'n_init': 10,
    'tol': 1.0,
    'size_min': 120,
    'random_state': 99
}

umap_params = {
    'n_neighbors': 75,
    'min_dist': 0.50, 
    'n_components': 2,
    'random_state': 99
}

score, labels, cluster_counts = preprocess_and_cluster(X, pca_params, kmeans_params, umap_params, use_constrained=True)
# assign labels and create markdowns

data = pd.read_csv("sales_data.csv")
data["ClusterLabel"] = labels
order_dates = data["OrderDate"]
lst = []
for i in range(data.shape[0]):
  date_obj = datetime.strptime(order_dates[i], '%Y-%m-%d')
  lst.append([date_obj.year, date_obj.month, date_obj.day])

data["OrderYear"] = [x[0] for x in lst]
data["OrderMonth"] = [x[1] for x in lst]

clusters = []
for label in np.unique(labels, return_counts=False):
  clusters.append(data[data["ClusterLabel"] == label])


