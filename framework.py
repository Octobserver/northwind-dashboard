# A Python framework that employs Optuna for optimizing the hyperparameters of a combined pipeline: 
#   1. data preprocessing (scaling/normalization)
#   2. dimensionality reduction 
#   3. clustering
import optuna
import traceback
import numpy as np
import pandas
import umap.umap_ as umap
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, HDBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering

raw_data = pandas.read_csv("./sales_data.csv")
COLS_TO_TRANSFORM = ['OrderDate', 'ShippedDate', 'CustomerCountry', 'CustomerCity', 'CustomerRegion', 'ProductName', 'CategoryName', 'SupplierCountry', 'SupplierRegion']
COLS_TO_RETAIN = ['TotalPrice', 'UnitPrice', 'Quantity', 'Discount']

def clean_data(data, cols_to_transform, cols_to_retain):
    # data preprocessing
    # use one-hot encoder on cols_to_transform
    # TODO: experiment with StandardScaler and Normalizer
    enc = OneHotEncoder(drop='first', sparse_output=False, min_frequency=int(data.shape[0]*0.005), handle_unknown='infrequent_if_exist', dtype = int)
    transformed = enc.fit_transform(data[cols_to_transform])
    processed_data = np.concatenate([data[cols_to_retain], transformed], axis=1)
    # scaled_processed_data = StandardScaler().fit_transform(processed_data)
    
    return processed_data

def objective(trial):
    # 3 dimensionality reduction algorithms (PCA, t-SNE, UMAP) x 5 clustering techniques (KMeans, HDBSCAN, OPTICS, Agglomerative Clustering, Spectral Clustering) experimental design
    # use OPTUNA to find optimal hyperparameters for each experiment
    # report maximal hyperparameters and silhouette score, and visualize clusters in 3D
    data = clean_data(raw_data, COLS_TO_TRANSFORM, COLS_TO_RETAIN)
    reducer_name = trial.suggest_categorical("reducer", ["PCA", "t-SNE", "UMAP"])
    clusterer_name = trial.suggest_categorical("clusterer", ["KMeans", "HDBSCAN", "OPTICS", "Agglomerative", "Spectral"])
    reducer_n, clusterer_n = 0, 0

    if reducer_name == 'PCA':

        reducer_n = trial.suggest_int("pca_n", 2, 20)
        whiten = trial.suggest_categorical("whiten", [True, False])
        solver = trial.suggest_categorical("solver", ["full", "arpack", "randomized"])
        pca_tol = trial.suggest_float("pca_tol", 0.0, 10.0)
        n_oversamples = trial.suggest_int("n_oversamples", 1, 20)
        normalizer = trial.suggest_categorical("normalizer", ["auto", "QR", "LU", "none"])
        reducer = PCA(n_components=reducer_n, whiten=whiten, svd_solver=solver, tol=pca_tol, n_oversamples=n_oversamples, power_iteration_normalizer=normalizer, random_state=99)

    elif reducer_name == 't-SNE':

        tsne_method = trial.suggest_categorical("tsne_method", ["barnes_hut", "exact"])
        if tsne_method == "barnes_hut":
            reducer_n = trial.suggest_int("tsne_n", 1, 3)
        else:
            reducer_n = trial.suggest_int("tsne_n", 2, 20)
        perplexity = trial.suggest_float("perplexity", 5.0, 50.0)
        tightness = trial.suggest_float("tightness", 5.0, 20.0)
        tsne_lr = trial.suggest_float("tsne_lr", 10.0, 100.0, step=5.0)
        tsne_metric = trial.suggest_categorical("tsne_metric", ['sokalsneath', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'yule', 'nan_euclidean', 'cosine', 'wminkowski', 'correlation', 'minkowski', 'sqeuclidean', 'chebyshev', 'haversine', 'dice', 'cityblock', 'matching', 'seuclidean', 'hamming', 'l2', 'euclidean', 'braycurtis', 'mahalanobis', 'jaccard', 'manhattan', 'canberra', 'l1'])
        tsne_init = trial.suggest_categorical("tsne_init", ["random", "pca"])
        reducer = TSNE(n_components = reducer_n, method=tsne_method, perplexity=perplexity, early_exaggeration=tightness, learning_rate=tsne_lr, metric=tsne_metric, init=tsne_init, random_state=99)

    else:

        reducer_n = trial.suggest_int("umap_n", 2, 20)
        umap_n_neighbors = trial.suggest_int("umap_n_neighbors", 2, 502, step = 5)
        min_dist= trial.suggest_float("min_dist", 0.0, 0.99, step=0.05)
        umap_metric =  trial.suggest_categorical("umap_metric", ["braycurtis", "canberra", "chebyshev", "correlation", "cosine", "dice", "euclidean", "hamming", "haversine", "jaccard", "mahalanobis", "manhattan", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "yule"])
        reducer = umap.UMAP(n_components = reducer_n, n_neighbors=umap_n_neighbors, min_dist=min_dist, metric=umap_metric)

    if clusterer_name == 'KMeans':

        clusterer_n = trial.suggest_int("kmeans_n", 2, 20)
        kmeans_init = trial.suggest_categorical("kmeans_init", ["random", "k-means++"])
        if kmeans_init == "random":
            kmeans_n_init = trial.suggest_int("kmeans_n_init", 1, 20)
        else:
            kmeans_n_init = 1
        kmeans_tol = trial.suggest_categorical("kmeans_tol", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
        kmeans_algo = trial.suggest_categorical("kmeans_algo", ["lloyd", "elkan"])
        clusterer = KMeans(n_clusters=clusterer_n, init=kmeans_init, n_init=kmeans_n_init, tol=kmeans_tol, algorithm=kmeans_algo, random_state=99)

    elif clusterer_name == 'HDBSCAN':

        hdbscan_n = trial.suggest_int("hdbscan_n", 2, 502, step = 5)
        clusterer_n = data.shape[0] // hdbscan_n
        min_samples = trial.suggest_int("min_samples", 2, 200, step = 2)
        epsilon = trial.suggest_float("epsilon", 0.0, 10.0, step=0.1)
        hdbscan_metric =  trial.suggest_categorical("hdbscan_metric", [ "l1", "l2", "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"])
        alpha = trial.suggest_float("alpha", 0.1, 10.0, step=0.1)
        hdbscan_algo = trial.suggest_categorical("hdbscan_algo", ["auto", "brute", "kd_tree", "ball_tree"])
        hdbscan_leaf_size = trial.suggest_int("hdbscan_leaf_size", 10, 100, step = 5)
        hdbscan_method = trial.suggest_categorical("hdbscan_method", ["eom", "leaf"])
        clusterer = HDBSCAN(min_cluster_size = hdbscan_n, min_samples=min_samples, cluster_selection_epsilon=epsilon, metric=hdbscan_metric, alpha=alpha, algorithm=hdbscan_algo, leaf_size=hdbscan_leaf_size, cluster_selection_method=hdbscan_method)

    elif clusterer_name == 'OPTICS':

        optics_n = trial.suggest_int("optics_n", 2, 502, step = 5)
        clusterer_n = data.shape[0] // optics_n
        max_eps = trial.suggest_categorical("max_eps", [1.0, 10.0, 100.0, 1000.0, 10000.0, np.inf])
        optics_metric = trial.suggest_categorical("optics_metric", ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"])
        if optics_metric == "manhattan":
            p = trial.suggest_int("p", 2, 10)
        else:
            p = 2
        optics_method = trial.suggest_categorical("optics_method", ["xi", "dbscan"])
        xi = trial.suggest_float("xi", 0.0, 1.0, step = 0.05)
        optics_algo = trial.suggest_categorical("optics_algo", ["auto", "ball_tree", "kd_tree", "brute"])
        optics_leaf_size = trial.suggest_int("optics_leaf_size", 10, 100, step = 5)
        clusterer = OPTICS(min_samples= optics_n, max_eps=max_eps, metric=optics_metric, p=p, cluster_method=optics_method, xi=xi, algorithm=optics_algo, leaf_size=optics_leaf_size)

    elif clusterer_name == 'Agglomerative':
        
        clusterer_n = trial.suggest_int("agglo_n", 2, 20)
        linkage = trial.suggest_categorical("linkage", ["ward", "complete", "average", "single"])
        if linkage == "ward":
            agglo_metric = "euclidean"
        else:
            agglo_metric =  trial.suggest_categorical("agglo_metric", ["euclidean", "l1", "l2", "manhattan", "cosine"])
        clusterer = AgglomerativeClustering(n_clusters = clusterer_n, metric=agglo_metric, linkage=linkage)

    else:
        clusterer_n = trial.suggest_int("spec_cluster", 2, 20)
        spec_n = trial.suggest_int("spec_n", 2, 20)
        eigen_solver = trial.suggest_categorical("eigen_solver", ["arpack", "lobpcg", "amg"])
        spec_n_init = trial.suggest_int("spec_n_init", 1, 20)
        spec_gamma = trial.suggest_float("spec_gamma", 0.1, 10.0, step=0.5)
        affinity = trial.suggest_categorical("affinity", ["nearest_neighbors", "rbf", "precomputed", "precomputed_nearest_neighbors"])
        spec_n_neighbors = trial.suggest_int("spec_n_neighbors", 2, 500, step = 5)
        assign_labels = trial.suggest_categorical("assign_lables", ["kmeans", "discretize", "cluster_qr"])
        degree = trial.suggest_categorical("degree", [1.0, 2.0, 3.0, 4.0, 5.0])
        coef0 = trial.suggest_float("coef0", 0.0, 100.0)
        clusterer = SpectralClustering(n_clusters = clusterer_n, n_components=spec_n, eigen_solver=eigen_solver, n_init=spec_n_init, gamma=spec_gamma, affinity=affinity, n_neighbors=spec_n_neighbors, assign_labels=assign_labels, degree=degree, coef0=coef0, random_state=99)

    try:
        embeddings = reducer.fit_transform(data)
        labels = clusterer.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        return score
    
    except Exception as e:
        print(f"Error with combination: {reducer} (n_components={reducer_n}), {clusterer} (n_clusters={clusterer_n})")
        print(traceback.format_exc())
        return -1  # Return a bad score if an error occurs

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000)
    print(study.best_trial)
    # Print the best parameters and the best score
    print("Best score:", study.best_value)
    print("Best params:", study.best_params)

    # Save dataframe
    session = study.trials_dataframe()
    session.to_csv("./session.csv")




    
    