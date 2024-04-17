# A Python framework that employs Optuna for optimizing the hyperparameters of a combined pipeline: 
#   1. data preprocessing (scaling/normalization)
#   2. dimensionality reduction 
#   3. clustering
import optuna
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
    if reducer_name == 'PCA':
        pca_n = trial.suggest_int("pca_n", 2, 20, log=True)
        # TODO add more hyperparameters
        reducer = PCA(n_components=pca_n)
    elif reducer_name == 't-SNE':
        tsne_n = trial.suggest_int("tsne_n", 1, 3, log=True)
        # TODO add more hyperparameters
        reducer = TSNE(n_components = tsne_n)
    else:
        umap_n = trial.suggest_int("umap_n", 2, 20, log=True)
        # TODO add more hyperparameters
        reducer = umap.UMAP(n_components = umap_n)
    
    if clusterer_name == 'KMeans':
        kmeans_n = trial.suggest_int("kmeans_n", 3, 10, log=True)
        clusterer = KMeans(n_clusters=8)
    elif clusterer_name == 'HDBSCAN':
        hdbscan_n = trial.suggest_int("hdbscan_n", 3, 10, log=True)
        clusterer = HDBSCAN(min_cluster_size = hdbscan_n)
    elif clusterer_name == 'OPTICS':
        optics_n = trial.suggest_int("optics_n", 3, 10, log=True)
        clusterer = OPTICS(min_samples= optics_n)
    elif clusterer_name == 'Agglomerative':
        agglo_n = trial.suggest_int("agglo_n", 3, 10, log=True)
        clusterer = AgglomerativeClustering(n_clusters = agglo_n)
    else:
        spec_n = trial.suggest_int("spec_n", 3, 10, log=True)
        clusterer = SpectralClustering(n_clusters = spec_n)

    
    embeddings = reducer.fit_transform(data)
    labels = clusterer.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)

    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print(study.best_trial)




    
    