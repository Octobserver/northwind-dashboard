# A Python framework that employs Optuna for optimizing the hyperparameters of a combined pipeline: 
#   1. data preprocessing (scaling/normalization)
#   2. dimensionality reduction 
#   3. clustering
import optuna
import umap.umap_ as umap
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def clean_data(data, cols_to_transform, cols_to_retain):
    # TODO: data preprocessing
    # TODO: use one-hot encoder on cols_to_transform
    # TODO: experiment with StandardScaler and Normalizer
    
    pass

def objective(trial):
    # TODO: 3 dimensionality reduction algorithms (PCA, t-SNE, UMAP) x 5 clustering techniques (KMeans, HDBSCAN, OPTICS, Agglomerative Clustering, Spectral Clustering) experimental design
    # TODO: use OPTUNA to find optimal hyperparameters for each experiment
    # TODO: report maximal hyperparameters and silhouette score, and visualize clusters in 3D
    reducer_name = trial.suggest_categorical("reducer", ["PCA", "t-SNE", "UMAP"])
    if reducer_name == 'PCA':
        pca_n = trial.suggest_int("pca_n", 3, 30, log=True)
        # TODO add more hyperparameters
        reducer = PCA(n_components=pca_n)
    elif reducer_name == 't-SNE':
        tsne_n = trial.suggest_int("tsne_n", 3, 30, log=True)
        # TODO add more hyperparameters
        reducer = TSNE(n_components = tsne_n)
    else:
        umap_n = trial.suggest_int("umap_n", 3, 30, log=True)
        # TODO add more hyperparameters
        reducer = umap.UMAP(n_components = umap_n)
    
    