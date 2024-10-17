# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap
import streamlit as st
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

def clean_data(data, cols_to_transform, cols_to_retain):
    """
    preprocesses and transforms data
    parameters:
        data: the data to process
        cols_to_transform: columns in the data to transform with a one-hot encoder
        cols_to_retain: columns in the data to retain their orignal formats
    """
    enc = OneHotEncoder(
        drop="first",
        sparse_output=False,
        min_frequency=int(data.shape[0] * 0.005),
        handle_unknown="infrequent_if_exist",
        dtype=int,
    )
    transformed = enc.fit_transform(data[cols_to_transform])
    processed_data = np.concatenate([data[cols_to_retain], transformed], axis=1)

    return processed_data

def reduce_and_cluster(data, pca_params, kmeans_params, umap_params, use_constrained=False):
    """
    performs dimensional reduction and clustering given a set of parameters and generate figures for the resulting clusters with 2d umap
    parameters:
        data: the dataset
        pca_params: the set of parameters for PCA
        kmeans_params: the set of parameters for KMeans
        umap_params: the set of parameters for 2d UMAP
        use_constrained: whether or not to use KMeans constrained
    """
    reducer = PCA(**pca_params)
    embeddings = reducer.fit_transform(data)
    
    clusterer = KMeansConstrained(**kmeans_params) if use_constrained else KMeans(**kmeans_params)
    labels = clusterer.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    
    cluster_counts = dict(zip(*np.unique(labels, return_counts=True)))
    
    umap_2d = umap.UMAP(**umap_params)
    umap_embeddings = umap_2d.fit_transform(data)
    
    df_plot = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
    df_plot['Cluster'] = labels
    
    return score, df_plot, cluster_counts

