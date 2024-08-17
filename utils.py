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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from k_means_constrained import KMeansConstrained
from sklearn.cluster import AgglomerativeClustering, KMeans
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
    # data preprocessing
    # use one-hot encoder on cols_to_transform
    # TODO: experiment with StandardScaler and Normalizer
    enc = OneHotEncoder(drop='first', sparse_output=False, min_frequency=int(data.shape[0]*0.005), handle_unknown='infrequent_if_exist', dtype = int)
    transformed = enc.fit_transform(data[cols_to_transform])
    processed_data = np.concatenate([data[cols_to_retain], transformed], axis=1)
    # scaled_processed_data = StandardScaler().fit_transform(processed_data)
    dump(processed_data, "serialized_data.joblib", compress=9)
    
    return processed_data

def dimentional_reduction(df, cols_to_transform, cols_to_retain):

    processed_data = clean_data(df, cols_to_transform, cols_to_retain)

    reducer = PCA(n_components=5, whiten=False, svd_solver='randomized', tol=9.633806975, n_oversamples=4, power_iteration_normalizer='none', random_state=99)
    embeddings = reducer.fit_transform(processed_data)
    return embeddings


def clustering(embeddings):
    #clusterer = KMeans(n_clusters=4, init="k-means++", n_init=8, tol=1.0, algorithm="elkan", random_state=99)
    clusterer = KMeansConstrained(n_clusters=5, init="k-means++", n_init=10, tol=1.0, size_min=100, random_state=99)
    labels = clusterer.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    u, c = np.unique(labels, return_counts=True)
    print(dict(zip(u, c)))
    return score, labels

