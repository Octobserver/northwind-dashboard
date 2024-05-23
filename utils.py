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
from framework import clean_data
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans


def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

def dimentional_reduction(df, cols_to_transform, cols_to_retain):

    processed_data = clean_data(df, cols_to_transform, cols_to_retain)

    reducer = reducer =  reducer = PCA(n_components=19, whiten=False, svd_solver='randomized', tol=7.33267839830423, n_oversamples=3, power_iteration_normalizer='none', random_state=99)
    embeddings = reducer.fit_transform(processed_data)
    return embeddings


def clustering(embeddings):
    clusterer = AgglomerativeClustering(n_clusters = 7, metric='euclidean', linkage='complete')
    #clusterer = KMeans(n_clusters=3, init="random", n_init=9, tol=1.0, algorithm="elkan", random_state=99)
    labels = clusterer.fit_predict(embeddings)
    u, c = np.unique(labels, return_counts=True)
    print(dict(zip(u, c)))
    return labels

