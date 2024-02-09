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
import numpy as np
import umap
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import streamlit as st


def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

def run_umap(df, cols_to_transform, cols_to_retain, n_components=3):

    enc = OneHotEncoder(drop='first', sparse_output=False, min_frequency=int(df.shape[0]*0.05), handle_unknown='infrequent_if_exist', dtype = int)
    transformed = enc.fit_transform(df[cols_to_transform])
    processed_data = np.concatenate([df[cols_to_retain], transformed], axis=1)

    reducer = umap.UMAP(n_components = n_components)
    scaled_processed_data = StandardScaler().fit_transform(processed_data)
    embedding = reducer.fit_transform(scaled_processed_data)
    return embedding
