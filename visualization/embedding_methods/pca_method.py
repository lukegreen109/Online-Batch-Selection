"""
visualization/embedding_methods/pca_method.py

PCA-based 2-D dimensionality reduction, registered under the name "pca".

This serves as a reference example for adding custom dimensionality reduction
methods.  To add your own:

    1. Create a new file here (e.g. my_method.py)
    2. Subclass EmbeddingMethod and implement fit_transform()
    3. Decorate the class with @register("my_method_name")
    4. Import it in __init__.py so it gets registered at startup
    5. Reference it by name in your visualization config:
           embedding_methods: ["my_method_name"]
"""

import numpy as np
from sklearn.decomposition import PCA

from .base import EmbeddingMethod
from .registry import register


@register("pca")
class PCAMethod(EmbeddingMethod):
    """
    PCA dimensionality reduction to 2-D (requires scikit-learn, already a
    project dependency).

    Unlike UMAP and t-SNE, PCA is linear and deterministic — useful as a
    fast sanity-check or baseline visualization.

    Config example::

        embedding_methods: ["pca"]
        embedding_params:
          pca:
            n_components: 2    # must be 2 for FiftyOne visualization
            whiten: false
    """

    def __init__(self, n_components: int = 2, **kwargs):
        self.params = dict(n_components=n_components, **kwargs)
        self._reducer = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        self._reducer = PCA(**self.params)
        return self._reducer.fit_transform(embeddings)

    def fit(self, embeddings: np.ndarray) -> "PCAMethod":
        self._reducer = PCA(**self.params)
        self._reducer.fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("Call fit() before transform().")
        return self._reducer.transform(embeddings)
