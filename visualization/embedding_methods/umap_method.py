"""
visualization/embedding_methods/umap_method.py

UMAP-based 2-D dimensionality reduction, registered under the name "umap".
"""

import numpy as np
from .base import EmbeddingMethod
from .registry import register


@register("umap")
class UMAPMethod(EmbeddingMethod):
    """
    UMAP dimensionality reduction (requires ``umap-learn``).

    Default parameters are tuned for image embedding spaces (cosine metric,
    moderate neighbourhood size).  Any parameter accepted by ``umap.UMAP``
    can be passed as a keyword argument, either here or via the
    ``embedding_params.umap`` key in the visualization config.

    Supports both one-shot ``fit_transform`` and the separate ``fit`` /
    ``transform`` API used by ``animate_embeddings.py`` for stable layouts.
    """

    def __init__(
        self,
        n_components: int = 2,
        metric: str = "cosine",
        n_neighbors: int = 30,
        min_dist: float = 0.05,
        random_state: int = 42,
        **kwargs,
    ):
        self.params = dict(
            n_components=n_components,
            metric=metric,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            **kwargs,
        )
        self._reducer = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        import umap as umap_lib
        self._reducer = umap_lib.UMAP(**self.params)
        return self._reducer.fit_transform(embeddings)

    def fit(self, embeddings: np.ndarray) -> "UMAPMethod":
        import umap as umap_lib
        self._reducer = umap_lib.UMAP(**self.params)
        self._reducer.fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("Call fit() before transform().")
        return self._reducer.transform(embeddings)
