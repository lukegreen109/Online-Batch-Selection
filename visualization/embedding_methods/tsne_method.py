"""
visualization/embedding_methods/tsne_method.py

t-SNE-based 2-D dimensionality reduction, registered under the name "tsne".
"""

import numpy as np
from .base import EmbeddingMethod
from .registry import register


@register("tsne")
class TSNEMethod(EmbeddingMethod):
    """
    t-SNE dimensionality reduction (via ``sklearn.manifold.TSNE``).

    t-SNE does not support a separate ``transform`` step (it must refit for
    each new set of points), so only ``fit_transform`` is implemented.

    Any parameter accepted by ``sklearn.manifold.TSNE`` can be passed as a
    keyword argument or via the ``embedding_params.tsne`` config key.
    """

    def __init__(
        self,
        n_components: int = 2,
        metric: str = "cosine",
        random_state: int = 42,
        **kwargs,
    ):
        self.params = dict(
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            **kwargs,
        )

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE
        reducer = TSNE(**self.params)
        return reducer.fit_transform(embeddings)
