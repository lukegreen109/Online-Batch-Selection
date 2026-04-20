"""
visualization/embedding_methods/base.py

Abstract base class for 2-D dimensionality-reduction methods used by the
Visualizer. Subclass this to plug in any custom reduction algorithm.

Minimal contract:
    fit_transform(embeddings)  →  (N, 2) array   [required]
    fit(embeddings)            →  self            [optional, for reuse]
    transform(embeddings)      →  (N, 2) array   [optional, for reuse]
"""

from abc import ABC, abstractmethod
import numpy as np


class EmbeddingMethod(ABC):
    """
    Base class for embedding dimensionality-reduction methods.

    Subclass and implement ``fit_transform`` at minimum.  Override ``fit``
    and ``transform`` if your method supports incremental / streaming use
    (e.g. for animate_embeddings.py which fits once and transforms each epoch).

    Example — registering a custom method::

        from visualization.embedding_methods import register, EmbeddingMethod

        @register("pca")
        class PCAMethod(EmbeddingMethod):
            def __init__(self, n_components=2, **kwargs):
                from sklearn.decomposition import PCA
                self._reducer = PCA(n_components=n_components, **kwargs)

            def fit_transform(self, embeddings):
                return self._reducer.fit_transform(embeddings)

            def fit(self, embeddings):
                self._reducer.fit(embeddings)
                return self

            def transform(self, embeddings):
                return self._reducer.transform(embeddings)
    """

    @abstractmethod
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit the reduction model and project ``embeddings`` to 2-D in one pass.

        Args:
            embeddings: ``(N, D)`` float array of high-dimensional embeddings.

        Returns:
            ``(N, 2)`` float array of 2-D coordinates.
        """
        ...

    def fit(self, embeddings: np.ndarray) -> "EmbeddingMethod":
        """
        Fit on reference embeddings without returning the projection.

        Override this if you want to fit once (e.g. on the final epoch) and
        then call ``transform`` on earlier epochs for a stable layout.

        Returns:
            self, for chaining.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement fit/transform separately. "
            "Use fit_transform instead, or override fit() and transform()."
        )

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project new embeddings using a previously fitted model.

        Must call ``fit`` (or ``fit_transform``) first.

        Returns:
            ``(N, 2)`` float array of 2-D coordinates.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement fit/transform separately. "
            "Use fit_transform instead, or override fit() and transform()."
        )
