"""
visualization/embedding_methods/
=================================

Plugin registry for 2-D dimensionality-reduction methods used by the
Visualizer (voxel51_vis.py).

Built-in methods
----------------
- ``"umap"``  — UMAP via umap-learn
- ``"tsne"``  — t-SNE via sklearn
- ``"pca"``   — PCA via sklearn (reference example)

Adding a custom method
----------------------
1. Create a new file here (e.g. my_method.py)
2. Subclass EmbeddingMethod and decorate with @register — no changes
   needed anywhere else:

        from visualization.embedding_methods.registry import register
        from visualization.embedding_methods.base import EmbeddingMethod

        @register("mymethod")
        class MyMethod(EmbeddingMethod):
            def __init__(self, **kwargs): ...
            def fit_transform(self, embeddings): ...

3. Import the file here so the decorator runs at startup.
4. Reference it by name in your visualization config:
        embedding_methods: ["mymethod"]
"""

from .base import EmbeddingMethod
from .registry import register, get_method, list_methods

# Importing each method triggers its @register decorator.
from .umap_method import UMAPMethod
from .tsne_method import TSNEMethod
from .pca_method import PCAMethod

__all__ = [
    "EmbeddingMethod",
    "UMAPMethod",
    "TSNEMethod",
    "PCAMethod",
    "register",
    "get_method",
    "list_methods",
]
