import fiftyone as fo
import torch
#import splinecam

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

from .fiftyone_utils import visualize_with_fiftyone
from .splinecam_utils import visualize_with_splinecam

class Visualizer:
    """
    A class for visualizing model predictions and data samples using FiftyOne and SplineCam.
    Accepts a pre-trained model and a dataset, and provides methods to visualize predictions
    """
    def __init__(self, config, logger, model=None, epoch=None):
        self.config = config
        self.logger = logger
        self.seed = config["seed"]
        self.dataset_name = config["dataset"]["name"].lower()
        self.epoch = epoch

        # visualization config
        vis_cfg = config['visualization']
        
        self.save_dir = vis_cfg["save_dir"]
        self.embedding_methods = vis_cfg.get("embedding_methods", ["tsne", "umap"])
        os.makedirs(self.save_dir, exist_ok=True) # for saving plots

        # embedding params
        embedding_params = vis_cfg["embedding_params"]
        self.tsne_params = embedding_params["tsne"]
        self.umap_params = embedding_params["umap"]

    def visualize_embeddings(
        self, 
        embs, 
        labels, 
        epoch=None, 
        selected_idx=None, 
        use_fiftyone=False, 
        filepaths=None,
    ):
        """
        Visualize embeddings of data points using dimensionality reduction or FiftyOne.

        Args:
            embs (np.ndarray or torch.Tensor): Embedding vectors of shape (N, D)
            labels (np.ndarray or torch.Tensor): Class labels of shape (N,)
            epoch (int, optional): Epoch number, used for filenames and dataset naming
            selected_idx (list or np.ndarray, optional): Indices of samples to highlight
            use_fiftyone (bool, optional): If True, visualize using FiftyOne; otherwise, matplotlib
            filepaths (list of str, optional): Paths to images corresponding to embeddings (required for FiftyOne)
        """
        def _get_reducer(method):
            if method == "tsne":
                return TSNE(**self.tsne_params, random_state=self.seed)
            if method == "umap":
                return umap.UMAP(**self.umap_params, random_state=self.seed)
            raise ValueError(f"Unknown method: {method}")

        embs = np.array(embs)
        labels = np.array(labels)

        if filepaths is None:
            filepaths = ["placeholder.jpg"] * len(embs) # dummy paths

        if use_fiftyone:
            for method in self.embedding_methods:
                self.logger.info(f"Visualizing embeddings with FiftyOne (method: {method})...")
                visualize_with_fiftyone(
                    embs,
                    labels,
                    filepaths=filepaths,
                    selected_idx=selected_idx,
                    epoch=self.epoch,
                    method=method,
                    persistent=True,
                )
        else:
            for method in self.embedding_methods:
                self.logger.info(f"Visualizing embeddings with {method}...")
                reduced = _get_reducer(method).fit_transform(embs)
                self._plot_and_save(reduced, labels, method, epoch, selected_idx)

    def _plot_and_save(self, reduced, labels, method, epoch, selected_idx=None):
        plt.figure(figsize=(6,6))

        # Plot all points
        scatter = plt.scatter(
            reduced[:,0], reduced[:,1],
            c=labels, cmap="tab10", s=5, alpha=0.6, label="All"
        )

        # Get handles & labels for class legend
        handles, labels_legend = scatter.legend_elements()
        
        # Plot selected points if provided
        if selected_idx is not None and len(selected_idx) > 0:
            selected_idx = np.array(selected_idx)
            selected_scatter = plt.scatter(
                reduced[selected_idx,0], reduced[selected_idx,1],
                facecolors='none', edgecolors='red', s=30, linewidths=0.8,
                label="Selected", marker='o'
            )
            handles.append(selected_scatter)
            labels_legend.append("Selected")

        # Add legend and set the title
        plt.legend(handles, labels_legend, title="Classes", loc="best")
        plt.title(f"{method} Embeddings (epoch={epoch})")

        fname = os.path.join(self.save_dir, f"embeddings_{method}_epoch{epoch}.png")
        plt.savefig(fname, dpi=300)
        plt.close()

