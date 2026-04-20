"""
tests/test_visualization.py

Integration tests for the visualization pipeline.

Uses a small MNIST subset (real data, auto-downloaded by torchvision) to
exercise the full pipeline end-to-end without any synthetic data:

  1. Download MNIST → take a 300-sample training slice
  2. Populate a FiftyOne dataset from that DataLoader
  3. Train a SmallCNN for a few epochs on the slice
  4. Extract 256-D feature embeddings at milestone epochs
  5. Compute UMAP brain runs and store in FiftyOne
  6. Export snapshot to disk
  7. Verify brain runs, embedding shapes, and snapshot files
  8. Register and use a custom PCA embedding method

Run (from project root):
    uv run pytest tests/test_visualization.py -v -s

The tests download MNIST on first run (~11 MB); subsequent runs use the
cache in ./_MNIST.
"""

import os
import sys
import pytest
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Ensure project root is importable regardless of where pytest is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Small_cnn import small_cnn
from visualization import Visualizer, embedding_methods, register
from visualization.embedding_methods import EmbeddingMethod


# ── Constants ──────────────────────────────────────────────────────────────

# 300 training samples is enough for UMAP and runs in seconds on CPU.
N_TRAIN = 300
N_TEST = 100
# n_neighbors must be < N_TRAIN; 8 is well under 300.
UMAP_N_NEIGHBORS = 8
# SmallCNN's penultimate layer (fc2) outputs 256 features for MNIST input.
FEATURE_DIM = 256


# ── Helpers ────────────────────────────────────────────────────────────────

class WrappedSubset(torch.utils.data.Dataset):
    """
    Wraps a ``torch.utils.data.Subset`` to yield the
    ``{'input', 'target', 'index'}`` dicts expected by Visualizer and the
    rest of the pipeline.
    """
    def __init__(self, subset: Subset):
        self.subset = subset

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> dict:
        img, label = self.subset[index]
        return {"input": img, "target": label, "index": index}


class PrintLogger:
    """Minimal logger that writes to stdout — no file I/O needed in tests."""
    def info(self, m):    print("[INFO]", m)
    def warning(self, m): print("[WARN]", m)
    def error(self, m):   print("[ERR]",  m)


def make_config(save_dir: str, embedding_methods_list=("umap",)) -> dict:
    """
    Produce a minimal config dict that mirrors what ``main.py`` builds.

    ``foz_name`` is set to a string that does not exist in the FiftyOne zoo
    so the zoo load always fails gracefully and the dataset gets populated
    from our DataLoader instead.  This gives us exact control over N.
    """
    return {
        "dataset": {
            "name": "MNIST",
            # Not a real zoo dataset name — forces fallback to DataLoader
            "foz_name": "test_custom_dataset_not_in_zoo",
            "num_classes": 10,
            "in_channels": 1,
            "im_size": [28, 28],
            "noise": False,
            "noise_percent": 0,
        },
        "save_dir": save_dir,
        "num_gpus": 0,
        "training_opt": {
            "num_epochs": 3,
            "batch_size": 64,
            "num_data_workers": 0,
        },
        "visualization": {
            "enable": True,
            "milestones": [0.5, 1.0],
            "embedding_methods": list(embedding_methods_list),
            "embedding_params": {
                "umap": {
                    "n_neighbors": UMAP_N_NEIGHBORS,
                    "min_dist": 0.1,
                    "metric": "cosine",
                },
            },
            "delete_stale_dataset": True,
            "launch_app": False,
        },
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One pass of SGD over the DataLoader."""
    model.train()
    for batch in loader:
        inputs  = batch["input"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()


# ── Module-scoped fixtures (shared across all tests) ───────────────────────

@pytest.fixture(scope="module")
def tmp_dir():
    """Temporary output directory — cleaned up after the whole module runs."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope="module")
def mnist_subsets():
    """
    Download MNIST once (cached to ./_MNIST) and return small
    WrappedSubset objects for training and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081]),
    ])
    raw_train = datasets.MNIST(
        "./_MNIST", train=True,  download=True, transform=transform
    )
    raw_test = datasets.MNIST(
        "./_MNIST", train=False, download=True, transform=transform
    )
    train = WrappedSubset(Subset(raw_train, list(range(N_TRAIN))))
    test  = WrappedSubset(Subset(raw_test,  list(range(N_TEST))))
    return train, test


# ── Tests ──────────────────────────────────────────────────────────────────

def test_full_umap_pipeline(tmp_dir, mnist_subsets):
    """
    End-to-end test using real MNIST data and UMAP:

      • FiftyOne dataset is populated from the DataLoader (300 real images)
      • SmallCNN is trained for 3 epochs on the 300-sample subset
      • 256-D feature embeddings are extracted at milestone epochs
      • UMAP brain runs are computed and stored in FiftyOne
      • A snapshot pickle + FiftyOne export are written to disk
    """
    import fiftyone as fo

    train_dset, _ = mnist_subsets
    config = make_config(tmp_dir, embedding_methods_list=["umap"])
    device = torch.device("cpu")

    train_loader = DataLoader(train_dset, batch_size=64, shuffle=True,  num_workers=0)
    vis_loader   = DataLoader(train_dset, batch_size=64, shuffle=False, num_workers=0)

    model     = small_cnn(in_channels=1, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Unique suffix prevents collisions if tests run in parallel
    dataset_suffix = f"_umap_{os.getpid()}"
    visualizer = Visualizer(
        config=config,
        logger=PrintLogger(),
        data_loader=vis_loader,
        split="train",
        dataset_suffix=dataset_suffix,
    )
    fo_name = visualizer.dataset_name

    try:
        # ── 1. FiftyOne dataset should be populated from the DataLoader ──
        assert visualizer.fo_dataset is not None, "FiftyOne dataset was not created"
        assert len(visualizer.fo_dataset) == N_TRAIN, (
            f"Expected {N_TRAIN} samples in FO dataset, "
            f"got {len(visualizer.fo_dataset)}"
        )

        # ── 2. Train and capture embeddings at milestone epochs ───────────
        num_epochs = config["training_opt"]["num_epochs"]
        milestones = config["visualization"]["milestones"]
        milestone_epochs = {max(1, round(m * num_epochs)) for m in milestones}
        milestone_epochs.add(num_epochs)

        for epoch in range(1, num_epochs + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            if epoch in milestone_epochs:
                embeddings, labels = visualizer.compute_embeddings_from_model(
                    model=model,
                    data_loader=vis_loader,
                )
                visualizer.add_run(epoch, embeddings, labels, "Uniform")

        # ── 3. Each run should cover all N_TRAIN samples ──────────────────
        assert len(visualizer.runs) == len(milestone_epochs), (
            f"Expected {len(milestone_epochs)} runs, got {len(visualizer.runs)}"
        )
        for run in visualizer.runs:
            assert run["embeddings"].shape == (N_TRAIN, FEATURE_DIM), (
                f"Bad embedding shape for '{run['key']}': "
                f"expected ({N_TRAIN}, {FEATURE_DIM}), "
                f"got {run['embeddings'].shape}"
            )
            assert run["labels"].shape == (N_TRAIN,), (
                f"Bad labels shape for '{run['key']}': {run['labels'].shape}"
            )
            # Labels should all be valid MNIST class indices (0–9)
            assert run["labels"].min() >= 0
            assert run["labels"].max() <= 9

        # ── 4. UMAP brain runs are created for each registered run ────────
        visualizer.compute_all_visualizations()

        brain_runs = visualizer.fo_dataset.list_brain_runs()
        for run in visualizer.runs:
            expected_key = f"Uniform_E{run['epoch']}_umap"
            assert expected_key in brain_runs, (
                f"Expected UMAP brain run '{expected_key}' not found. "
                f"Available: {brain_runs}"
            )

        # ── 5. Snapshot is written to disk ────────────────────────────────
        visualizer._export_snapshot()

        snap_dir  = os.path.join(tmp_dir, "visualization_snapshots")
        snap_file = os.path.join(snap_dir, f"{fo_name}_snapshot.pkl")
        fo_export = os.path.join(snap_dir, f"{fo_name}_fo_export")

        assert os.path.isfile(snap_file), (
            f"Snapshot pickle not found at: {snap_file}"
        )
        assert os.path.isdir(fo_export), (
            f"FiftyOne export directory not created at: {fo_export}"
        )

        # ── 6. Snapshot content is valid ──────────────────────────────────
        import pickle
        with open(snap_file, "rb") as f:
            snap = pickle.load(f)

        assert snap["dataset_name"] == fo_name
        assert len(snap["runs_meta"]) == len(visualizer.runs)
        # in_memory_embeddings should include one entry per registered run key
        for run in visualizer.runs:
            assert run["key"] in snap["in_memory_embeddings"], (
                f"Embedding key '{run['key']}' missing from snapshot"
            )

    finally:
        # Clean up FiftyOne DB entry so this run doesn't affect others
        if fo.dataset_exists(fo_name):
            fo.delete_dataset(fo_name)


def test_custom_embedding_method(tmp_dir, mnist_subsets):
    """
    Register a PCA method via the ``@register`` decorator and confirm that:
      • The method appears in ``embedding_methods.list_methods()``
      • The Visualizer routes to it via the registry
      • A brain run with the correct key is created in FiftyOne
    """
    import fiftyone as fo

    # Register a custom PCA reducer under the name "pca_test"
    @register("pca_test")
    class PCATestMethod(EmbeddingMethod):
        def __init__(self, n_components: int = 2, **kwargs):
            from sklearn.decomposition import PCA
            self._reducer = PCA(n_components=n_components)

        def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
            return self._reducer.fit_transform(embeddings).astype(np.float32)

    assert "pca_test" in embedding_methods.list_methods(), (
        "Registered method 'pca_test' not found in registry"
    )

    train_dset, _ = mnist_subsets
    config = make_config(tmp_dir, embedding_methods_list=["pca_test"])
    device = torch.device("cpu")

    vis_loader   = DataLoader(train_dset, batch_size=64, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dset, batch_size=64, shuffle=True,  num_workers=0)

    model     = small_cnn(in_channels=1, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    dataset_suffix = f"_pca_{os.getpid()}"
    visualizer = Visualizer(
        config=config,
        logger=PrintLogger(),
        data_loader=vis_loader,
        split="train",
        dataset_suffix=dataset_suffix,
    )
    fo_name = visualizer.dataset_name

    try:
        # One epoch of actual training
        train_one_epoch(model, train_loader, optimizer, criterion, device)

        embeddings, labels = visualizer.compute_embeddings_from_model(
            model=model,
            data_loader=vis_loader,
        )
        assert embeddings.shape == (N_TRAIN, FEATURE_DIM)

        visualizer.add_run(1, embeddings, labels, "PCATest")
        visualizer.compute_all_visualizations()

        brain_runs = visualizer.fo_dataset.list_brain_runs()
        assert "PCATest_E1_pca_test" in brain_runs, (
            f"PCA brain run not found. Available: {brain_runs}"
        )

    finally:
        if fo.dataset_exists(fo_name):
            fo.delete_dataset(fo_name)


def test_snapshot_roundtrip(tmp_dir, mnist_subsets):
    """
    Export a snapshot then reload it.  Confirm that run metadata and
    embedding arrays survive the pickle roundtrip intact.
    """
    import fiftyone as fo
    import pickle

    train_dset, _ = mnist_subsets
    config = make_config(tmp_dir, embedding_methods_list=["umap"])
    device = torch.device("cpu")

    vis_loader = DataLoader(train_dset, batch_size=64, shuffle=False, num_workers=0)
    model = small_cnn(in_channels=1, num_classes=10).to(device)

    dataset_suffix = f"_snap_{os.getpid()}"
    visualizer = Visualizer(
        config=config,
        logger=PrintLogger(),
        data_loader=vis_loader,
        split="train",
        dataset_suffix=dataset_suffix,
    )
    fo_name = visualizer.dataset_name

    try:
        embeddings, labels = visualizer.compute_embeddings_from_model(
            model=model,
            data_loader=vis_loader,
        )
        visualizer.add_run(1, embeddings, labels, "SnapTest")

        # Export
        visualizer._export_snapshot()
        snap_dir  = os.path.join(tmp_dir, "visualization_snapshots")
        snap_file = os.path.join(snap_dir, f"{fo_name}_snapshot.pkl")

        # Reload and verify
        with open(snap_file, "rb") as f:
            snap = pickle.load(f)

        assert snap["dataset_name"] == fo_name
        assert len(snap["runs_meta"]) == 1

        run_meta = snap["runs_meta"][0]
        assert run_meta["key"] == "SnapTest_E1"
        assert run_meta["epoch"] == 1
        assert run_meta["selection_method"] == "SnapTest"

        # Embedding arrays should be preserved in the snapshot
        restored = snap["in_memory_embeddings"]["SnapTest_E1"]
        assert restored.shape == embeddings.shape
        np.testing.assert_array_almost_equal(restored, embeddings, decimal=5)

    finally:
        if fo.dataset_exists(fo_name):
            fo.delete_dataset(fo_name)
