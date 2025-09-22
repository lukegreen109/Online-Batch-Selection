import fiftyone as fo
import fiftyone.brain as fob
import numpy as np
from typing import List, Optional, Union, Tuple
import logging

def visualize_with_fiftyone(
    embs: np.ndarray,
    labels: Union[np.ndarray, List[Union[int, str]]],
    filepaths: List[str],
    selected_idx: Optional[List[int]] = None,
    epoch: int = 0,
    method: str = "umap",
    persistent: bool = True,
    params: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
    notebook: bool = False
) -> Tuple[fo.Dataset, fob.VisualizationResults, fo.Session]:
    """
    Visualize embeddings interactively in FiftyOne.

    Automatically slices embeddings, labels, and filepaths if `selected_idx` is provided.

    Args:
        embs: np.ndarray of shape (N, D) – embedding vectors
        labels: array-like of shape (N,) – class labels (int or str)
        filepaths: list of file paths (length N) – must exist or be placeholders
        selected_idx: indices to visualize (optional)
        epoch: identifier for dataset naming
        method: one of {"umap", "tsne"}
        persistent: whether dataset persists across App sessions
        params: optional method-specific params for compute_visualization
        logger: optional logger (defaults to print if not provided)
        notebook: boolean determining whether to set context to notebook or not

    Returns:
        (dataset, results, session)
    """
    log = logger.info if logger else print

    if notebook: fo.context = "Notebook"

    # Convert to arrays
    labels = np.array(labels)
    filepaths = np.array(filepaths)

    # Apply selection
    if selected_idx is not None:
        selected_idx = np.asarray(selected_idx)
        embs = embs[selected_idx]
        labels = labels[selected_idx]
        filepaths = filepaths[selected_idx]

    # Sanity checks
    if len(embs) == 0:
        raise ValueError("No embeddings provided.")
    if len(embs) != len(labels) or len(embs) != len(filepaths):
        raise ValueError("embs, labels, and filepaths must have same length")

    N, D = embs.shape
    ds_name = f"embeddings_epoch{epoch}"
    brain_key = f"viz_epoch{epoch}_{method}"

    log(f"[FiftyOne] Creating dataset '{ds_name}' with {N} samples")

    # Delete old dataset only if not persistent
    if not persistent and ds_name in fo.list_datasets():
        log(f"[FiftyOne] Deleting existing dataset '{ds_name}' (non-persistent)")
        fo.delete_dataset(ds_name)

    ds = fo.Dataset(ds_name, persistent=persistent)

    # Mark selected indices
    sel_mask = np.zeros(N, dtype=bool)
    if selected_idx is not None:
        sel_mask[np.arange(N)] = True
        log(f"[FiftyOne] {sel_mask.sum()} samples marked as 'selected'")

    # Add samples
    # Create samples from filepaths first
    samples = [fo.Sample(filepath=fp) for fp in filepaths]
    ds.add_samples(samples)

    # Add data in bulk (much faster than looping)
    ds.set_values("embedding", embs)
    ds.set_values("ground_truth", [fo.Classification(label=str(l)) for l in labels])

    # Tag selected samples in bulk
    if selected_idx is not None:
        selected_view = ds.select(selected_idx)
        selected_view.tag_samples("selected")
        log(f"[FiftyOne] Tagged {len(selected_view)} samples as 'selected'")

    log(f"[FiftyOne] Added {len(ds)} samples and their data to dataset '{ds_name}'")

    # Compute visualization
    log(f"[FiftyOne] Computing {method.upper()} visualization (brain_key='{brain_key}')")
    results = fob.compute_visualization(
        ds,
        embeddings="embedding",
        method=method,
        brain_key=brain_key,
        **(params or {}),
    )

    # Launch interactive app
    log("[FiftyOne] Launching App...")
    session = fo.launch_app(ds)

    # Display the embedding visualization directly
    results.visualize(labels=labels)

    log(f"[FiftyOne] Visualization ready in App (dataset='{ds_name}', brain_key='{brain_key}')")

    return ds, results, session
