import fiftyone as fo
import fiftyone.zoo as foz
import torch
import numpy as np

class Visualizer:
    """
    Manages a fiftyone dataset to log and compare model results from
    different training milestones using 'runs'.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.epochs = config.get("training_opt", {}).get("num_epochs", 1) # avoid none
        self.seed = config.get("seed", None)

        # dataset config
        dataset_info = config.get("dataset", {})
        # use a generic name for the local copy, but specify what to download
        self.dataset_name = dataset_info.get("name", "mnist").lower()
        self.zoo_dataset_name = dataset_info.get("foz_name", "mnist").lower()

        # visualization config
        vis_cfg = config.get("visualization", {})
        self.persistent = vis_cfg.get("persistent", True)
        self.embedding_methods = vis_cfg.get("embedding_methods", ["tsne", "umap"])
        self.milestones = vis_cfg.get('milestones', [])
        self.milestone_epochs = [int(p * self.epochs) for p in self.milestones if self.epochs]

        # embedding params
        embedding_params = vis_cfg.get("embedding_params", {})
        self.umap_params = embedding_params.get("umap", {})
        self.tsne_params = embedding_params.get("tsne", {})
        
        # initialize FiftyOne objects
        self.fo_dataset = None
        self.fo_session = None
        self._setup_dataset()

    def _setup_dataset(self):
        """
        Loads a persistent fiftyone dataset. if it doesn't exist,
        it downloads it from the zoo; clears any old runs.
        """
        self.logger.info(f"Setting up fiftyone dataset '{self.dataset_name}'...")

        if fo.dataset_exists(self.dataset_name):
            self.logger.info("Found existing dataset. Loading it.")
            self.fo_dataset = fo.load_dataset(self.dataset_name)
        else:
            self.logger.info(f"Dataset not found. downloading '{self.zoo_dataset_name}' from the zoo.")
            # load the base dataset (e.g., mnist test split)
            self.fo_dataset = foz.load_zoo_dataset(
                self.zoo_dataset_name,
                split="test",
                dataset_name=self.dataset_name,
            )
        
        self.fo_dataset.persistent = self.persistent
        self.fo_dataset.delete_runs() # clear old runs for a clean experiment
        self.fo_dataset.reload() # ensure we have the latest from the db

    def add_run(
            self,
            epoch=-1,
            embeddings=None,
            labels=None,
            selection_method=None
            ):
        """
        Saves model outputs (embeddings & labels) as a new run to the dataset.
        """
        if self.fo_dataset is None:
            raise ValueError("Dataset not initialized. call _setup_dataset first.")

        if embeddings is None:
            raise ValueError("You must provide embeddings to create a run.")

        # 1. create a unique key for this run
        run_key = f"epoch_{epoch}_selection_{selection_method}"
        self.logger.info(f"Adding new run with key: '{run_key}'...")

        # 2. validate and prepare data
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        if embeddings.shape[0] != len(self.fo_dataset):
            raise ValueError(f"Embeddings mismatch: {embeddings.shape[0]} != {len(self.fo_dataset)}")

        if labels is not None:
            # convert labels to fiftyone format if they aren't already
            if not isinstance(labels[0], fo.classification):
                 # assuming labels is a list/array of class strings or ints
                 labels = [fo.classification(label=str(l)) for l in labels]

        # 3. create the run and set the data fields within it
        with self.fo_dataset.new_run(run_key) as run:
            # the field is simply named "embeddings", namespaced within this run
            run.set_values("embeddings", embeddings)
            if labels is not None:
                # the field is simply named "predictions", namespaced within this run
                run.set_values("predictions", labels)
        
        self.logger.info(f"Run '{run_key}' successfully saved.")

    def compute_all_visualizations(self):
        """
        Computes the 2d visualizations for all runs that have an 'embeddings' field.
        We call this method after all training and add_run calls are complete.
        """
        self.logger.info("Starting computation of visualizations for all runs...")
        if self.fo_dataset is None:
            raise ValueError("Dataset not initialized.")
            
        for run_key in self.fo_dataset.list_runs():
            self.logger.info(f"--- Processing run: '{run_key}' ---")
            run_info = self.fo_dataset.get_run_info(run_key)
            
            # check if this run has embeddings that we can visualize
            if "embeddings" not in run_info.config.fields:
                self.logger.warning(f"Skipping run '{run_key}', no 'embeddings' field found.")
                continue
            
            # get the exact name of the embedding field within the run
            embedding_field = run_info.config.fields["embeddings"]["field"]

            for embedding_method in self.embedding_methods:
                brain_key = f"{run_key}_{embedding_method}" # e.g., "epoch_10_tsne"
                params = self.umap_params if embedding_method == "umap" else self.tsne_params

                self.logger.info(f"Computing '{embedding_method}' visualization (brain_key='{brain_key}')...")
                self.fo_dataset.compute_visualization(
                    embeddings=embedding_field,
                    brain_key=brain_key,
                    model=embedding_method,
                    seed=self.seed,
                    run_key=run_key,
                    **params,
                )
        
        self.logger.info("All visualizations computed.")

    def close_session(self):
        """Closes the fiftyone app session if it is running."""
        if self.fo_session is not None:
            self.logger.info("Closing the fiftyone app session...")
            self.fo_session.close()
            self.fo_session = None
            self.logger.info("Fiftyone app session closed.")

    def launch_app(self):
        """launches the fiftyone app and waits for it to be closed."""
        self.logger.info("Launching the fiftyone app...")
        self.fo_session = fo.launch_app(self.fo_dataset, auto=False)
        self.logger.info(f"Fiftyone app is running at: {self.fo_session.url}")
        self.logger.info("Script will block here until you close the app browser tab.")
        self.fo_session.wait()
        self.logger.info("Fiftyone app session closed.")