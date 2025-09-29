import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import torch

import numpy as np

class Visualizer:
    """
    Accepts a dataset, then makes a FiftyOne session & dataset object.
    At each milestone, we'll pass in the updated embeddings and compare them.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.seed = config.get("seed", None)

        # get names of datasets
        dataset_info = config.get("dataset", {})
        self.dataset_name = dataset_info.get("name", "mnist").lower()
        self.foz_name = dataset_info.get("foz_name", self.dataset_name).lower()

        # visualization config
        vis_cfg = config.get("visualization", {})
        self.persistent = vis_cfg.get("persistent", True)
        self.save_dir = vis_cfg.get("save_dir", "./vis")
        self.embedding_methods = vis_cfg.get("embedding_methods", ["tsne", "umap"])

        # embedding params
        embedding_params = vis_cfg.get("embedding_params", {})
        self.umap_params = embedding_params.get("umap", {})
        self.tsne_params = embedding_params.get("tsne", {})

    def setup_fiftyone_session(self, dataset_name="mnist"):
        """
        Initializes a persistent FiftyOne dataset and session one time.
        """
        self.logger.info("Setting up persistent FiftyOne session...")

        # delete previous dataset
        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)
        
        # load in FiftyOne dataset (e.g. MNIST test split)
        self.fo_dataset = foz.load_zoo_dataset(
            dataset_name,
            split="test",
            download_if_necessary=True,
            persistent=True,
        )

        # Launch the app and store the session object
        self.fo_session = fo.launch_app(self.fo_dataset, auto=False)
        self.logger.info(f"FiftyOne App launched. Point browser to: {self.fo_session.url}")

        return self.fo_session, self.fo_dataset

    def add_run(self, epoch=-1, embeddings=None, labels=None):
        """
        Calculates embeddings and adds them (and optional labels) to FiftyOne.
        """

        if self.fo_dataset is None or self.fo_session is None:
            raise ValueError("Visualizer must be initialized with setup_fiftyone_session first")

        if embeddings is None:
            raise ValueError("You must provide exactly embeddings")

        if labels is not None:
            self.logger.info(f"Adding labels for epoch {epoch}...")
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            
            # create a dynamic field name for this epoch's labels/predictions
            label_field = f"predictions_epoch_{epoch}"
            
            # attach the labels to the FiftyOne dataset
            self.fo_dataset.set_values(label_field, labels)
            self.logger.info(f"Labels stored in field '{label_field}'")
        
        # validate embeddings
        embedding_field = f"embs_epoch_{epoch}"
        if embeddings is not None:
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()
            if not isinstance(embeddings, np.ndarray):
                raise TypeError("embeddings must be a numpy array or torch.Tensor")
            if embeddings.shape[0] != len(self.fo_dataset):
                raise ValueError(
                    f"Embeddings dimension mismatch: got {embeddings.shape[0]} "
                    f"vs dataset size {len(self.fo_dataset)}"
                )
            self.fo_dataset.set_values(embedding_field, embeddings)

        for method in self.embedding_methods:
            # for naming conventions
            brain_key = f"{method}_epoch_{epoch}"
            params = self.umap_params if method == "umap" else self.tsne_params
            
            results = fob.compute_visualization(
                self.fo_dataset,
                embeddings=embeddings,
                brain_key=brain_key,
                method=method,
                seed=self.seed,
                **params,
            )

        self.fo_dataset.load_brain_results(model_name="test123", brain_key=brain_key)
        #self.fo_session.wait()
        #self.fo_session.view = self.fo_dataset.view()

