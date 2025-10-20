import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import fiftyone.core.fields as fof # Make sure this is imported
import torch
import numpy as np

# 1. Define the schema for the data you want to store in each run
class RunResults(fo.EmbeddedDocument):
    """A container for the results of a single model run."""
    embeddings = fof.VectorField()
    predictions = fof.EmbeddedDocumentField(document_type=fo.Classification)


class Visualizer:
    """
    Manages a fiftyone dataset to log and compare model results from
    different training milestones using 'runs'.
    """
    def __init__(self, config, logger):
        # ... (the rest of your __init__ method is unchanged) ...
        self.config = config
        self.logger = logger
        self.epochs = config.get("training_opt", {}).get("num_epochs", 1) # avoid none
        self.seed = config.get("seed", None)
        dataset_info = config.get("dataset", {})
        self.dataset_name = dataset_info.get("name", "mnist").lower()
        self.zoo_dataset_name = dataset_info.get("foz_name", "mnist").lower()
        vis_cfg = config.get("visualization", {})
        self.persistent = vis_cfg.get("persistent", True)
        self.embedding_methods = vis_cfg.get("embedding_methods", ["tsne", "umap"])
        self.milestones = vis_cfg.get('milestones', [])
        self.milestone_epochs = [int(p * self.epochs) for p in self.milestones if self.epochs]
        embedding_params = vis_cfg.get("embedding_params", {})
        self.umap_params = embedding_params.get("umap", {})
        self.tsne_params = embedding_params.get("tsne", {})
        self.fo_dataset = None
        self.fo_session = None
        self._setup_dataset()

    def _setup_dataset(self):
        self.logger.info(f"Setting up fiftyone dataset '{self.dataset_name}'...")
        if fo.dataset_exists(self.dataset_name):
            self.logger.info("Found existing dataset. Loading it.")
            self.fo_dataset = fo.load_dataset(self.dataset_name)
        else:
            self.logger.info(f"Dataset not found. downloading '{self.zoo_dataset_name}' from the zoo.")
            self.fo_dataset = foz.load_zoo_dataset(
                self.zoo_dataset_name,
                split="test",
                dataset_name=self.dataset_name,
            )
        self.fo_dataset.persistent = self.persistent
        self.fo_dataset.reload()

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

        run_key = f"epoch_{epoch}_selection_{selection_method}"
        self.logger.info(f"Adding new run with key: '{run_key}'...")
        
        # 2. Update the `add_sample_field` call here
        if run_key not in self.fo_dataset.get_field_schema():
            self.logger.info(f"Field '{run_key}' not in schema. Adding it.")
            self.fo_dataset.add_sample_field(
                run_key, 
                fof.EmbeddedDocumentField, 
                embedded_doc_type=RunResults
            )

        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        if embeddings.shape[0] != len(self.fo_dataset):
            raise ValueError(f"Embeddings mismatch: {embeddings.shape[0]} != {len(self.fo_dataset)}")

        if labels is not None:
            # Check if we need to convert the labels to FiftyOne's format
            if len(labels) > 0 and not isinstance(labels[0], fo.Classification):
                # It's also good practice to handle tensors here too
                if torch.is_tensor(labels):
                    labels = labels.cpu().numpy()
                
                labels = [fo.Classification(label=str(l)) for l in labels]

        self.fo_dataset.set_values(f"{run_key}.embeddings", embeddings)
        if labels is not None:
            self.fo_dataset.set_values(f"{run_key}.predictions", labels)
        
        self.logger.info(f"Run '{run_key}' successfully saved.")
    
    def compute_all_visualizations(self):
        """
        Computes the 2d visualizations for all runs that have an 'embeddings' field.
        We call this method after all training and add_run calls are complete.
        """
        self.logger.info("Starting computation of visualizations for all runs...")
        if self.fo_dataset is None:
            raise ValueError("Dataset not initialized.")
            
        # reload prior to getting scheme (apparently helps to avoid some issues)
        self.fo_dataset.reload()

        # Get schema for all fields, including those in runs
        schema = self.fo_dataset.get_field_schema(include_private=True)

        # for testing...
        print(schema)

        for field_name, field in schema.items():
            if field_name.startswith("epoch_"):
                run_key = field_name
                self.logger.info(f"--- Processing run: '{run_key}' ---")
                
                # The path to the embeddings is simply the run_key + ".embeddings"
                embedding_field = f"{run_key}.embeddings"

                for embedding_method in self.embedding_methods:
                    brain_key = f"{run_key}_{embedding_method}"
                    params = self.umap_params if embedding_method == "umap" else self.tsne_params

                    self.logger.info(f"Computing '{embedding_method}' visualization (brain_key='{brain_key}')...")

                    fob.compute_visualization(
                        self.fo_dataset,
                        embeddings=embedding_field,
                        brain_key=brain_key,
                        method=embedding_method,
                        seed=self.seed,
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
        self.fo_dataset.save() # ensure dataset is saved before launching (apparently helps fix some issues)
        self.fo_session = fo.launch_app(dataset=self.fo_dataset, auto=False)
        self.logger.info(f"Fiftyone app is running at: {self.fo_session.url}")
        self.logger.info("Script will block here until you close the app browser tab.")
        self.fo_session.wait()
        self.logger.info("Fiftyone app session closed.")