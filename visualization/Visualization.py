import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import fiftyone.core.fields as fof
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms.functional import normalize

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
    def __init__(self, config, logger, test_loader=None):
        self.config = config
        self.logger = logger
        self.test_loader = test_loader
        self.epochs = config.get("training_opt", {}).get("num_epochs", 1) # avoid none
        self.seed = config.get("seed", None)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Dataset Stuff
        dataset_info = config.get("dataset", {})
        self.dataset_name = dataset_info.get("name", "mnist").lower()
        self.zoo_dataset_name = dataset_info.get("foz_name", "mnist").lower()
        vis_cfg = config.get("visualization", {})
        self.persistent = vis_cfg.get("persistent", True)

        # Visualization Config
        self.embedding_methods = vis_cfg.get("embedding_methods", ["tsne", "umap"])
        self.milestones = vis_cfg.get('milestones', [])
        self.milestone_epochs = [int(p * self.epochs) for p in self.milestones if self.epochs]
        embedding_params = vis_cfg.get("embedding_params", {})
        self.umap_params = embedding_params.get("umap", {})
        self.tsne_params = embedding_params.get("tsne", {})

        # FiftyOne stuff
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

    def load_huggingface_model(self):
        """
        Loads a pre-trained model from Hugging Face for calculating embeddings.
        """
        self.logger.info("Loading pre-trained model from Hugging Face...")
        model_map = {
            'mnist': 'google/vit-base-patch16-224',
            'fashionmnist': 'google/vit-base-patch16-224',
            'cifar10': 'microsoft/resnet-50',
            'cifar100': 'microsoft/resnet-50',
            # Add other mappings as needed
        }
        model_name = model_map.get(self.dataset_name, 'google/vit-base-patch16-224')
        self.logger.info(f"Loading model: {model_name} for dataset: {self.dataset_name}")

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.hf_model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load model '{model_name}': {e}")
            return
        
        self.hf_model = self.hf_model.to(self.device)
        self.hf_model.eval()
        self.logger.info(f"Successfully loaded pre-trained Hugging Face model {model_name}")

    def compute_huggingface_embeddings(self):
        """
        Computes embeddings using our pre-trained HF model.
        """
        if not hasattr(self, 'hf_model') or not hasattr(self, 'processor'):
            self.logger.info("Hugging Face model not loaded. Please call _load_huggingface_model() first.")
            raise AttributeError("Hugging Face model not loaded. Please call _load_huggingface_model() first.")
        
        if self.test_loader is None:
            self.logger.info("Visualizer does not have a test_loader. Pass one during init.")
            raise ValueError("self.test_loader is None.")

        self.logger.info("Computing embeddings with Hugging Face model...")

        # get image size from our processor
        try:
            target_size = self.processor.size["height"]
        except TypeError:
            target_size = self.processor.size

        mean = self.processor.image_mean
        std = self.processor.image_std
        self.logger.info(f"Processing images to size={target_size} with mean={mean}, std={std}")

        all_embs, all_labels = [], []

        # calculate embeddings
        with torch.no_grad():
            for datas in self.test_loader:
                
                # Handle different data_loader formats
                if isinstance(datas, dict):
                    images = datas['input']
                    labels = datas['target']
                elif isinstance(datas, (list, tuple)) and len(datas) == 2:
                    images, labels = datas
                else:
                    self.logger.error("Dataloader batch format not recognized. Expected dict or (img, label) tuple.")
                    raise ValueError("Invalid batch format")

                images = images.to(self.device)
                
                # 3. Preprocess images
                # Convert grayscale to RGB
                if self.dataset_name in ['mnist', 'fashionmnist']:
                    if images.shape[1] == 1:
                        images = images.repeat(1, 3, 1, 1) # (B, 1, H, W) -> (B, 3, H, W)
                
                # Resize to the model's expected input size
                images = torch.nn.functional.interpolate(
                    images, 
                    size=(target_size, target_size), 
                    mode='bilinear', 
                    align_corners=False
                )

                images = normalize(images, mean=mean, std=std)
                outputs = self.hf_model(images)

                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    # Use the 'pooler_output' if it exists (e.g., ViT, ResNet)
                    embeddings = outputs.pooler_output
                elif 'resnet' in self.hf_model.config.model_type:
                    # Fallback for ResNet: Global Average Pooling
                    embeddings = outputs.last_hidden_state.mean(dim=[2, 3])
                elif 'vit' in self.hf_model.config.model_type:
                    # Fallback for ViT: Use the [CLS] token (first token)
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    # General-purpose fallback
                    self.logger.warning("Using mean(dim=1) on last_hidden_state as default embedding.")
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                
                all_embs.append(embeddings.cpu())
                all_labels.append(labels.cpu())

            all_embs = torch.cat(all_embs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            self.hf_embs, self.hf_labels = all_embs, all_labels
            self.logger.info(f"Computed Hugging Face embeddings: {all_embs.shape}")
            return all_embs, all_labels

    def add_huggingface_ground_truth_run(self):
        """
        Computes and adds the Hugging Face embeddings as a run.
        This allows us to use the embeddings from our "ground truth"
        HF model.
        """
        self.logger.info("Computing HF ground truth embeddings...")
        try:
            self.load_huggingface_model()
            self.compute_huggingface_embeddings()
            self.add_run(
                epoch="hf",
                embeddings=self.hf_embs,
                labels=self.hf_labels,
                selection_method="hf_ground_truth"
            )
            self.logger.info("Successfully added HF ground truth run.")

        except Exception as e:
            self.logger.info(f"Failed to add HF ground truth run {e}")

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

    