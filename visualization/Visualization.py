"""
visualization/Visualization.py

Visualizer class that integrates with SelectionMethod to provide:
  - FiftyOne dataset management (with original_index field)
  - HuggingFace/timm ground-truth embedding runs
  - Per-epoch prediction runs (add_run)
  - UMAP / t-SNE brain visualization (compute_all_visualizations)
  - Snapshot export / load
  - FiftyOne app launch

Expected config keys (under 'visualization'):
    enable            : bool
    milestones        : list[float]   – fractional epoch checkpoints
    embedding_methods : list[str]     – e.g. ["umap", "tsne"]
    embedding_params  : dict          – per-method kwargs for fob.compute_visualization
    hf_model_name     : str           – timm model for ground-truth embeddings
    load_snapshot     : bool
    snapshot_path     : str | None
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    
try:
    from transformers import AutoModel, AutoFeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class Visualizer:
    def __init__(self, config, logger, data_loader=None, split="test", dataset_suffix=""):
        """
        Args:
            config      : full experiment config dict
            logger      : experiment logger
            data_loader : DataLoader whose batches are dicts with keys
                          'input', 'target', 'index'
            split       : "train" or "test"
            dataset_suffix : appended to the FiftyOne dataset name to make
                             it unique (e.g. "_train0")
        """
        self.config = config
        self.logger = logger
        self.data_loader = data_loader
        self.split = split

        # ── Dataset naming ──────────────────────────────────────────────
        dataset_info = config.get("dataset", {})
        raw_name = dataset_info.get("name", "mnist").lower()
        self.foz_name = dataset_info.get("foz_name", raw_name).lower()
        # e.g.  "cifar10_train_train0"
        self.dataset_name = f"{self.foz_name}_{split}{dataset_suffix}"

        # ── Visualization config ─────────────────────────────────────────
        vis_cfg = config.get("visualization", {})
        self.embedding_methods = [m.lower() for m in vis_cfg.get("embedding_methods", ["umap"])]
        self.embedding_params  = vis_cfg.get("embedding_params", {})
        
        # ── HF Model Config ──────────────────────────────────────────────
        # Map dataset names (lowercase) to HF model IDs or custom keys
        self.DATASET_MODEL_MAPPING = {
            "cifar10": "edadaltocg/resnet50_supcon_cifar10",
            "cifar100": "edadaltocg/resnet34_cifar100",
            "mnist": "farleyknight/mnist-digit-classification-2022-09-04",
            "fashion-mnist": "Kankanaghosh/vit-fashion-mnist",
            "svhn": "edadaltocg/resnet18_svhn",
        }
        
        default_model = "microsoft/resnet-18"
        self.hf_model_name = vis_cfg.get("hf_model_name", self.DATASET_MODEL_MAPPING.get(self.foz_name, default_model))
        self.logger.info(f"[Visualizer] Selected HF model '{self.hf_model_name}' for dataset '{self.foz_name}'")
        self.output_dir        = config.get("output_dir", "./exp/output")

        # ── In-memory stores ─────────────────────────────────────────────
        # Maps string key → np.ndarray of shape (N, D)
        self.in_memory_embeddings: dict = {}
        # List of run dicts: {epoch, embeddings, labels, selection_method, key}
        self.runs: list = []

        # ── FiftyOne dataset ─────────────────────────────────────────────
        self.fo_dataset = None
        self._setup_fo_dataset()

    # ------------------------------------------------------------------ #

    def _setup_fo_dataset(self):
        """Load or create the FiftyOne dataset and ensure original_index field."""
        name = self.dataset_name

        if fo.dataset_exists(name):
            self.logger.info(f"[Visualizer] Loading existing FiftyOne dataset: '{name}'")
            self.fo_dataset = fo.load_dataset(name)
            # Ensure the field exists in case the dataset was created elsewhere
            if not self.fo_dataset.has_field("original_index"):
                self._set_original_index()
            
            # If empty, try populate
            if len(self.fo_dataset) == 0 and self.data_loader is not None:
                 self._populate_from_dataloader()
            return

        self.logger.info(f"[Visualizer] Creating FiftyOne dataset: '{name}'")

        # Try to load from the zoo
        tmp_name = name + "__zoo_tmp"
        try:
            zoo_ds = foz.load_zoo_dataset(
                self.foz_name,
                split=self.split,
                dataset_name=tmp_name,
            )
            self.fo_dataset = zoo_ds.clone(name)
            fo.delete_dataset(tmp_name)
            self.logger.info(f"[Visualizer] Loaded zoo dataset '{self.foz_name}' ({self.split}) → '{name}'")
        except Exception as e:
            self.logger.warning(
                f"[Visualizer] Could not load zoo dataset '{self.foz_name}' "
                f"(split='{self.split}'): {e}\n"
                f"Creating an empty FiftyOne dataset instead."
            )
            self.fo_dataset = fo.Dataset(name)

        self.fo_dataset.persistent = True
        
        # If dataset is empty (e.g. created manually because not in zoo), populate it from dataloader
        if len(self.fo_dataset) == 0:
            if self.data_loader is not None:
                self._populate_from_dataloader()
            else:
                 self.logger.warning("[Visualizer] Dataset is empty and no data_loader provided to populate it!")

        self._set_original_index()
        self.logger.info(
            f"[Visualizer] Dataset '{name}' ready — {len(self.fo_dataset)} samples."
        )

    def _populate_from_dataloader(self):
        """Populate the FiftyOne dataset from the PyTorch data_loader."""
        self.logger.info(f"[Visualizer] Populating dataset '{self.fo_dataset.name}' from dataloader... (this may take a while)")
        
        # Directory to save images (since FO needs filepaths)
        # using output_dir/fo_images/dataset_name
        img_root = os.path.join(self.output_dir, "fo_images", self.dataset_name)
        os.makedirs(img_root, exist_ok=True)
        
        samples = []
        
        # We assume the loader yields batches
        # We simply perform one pass
        
        # Need to handle potential normalization to viewable range?
        # For now, just saving what we get (it might be normalized).
        # To invert normalization, we'd need mean/std from config.
        # Let's try to un-normalize if possible, valid logic:
        # img = img * std + mean
        # But we don't have them easily. We'll save as is and hope FO handles floats or we clip.
        
        import torchvision
        from PIL import Image
        
        cnt = 0
        for batch in self.data_loader:
            inputs = batch['input']
            targets = batch['target']
            indices = batch['index']
            
            # inputs: (B, C, H, W)
            for i in range(len(inputs)):
                idx = int(indices[i])
                target = int(targets[i])
                img_t = inputs[i]

                # Denormalize simple heuristic: move min to 0
                # or just clip to 0-1 if it's float
                img_t = img_t.clone()
                img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8)
                
                # Save image
                fname = f"{idx:06d}.png"
                fpath = os.path.join(img_root, fname)
                
                # Convert to PIL
                # (C, H, W) -> (H, W, C)
                # But pytorch tensor might be on GPU
                if img_t.device != torch.device("cpu"):
                    img_t = img_t.cpu()
                    
                ndarr = img_t.permute(1, 2, 0).numpy()
                ndarr = (ndarr * 255).astype(np.uint8)
                
                # Helper to save
                # Handle 1-channel
                if ndarr.shape[2] == 1:
                    ndarr = ndarr[:, :, 0]
                    
                im = Image.fromarray(ndarr)
                im.save(fpath)
                
                # Create Sample
                sample = fo.Sample(filepath=fpath)
                sample["ground_truth"] = fo.Classification(label=str(target))
                sample["original_index"] = idx
                samples.append(sample)
                cnt += 1
                
        self.fo_dataset.add_samples(samples)
        self.fo_dataset.save()
        self.logger.info(f"[Visualizer] Populated {len(samples)} samples.")

    def _set_original_index(self):
        """Assign sequential original_index values to every sample."""
        if not self.fo_dataset.has_field("original_index"):
            self.fo_dataset.add_sample_field("original_index", fo.IntField)
        for i, sample in enumerate(self.fo_dataset):
            sample["original_index"] = i
            sample.save()
        self.fo_dataset.save()

    # ------------------------------------------------------------------ #
    #  Embedding runs                                                       #
    # ------------------------------------------------------------------ #

    def _load_custom_resnet_small(self, model_name: str, device):
        """
        Load a custom ResNet (18/34/50) adapted for small images (CIFAR/SVHN).
        These models typically use:
          - conv1 kernel_size=3, stride=1, padding=1
          - no maxpool
        """
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for custom ResNet loading.")
            
        self.logger.info(f"[Visualizer] Loading custom small-input ResNet: {model_name}")
        
        # 1. Infer architecture from name
        if "resnet50" in model_name:
            arch = "resnet50"
        elif "resnet34" in model_name:
            arch = "resnet34"
        elif "resnet18" in model_name:
            arch = "resnet18"
        else:
            self.logger.warning(f"[Visualizer] Could not infer ResNet depth from '{model_name}'. Defaulting to resnet50.")
            arch = "resnet50"

        # 2. Create standard model
        model = timm.create_model(arch, pretrained=False, num_classes=0)
        
        # 3. Patch input layer for small images (3x3 kernel instead of 7x7, no pooling)
        # Standard: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Small:    Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        if hasattr(model, "conv1"):
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if hasattr(model, "maxpool"):
            model.maxpool = torch.nn.Identity()
        
        model = model.to(device)
        
        # 4. Load weights from Hugging Face
        if HF_HUB_AVAILABLE:
            try:
                # Try downloading model.safetensors first, then pytorch_model.bin
                try:
                    weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_path)
                except Exception:
                    weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
                    state_dict = torch.load(weights_path, map_location=device)
                    
                # Fix keys: remove 'module.' prefix if present, handle 'encoder.' etc.
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace("module.", "").replace("encoder.", "")
                    new_state_dict[name] = v
                    
                model.load_state_dict(new_state_dict, strict=False)
                self.logger.info(f"[Visualizer] Loaded weights for {model_name} successfully.")
                
            except Exception as e:
                self.logger.error(f"[Visualizer] Failed to download/load weights from HF for {model_name}: {e}")
                self.logger.warning("[Visualizer] Using random initialization!")
        else:
            self.logger.warning("[Visualizer] huggingface_hub not installed. Cannot download weights.")
            
        return model.eval()

    def add_huggingface_ground_truth_run(self):
        """
        Extract embeddings from a pretrained model and store them.
        Supports:
          1. Custom CIFAR-10 ResNet (via timm patch)
          2. MNIST (via transformers)
          3. Standard ImageNet models (via timm)
        """
        if self.data_loader is None:
            self.logger.warning("[Visualizer] No data_loader — skipping HF ground-truth run.")
            return

        self.logger.info(
            f"[Visualizer] Computing HF ground-truth embeddings "
            f"with model '{self.hf_model_name}' …"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = None

        # ── Strategy Selection ───────────────────────────────────────────
        
        # Strategy 1: custom small-input ResNets (CIFAR-10/100, SVHN)
        # e.g. edadaltocg/resnetXX...
        if ("cifar" in self.foz_name or "svhn" in self.foz_name) and "edadaltocg" in self.hf_model_name:
            try:
                model = self._load_custom_resnet_small(self.hf_model_name, device)
            except Exception as e:
                self.logger.error(f"[Visualizer] Custom ResNet load failed: {e}")
                return

        # Strategy 2: MNIST / Fashion-MNIST (Transformers)
        elif ("mnist" in self.foz_name or "fashion" in self.foz_name) and TRANSFORMERS_AVAILABLE:
            try:
                # MNIST models on HF are often specific and work best with AutoModel
                self.logger.info(f"[Visualizer] Using transformers.AutoModel for {self.hf_model_name}")
                model = AutoModel.from_pretrained(self.hf_model_name)
                model = model.eval().to(device)
            except Exception as e:
                self.logger.error(f"[Visualizer] Transformers load failed: {e}")
                # Fallback to timm if possible? Usually not for MNIST.
                return

        # Strategy 3: Standard timm (ImageNet etc.)
        else:
            if not TIMM_AVAILABLE:
                self.logger.error("[Visualizer] timm is not installed.")
                return
            try:
                model = timm.create_model(self.hf_model_name, pretrained=True, num_classes=0)
                model = model.eval().to(device)
            except Exception as e:
                self.logger.error(f"[Visualizer] timm load failed for '{self.hf_model_name}': {e}")
                return

        # ── Inference ────────────────────────────────────────────────────
        
        # Infer expected spatial size for this backbone
        try:
            if hasattr(model, "default_cfg"):
                # timm model
                cfg = model.default_cfg
                min_size = cfg.get("input_size", (3, 32, 32))[1]
            elif hasattr(model, "config") and hasattr(model.config, "image_size"):
                # transformers model
                min_size = model.config.image_size
            else:
                 min_size = 32
        except Exception:
            min_size = 32
        
        all_embs   = []
        all_labels = []

        with torch.no_grad():
            for batch in self.data_loader:
                inputs  = batch["input"].to(device)
                targets = batch["target"]

                # Preprocessing handling
                # CIFAR custom model expects 32x32 (no resize needed if inputs are that size)
                # MNIST transformers model might expect 1 channel? 
                # If inputs are 1 channel and model expects 3, or vice versa, we might need adjustments.
                
                # Check model expectation if possible, or just try/catch
                
                # Proactive handling for Transformers models on MNIST/Fashion-MNIST (1-channel to 3-channel)
                # The farleyknight/mnist model is a ViT that expects 3 channels
                is_mnist_like = "mnist" in self.foz_name.lower() or "fashion" in self.foz_name.lower()
                if is_mnist_like and inputs.shape[1] == 1:
                     inputs = inputs.repeat(1, 3, 1, 1)

                # Upsample only if the backbone requires larger inputs
                if inputs.shape[-1] < min_size:
                    inputs = F.interpolate(
                        inputs, size=(min_size, min_size),
                        mode="bilinear", align_corners=False,
                    )

                try:
                    outputs = model(inputs)
                    
                    # Output handling
                    if isinstance(outputs, torch.Tensor):
                        embs = outputs
                    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        # Transformers
                        embs = outputs.pooler_output
                    elif hasattr(outputs, "last_hidden_state"):
                         # Transformers (e.g. ViT without pooler)
                         # Mean pool
                         embs = outputs.last_hidden_state.mean(dim=1)
                    else:
                        # Tuple or unknown
                        embs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                    # Ensure 2D (N, D)
                    if len(embs.shape) > 2:
                        # Global average pool if spatial dims remain (e.g. from a Conv net that wasn't fully pooled)
                        embs = embs.mean(dim=list(range(2, len(embs.shape))))
                        
                except Exception as e:
                    # Catch-all for other shape/channel errors that might slip through
                    msg = str(e).lower()
                    if "channel" in msg or "shape" in msg or "dimension" in msg:
                        # Last ditch effort: if we haven't already repeated (though we should have above)
                        if inputs.shape[1] == 1:
                             inputs_3 = inputs.repeat(1, 3, 1, 1)
                             try:
                                 outputs = model(inputs_3)
                                 if isinstance(outputs, torch.Tensor): embs = outputs
                                 elif hasattr(outputs, "pooler_output"): embs = outputs.pooler_output
                                 else: embs = outputs.last_hidden_state.mean(dim=1)
                             except Exception:
                                 self.logger.warning(f"[Visualizer] Retry with 3 channels failed: {e}. Skipping batch.")
                                 continue
                        else:
                             self.logger.warning(f"[Visualizer] Inference failed with shape error: {e}. Skipping batch.")
                             continue
                    else:
                        self.logger.warning(f"[Visualizer] Forward pass failed: {e}. Skipping batch.")
                        continue

                all_embs.append(embs.cpu().numpy())
                all_labels.append(targets.numpy() if isinstance(targets, torch.Tensor)
                                  else np.asarray(targets))

        if not all_embs:
             self.logger.error("[Visualizer] No embeddings computed.")
             return

        embeddings = np.concatenate(all_embs,   axis=0)
        labels     = np.concatenate(all_labels, axis=0)

        self.in_memory_embeddings["hf_ground_truth"] = embeddings
        self.logger.info(
            f"[Visualizer] HF embeddings computed: shape={embeddings.shape}"
        )

        # Persist a brain run on the FO dataset
        self._compute_fo_visualization(
            embeddings=embeddings,
            brain_key="hf_ground_truth",
        )

    # ------------------------------------------------------------------ #

    def add_run(self, epoch: int, embeddings, labels, selection_method: str):
        """
        Register a prediction run for later batch-visualization.

        Args:
            epoch            : training epoch this run corresponds to
            embeddings       : np.ndarray or torch.Tensor  (N, D)
            labels           : np.ndarray or torch.Tensor  (N,)  – predicted classes
            selection_method : name of the selection method (e.g. "DivBS")
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        key = f"{selection_method}_E{epoch}"
        run = {
            "epoch"            : epoch,
            "embeddings"       : embeddings,
            "labels"           : labels,
            "selection_method" : selection_method,
            "key"              : key,
        }
        self.runs.append(run)
        self.in_memory_embeddings[key] = embeddings
        self.logger.info(
            f"[Visualizer] Registered run '{key}' — {len(labels)} samples."
        )

    # ------------------------------------------------------------------ #

    def compute_all_visualizations(self):
        """
        Run UMAP / t-SNE for every registered run (all methods × all epochs)
        and persist the results as FiftyOne brain keys.
        """
        if not self.runs:
            self.logger.warning(
                "[Visualizer] compute_all_visualizations called but no runs "
                "have been added — nothing to do."
            )
            return

        self.logger.info(
            f"[Visualizer] Computing visualizations for {len(self.runs)} run(s) "
            f"× {len(self.embedding_methods)} method(s) …"
        )

        for run in self.runs:
            for method in self.embedding_methods:
                # FO brain keys must be alphanumeric + underscores, ≤ 64 chars
                raw_key = f"{run['selection_method']}_E{run['epoch']}_{method}"
                brain_key = self._sanitize_key(raw_key)

                self.logger.info(
                    f"[Visualizer] {method.upper()} for '{run['key']}' → brain_key='{brain_key}'"
                )
                try:
                    self._compute_fo_visualization(
                        embeddings=run["embeddings"],
                        brain_key=brain_key,
                        method=method,
                    )
                except Exception as e:
                    self.logger.error(
                        f"[Visualizer] Visualization '{brain_key}' failed: {e}"
                    )

        self.logger.info("[Visualizer] All visualizations computed.")

    # ------------------------------------------------------------------ #
    #  Snapshot                                                             #
    # ------------------------------------------------------------------ #

    def _export_snapshot(self):
        """Persist in-memory state (embeddings + run metadata) to disk."""
        snap_dir = os.path.join(self.output_dir, "visualization_snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        snap_path = os.path.join(snap_dir, f"{self.dataset_name}_snapshot.pkl")

        # Omit raw embeddings from run list (already in in_memory_embeddings)
        snapshot = {
            "dataset_name"        : self.dataset_name,
            "in_memory_embeddings": self.in_memory_embeddings,
            "runs_meta"           : [
                {k: v for k, v in run.items() if k != "embeddings"}
                for run in self.runs
            ],
        }

        with open(snap_path, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"[Visualizer] Snapshot saved → {snap_path}")

        # Also export the FiftyOne dataset (brain runs included)
        fo_export_dir = os.path.join(snap_dir, f"{self.dataset_name}_fo_export")
        try:
            self.fo_dataset.export(
                export_dir=fo_export_dir,
                dataset_type=fo.types.FiftyOneDataset,
                overwrite=True,
            )
            self.logger.info(f"[Visualizer] FO dataset exported → {fo_export_dir}")
        except Exception as e:
            self.logger.warning(f"[Visualizer] FO dataset export failed: {e}")

    def _load_snapshot_and_visualize(self, path=None):
        """
        Restore a previously saved snapshot and launch the FiftyOne app.

        Args:
            path : explicit path to the .pkl snapshot file.
                   Defaults to the standard location derived from dataset_name.
        """
        if path is None:
            snap_dir  = os.path.join(self.output_dir, "visualization_snapshots")
            path      = os.path.join(snap_dir, f"{self.dataset_name}_snapshot.pkl")

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[Visualizer] Snapshot not found: '{path}'"
            )

        self.logger.info(f"[Visualizer] Loading snapshot from '{path}' …")
        with open(path, "rb") as f:
            snapshot = pickle.load(f)

        self.in_memory_embeddings = snapshot.get("in_memory_embeddings", {})

        # Reconstruct runs (without embeddings — they live in in_memory_embeddings)
        self.runs = []
        for meta in snapshot.get("runs_meta", []):
            key = meta.get("key", "")
            run = dict(meta)
            run["embeddings"] = self.in_memory_embeddings.get(key, np.array([]))
            self.runs.append(run)

        self.logger.info(
            f"[Visualizer] Snapshot loaded: "
            f"{len(self.runs)} run(s), "
            f"{len(self.in_memory_embeddings)} embedding set(s)."
        )
        self.launch_app()

    # ------------------------------------------------------------------ #
    #  App                                                                  #
    # ------------------------------------------------------------------ #

    def launch_app(self):
        """Launch the FiftyOne interactive app and wait."""
        self.logger.info(
            f"[Visualizer] Launching FiftyOne app for '{self.dataset_name}' …"
        )
        session = fo.launch_app(self.fo_dataset)
        session.wait()

    # ------------------------------------------------------------------ #
    #  Internals                                                            #
    # ------------------------------------------------------------------ #

    def _compute_fo_visualization(
        self,
        embeddings: np.ndarray,
        brain_key: str,
        method: str = None,
    ):
        """
        Call fob.compute_visualization and save the result on fo_dataset.

        Args:
            embeddings : (N, D) float array
            brain_key  : key to store result under in FiftyOne
            method     : "umap" | "tsne" | etc.  Defaults to first configured method.
        """
        if method is None:
            method = self.embedding_methods[0] if self.embedding_methods else "umap"

        brain_key = self._sanitize_key(brain_key)

        # Skip if we'd recompute an identical key
        if brain_key in self.fo_dataset.list_brain_runs():
            self.logger.info(
                f"[Visualizer] Brain run '{brain_key}' already exists — skipping."
            )
            return

        extra_kwargs = self.embedding_params.get(method, {})

        fob.compute_visualization(
            self.fo_dataset,
            embeddings=embeddings,
            num_dims=2,
            method=method,
            brain_key=brain_key,
            verbose=False,
            seed=51,
            **extra_kwargs,
        )
        self.fo_dataset.save()
        self.logger.info(
            f"[Visualizer] Brain run '{brain_key}' saved (method={method})."
        )

    @staticmethod
    def _sanitize_key(key: str, max_len: int = 64) -> str:
        """Make a string safe to use as a FiftyOne brain key."""
        import re
        key = re.sub(r"[^A-Za-z0-9_]", "_", key)
        if not key:
            key = "unnamed_run"
        return key[:max_len]