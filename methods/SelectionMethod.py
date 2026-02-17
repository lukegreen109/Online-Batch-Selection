import torch
import models
import os
import shutil
import numpy as np
import time
from .method_utils import *
from visualization.Visualization import Visualizer

import data
import fiftyone.zoo as foz
import fiftyone as fo

class SelectionMethod(object):
    method_name = 'SelectionMethod'

    def __init__(self, config, logger):
        logger.info(f'Creating {self.method_name}...')
        self.config = config
        self.logger = logger

        # Basic training attributes
        self.epoch = 0
        self.start_epoch = 0
        self.best_acc = 0
        self.best_epoch = 0
        self.need_features = False
        self.last_selection_method = config.get("methods")[-1]

        # Model creation
        model_type = config['networks']['type']
        model_args = config['networks']['params']
        self.model = getattr(models, model_type)(**model_args)
        self.in_channels = model_args.get('in_channels', 3)

        # GPU setup
        self.num_gpus = config['num_gpus']
        if self.num_gpus == 0:
            self.model = self.model.cpu()
        elif self.num_gpus == 1:
            self.model = self.model.cuda()
        elif self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            raise ValueError(f'Wrong number of GPUs: {self.num_gpus}')

        # Optimizer and scheduler
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)

        # Resume from checkpoint
        config['training_opt']['resume'] = config['training_opt'].get('resume', None)
        if config['training_opt']['resume'] is not None:
            self.resume(config['training_opt']['resume'])

        # Training parameters
        self.epochs = config['training_opt'].get('num_epochs', None)
        self.num_steps = config['training_opt'].get('num_steps', None)
        if self.epochs is None and self.num_steps is None:
            raise ValueError('Must specify either num_epochs or num_steps in training_opt')
        self.num_data_workers = config['training_opt']['num_data_workers']
        self.batch_size = config['training_opt']['batch_size']
        self.training_opt = config.get('training_opt', None)

        # Criterion
        self.criterion = create_criterion(config, logger)

        # Visualization
        self.test_visualizer = None
        self.train_visualizer = None
        self.milestones = []
        self.milestone_epochs = []
        self.embedding_methods = []
        self.embedding_params = {}

        # get names of datasets
        dataset_info = config.get("dataset", {})
        self.dataset_name = dataset_info.get("name", "mnist").lower()
        self.foz_name = dataset_info.get("foz_name", self.dataset_name).lower()

        # dataset handling
        self.data_info = getattr(data, config['dataset']['name'])(config, logger) 
        self.num_classes = self.data_info['num_classes']
        self.train_dset = self.data_info['train_dset']
        self.test_loader = self.data_info['test_loader']
        self.num_train_samples = self.data_info['num_train_samples']

        # visualization handling
        vis_cfg = config.get('visualization', {})
        self.visualization_enabled = vis_cfg.get('enable', False)

        if self.visualization_enabled:
            # Common config
            self.milestones = vis_cfg.get('milestones', [])
            if not self.milestones:
                self.logger.warning("Visualization is enabled, but no 'milestones' are configured. No checkpoints will be saved for visualization, and no embeddings will be generated.")
            self.milestone_epochs = [int(p * self.epochs) for p in self.milestones]
            self.embedding_params = vis_cfg.get("embedding_params", {})
            self.embedding_methods = [m.lower() for m in vis_cfg.get("embedding_methods", [])]
            if not self.embedding_methods:
                self.logger.warning("Visualization is enabled, but no 'embedding_methods' (e.g., 'umap', 'tsne') are configured. No embeddings will be computed.")

            # Train Visualizer
            try:
                # Create a non-shuffled loader for the training set
                vis_train_loader = torch.utils.data.DataLoader(
                    self.train_dset, 
                    batch_size=self.batch_size, 
                    shuffle=False, 
                    num_workers=self.num_data_workers, 
                    pin_memory=True
                )
                self.train_visualizer = Visualizer(
                    config, logger, 
                    data_loader=vis_train_loader, 
                    split="train", 
                    dataset_suffix="_train0"
                )
                self.logger.info(f"Train set visualizer enabled. Dataset: {self.train_visualizer.dataset_name}")
            except Exception as e:
                self.logger.info(f"Train set visualization was disabled because init failed: {e}")
                self.train_visualizer = None

            # Load snapshot
            if vis_cfg.get("load_snapshot", False):
                logger.info("Configuration set to 'load_snapshot'. Skipping training.")
                snap_path = vis_cfg.get("snapshot_path", None)
                try:
                    self.train_visualizer._load_snapshot_and_visualize(path=snap_path)
                    logger.info("Visualization session ended.")
                    return
                except Exception as e:
                    logger.error(f"Failed to load snapshot: {e}")
                    return

            # If creating visualizer failed, disable visualization entirely
            if not self.train_visualizer:
                self.visualization_enabled = False
                self.logger.info("All visualizations failed to initialize and are disabled.")
        
        else:
            self.logger.info(f"Not running visualization because it was disabled in config.")

    def resume(self, resume_path):
        if os.path.isfile(resume_path):
            self.logger.info(("=> loading checkpoint '{}'".format(resume_path)))
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
            self.best_epoch = checkpoint['best_epoch']
            self.model.module.load_state_dict(checkpoint['state_dict']) if hasattr(self.model, 'module') else self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.logger.info(("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch'])))
        else:
            self.logger.info(("=> no checkpoint found at '{}'".format(resume_path)))

    def _save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """
        Saves the model state and whether it is the best.
        Helper function for _save_model method.
        
        :param state: model state object
        :param is_best: boolean describing whether the current model is the best amongst all epochs
        :param filename: Filename of checkpoint file to save
        """
        base_filename = f"{self.method_name}_{filename}"
        filename = os.path.join(self.config['output_dir'], base_filename)
        best_filename = os.path.join(self.config['output_dir'], f'{self.method_name}_model_best.pth.tar')
        torch.save(state, filename)
        self.logger.info(f'Save checkpoint to {filename}')
        if is_best:
            shutil.copyfile(filename, best_filename)
            self.logger.info(f'Save best checkpoint to {best_filename}')

    def _save_model(self, filename=None):
        checkpoint = {
            'epoch': self.epoch,
            'state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch
        }
        
        filename = filename if filename is not None else f'checkpoint_epoch_{self.epoch}.pth.tar'
        self._save_checkpoint(checkpoint, is_best=False, filename=filename)

    def _load_state_dict_safe(self, checkpoint):
        """Load a state dict from a checkpoint, handling DataParallel wrapping."""
        state_dict = checkpoint['state_dict']
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def run_final_evaluations(self):
        """
        Iterates over all trained models, loads their checkpoints,
        and saves their predictions/losses as new fields in the
        FiftyOne *TEST* dataset.
        """
        if not self.test_visualizer:
            self.logger.info("Skipping final evaluations: no test visualizer.")
            return
            
        self.logger.info("Running final model evaluations on test set...")
        
        test_view = self.test_visualizer.fo_dataset
        test_sample_ids = test_view.values("id")
        
        # Get the ground truth labels from the dataset
        gt_labels = test_view.values("ground_truth.label")
        
        # Class map
        classes = self.data_info.get("classes", [str(i) for i in range(self.num_classes)])
        class_map_int_to_str = {i: label for i, label in enumerate(classes)}

        all_method_names = self.config.get("methods", [self.method_name])
        base_exp_dir = os.path.dirname(self.config['output_dir'].rstrip('/'))
        
        for method_name in all_method_names:
            self.logger.info(f"Processing evaluations for: {method_name}")
            
            try:
                checkpoint_path = os.path.join(
                    base_exp_dir,
                    method_name,
                    f'{method_name}_model_best.pth.tar'
                )
                
                if not os.path.exists(checkpoint_path):
                    self.logger.info(f"Checkpoint not found for {method_name}. Skipping.")
                    continue
                    
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                self._load_state_dict_safe(checkpoint)
                
                outputs, probs, preds, labels = self.extract_predictions()
                
                pred_field = f"{method_name}_prediction"
                loss_field = f"{method_name}_loss"
                eval_key   = f"eval_{method_name}"

                fo_predictions = []
                losses = []
                
                fo_predictions = [
                    fo.Classification(
                        label=class_map_int_to_str[preds[i].item()],
                        confidence=probs[i][preds[i].item()].item(),
                        logits=outputs[i].tolist()
                    )
                    for i in range(len(test_sample_ids))
                ]
                
                losses = []
                criterion = self.criterion
                outputs_cpu = outputs.cpu()
                labels_cpu = labels.cpu()
                for i in range(len(labels)):
                    sample_loss = criterion(
                        outputs_cpu[i].unsqueeze(0), 
                        labels_cpu[i].unsqueeze(0)
                    )
                    losses.append(sample_loss.item())

                test_view.set_values(pred_field, dict(zip(test_sample_ids, fo_predictions)))
                test_view.set_values(loss_field, dict(zip(test_sample_ids, losses)))

                test_view.evaluate_classifications(
                    pred_field,
                    gt_field="ground_truth",
                    eval_key=eval_key
                )
                self.logger.info(f"Saved results to fields: {pred_field}, {loss_field}, {eval_key}")

            except Exception as e:
                self.logger.info(f"Failed to process evaluations for {method_name}: {e}")
                continue

        self.logger.info("All model evaluations complete.")

    def run(self):
        self.before_run()
        self.run_begin_time = time.time()
        self.total_step = 0
        self.logger.info(f'Begin training for {self.method_name}...')
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            list_of_train_idx = self.before_epoch()
            selected_indices = self.train(list_of_train_idx)
            self.after_epoch(selected_indices)
            if self.total_step >= self.num_steps:
                self.logger.info(f'Finish training for {self.method_name} because num_steps {self.num_steps} is reached')
                break

        self.after_run()

    def before_run(self):
        pass

    def _get_loader_for_visualizer(self, visualizer):
        if self.train_visualizer and visualizer == self.train_visualizer:
            return torch.utils.data.DataLoader(
                self.train_dset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_data_workers, 
                pin_memory=True
            )
        return self.test_loader

    def process_milestones(self, visualizer):
        """
        Loads model checkpoints from milestones, runs inference once per milestone,
        and:
          1. Adds predictions (embeddings + labels) to the visualizer.
          2. Computes and stores per-sample loss (hardness) in FiftyOne.
        """
        self.logger.info("Processing milestone runs (Visualizer + Hardness tagging)...")
        all_method_names = self.config.get("methods", [self.method_name])
        loader = self._get_loader_for_visualizer(visualizer)
        base_exp_dir = os.path.dirname(self.config['output_dir'].rstrip('/'))
        
        for method_name in all_method_names:
            self.logger.info(f"Processing method: {method_name}")
            for milestone_epoch in self.milestone_epochs:
                try:
                    self.logger.info(f'Loading model weights for {method_name} epoch {milestone_epoch}')
                    checkpoint_path = os.path.join(
                        base_exp_dir,
                        method_name,
                        f'{method_name}_checkpoint_epoch_{milestone_epoch}.pth.tar'
                    )
                    
                    if not os.path.exists(checkpoint_path):
                        self.logger.info(f'Checkpoint not found: {checkpoint_path}')
                        continue
                        
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    self._load_state_dict_safe(checkpoint)
                    
                    # Run inference ONCE for both tasks
                    self.logger.info(f'Extracting predictions for epoch {milestone_epoch}')
                    # Note: We need features for visualization
                    outputs, probs, preds, labels, embeddings = self.extract_predictions(
                        loader, return_features=True
                    )

                    if outputs is None:
                        self.logger.warning(
                            f"extract_predictions returned None for epoch {milestone_epoch}. "
                            f"Skipping."
                        )
                        continue

                    # --- Task 1: Visualization ---
                    # Prefer the HF ground-truth embedding space for UMAP positions
                    hf_embs = visualizer.in_memory_embeddings.get("hf_ground_truth")
                    vis_embeddings = hf_embs if hf_embs is not None else embeddings

                    if vis_embeddings is not None:
                        visualizer.add_run(
                            epoch=milestone_epoch, 
                            embeddings=vis_embeddings, 
                            labels=preds,
                            selection_method=method_name
                        )
                        self.logger.info(f'Added visualization run for {method_name} epoch {milestone_epoch}')
                    else:
                        self.logger.warning(f"No embeddings available for visualization (epoch {milestone_epoch}).")

                    # --- Task 2: Hardness Tagging ---
                    # Compute losses
                    losses = []
                    criterion = self.criterion
                    # Move to CPU for loop processing to avoid potential GPU sync issues in loop
                    outputs_cpu = outputs.cpu()
                    labels_cpu = labels.cpu()
                    
                    for output, label in zip(outputs_cpu, labels_cpu):
                         losses.append(criterion(output.unsqueeze(0), label.unsqueeze(0)).item())

                    loss_field = f"loss_{method_name}_E{milestone_epoch}"
                    ids = visualizer.fo_dataset.values("id")
                    
                    if len(ids) == len(losses):
                        visualizer.fo_dataset.set_values(loss_field, dict(zip(ids, losses)), key_field="id")
                        visualizer.fo_dataset.save()
                        self.logger.info(f"Tagged '{loss_field}' on {len(losses)} samples.")
                    else:
                         self.logger.warning(
                            f"Loss count ({len(losses)}) != FO dataset size ({len(ids)}) "
                            f"for {loss_field}. Skipping tagging."
                        )

                except Exception as e:
                    self.logger.exception(
                        f"Failed to process milestone epoch {milestone_epoch} "
                        f"for {method_name}: {e}"
                    )
                    continue

    def after_run(self):
        """
        Called after all training epochs are complete.
        Handles final FiftyOne visualization computations and session closure.
        This only runs *once* for the *last* method in the config.
        """
        if self.method_name == self.last_selection_method: 
            # Training Set Visualizations
            if self.train_visualizer:
                self.logger.info("--- Processing Training Set Visualizations ---")
                try:
                    self.train_visualizer.add_huggingface_ground_truth_run()
                    self.process_milestones(self.train_visualizer)
                    self.train_visualizer.compute_all_visualizations()
                    self.run_final_evaluations()
                    self.train_visualizer._export_snapshot()
                except Exception as e:
                    self.logger.info(f"Failed to process train set visualizations: {e}")
                finally:
                    self.train_visualizer.launch_app()

            self.logger.info("All visualization runs complete.")
        else:
            self.logger.info(
                f"Skipping final visualization processing for method '{self.method_name}' "
                f"because it is not the last method ('{self.last_selection_method}'). "
                f"Final processing will run after the last method is complete."
            )

    def get_ratio_per_epoch(self):
        if self.epoch < self.warmup_epochs:
            return 1.0
        if self.ratio_scheduler == 'constant':
            return self.ratio
        elif self.ratio_scheduler == 'increase_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * self.epoch / self.epochs
        elif self.ratio_scheduler == 'decrease_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * self.epoch / self.epochs
        elif self.ratio_scheduler == 'increase_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * np.exp(self.epoch / self.epochs)
        elif self.ratio_scheduler == 'decrease_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * np.exp(self.epoch / self.epochs)
        else:
            raise NotImplementedError

    def before_epoch(self):
        self.logger.info(f'getting ratio per epoch.')
        self.ratio = self.get_ratio_per_epoch()
        self.logger.info(f'done. epoch={self.epoch}, ratio={self.ratio}')
        return np.arange(self.num_train_samples)
    
    def log_epoch_selection(self, selected_indices):
        """
        Logs the selected training samples for the current epoch to FiftyOne.
        """
        if not self.train_visualizer:
            self.logger.info("Skipping selection logging: no train visualizer.")
            return
        
        if selected_indices is None:
            self.logger.warning("`selected_indices` is None, cannot log epoch selection.")
            return

        self.logger.info(
            f"Logging {len(selected_indices)} selected samples for "
            f"{self.method_name} at epoch {self.epoch}..."
        )
        
        selection_field = f"selected_{self.method_name}_E{self.epoch}"
        self.logger.info(f"Storing selection status in field '{selection_field}'...")

        fo_dataset = self.train_visualizer.fo_dataset
        
        selection_dict = {}
        for sample in fo_dataset.select_fields(["original_index", "id"]):
            is_selected = sample.original_index in selected_indices
            selection_dict[sample.id] = is_selected
        
        fo_dataset.set_values(
            field_name=selection_field,
            values=selection_dict,
            key_field="id"
        )
        fo_dataset.save()
        
        num_selected = sum(selection_dict.values())
        self.logger.info(
            f"Done. Marked {num_selected} samples as selected in field '{selection_field}'."
        )

    def after_epoch(self, selected_indices=None):
        if self.visualization_enabled and self.epoch in self.milestone_epochs:
            self.logger.info(f'Saving model weights for milestone epoch {self.epoch}')
            self._save_model()
            self.log_epoch_selection(selected_indices)
        
    def before_batch(self, i, inputs, targets, indexes):
        return inputs, targets, indexes
    
    def after_batch(self, i, inputs, targets, indexes, outputs):
        pass

    def train(self, list_of_train_idx):
        """Train for one epoch and return the set of selected sample indices."""
        self.model.train()
        self.logger.info(
            'Epoch: [{} | {}] LR: {}'.format(
                self.epoch, self.epochs, self.optimizer.param_groups[0]['lr']
            )
        )

        list_of_train_idx = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(
            list_of_train_idx, batch_size=self.batch_size, drop_last=False
        )

        train_loader = torch.utils.data.DataLoader(
            self.train_dset,
            num_workers=self.num_data_workers,
            pin_memory=True,
            batch_sampler=batch_sampler
        )
        total_batch = len(train_loader)
        epoch_begin_time = time.time()

        epoch_selected_indices = set() if self.visualization_enabled else None
                
        for i, datas in enumerate(train_loader):
            inputs  = datas['input'].cuda()
            targets = datas['target'].cuda()
            indexes = datas['index']

            inputs, targets, indexes = self.before_batch(i, inputs, targets, indexes)

            if epoch_selected_indices is not None:
                epoch_selected_indices.update(indexes.cpu().numpy())

            outputs, features = (
                self.model(inputs, self.need_features)
                if self.need_features
                else (self.model(inputs, False), None)
            )
            loss = self.criterion(outputs, targets)
            self.while_update(outputs, loss, targets, features, indexes,
                              batch_idx=i, batch_size=self.batch_size)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.after_batch(i, inputs, targets, indexes, outputs.detach())

            if i % self.config['logger_opt']['print_iter'] == 0:
                _, predicted = torch.max(outputs.data, 1)
                total    = targets.size(0)
                correct  = (predicted == targets).sum().item()
                train_acc = correct / total
                self.logger.info(
                    f'Epoch: {self.epoch}/{self.epochs}, '
                    f'Iter: {i}/{total_batch}, '
                    f'global_step: {self.total_step + i}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Train acc: {train_acc:.4f}, '
                    f'lr: {self.optimizer.param_groups[0]["lr"]:.6f}'
                )
                    
        self.scheduler.step()
        self.total_step = self.total_step + total_batch

        now = time.time()
        self.logger.wandb_log({
            'loss'     : loss.item(),
            'epoch'    : self.epoch,
            'lr'       : self.optimizer.param_groups[0]['lr'],
            self.training_opt['loss_type']: loss.item()
        })

        val_acc = self.test()
        self.logger.wandb_log({
            'val_acc'      : val_acc,
            'epoch'        : self.epoch,
            'total_time'   : now - self.run_begin_time,
            'total_step'   : self.total_step,
            'time_epoch'   : now - epoch_begin_time,
            'best_val_acc' : max(self.best_acc, val_acc)
        })
        self.logger.info(
            f'=====> Time: {now - self.run_begin_time:.4f} s, '
            f'Time this epoch: {now - epoch_begin_time:.4f} s, '
            f'Total step: {self.total_step}'
        )

        self.logger.info('=====> Save model')
        is_best = False
        if val_acc > self.best_acc:
            self.best_epoch = self.epoch
            self.best_acc   = val_acc
            is_best = True

        checkpoint = {
            'epoch'     : self.epoch,
            'state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'scheduler' : self.scheduler.state_dict(),
            'best_acc'  : self.best_acc,
            'best_epoch': self.best_epoch
        }
        self._save_checkpoint(checkpoint, is_best)
        self.logger.info(
            f'=====> Epoch: {self.epoch}/{self.epochs}, '
            f'Best val acc: {self.best_acc:.4f}, '
            f'Current val acc: {val_acc:.4f}'
        )
        self.logger.info(f'=====> Best epoch: {self.best_epoch}')

        return epoch_selected_indices

    def while_update(self, outputs, loss, targets, features, indexes, batch_idx, batch_size):
        pass

    def test(self):
        self.logger.info('=====> Start Validation')
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for i, datas in enumerate(self.test_loader):
                inputs  = datas['input'].cuda()
                targets = datas['target'].cuda()
                outputs = model(inputs)
                preds   = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = np.mean(all_preds == all_labels)
        self.logger.info(f'=====> Validation Accuracy: {acc:.4f}')
        return acc

    def extract_predictions(self, data_loader=None, return_features=False):
        """
        Extract predictions (and optionally features) from the model.

        Returns:
            Without return_features:
                (all_outputs, all_probs, all_preds, all_labels)
            With return_features:
                (all_outputs, all_probs, all_preds, all_labels, all_features)

            On failure (only possible with return_features=True):
                (None, None, None, None, None)
        """
        if data_loader is None:
            data_loader = self.test_loader

        model = self.model
        self.logger.info('Extracting Predictions')
        unwrapped_model = model.module if hasattr(model, 'module') else model
        unwrapped_model.eval()

        all_outputs  = []
        all_probs    = []
        all_preds    = []
        all_labels   = []
        all_features = [] if return_features else None
        features_ok  = True  # tracks whether feature extraction is working

        with torch.no_grad():
            for i, datas in enumerate(data_loader):
                inputs  = datas['input'].cuda()
                targets = datas['target'].cuda()
                
                if return_features and features_ok:
                    try:
                        outputs, features = unwrapped_model(inputs, True)
                        if features is None:
                            self.logger.warning(
                                "Model returned None for features; "
                                "disabling feature extraction."
                            )
                            features_ok = False
                            outputs = unwrapped_model(inputs)
                        else:
                            all_features.append(features.cpu())
                    except (TypeError, ValueError) as e:
                        self.logger.error(
                            f"Model forward with return_features=True failed: {e}. "
                            f"Falling back to standard forward."
                        )
                        features_ok = False
                        outputs = unwrapped_model(inputs)
                else:
                    outputs = unwrapped_model(inputs)

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_outputs.append(outputs.cpu())
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(targets.cpu())

        all_outputs = torch.cat(all_outputs, dim=0)
        all_probs   = torch.cat(all_probs,   dim=0)
        all_preds   = torch.cat(all_preds,   dim=0)
        all_labels  = torch.cat(all_labels,  dim=0)
        
        self.logger.info('Done extracting predictions')

        if return_features:
            if not features_ok or not all_features:
                return None, None, None, None, None
            return (
                all_outputs, all_probs, all_preds, all_labels,
                torch.cat(all_features, dim=0)
            )

        return all_outputs, all_probs, all_preds, all_labels