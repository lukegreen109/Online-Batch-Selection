import torch
import models
import os
import shutil
import numpy as np
import time
from .method_utils import *
from visualization.Visualization import Visualizer
import matplotlib.pyplot as plt
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
        self.visualizer = None
        self.milestones = []
        self.milestone_epochs = []
        self.embedding_methods = []
        self.embedding_params = {}

        # get names of datasets
        dataset_info = config.get("dataset", {})
        self.dataset_name = dataset_info.get("name", "mnist").lower()
            # arg passed to fetch dataset
        self.foz_name = dataset_info.get("foz_name", self.dataset_name).lower()
            # arg passed to fetch FO dataset

        # dataset handling
        self.data_info = getattr(data, config['dataset']['name'])(config, logger) 
            # load in normal dataset
        self.num_classes = self.data_info['num_classes']
        self.train_dset = self.data_info['train_dset']
        self.test_loader = self.data_info['test_loader']
        self.num_train_samples = self.data_info['num_train_samples']

        # visualization handling
        vis_cfg = config.get('visualization', {})
        self.visualization_enabled = vis_cfg.get('enable', False)

        if self.visualization_enabled:
            try:
                self.visualizer = Visualizer(config, logger)
                self.milestones = vis_cfg.get('milestones', [])
                self.milestone_epochs = [int(p * self.epochs) for p in self.milestones]
                self.embedding_params = vis_cfg["embedding_params"]
                self.embedding_methods = [m.lower() for m in vis_cfg["embedding_methods"]]
                self.logger.info(f'Visualization enabled. Milestone epochs: {self.milestone_epochs}, params: {self.embedding_params}')
            except Exception as e:
                self.logger.info(f"Visualization was disabled because init failed: {e}")
                self.visualization_enabled = False
        else:
            self.logger.info(f"Not running visualization because it was disabled in config.")

    def resume(self, resume_path):
        if os.path.isfile(resume_path):
            self.logger.info(("=> loading checkpoint '{}'".format(resume_path)))
            checkpoint = torch.load(resume_path, map_location='cpu')
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
            self.best_epoch = checkpoint['best_epoch']
            # self.model.load_state_dict(checkpoint['state_dict'])
            self.model.module.load_state_dict(checkpoint['state_dict']) if hasattr(self.model, 'module') else self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.logger.info(("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch'])))
        else:
            self.logger.info(("=> no checkpoint found at '{}'".format(resume_path)))

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
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
        
        # Use the existing save_checkpoint function
        filename = filename if filename is not None else f'checkpoint_epoch_{self.epoch}.pth.tar'
        self.save_checkpoint(checkpoint, is_best=False, filename=filename)

    def _export_snapshot(self, path=None):
        """Saves the entire dataset with all runs and embeddings."""
        if self.visualization_enabled:
            path = path if path is not None else f"./saved_runs/{self.dataset_name}"
            self.visualizer.fo_dataset.export(
                export_dir=path,
                dataset_type=fo.types.FiftyOneDataset
            )
            self.logger.info(f"Saved to: {path}")
            return path

    def run(self):
        self.before_run()
        self.run_begin_time = time.time()
        self.total_step = 0
        self.logger.info(f'Begin training for {self.method_name}...')
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            list_of_train_idx = self.before_epoch()
            self.train(list_of_train_idx)
            self.after_epoch()
            if self.total_step >= self.num_steps:
                self.logger.info(f'Finish training for {self.method_name} because num_steps {self.num_steps} is reached')
                break

        self.after_run()

    def before_run(self):
        pass
            
    def after_run(self):
        """
        Called after all training epochs are complete.
        Handles final FiftyOne visualization computations and session closure.
        """
        if self.method_name == self.last_selection_method: # After all runs...
            if self.visualization_enabled:
                # 1. Compute ground truth embeddings using HF model
                self.visualizer.add_huggingface_ground_truth_run()

                # 2. Compute embeddings after training all models & all milestones
                all_method_names = self.config.get("methods", [self.method_name])
                self.logger.info(f"Running final visualization for all methods: {all_method_names}")
                for method_name in all_method_names:
                    self.logger.info(f"Processing method: {method_name}")
                    for milestone_epoch in self.milestone_epochs:
                        try:
                            self.logger.info(f'Loading model weights for epoch {milestone_epoch}')
                            
                            # Load model weights from saved checkpoint
                            checkpoint_path = os.path.join(
                                self.config['output_dir'], 
                                f'{method_name}_checkpoint_epoch_{milestone_epoch}.pth.tar'
                            )
                            
                            if os.path.exists(checkpoint_path):
                                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                                
                                # Load the saved weights into the model
                                if hasattr(self.model, 'module'):
                                    self.model.module.load_state_dict(checkpoint['state_dict'])
                                else:
                                    self.model.load_state_dict(checkpoint['state_dict'])
                                
                                self.logger.info(f'Successfully loaded weights for epoch {milestone_epoch}')
                                
                                # Compute embeddings with the loaded model
                                # self.logger.info(f'Computing embeddings for epoch {milestone_epoch}')
                                # embeddings, labels, images = self.extract_embeddings()

                                # Get predictions for checkpoint model
                                self.logger.info(f'Extracting predictions for epoch {milestone_epoch}')
                                outputs, preds, pred_labels = self.extract_predictions()
                                self.logger.info(f'Extracted {preds.shape[0]} predictions.')

                                embeddings = self.visualizer.hf_embs # Use HF model's embeddings
                                labels = preds # Use our own model's predictions
                                
                                # Add run to visualizer
                                self.visualizer.add_run(
                                    epoch=milestone_epoch, 
                                    embeddings=embeddings, 
                                    labels=labels, 
                                    selection_method=method_name
                                )
                                
                                self.logger.info(f'Successfully added visualization for epoch {milestone_epoch}')
                            else:
                                self.logger.info(f'Checkpoint not found for epoch {milestone_epoch}: {checkpoint_path}')
                            
                        except Exception as e:
                            self.logger.info(f"Failed to process milestone epoch {milestone_epoch}: {e}")
                            continue

                self.logger.info('Computing all visualizations & launching FiftyOne app...')
                self.visualizer.compute_all_visualizations()

                self.logger.info("Saving snapshot of Voxel51 run...")
                self._export_snapshot()

                self.logger.info("Launching Voxel51 Application...")
                self.visualizer.launch_app()

                self.logger.info("Done")

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
        # get ratio per epoch
        self.logger.info(f'getting ratio per epoch.')
        self.ratio = self.get_ratio_per_epoch()
        self.logger.info(f'done. epoch={self.epoch}, ratio={self.ratio}')

        # select samples for this epoch
        return np.arange(self.num_train_samples)
    
    def after_epoch(self):
        # save model weights instead of saving embedding
        if self.visualization_enabled and self.epoch in self.milestone_epochs:
            self.logger.info(f'Saving model weights for milestone epoch {self.epoch}')
            self._save_model()
            #embeddings, labels, images = self.extract_embeddings()
            #self.visualizer.add_run(epoch=self.epoch, embeddings=embeddings, labels=labels, selection_method=self.method_name)
        
    def before_batch(self, i, inputs, targets, indexes):
        # online batch selection
        return inputs, targets, indexes
    
    def after_batch(self, i,inputs, targets, indexes, outputs):
        pass

    def train(self, list_of_train_idx):
        # train for one epoch
        self.model.train()
        self.logger.info('Epoch: [{} | {}] LR: {}'.format(self.epoch, self.epochs, self.optimizer.param_groups[0]['lr']))

        # data loader
        # sub_train_dset = torch.utils.data.Subset(self.train_dset, list_of_train_idx)
        list_of_train_idx = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(list_of_train_idx, batch_size=self.batch_size,
                                                      drop_last=False)
        # list_of_train_idx = list(batch_sampler)

        train_loader = torch.utils.data.DataLoader(self.train_dset, num_workers=self.num_data_workers, pin_memory=True, batch_sampler=batch_sampler)
        total_batch = len(train_loader)
        epoch_begin_time = time.time()
                
        # train
        for i, datas in enumerate(train_loader):
            inputs = datas['input'].cuda()
            targets = datas['target'].cuda()
            indexes = datas['index']
            inputs, targets, indexes = self.before_batch(i, inputs, targets, indexes)
            ########### Might need to change this to not use features
            outputs, features = self.model(inputs, self.need_features) if self.need_features else (self.model(inputs, False), None)
            loss = self.criterion(outputs, targets)
            self.while_update(outputs, loss, targets, features, indexes, batch_idx=i, batch_size=self.batch_size)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.after_batch(i,inputs, targets, indexes,outputs.detach())
            if i % self.config['logger_opt']['print_iter'] == 0:
                # train acc
                _, predicted = torch.max(outputs.data, 1)
                total = targets.size(0)
                correct = (predicted == targets).sum().item()
                train_acc = correct / total
                self.logger.info(f'Epoch: {self.epoch}/{self.epochs}, Iter: {i}/{total_batch}, global_step: {self.total_step+i}, Loss: {loss.item():.4f}, Train acc: {train_acc:.4f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}')
                    
        self.scheduler.step()
        self.total_step = self.total_step + total_batch
        # test
        now = time.time()
        self.logger.wandb_log({'loss': loss.item(), 'epoch': self.epoch, 'lr': self.optimizer.param_groups[0]['lr'], self.training_opt['loss_type']: loss.item()})
        val_acc = self.test()
        self.logger.wandb_log({'val_acc': val_acc, 'epoch': self.epoch, 'total_time': now - self.run_begin_time, 'total_step': self.total_step, 'time_epoch': now - epoch_begin_time, 'best_val_acc': max(self.best_acc, val_acc)})
        # self.logger.info(f'=====> Best val acc: {max(self.best_acc, val_acc):.4f}, Current val acc: {val_acc:.4f}')
        self.logger.info(f'=====> Time: {now - self.run_begin_time:.4f} s, Time this epoch: {now - epoch_begin_time:.4f} s, Total step: {self.total_step}')
            # save model
        self.logger.info('=====> Save model')
        is_best = False
        if val_acc > self.best_acc:
            self.best_epoch = self.epoch
            self.best_acc = val_acc
            is_best = True
        checkpoint = {
                'epoch': self.epoch,
                'state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_acc': self.best_acc,
                'best_epoch': self.best_epoch
            }
        self.save_checkpoint(checkpoint, is_best)
        self.logger.info(f'=====> Epoch: {self.epoch}/{self.epochs}, Best val acc: {self.best_acc:.4f}, Current val acc: {val_acc:.4f}')
        self.logger.info(f'=====> Best epoch: {self.best_epoch}')

    def while_update(self, outputs, loss, targets, features, indexes, batch_idx, batch_size):
        pass

    def test(self):
        self.logger.info('=====> Start Validation')
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, datas in enumerate(self.test_loader):
                inputs = datas['input'].cuda()
                targets = datas['target'].cuda()
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = np.mean(all_preds == all_labels)
        self.logger.info(f'=====> Validation Accuracy: {acc:.4f}')

        return acc

    def extract_predictions(self):
        """
        Extract predictions and corresponding labels from the test set
        using the provided model.

        Args:
            model (torch.nn.Module): The model (potentially wrapped) to use.

        Returns:
            all_outputs (torch.Tensor): shape (N, num_classes) - raw logits
            all_preds (torch.Tensor): shape (N,) - predicted class indices
            all_labels (torch.Tensor): shape (N,) - ground truth labels
        """
        model = self.model
        self.logger.info('Extracting Predictions')
        unwrapped_model = model.module if hasattr(model, 'module') else model
        unwrapped_model.eval()
        
        all_preds = []
        all_labels = []
        all_outputs = [] 

        with torch.no_grad():
            for i, datas in enumerate(self.test_loader):
                inputs = datas['input'].cuda()
                targets = datas['target'].cuda()
                
                # Use the unwrapped 'model'
                outputs = unwrapped_model(inputs) 
                preds = torch.argmax(outputs, dim=1) 
                
                all_outputs.append(outputs.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(targets.cpu())

        # Concatenate results
        all_outputs = torch.cat(all_outputs, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        self.logger.info('Done extracting predictions')
        return all_outputs, all_preds, all_labels

