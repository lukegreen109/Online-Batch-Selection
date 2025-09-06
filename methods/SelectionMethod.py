import torch
import models
import os
import shutil
import numpy as np
import time
from .method_utils import *
from visualization.Visualization import Visualizer
import matplotlib.pyplot as plt

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

        # Criterion
        self.criterion = create_criterion(config, logger)

        # Visualization
        self.visualizer = None
        self.milestones = []
        self.milestone_epochs = []
        self.embedding_methods = []
        self.embedding_params = {}

        vis_cfg = config.get('visualization', {})
        self.visualization_enabled = vis_cfg.get('enable', False)
        self.use_fiftyone = vis_cfg.get('use_fiftyone', False)
        self.use_fiftyone_dataloader = vis_cfg.get('use_fiftyone_dataloader', False)

        if self.visualization_enabled:
            try:
                self.visualizer = Visualizer(config, logger, self.model)
                self.milestones = vis_cfg.get('milestones', [])
                self.milestone_epochs = [int(p * self.epochs) for p in self.milestones]
                self.embedding_params = vis_cfg["embedding_params"]
                self.embedding_methods = [m.lower() for m in vis_cfg["embedding_methods"]]
                self.logger.info(f'Visualization enabled. Milestone epochs: {self.milestone_epochs}, params: {self.embedding_params}')
            except Exception as e:
                self.logger.info(f"Visualization was disabled because init failed: {e}")
                self.visualization_enabled = False

        # Dataset handling
        self.dataset_name = config['dataset']['name'].lower()

        if self.use_fiftyone_dataloader:
            import fiftyone as fo
            from fiftyone.utils.torch import FiftyOneTorchDataset
            import fiftyone.zoo as foz
            self.logger.info(f'Loading {self.dataset_name} from FiftyOne Zoo')
            self.fo_test_dataset = foz.load_zoo_dataset(self.dataset_name, split="test", shuffle=True)
            self.fo_train_dataset = foz.load_zoo_dataset(self.dataset_name, split="train", shuffle=True)
            self.train_dset = FiftyOneTorchDataset(self.fo_train_dataset, label_field="ground_truth")
            self.test_loader = torch.utils.data.DataLoader(
                FiftyOneTorchDataset(self.fo_test_dataset, label_field="ground_truth"),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_data_workers,
                pin_memory=True
            )
            self.num_train_samples = len(self.fo_train_dataset)
        else:
            import data
            self.data_info = getattr(data, config['dataset']['name'])(config, logger)
            self.num_classes = self.data_info['num_classes']
            self.train_dset = self.data_info['train_dset']
            self.test_loader = self.data_info['test_loader']
            self.num_train_samples = self.data_info['num_train_samples']

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
        filename = os.path.join(self.config['output_dir'],filename)
        best_filename = os.path.join(self.config['output_dir'],'model_best.pth.tar')
        torch.save(state, filename)
        self.logger.info(f'Save checkpoint to {filename}')
        if is_best:
            shutil.copyfile(filename, best_filename)
            self.logger.info(f'Save best checkpoint to {best_filename}')

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
        
    def before_epoch(self):
        # select samples for this epoch
        return np.arange(self.num_train_samples)
    
    def _extract_embeddings_and_maybe_save_images(self):
        """
        Extract embeddings and save images if FiftyOne is enabled.
        Images are used for thumbnail visualization in FiftyOne.
        Returns:
            all_embs (torch.Tensor): shape (N, D)
            all_labels (torch.Tensor): shape (N,)
            filepaths (list): list of file paths if FiftyOne is enabled, else empty list
        """
        all_embs, all_labels, filepaths = [], [], []
        save_dir = os.path.join(self.config['output_dir'], "fiftyone_images", f"{self.dataset_name}", f"epoch_{self.epoch}")
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for datas in self.test_loader:
                x_batch = datas['input'].cuda()
                y_batch = datas['target'].cuda()
                indexes = datas['index']

                _, embs = self.model.feat_nograd_forward(x_batch)

                if self.use_fiftyone:
                    for i, idx in enumerate(indexes):
                        img_path = os.path.join(save_dir, f"sample_{idx}.png")
                        if self.in_channels == 1: # 1 channel
                            image = x_batch[i].cpu().squeeze().numpy()
                        else: # 3 channels
                            image = np.transpose(x_batch[i].cpu().numpy(), (1, 2, 0))
                        plt.imsave(img_path, image, cmap="gray")
                        filepaths.append(img_path)

                all_embs.append(embs.cpu())
                all_labels.append(y_batch.cpu())

        return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0), filepaths
    
    def _extract_embeddings(self):
        """
        Extract embeddings and corresponding images.

        Returns:
            all_embs (torch.Tensor): shape (N, D)
            all_labels (torch.Tensor): shape (N,)
            all_images (list[np.ndarray]): list of N images (H, W) or (H, W, 3)
        """
        all_embs, all_labels, all_images = [], [], []

        with torch.no_grad():
            for datas in self.test_loader:
                x_batch = datas['input'].cuda()
                y_batch = datas['target'].cuda()

                # Forward pass for embeddings
                _, embs = self.model.feat_nograd_forward(x_batch)

                # Collect embeddings + labels
                all_embs.append(embs.cpu())
                all_labels.append(y_batch.cpu())

                # Convert batch to numpy images (handle channels)
                for i in range(x_batch.size(0)):
                    if self.in_channels == 1:  # grayscale
                        image = x_batch[i].cpu().squeeze().numpy()
                    else:  # RGB
                        image = np.transpose(x_batch[i].cpu().numpy(), (1, 2, 0))
                    all_images.append(image)

        return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0), all_images


    def after_epoch(self):
        if self.visualization_enabled and self.epochs is not None and self.epoch in self.milestone_epochs:
            self.logger.info(f'Visualizing embeddings at epoch {self.epoch} using {self.embedding_methods}')

            all_embs, all_labels, filepaths = self._extract_embeddings_and_maybe_save_images()

            # Visualize embeddings
            self.visualizer.visualize_embeddings(
                all_embs,
                all_labels,
                epoch=self.epoch,
                selected_idx=None,
                use_fiftyone=self.use_fiftyone,
                filepaths=filepaths if self.use_fiftyone else None
            )

    def after_run(self):
        pass

    def before_batch(self, i, inputs, targets, indexes):
        # online batch selection
        return inputs, targets, indexes
    
    def after_batch(self, i,inputs, targets, indexes,outputs):
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
                self.logger.info(f'Epoch: {self.epoch}/{self.training_opt["num_epochs"]}, Iter: {i}/{total_batch}, global_step: {self.total_step+i}, Loss: {loss.item():.4f}, Train acc: {train_acc:.4f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}')
                    
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
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = np.mean(all_preds == all_labels)
        self.logger.info(f'=====> Validation Accuracy: {acc:.4f}')

        return acc

