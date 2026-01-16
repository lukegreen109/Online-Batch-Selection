import torch
import models
import os
import shutil
import numpy as np
import time
from .method_utils import *
import data
from torch.utils.data import DataLoader, Subset
from ema_pytorch import EMA



class ReweightMethod(object):
    method_name = 'ReweightMethod'
    def __init__(self, config, logger):
        logger.info(f'Creating {self.method_name}...')
        self.config = config
        self.logger = logger
        # create model
        model_type = config['networks']['type']
        model_args = config['networks']['params']
        self.training_opt = config['training_opt']
        self.model = getattr(models, model_type)(**model_args)
        self.start_epoch = 0
        self.best_acc = 0
        self.best_epoch = 0
        # gpu
        self.num_gpus = config['num_gpus']
        if self.num_gpus == 0:
            self.model = self.model.cpu()
        elif self.num_gpus == 1:
            self.model = self.model.cuda()
        elif self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            raise ValueError(f'Wrong number of GPUs: {self.num_gpus}')
        
        # create optimizer
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        # resume
        config['training_opt']['resume'] = config['training_opt']['resume'] if 'resume' in config['training_opt'] else None
        if config['training_opt']['resume'] is not None:
            self.resume(config['training_opt']['resume'])
        
        # create EMA model
        self.ema_net = EMA(
            self.model,
            beta=config["ema_momentum"],
            update_after_step=0,
            update_every=5,
        )
        self.ema_net.eval()

        self.epochs = config['training_opt']['num_epochs'] if 'num_epochs' in config['training_opt'] else None
        self.num_steps = config['training_opt']['num_steps'] if 'num_steps' in config['training_opt'] else None
        if self.epochs is None and self.num_steps is None:
            raise ValueError('Must specify either num_epochs or num_steps in training_opt')
        self.num_data_workers = config['training_opt']['num_data_workers']
        self.batch_size = config['training_opt']['batch_size']

        # data
        self.data_info = getattr(data, config['dataset']['name'])(config, logger)
        self.num_classes = self.data_info['num_classes']
        self.train_dset = self.data_info['train_dset']
        self.test_loader = self.data_info['test_loader']
        self.num_train_samples = self.data_info['num_train_samples']

        # If including holdout dataset for rholoss holdout model
        if config['dataset']['include_holdout'] == True:
            
            train_dset_unaugmented = self.data_info['train_dset_unaugmented']
            
            self.holdout_percentage = config['rholoss']['holdout_percentage']
            self.holdout_batch_size = config['rholoss']['holdout_batch_size']

            # Split data into main model and holdout model
            holdout_len = int(self.num_train_samples * self.holdout_percentage)
            main_len = int(self.num_train_samples - holdout_len)
            
            # Create augmented and non-augmented datasets
            self.original_train_dset = self.train_dset

            indices = torch.randperm(len(self.original_train_dset), generator=torch.Generator().manual_seed(config['seed']))
            main_indices = indices[:main_len]
            holdout_indices = indices[main_len:]

            # Apply same indices to both datasets
            self.train_dset = Subset(self.original_train_dset, main_indices)
            self.train_dset_unaugmented = Subset(train_dset_unaugmented, main_indices)
            self.holdout_dataset_augmented = Subset(self.original_train_dset, holdout_indices)
            self.holdout_dataset_unaugmented = Subset(train_dset_unaugmented, holdout_indices)

            self.num_train_samples = main_len

            self.holdout_dataloader_augmented = DataLoader(self.holdout_dataset_augmented, batch_size=self.holdout_batch_size, shuffle=True)
            self.holdout_dataloader_unaugmented = DataLoader(self.holdout_dataset_unaugmented, batch_size=self.holdout_batch_size, shuffle=True)
            self.train_dataloader_augmented = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, pin_memory = True, num_workers=self.num_data_workers)
            self.train_dataloader_unaugmented = DataLoader(self.train_dset_unaugmented, batch_size=self.batch_size, shuffle=True, pin_memory = True, num_workers=self.num_data_workers)      

        self.criterion = create_criterion(config, logger)

        self.need_features = False
                

        
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
            list_of_train_idx = self.before_epoch(epoch)
            self.train(epoch, list_of_train_idx)
            self.after_epoch(epoch)
            if self.num_steps is not None and self.total_step >= self.num_steps:
                self.logger.info(f'Finish training for {self.method_name} because num_steps {self.num_steps} is reached')
                break

        self.after_run()

    def before_run(self):
        pass
        

    def before_epoch(self,epoch):
        # select samples for this epoch
        return np.arange(self.num_train_samples)

    def after_epoch(self, epoch):
        pass

    def after_run(self):
        pass

    def before_batch(self, i, inputs, targets, indexes, epoch):
        return torch.ones_like(targets).float().cuda() / len(targets), inputs, targets, indexes
    
    def after_batch(self, i, inputs, targets, indexes, outputs):
        self.ema_net.update()


    def train(self, epoch, list_of_train_idx):
        # train for one epoch
        self.model.train()
        self.logger.info('Epoch: [{} | {}] LR: {}'.format(epoch, self.epochs, self.optimizer.param_groups[0]['lr']))

        # data loader
        # sub_train_dset = torch.utils.data.Subset(self.train_dset, list_of_train_idx)
        list_of_train_idx = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(list_of_train_idx, batch_size=self.batch_size,
                                                      drop_last=False)
        # list_of_train_idx = list(batch_sampler)

        train_loader = torch.utils.data.DataLoader(self.train_dset, num_workers=self.num_data_workers, pin_memory=True, batch_sampler=batch_sampler)
        total_batch = len(train_loader)
        epoch_begin_time = time.time()
        epoch_loss = 0.0
        num_samples = 0
        # train
        epoch_train_acc = 0.0
        for i, datas in enumerate(train_loader):
            inputs = datas['input'].cuda()
            targets = datas['target'].cuda()
            indexes = datas['index']
            weights, inputs, targets, indexes = self.before_batch(i, inputs, targets, indexes, epoch)
            # outputs, features = self.model(x=inputs, need_features=self.need_features, targets=targets) if self.need_features else (self.model(x=inputs, need_features=False, targets=targets), None)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            weighted_loss = (loss * weights).sum()
            # self.while_update(outputs, loss, targets, epoch, features, indexes, batch_idx=i, batch_size=self.batch_size)
            self.optimizer.zero_grad()
            weighted_loss.backward()
            # Extra gradient clipping can be added here to safeguard against exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.after_batch(i,inputs, targets, indexes,outputs.detach())
            # record statistics
            batch_size = targets.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            epoch_train_acc += correct
            if i % self.config['logger_opt']['print_iter'] == 0:
                self.logger.info(f'Epoch: {epoch}/{self.training_opt["num_epochs"]}, Iter: {i}/{total_batch}, global_step: {self.total_step+i}, Loss: {epoch_loss / num_samples}, Train acc: {epoch_train_acc / num_samples}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}')          
        avg_epoch_train_loss = epoch_loss / num_samples
        avg_epoch_train_acc = epoch_train_acc / num_samples
        self.scheduler.step()
        self.total_step = self.total_step + total_batch
        # test
        now = time.time()
        self.logger.wandb_log({'train_loss': avg_epoch_train_loss, 'train_acc': avg_epoch_train_acc, 'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr'], self.training_opt['loss_type']: loss.item()}, commit=False)
        val_acc, ema_val_acc, avg_epoch_test_loss = self.test()
        self.logger.wandb_log({'val_loss': avg_epoch_test_loss, 'val_acc': val_acc, 'ema_val_acc': ema_val_acc, 'epoch': epoch, 'total_time': now - self.run_begin_time, 'total_step': self.total_step, 'time_epoch': now - epoch_begin_time, 'best_val_acc': max(self.best_acc, val_acc)})
        self.logger.info(f'=====> Time: {now - self.run_begin_time:.4f} s, Time this epoch: {now - epoch_begin_time:.4f} s, Total step: {self.total_step}')
            # save model
        self.logger.info('=====> Save model')
        is_best = False
        if val_acc > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = val_acc
            is_best = True
        checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_acc': self.best_acc,
                'best_epoch': self.best_epoch
            }
        self.save_checkpoint(checkpoint, is_best)
        self.logger.info(f'=====> Epoch: {epoch}/{self.epochs}, Best val acc: {self.best_acc:.4f}, Current val acc: {val_acc:.4f}')
        self.logger.info(f'=====> Best epoch: {self.best_epoch}')

    def while_update(self, outputs, loss, targets, epoch, features, indexes, batch_idx, batch_size):
        pass

    def test(self):
        # customize to use ema model for testing
        self.logger.info('=====> Start Validation')
        model = self.model.module if hasattr(self.model, 'module') else self.model
        ema_model = self.ema_net.ema_module if hasattr(self.ema_net, 'ema_module') else self.ema_net
        model.eval()
        all_preds = []
        all_ema_preds = []
        all_labels = []
        epoch_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for i, datas in enumerate(self.test_loader):
                inputs = datas['input'].cuda()
                targets = datas['target'].cuda()
                outputs = model(inputs)
                test_loss = self.criterion(outputs, targets)
                ema_outputs = ema_model(inputs)
                ema_preds = torch.argmax(ema_outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_ema_preds.append(ema_preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
                batch_size = targets.size(0)
                epoch_loss += test_loss.item() * batch_size
                num_samples += batch_size
        avg_epoch_test_loss = epoch_loss / num_samples
        all_preds = np.concatenate(all_preds)
        all_ema_preds = np.concatenate(all_ema_preds)
        all_labels = np.concatenate(all_labels)
        acc = np.mean(all_preds == all_labels)
        ema_acc = np.mean(all_ema_preds == all_labels)
        self.logger.info(f'=====> Validation Accuracy: {acc:.4f}')
        self.logger.info(f'=====> EMA Validation Accuracy: {ema_acc:.4f}')

        return acc, ema_acc, avg_epoch_test_loss