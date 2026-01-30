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
from collections import Counter




class SelectionMethod(object):
    method_name = 'SelectionMethod'
    def __init__(self, config, logger):
        logger.info(f'Creating {self.method_name}...')
        self.config = config
        self.logger = logger
        # create model
        model_type = config['networks']['type']
        model_args = config['networks']['params']
        model_args['in_channels'] = config['dataset']['in_channels']
        model_args['num_classes'] = config['dataset']['num_classes']
        self.training_opt = config['training_opt']
        self.model = getattr(models, model_type)(**model_args)
        self.start_epoch = 0
        self.best_acc = 0
        self.best_epoch = 0
        self.num_selected_noisy_indexes = 0
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
            beta=0.99,
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


        if config['dataset']['noise'] == True:
            self.noise == True
            all_dataset = self.train_dset
            all_dataset = all_dataset.dataset
            noise_percentage = config['dataset']["noise_percent"]
            # Check it works
            ##print("Label distribution BEFORE noise:")
            ##print(Counter(all_dataset.targets))
            # Get Labels
            targets = all_dataset.targets
            # Get % of all_dataset 
            num_samples = len(targets)
            num_noisy = int(noise_percentage * num_samples)
            noisy_indices = torch.randperm(num_samples)[:num_noisy] 
            # For the 10% chosen we will swap
            for i in noisy_indices:
                current_target = targets[i]
                new_target = np.random.randint(self.num_classes)
                # Repeat swapping labels until it is different
                while targets[i] == current_target:
                    new_target = np.random.randint(self.num_classes)
                    targets[i] = new_target
            if(self.noise == True):
                self.noisy_indices = noisy_indices
            # Check it works
            #print("Label distribution AFTER noise:")
            #print(Counter(all_dataset.targets))
            
            ## Reset seeds of random number generator
        else:
            self.noise = False
            self.noisy_indices = np.array([])

        self.criterion = create_criterion(config, logger)
        self.need_features = False

        self.train_loader = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_data_workers, pin_memory=True, drop_last=False)

        
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
        filename = os.path.join(self.config['save_dir'],filename)
        best_filename = os.path.join(self.config['save_dir'],'model_best.pth.tar')
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
        # Log first epoch before training to confirm same initialization
        self.logger.info('Epoch: [{} | {}] LR: {}'.format(self.start_epoch, self.epochs, self.optimizer.param_groups[0]['lr']))
        val_acc, val_loss, ema_val_acc = self.test_val()
        self.logger.wandb_log({'epoch': self.start_epoch, 'lr': self.optimizer.param_groups[0]['lr'], 'val_loss': val_loss, 'val_acc': val_acc, 
                                'ema_val_acc': ema_val_acc, 'best_val_acc': max(self.best_acc, val_acc),'total_step': self.total_step})

        for epoch in range(self.start_epoch+1, self.epochs+1):
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
        # online batch selection
        return inputs, targets, indexes
    
    def after_batch(self, i, inputs, targets, indexes, outputs):
        self.ema_net.update()


    def train(self, epoch, list_of_train_idx):
        # train for one epoch
        self.logger.info('Epoch: [{} | {}] LR: {}'.format(epoch, self.epochs, self.optimizer.param_groups[0]['lr']))

        total_batch = len(self.train_loader)
        epoch_begin_time = time.time()
        self.num_selected_noisy_indexes = 0
        all_preds = []
        all_labels = []
        epoch_loss = 0.0
        num_samples = 0
        
        # train
        for i, datas in enumerate(self.train_loader):
            self.model.train()
            batch_inputs = datas['input'].cuda()
            batch_targets = datas['target'].cuda()
            batch_indexes = datas['index']
            selected_inputs, selected_targets, selected_indexes = self.before_batch(i, batch_inputs, batch_targets, batch_indexes, epoch)
            selected_outputs, features = self.model(x=selected_inputs, need_features=self.need_features, targets=selected_targets) if self.need_features else (self.model(x=selected_inputs, need_features=False, targets=selected_targets), None)
            loss = self.criterion(selected_outputs, selected_targets)
            self.while_update(selected_outputs, loss, selected_targets, epoch, features, selected_indexes, batch_idx=i, batch_size=self.batch_size)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.after_batch(i,selected_inputs, selected_targets, selected_indexes, selected_outputs.detach())
            
            # record epoch statistics
            self.model.eval()
            self.num_selected_noisy_indexes = self.num_selected_noisy_indexes + np.intersect1d(selected_indexes, self.noisy_indices).size
            batch_outputs = self.model(x=batch_inputs, targets=batch_targets)
            train_loss = self.criterion(batch_outputs, batch_targets)
            preds = torch.argmax(batch_outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_targets.cpu().numpy())
            batch_size = batch_targets.size(0)
            epoch_loss += train_loss.item() * batch_size
            num_samples += batch_size
        
        # Log epoch statistics
        now = time.time()
        self.scheduler.step()
        self.total_step = self.total_step + total_batch
        train_loss = epoch_loss / num_samples
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        train_acc = np.mean(all_preds == all_labels)
        self.logger.info(f'=====> Epoch: {epoch}/{self.training_opt["num_epochs"]}, global_step: {self.total_step+i}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}, Train Loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')
        self.logger.info(f'=====> Time: {now - self.run_begin_time:.4f} s, Time this epoch: {now - epoch_begin_time:.4f} s, Total step: {self.total_step}')

        # test on validation data and log statistics for WandB
        val_acc, val_loss, ema_val_acc = self.test_val()
        # self.num_selected_noisy_indexes += np.intersect1d(indexes.cpu().numpy(), self.noisy_indices.cpu().numpy()).size
        self.logger.wandb_log({'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr'], 'val_loss': val_loss, 'val_acc': val_acc, 'train_loss': train_loss,
                                'train_acc': train_acc, 'ema_val_acc': ema_val_acc, 'best_val_acc': max(self.best_acc, val_acc), 'total_time': now - self.run_begin_time,
                                'total_step': self.total_step, 'time_epoch': now - epoch_begin_time, 'percent noisy points selected': self.num_selected_noisy_indexes / len(self.train_dset)})

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

    def test_train(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        all_preds = []
        all_labels = []
        epoch_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for i, datas in enumerate(self.train_loader):
                inputs = datas['input'].cuda()
                targets = datas['target'].cuda()
                outputs = model(inputs)
                train_loss = self.criterion(outputs, targets)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
                batch_size = targets.size(0)
                epoch_loss += train_loss.item() * batch_size
                num_samples += batch_size
        avg_train_loss = epoch_loss / num_samples
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        train_acc = np.mean(all_preds == all_labels)
        self.logger.info(f'=====> Training Accuracy: {train_acc:.4f}')
        return train_acc, avg_train_loss

    def test_val(self):
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
        all_preds =  np.concatenate(all_preds)
        all_ema_preds = np.concatenate(all_ema_preds)
        all_labels = np.concatenate(all_labels)
        acc = np.mean(all_preds == all_labels)
        ema_acc = np.mean(all_ema_preds == all_labels)
        self.logger.info(f'=====> Validation Accuracy: {acc:.4f}')
        self.logger.info(f'=====> EMA Validation Accuracy: {ema_acc:.4f}')

        return acc, avg_epoch_test_loss,ema_acc

