from .SelectionMethod import SelectionMethod
import numpy as np
import torch
import torch.nn.functional as F
import time

class TrainLoss(SelectionMethod):
    method_name = 'TrainLoss'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.balance = config['method_opt']['balance']
        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config['method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0
        
    def get_ratio_per_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            return 1.0
        if self.ratio_scheduler == 'constant':
            return self.ratio
        elif self.ratio_scheduler == 'increase_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'decrease_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'increase_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        elif self.ratio_scheduler == 'decrease_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        else:
            raise NotImplementedError
        
    # Method 1
    # Overview: Override the while_update function to reevaluate the losses and only get the losses of the most important points
    # Potential Errors: Would require modification to train function 

    # def while_update(self, outputs, loss, targets, epoch, features, indexes, batch_idx, batch_size):
    #     ratio = self.get_ratio_per_epoch(epoch)
    #     selected_num_samples = int(targets.shape[0] * ratio)
    #     losses = F.cross_entropy(outputs, targets, reduction = 'none')
    #     _, top_indices = torch.topk(losses, selected_num_samples, largest = True)
    #     select_losses = losses[top_indices].mean()
    #     return select_losses
    
    # Method 2
    # Overview: Override the before_batch function to calculate losses and return the incides that correspond
    # Potential Errors: Might be calculating gradiants twice (once here and once in train function)
        
    def selection(self, inputs, targets, selected_num_samples):
        with torch.no_grad():
            outputs = self.model(inputs)
            losses = F.cross_entropy(outputs, targets, reduction = 'none')
        _, indices = torch.sort(losses, descending = True)
        return indices[:selected_num_samples]
        
    def before_batch(self, i, inputs, targets, indexes, epoch):
        ratio = self.get_ratio_per_epoch(epoch)
        if ratio == 1.0:
            if i == 0:
                self.logger.info('using all samples')
            return super().before_batch(i, inputs, targets, indexes, epoch)
        else:
            if i == 0:
                self.logger.info(f'balance: {self.balance}')
                self.logger.info('selecting samples for epoch {}, ratio {}'.format(epoch, ratio))
        selected_num_samples = int(inputs.shape[0] * ratio)
        indices = self.selection(inputs, targets, selected_num_samples)
        inputs = inputs[indices]
        targets = targets[indices]
        indices = indices.to(indexes.device)
        indexes = indexes[indices]
        return inputs, targets, indexes
    

    # Method 3
    # Overview: Override the whole train function to include high loss selection
    # Potential Errors: We do not want to have to override this whole function if we can help it

    # def train(self, epoch, list_of_train_idx):
    #     # Set model to training mode
    #     self.model.train()
    #     self.logger.info(
    #         f'Epoch: [{epoch} | {self.epochs}] LR: {self.optimizer.param_groups[0]["lr"]}'
    #     )

    #     # Shuffle training indices and create batch sampler
    #     list_of_train_idx = np.random.permutation(list_of_train_idx)
    #     batch_sampler = torch.utils.data.BatchSampler(
    #         list_of_train_idx, batch_size=self.batch_size, drop_last=False
    #     )

    #     train_loader = torch.utils.data.DataLoader(
    #         self.train_dset, 
    #         num_workers=self.num_data_workers, 
    #         pin_memory=True, 
    #         batch_sampler=batch_sampler
    #     )

    #     total_batch = len(train_loader)
    #     epoch_begin_time = time.time()

    #     for i, datas in enumerate(train_loader):
    #         # Move data to GPU
    #         inputs = datas['input'].cuda()
    #         targets = datas['target'].cuda()
    #         indexes = datas['index']

    #         # Pre-batch processing (optional)
    #         inputs, targets, indexes = self.before_batch(i, inputs, targets, indexes, epoch)

    #         # Forward pass
    #         outputs, features = (
    #             self.model(x=inputs, need_features=self.need_features, targets=targets)
    #             if self.need_features else
    #             (self.model(x=inputs, need_features=False, targets=targets), None)
    #         )

    #         # ----------------------------
    #         # TOP-10% LOSS SELECTION
    #         # ----------------------------
    #         per_sample_losses = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
    #         selected_num_samples = max(1, int(outputs.shape[0] * 0.1))
    #         _, top_indices = torch.sort(per_sample_losses, descending=True)
    #         top_indices = top_indices[:selected_num_samples]

    #         # Use only top-loss samples for backward
    #         loss = per_sample_losses[top_indices].mean()
    #         # ----------------------------

    #         # Hook for logging, metrics, or extra updates
    #         self.while_update(outputs, loss, targets, epoch, features, indexes, batch_idx=i, batch_size=self.batch_size)

    #         # Backpropagation and optimizer step
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

    #         # Post-batch hook
    #         self.after_batch(i, inputs, targets, indexes, outputs.detach())

    #         # Logging
    #         if i % self.config['logger_opt']['print_iter'] == 0:
    #             _, predicted = torch.max(outputs.data, 1)
    #             train_acc = (predicted == targets).sum().item() / targets.size(0)
    #             self.logger.info(
    #                 f'Epoch: {epoch}/{self.training_opt["num_epochs"]}, '
    #                 f'Iter: {i}/{total_batch}, '
    #                 f'global_step: {self.total_step + i}, '
    #                 f'Loss: {loss.item():.4f}, '
    #                 f'Train acc: {train_acc:.4f}, '
    #                 f'lr: {self.optimizer.param_groups[0]["lr"]:.6f}'
    #             )

    #     # Scheduler step
    #     self.scheduler.step()
    #     self.total_step += total_batch

    #     # Validation & checkpointing
    #     now = time.time()
    #     self.logger.wandb_log({
    #         'loss': loss.item(),
    #         'epoch': epoch,
    #         'lr': self.optimizer.param_groups[0]['lr'],
    #         self.training_opt['loss_type']: loss.item()
    #     })
    #     val_acc, ema_val_acc = self.test()
    #     self.logger.wandb_log({
    #         'val_acc': val_acc,
    #         'ema_val_acc': ema_val_acc,
    #         'epoch': epoch,
    #         'total_time': now - self.run_begin_time,
    #         'total_step': self.total_step,
    #         'time_epoch': now - epoch_begin_time,
    #         'best_val_acc': max(self.best_acc, val_acc)
    #     })

    #     self.logger.info(
    #         f'=====> Time: {now - self.run_begin_time:.4f} s, '
    #         f'Time this epoch: {now - epoch_begin_time:.4f} s, '
    #         f'Total step: {self.total_step}'
    #     )

    #     # Save model checkpoint
    #     self.logger.info('=====> Save model')
    #     is_best = val_acc > self.best_acc
    #     if is_best:
    #         self.best_epoch = epoch
    #         self.best_acc = val_acc

    #     checkpoint = {
    #         'epoch': epoch,
    #         'state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
    #         'optimizer': self.optimizer.state_dict(),
    #         'scheduler': self.scheduler.state_dict(),
    #         'best_acc': self.best_acc,
    #         'best_epoch': self.best_epoch
    #     }
    #     self.save_checkpoint(checkpoint, is_best)

    #     self.logger.info(f'=====> Epoch: {epoch}/{self.epochs}, Best val acc: {self.best_acc:.4f}, Current val acc: {val_acc:.4f}')
    #     self.logger.info(f'=====> Best epoch: {self.best_epoch}')
