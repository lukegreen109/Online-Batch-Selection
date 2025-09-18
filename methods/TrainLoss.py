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
        elif self.ratio_scheduler == 'constant':
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