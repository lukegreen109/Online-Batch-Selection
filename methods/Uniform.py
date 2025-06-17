from .SelectionMethod import SelectionMethod
import numpy as np

class Uniform(SelectionMethod):
    method_name = 'Uniform'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.balance = config['method_opt']['balance']
        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config['method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0
        self.current_train_indices = np.arange(self.num_train_samples)
    
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

        
    def before_batch(self, i, inputs, targets, indexes, epoch):
        if self.balance:
            ratio = self.get_ratio_per_epoch(epoch)
            if i == 0:
                self.logger.info(f'selecting samples for epoch {epoch}')
                self.logger.info(f'balance: {self.balance}')
                self.logger.info(f'ratio: {ratio}')
            all_indices = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                indices = np.where(targets == c)[0]
                num_samples = int(len(indices) * ratio)
                selected_indices = np.random.choice(indices, num_samples, replace=False)
                all_indices = np.append(all_indices, selected_indices)
            return inputs[all_indices], targets[all_indices], indexes[all_indices]
        else:
            ratio = self.get_ratio_per_epoch(epoch)
            if i == 0:
                self.logger.info(f'selecting samples for epoch {epoch}')
                self.logger.info(f'balance: {self.balance}')
                self.logger.info(f'ratio: {ratio}')
            num_samples = int(inputs.shape[0] * ratio)
            selected_indices = np.random.choice(np.arange(inputs.shape[0]), num_samples, replace=False)
            return inputs[selected_indices], targets[selected_indices], indexes[selected_indices]
        