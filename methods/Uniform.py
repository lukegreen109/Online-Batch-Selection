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
        
    def before_batch(self, i, inputs, targets, indexes):
        ratio = self.ratio
        if self.balance:
            if i == 0:
                self.logger.info(f'selecting samples for epoch {self.epoch}')
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
            if i == 0:
                self.logger.info(f'selecting samples for epoch {self.epoch}')
                self.logger.info(f'balance: {self.balance}')
                self.logger.info(f'ratio: {ratio}')
            num_samples = int(inputs.shape[0] * ratio)
            selected_indices = np.random.choice(np.arange(inputs.shape[0]), num_samples, replace=False)
            return inputs[selected_indices], targets[selected_indices], indexes[selected_indices]
        