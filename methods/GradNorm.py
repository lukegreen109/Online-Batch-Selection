from methods.SelectionMethod import SelectionMethod
import torch
import numpy as np
class GradNorm(SelectionMethod):
    method_name = "GradNorm"
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.selection_method = 'GradNorm'
        self.balance = config['method_opt']['balance']
        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config['method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0
        self.reduce_dim = config['method_opt']['reduce_dim'] if 'reduce_dim' in config['method_opt'] else False 
    
    def get_ratio_per_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            self.logger.info('warming up')
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

    def calc_grad(self, inputs, targets, indexes):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model.eval()
        outputs, features = model.feat_nograd_forward(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        with torch.no_grad():
            grad_out = torch.autograd.grad(loss, outputs, retain_graph=True)[0] 
            grad = grad_out.unsqueeze(-1) * features.unsqueeze(1)
            grad = grad.view(grad.shape[0], -1)
        model.train()
        if self.reduce_dim:
            dim = grad.shape[1]
            dim_reduced = dim // self.reduce_dim
            index = np.random.choice(dim, dim_reduced, replace=False)
            grad = grad[:, index]
        grad_mean = grad.mean(dim=0)
        return grad_mean, grad
    
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
        grad_mean, grad = self.calc_grad(inputs, targets, indexes)
        grad_norm = torch.norm(grad, dim=1)
        k = int(ratio * len(inputs))
        k = max(1, min(k, len(inputs)))  # at least 1, at most batch size
        _, indices = torch.topk(grad_norm, k)
        indices = indices.cpu()
        inputs = inputs[indices]
        targets = targets[indices]
        indexes = indexes[indices]

        return inputs, targets, indexes
    