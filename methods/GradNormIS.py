from methods.SelectionMethod import SelectionMethod
import torch
import time
import numpy as np
class GradNormIS(SelectionMethod):
    method_name = "GradNormIS"
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.selection_method = 'GradNormIS'
        self.prev_params = [p.data.clone() for p in self.model.parameters()]
        self.probs = None
        self.tau = 1   # Predetermined threshold that is updated
        self.a_tau = config["a_tau"] # Parameter that is "moving" similar to EMA momentum
        self.current_grad_norm = None
        self.current_selected_probs = None
        self.balance = config['method_opt']['balance']
        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config['method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0
        self.current_train_indices = np.arange(self.num_train_samples)
        self.reduce_dim = config['method_opt']['reduce_dim'] if 'reduce_dim' in config['method_opt'] else False 
    
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
        self.num_selected_noisy_indexes = 0
        # train
        for i, datas in enumerate(train_loader):
            inputs = datas['input'].cuda()
            targets = datas['target'].cuda()
            indexes = datas['index']
            B = self.batch_size
            b = int(self.batch_size * self.ratio)
            tau_th = (B + 3 * b) / (3*b)
            if self.tau > tau_th:
                # Importance Sampling
                inputs, targets, indexes, grad, grad_norm = self.before_batch(i, inputs, targets, indexes, epoch)
                outputs, features = self.model(x=inputs, need_features=self.need_features, targets=targets) if self.need_features else (self.model(x=inputs, need_features=False, targets=targets), None)
                loss = self.criterion(outputs, targets)
                self.while_update(outputs, loss, targets, epoch, features, indexes, batch_idx=i, batch_size=self.batch_size)
                weighted_loss = self.while_update(outputs, loss, targets, epoch, features, indexes, batch_idx=i, batch_size=self.batch_size)
                self.optimizer.zero_grad()
                #loss.backward()
                weighted_loss.backward()
                self.optimizer.step()
                self.logger.wandb_log({"sampling": 1, "epoch": epoch})
            else:
                # Uniform Sampling
                inputs, targets, indexes = self.uniform_sample(i, inputs, targets, indexes, epoch)
                outputs, features = self.model(x=inputs, need_features=self.need_features, targets=targets) if self.need_features else (self.model(x=inputs, need_features=False, targets=targets), None)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                grad_mean, grad = self.calc_grad(inputs, targets, indexes)
                grad_norm = torch.norm(grad, dim=1)
                #grad_norm = grad_norm ** 3
                self.logger.wandb_log({"sampling": 0, "epoch": epoch})
            outputs, features = self.model(x=inputs, need_features=self.need_features, targets=targets) if self.need_features else (self.model(x=inputs, need_features=False, targets=targets), None)
            #loss = self.criterion(outputs, targets)
            # I changed this line from previous including the "loss =" before the whileupdate. 
            #if self.selection_method == "GradNormIS":
                #loss = self.while_update(outputs, loss, targets, epoch, features, indexes, batch_idx=i, batch_size=self.batch_size)
            #else:
                #self.while_update(outputs, loss, targets, epoch, features, indexes, batch_idx=i, batch_size=self.batch_size)
            #self.optimizer.zero_grad()
            #loss.backward()
            total_norm = 0.0
            #for p in self.model.parameters():
                #if p.grad is not None:
                    #param_norm = p.grad.data.norm(2)   # L2 norm of each param’s grad
                    #total_norm += param_norm.item() ** 2
            #total_norm = total_norm ** 0.5
            #self.logger.info(f"Gradient Norm: {total_norm:.6f}")
            #self.logger.wandb_log({"grad_norm": total_norm, "epoch": epoch})
            #self.optimizer.step()
            uniform = torch.full_like(grad_norm, 1 / self.batch_size)
            # Update tau threshold to determine if we should IS or Uniform select
            self.tau = self.a_tau * self.tau + (1 - self.a_tau) * ((1 - (1 / (grad_norm ** 2).sum()) * torch.norm((grad_norm - uniform)) ** 2) ** -1/2)
            self.logger.wandb_log({"tau": self.tau, "tauth": tau_th})
            self.after_batch(i,inputs, targets, indexes,outputs.detach())
            self.num_selected_noisy_indexes += int(len(np.intersect1d(indexes.cpu().numpy(), self.noisy_indices.cpu().numpy())))
            if i % self.config['logger_opt']['print_iter'] == 0:
                # train acc
                _, predicted = torch.max(outputs.data, 1)
                total = targets.size(0)
                correct = (predicted == targets).sum().item()
                train_acc = correct / total
                self.logger.info(f'Epoch: {epoch}/{self.training_opt["num_epochs"]}, Iter: {i}/{total_batch}, global_step: {self.total_step+i}, Loss: {loss.item():.4f}, Train acc: {train_acc:.4f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}')
                    
        self.scheduler.step()
        self.total_step = self.total_step + total_batch
        # test
        now = time.time()
        self.logger.wandb_log({'loss': loss.item(), 'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr'], self.training_opt['loss_type']: loss.item()})
        val_acc, ema_val_acc = self.test()
        self.logger.wandb_log({'percent noisy points selected': self.num_selected_noisy_indexes / int(len(self.train_dset)), 'val_acc': val_acc, 'ema_val_acc': ema_val_acc, 'epoch': epoch, 'total_time': now - self.run_begin_time, 'total_step': self.total_step, 'time_epoch': now - epoch_begin_time, 'best_val_acc': max(self.best_acc, val_acc)})
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
        # Compute how far the parameters moved this epoch
        param_diff_norm = 0.0
        for p, prev_p in zip(self.model.parameters(), self.prev_params):
            diff = torch.norm(p.data - prev_p).item()
            param_diff_norm += diff ** 2

        param_diff_norm = param_diff_norm ** 0.5
        self.logger.info(f"Parameter delta (epoch {epoch}): {param_diff_norm:.6f}")
        self.logger.wandb_log({"param_delta": param_diff_norm, "epoch": epoch})
        # Update stored parameters for next epoch
        self.prev_params = [p.data.clone() for p in self.model.parameters()]


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
        #grad_norm = grad_norm ** 3
        batch_probs = grad_norm / (grad_norm.sum() + 1e-12)
        batch_probs = batch_probs.detach().cpu().numpy()  # only use probabilities for this batch
        mini_batch_size = max(1,int(ratio * len(inputs)))
        indices = self.sample_batch_indices(batch_probs, mini_batch_size, epoch)
        indices = torch.tensor(indices, device=inputs.device, dtype=torch.long)
        selected_probs = torch.as_tensor(batch_probs[indices.cpu().numpy()], dtype=torch.float32, device=inputs.device)
        self.current_selected_probs = torch.clamp(selected_probs, min=1e-8)
        self.current_selected_grads = torch.as_tensor(grad_norm[indices.cpu().numpy()], dtype=torch.float32, device=inputs.device)
        if isinstance(indexes, np.ndarray):
            indexes = torch.tensor(indexes, device=inputs.device, dtype=torch.long)
        else:
            indexes = indexes.to(inputs.device)

        inputs = inputs[indices]
        targets = targets[indices]
        indexes = indexes[indices]

        return inputs, targets, indexes, grad, grad_norm
    
    def sample_batch_indices(self, importance_scores, batch_size, epoch):
        """
        importance_scores: numpy array summing to 1 over the candidate set (here candidates are local batch elements)
        batch_size: int
        returns numpy array indices in [0, len(importance_scores)-1]
        """
        batch_size = int(batch_size)
        # safety: when batch_size equals len(importance_scores), just return all indices
        n = len(importance_scores)
        if batch_size >= n:
            return np.arange(n, dtype=np.int64)
        idx = np.random.choice(n, size=batch_size, replace=False, p=importance_scores)
        selected_probs = importance_scores[idx]
        min_p = selected_probs.min()
        mean_p = selected_probs.mean()
        max_p = selected_probs.max()
        self.logger.wandb_log({"Min Prob Selected": min_p, "Mean Prob Selected": mean_p, "Max Prob Selected": max_p, "Epoch": epoch})

        # Find when low prob are chosen.
        return idx

    def while_update(self, outputs, loss, targets, epoch, features, indexes, batch_idx, batch_size):
        p_i = self.current_selected_grads
        weights = 1.0 / (batch_size * p_i)
        #weights = weights.to(loss.device, dtype=loss.dtype)
        # Lets add some bias in rewieghting to weaken variance
        #weights = weights ** (2)
        #weights = weights / (weights * p_i).sum() * (batch_size * self.ratio)
        weighted_loss = (loss * weights).sum()
        return weighted_loss

    def uniform_sample(self, i, inputs, targets, indexes, epoch):
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
