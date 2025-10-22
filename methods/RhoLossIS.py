from methods.SelectionMethod import SelectionMethod
from methods.method_utils.optimizer import *
from methods.method_utils.loss import *
import models
import data
import torch
import numpy as np
import torch
import os
import torch.nn.functional as F
import torch.serialization


class RhoLossIS(SelectionMethod):
    """A class for implementing the RhoLoss selection method, which selects samples based on reducible loss.

    This class inherits from `SelectionMethod` and uses an irreducible loss model (ILmodel) and a target model
    to compute reducible loss for sample selection during training. It supports various ratio scheduling strategies
    for dynamic sample selection and handles model training and loading for specific datasets.

    Args:
        config (dict): Configuration dictionary containing method and dataset parameters.
            Expected keys include:
                - 'method_opt': Dictionary with keys 'ratio', 'balance', 'ratio_scheduler', 'warmup_epochs'.
                - 'rho_loss': Dictionary with keys for the irreducible holdout model.
                - 'dataset': Dictionary with keys 'name' and 'root'.
                - 'networks': Dictionary with key 'type' and 'params' containing 'm_type' and 'num_classes'.
        logger (logging.Logger): Logger instance for logging training and selection information.
    """
    method_name = 'RhoLossIS'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.balance = config['method_opt']['balance']
        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config['method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0
        self.temperature = config['method_opt']['temperature'] if 'temperature' in config['method_opt'] else 0
        self.current_train_indices = np.arange(self.num_train_samples)
        self.reduce_dim = config['method_opt']['reduce_dim'] if 'reduce_dim' in config['method_opt'] else False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_holdout_model(config, logger)
        
        self.precompute_losses()

        self.warming_up = False

    def precompute_losses(self):
        """Precompute irreducible losses for the training dataset using the holdout model."""
        self.holdout_model.eval()
        losses_tensor = torch.zeros(self.original_train_dset.__len__())

        with torch.no_grad():
            for datas in self.train_dataloader_unaugmented:
                inputs = datas['input'].to(self.device)
                targets = datas['target'].to(self.device)
                indexes = datas['index']
                outputs = self.holdout_model(inputs)
                loss = F.cross_entropy(outputs, targets, reduction='none')
                losses_tensor[indexes] = loss.cpu()

        # Attach the losses tensor directly to the dataset object
        self.original_train_dset.irreducible_loss_cache = losses_tensor
        self.logger.info(f"Cached irreducible losses for {len(losses_tensor)} samples in dataset.")


    def setup_holdout_model(self, config, logger):
        """"
        Retrieve the holdout model for computing irreducible loss.
        Args:
            config (dict): Configuration dictionary containing model parameters.
            logger (logging.Logger): Logger instance for logging information.
        """
        holdout_model_path = config['rholoss']['holdout_model_path']
        model_type = config['rholoss']['networks']['type']
        model_args = config['rholoss']['networks']['params']

        # Create model
        self.holdout_model = getattr(models, model_type)(**model_args)
        self.holdout_model.to(self.device)

        if holdout_model_path == 'None':
            self.train_holdout_model(config, logger)
        else:
            try:
                logger.info(f"Loading holdout model from {holdout_model_path}")
                self.holdout_model.load_state_dict(torch.load(holdout_model_path, map_location=self.device))
            except FileNotFoundError:
                raise ValueError("Invalid holdout_model configuration.")

    def train_holdout_model(self, config, logger):
        """
        Train the holdout model for estimating irreducible loss.
        Args:
            config (dict): Configuration settings for training.
            logger (Logger): Logging utility.
        """
        
        optimizer = create_optimizer(self.holdout_model, config)
        criterion = create_holdout_criterion(config, logger)

        epochs = config['rholoss']['holdout_num_epochs']
        total_batch = len(self.holdout_dataloader_augmented)

        logger.info(f"[Holdout Model] Starting training for {epochs} epochs on {total_batch} batches per epoch.")

        best_loss = float('inf')
        best_model_path = os.path.join(self.config['output_dir'], f'best_holdout.pth.tar')

        self.holdout_model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, datas in enumerate(self.holdout_dataloader_augmented):
                inputs = datas['input'].to(self.device)
                targets = datas['target'].to(self.device)
                
                with torch.set_grad_enabled(True):
                    outputs = self.holdout_model(inputs)
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss

                if (i + 1) % 50 == 0 or (i + 1) == total_batch:
                    logger.info(f"[Holdout Model][Epoch {epoch+1}/{epochs}] "
                                f"Batch {i+1}/{total_batch}, Batch Loss: {batch_loss:.4f}")

            avg_epoch_loss = epoch_loss / total_batch
            logger.info(f"[Holdout Model][Epoch {epoch+1}/{epochs}] "
                        f"Average Loss: {avg_epoch_loss:.4f}")
            
            # Track lowest loss and save best model path
            # Evaluate on validation set if available
            self.holdout_model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for val_datas in self.train_dataloader_unaugmented:
                    val_inputs = val_datas['input'].to(self.device)
                    val_targets = val_datas['target'].to(self.device)
                    outputs = self.holdout_model(val_inputs)
                    loss = criterion(outputs, val_targets)
                    val_loss += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / max(1, val_batches)
            logger.info(f"[Holdout Model][Epoch {epoch+1}/{epochs}] Validation Loss: {avg_val_loss:.4f}")

            # Save best model if average validation loss improves
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(self.holdout_model.state_dict(), best_model_path)
                logger.info(f"Best holdout model updated at epoch {epoch+1}")
        
        # Save model state with least validation loss
        best_model_state = torch.load(best_model_path, map_location=self.device)

        self.holdout_model.load_state_dict(best_model_state)

        # Freeze parameters
        for param in self.holdout_model.parameters():
            param.requires_grad = False

        self.holdout_model.eval()
    
    def get_ratio_per_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            self.logger.info('warming up')
            self.warming_up = True
            return 0.1
        self.warming_up = False
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

    def reducible_loss_selection(self, inputs, targets, indexes, selected_num_samples):
        """
        Select sub-batch with highest reducible loss.
        Args:
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
            indexes (torch.Tensor): Indices of the samples in the current batch.
            selected_num_samples (int): Number of samples to select based on reducible loss.
        Returns:
            torch.Tensor: Indices of the selected samples.
        """
        # Set models to eval mode
        self.model.eval()
        self.holdout_model.eval()

        # Get total loss from main model
        with torch.no_grad():
            total_loss = F.cross_entropy(self.model(inputs), targets, reduction='none')

        # Get irreducible loss from holdout model
        irreducible_loss = self.original_train_dset.irreducible_loss_cache[indexes].to(total_loss.device)

        # Compute reducible loss
        reducible_loss = total_loss - irreducible_loss

        # Method 1 - Create ranking of reducible losses and create softmax rankings with temperature
        sorted_reducible_loss, ranking = torch.sort(reducible_loss, descending=True)
        temperature = self.temperature
        adjusted_ranking = F.softmax(temperature * sorted_reducible_loss, dim=0)
        # adjusted_ranking = torch.exp(temperature * (sorted_reducible_loss-sorted_reducible_loss.max())) / sum(torch.exp(temperature * (sorted_reducible_loss-sorted_reducible_loss.max())))
 
        # Method 2 - Use polynomial decay on reducible loss to create probabilities
        # adjusted_ranking = sorted_reducible_loss.pow(temperature) / torch.sum(sorted_reducible_loss.pow(temperature))

        # Method 3 - Use the rankings directly as probabilities
        # adjusted_ranking = (len(ranking) - torch.arange(len(rank  ing)).float().to(ranking.device)) / torch.sum(len(ranking) - torch.arange(len(ranking)).float().to(ranking.device))
        # Sample without replacement based on probability density
        
        indices = np.random.choice(ranking.cpu().numpy(), size=selected_num_samples, replace=False, p=adjusted_ranking.cpu().numpy())
        # Uniform selection during warmup
        if self.warming_up:
            self.logger.info('warming up')
            indices = np.random.choice(len(inputs), size=(selected_num_samples)*self.ratio, replace=False)
        
        # Return to train mode and return selected indices
        self.model.train()
        return indices

    def before_batch(self, i, inputs, targets, indexes, epoch):
        """
        Prepare the batch for training by selecting samples based on reducible loss.
        Args:
            i (int): Current batch index.
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
            indexes (torch.Tensor): Indices of the samples in the current batch.
            epoch (int): Current epoch number.
        Returns:
            tuple: Selected inputs, targets, and indexes for the current batch.
        """
        # Get the ratio for the current epoch
        # print(f"Indexes: {indexes}")
        ratio = self.get_ratio_per_epoch(epoch)
        if ratio == 1.0:
            if i == 0:
                self.logger.info('using all samples')
            return super().before_batch(i, inputs, targets, indexes, epoch)
        else:
            if i == 0:
                self.logger.info(f'balance: {self.balance}')
                self.logger.info('selecting samples for epoch {}, ratio {}'.format(epoch, ratio))

        # Get indices based on reducible loss
        selected_num_samples = max(1, int(inputs.shape[0] * ratio))

        # Uniform selection during warmup
        if self.warming_up:
            self.logger.info('warming up')
            indices = np.random.choice(len(inputs), size=(selected_num_samples), replace=False)
        else:
            indices = self.reducible_loss_selection(inputs, targets, indexes, selected_num_samples)

        inputs = inputs[indices]
        targets = targets[indices]
        indexes = indexes[indices]
        return inputs, targets, indexes