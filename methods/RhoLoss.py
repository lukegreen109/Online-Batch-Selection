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
from torch.utils.data import random_split, DataLoader

class RhoLoss(SelectionMethod):
    """A class for implementing the RhoLoss selection method, which selects samples based on reducible loss.

    This class inherits from `SelectionMethod` and uses an irreducible loss model (ILmodel) and a target model
    to compute reducible loss for sample selection during training. It supports various ratio scheduling strategies
    for dynamic sample selection and handles model training and loading for specific datasets.

    Args:
        config (dict): Configuration dictionary containing method and dataset parameters.
            Expected keys include:
                - 'method_opt': Dictionary with keys 'ratio', 'budget', 'epochs', 'ratio_scheduler',
                  'warmup_epochs', 'iter_selection', 'balance'.
                - 'rho_loss': Dictionary with key 'training_budget'.
                - 'dataset': Dictionary with keys 'name' and 'num_classes'.
                - 'networks': Dictionary with key 'params' containing 'm_type'.
        logger (logging.Logger): Logger instance for logging training and selection information.
    """
    method_name = 'RhoLoss'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.balance = config['method_opt']['balance']
        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config['method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0
        self.current_train_indices = np.arange(self.num_train_samples)
        self.reduce_dim = config['method_opt']['reduce_dim'] if 'reduce_dim' in config['method_opt'] else False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.holdout_model_type = config['rholoss']['networks']['type']
        self.holdout_model_args = config['rholoss']['networks']['params']
        self.holdout_epochs = config['rholoss']['holdout_num_epochs']
        # self.holdout_model = self.setup_holdout_model(config, logger)
        # Load checkpoint
        ckpt = torch.load(
            "/home/phancock/Online-Batch-Selection/RhoLoss_pretrained_holdout.ckpt",
            map_location=self.device,
            weights_only=False
        )
        state_dict = ckpt["state_dict"]

        # Remove "model." prefix from keys if present
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # Create same architecture and load weights
        self.holdout_model = getattr(models, self.holdout_model_type)(**self.holdout_model_args)
        self.holdout_model.load_state_dict(new_state_dict)
        self.holdout_model.to(self.device)



    def setup_holdout_model(self, config, logger):
        """"Retrieve the holdout model for computing irreducible loss.
        Args:
            config (dict): Configuration dictionary containing model parameters.
            logger (logging.Logger): Logger instance for logging information.
        """
        holdout_model_path = config['rholoss']['holdout_model_path']

        # Create model
        holdout_model = getattr(models, self.holdout_model_type)(**self.holdout_model_args)
        self.holdout_model = holdout_model
        self.holdout_model.to(self.device)

        if holdout_model_path == 'None':
            best_model = self.train_holdout_model(config, logger)

            return best_model
        else:
            try:
                logger.info(f"Loading holdout model from {holdout_model_path}")
                holdout_model.load_state_dict(torch.load(holdout_model_path, map_location=self.device))
                return holdout_model
            except FileNotFoundError:
                raise ValueError("Invalid holdout_model configuration.")

    def train_holdout_model(self, config, logger):
        """
        Train the holdout model for estimating irreducible loss.

        Args:
            config (dict): Configuration settings for training.
            logger (Logger): Logging utility.
            holdout_dataloader (DataLoader): DataLoader for holdout set.
            holdout_epochs (int): Number of epochs to train the holdout model.
        """

        holdout_optimizer = create_holdout_optimizer(self.holdout_model, config)
        holdout_criterion = create_holdout_criterion(config, logger)

        total_batch = len(self.holdout_dataloader)

        logger.info(f"[Holdout Model] Starting training for {self.holdout_epochs} epochs on {total_batch} batches per epoch.")

        self.best_loss = float('inf')
        self.best_model_path = None

        self.holdout_model.train()
        for epoch in range(self.holdout_epochs):
            epoch_loss = 0.0
            for i, datas in enumerate(self.holdout_dataloader):
                inputs = datas['input'].to(self.device)
                targets = datas['target'].to(self.device)
                
                with torch.set_grad_enabled(True):
                    outputs = self.holdout_model(inputs)
                    holdout_loss = holdout_criterion(outputs, targets)
                    holdout_optimizer.zero_grad()
                    holdout_loss.backward()
                    holdout_optimizer.step()

                batch_loss = holdout_loss.item()
                epoch_loss += batch_loss

                if (i + 1) % 50 == 0 or (i + 1) == total_batch:
                    logger.info(f"[Holdout Model][Epoch {epoch+1}/{self.holdout_epochs}] "
                                f"Batch {i+1}/{total_batch}, Batch Loss: {batch_loss:.4f}")

            avg_epoch_loss = epoch_loss / total_batch
            logger.info(f"[Holdout Model][Epoch {epoch+1}/{self.holdout_epochs}] "
                        f"Average Loss: {avg_epoch_loss:.4f}")
            
            # Track lowest loss and save best model path
            # Evaluate on validation set if available
            self.holdout_model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_datas in self.test_loader:
                    val_inputs = val_datas['input'].to(self.device)
                    val_targets = val_datas['target'].to(self.device)
                    outputs = self.holdout_model(val_inputs)
                    loss = holdout_criterion(outputs, val_targets)
                    val_loss += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / max(1, val_batches)
            logger.info(f"[Holdout Model][Epoch {epoch+1}/{self.holdout_epochs}] Validation Loss: {avg_val_loss:.4f}")

            # Save best model if average loss improves
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                best_model_path = os.path.join(self.config['output_dir'], f'best_holdout.pth.tar')
                torch.save(self.holdout_model.state_dict(), best_model_path)
                self.best_model_path = best_model_path
                logger.info(f"Best holdout model updated at epoch {epoch+1}")
        
        # Save model state with least validation loss
        best_model_state = torch.load(self.best_model_path, map_location=self.device)
        
        # Create a new model instance
        holdout_model = getattr(models, self.holdout_model_type)(**self.holdout_model_args)

        # Load the weights into the new model
        holdout_model.load_state_dict(best_model_state)

        # Freeze parameters
        for param in holdout_model.parameters():
            param.requires_grad = False
        holdout_model.to(self.device)
        holdout_model.eval()
        return holdout_model

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

    def reducible_loss_selection(self, inputs, targets, selected_num_samples):
        """Select sub-batch with highest reducible loss.
        Args:
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
        Returns:
            torch.Tensor: Indices of the selected samples.
        """
        # Get total loss from main model
        self.model.eval()
        with torch.no_grad():
            total_loss = F.cross_entropy(self.model(inputs), targets, reduction='none')
        # Get irreducible loss from holdout model
        self.holdout_model.eval()
        with torch.no_grad():
            irreducible_loss = F.cross_entropy(self.holdout_model(inputs), targets, reduction='none')
        reducible_loss = total_loss - irreducible_loss

        # Select samples with highest reducible loss
        _, indices = torch.topk(reducible_loss, selected_num_samples)

        return indices

    def before_batch(self, i, inputs, targets, indexes, epoch):
        """Prepare the batch for training by selecting samples based on reducible loss.
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
        indices = self.reducible_loss_selection(inputs, targets, selected_num_samples)
        inputs = inputs[indices]
        targets = targets[indices]
        indices = indices.to(indexes.device)
        indexes = indexes[indices]
        return inputs, targets, indexes