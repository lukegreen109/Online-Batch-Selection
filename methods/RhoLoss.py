from methods.SelectionMethod import SelectionMethod
from methods.method_utils.optimizer import *
from methods.method_utils.loss import *
import models
import data
import torch
import numpy as np
import torch
import copy
import torch.nn.functional as F
import torch.nn as nn
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

        # data
        self.data_info = getattr(data, config['dataset']['name'])(config, logger)
        self.num_classes = self.data_info['num_classes']
        self.train_dset = self.data_info['train_dset']
        self.test_loader = self.data_info['test_loader']
        self.num_train_samples = self.data_info['num_train_samples']
        self.holdout_model = self.setup_holdout_model(config, logger)

    def setup_holdout_model(self, config, logger):
        """"Retrieve the holdout model for computing irreducible loss.
        Args:
            config (dict): Configuration dictionary containing model parameters.
            logger (logging.Logger): Logger instance for logging information.
        """
        holdout_model_path = config['rholoss']['holdout_model_path']

        # Create model
        holdout_model_type = config['rholoss']['networks']['type']
        holdout_model_args = config['rholoss']['networks']['params']
        holdout_model = getattr(models, holdout_model_type)(**holdout_model_args)
        holdout_model.to(self.device)

        if holdout_model_path == 'None':
            # Set up holdout model
            holdout_percentage = config['rholoss']['holdout_percentage']
            holdout_batch_size = config['rholoss']['holdout_batch_size']

            # Split data into main model and holdout model
            holdout_len = int(self.num_train_samples * holdout_percentage)
            main_len = int(self.num_train_samples - holdout_len)
            main_model_dataset, holdout_dataset = random_split(self.train_dset, [main_len, holdout_len])

            self.train_dset = main_model_dataset
            self.num_train_samples = main_len
            # Create DataLoader
            holdout_dataloader = DataLoader(holdout_dataset, batch_size=holdout_batch_size, shuffle=False)

            # Train holdout model
            holdout_epochs = config['rholoss']['holdout_num_epochs']
            self.train_holdout_model(config, logger, holdout_model, holdout_dataloader, holdout_epochs)

            return holdout_model
        else:
            try:
                logger.info(f"Loading holdout model from {holdout_model_path}")
                holdout_model.load_state_dict(torch.load(holdout_model_path, map_location=self.device))
                return holdout_model
            except FileNotFoundError:
                raise ValueError("Invalid holdout_model configuration.")

    def train_holdout_model(self, config, logger, holdout_model, holdout_dataloader, holdout_epochs):
        """
        Train the holdout model for estimating irreducible loss.

        Args:
            config (dict): Configuration settings for training.
            logger (Logger): Logging utility.
            holdout_dataloader (DataLoader): DataLoader for holdout set.
            holdout_epochs (int): Number of epochs to train the holdout model.
        """

        holdout_optimizer = create_holdout_optimizer(holdout_model, config)
        holdout_criterion = create_holdout_criterion(config, logger)

        total_batch = len(holdout_dataloader)

        logger.info(f"[Holdout Model] Starting training for {holdout_epochs} epochs on {total_batch} batches per epoch.")

        holdout_model.train()
        for epoch in range(holdout_epochs):
            epoch_loss = 0.0
            for i, datas in enumerate(holdout_dataloader):
                inputs = datas['input'].to(self.device)
                targets = datas['target'].to(self.device)
                indexes = datas['index'].to(self.device)

                with torch.set_grad_enabled(True):
                    holdout_optimizer.zero_grad()
                    outputs = holdout_model(inputs)
                    holdout_loss = holdout_criterion(outputs, targets)
                    holdout_loss.backward()
                    holdout_optimizer.step()

                batch_loss = holdout_loss.item()
                epoch_loss += batch_loss

                if (i + 1) % 50 == 0 or (i + 1) == total_batch:
                    logger.info(f"[Holdout Model][Epoch {epoch+1}/{holdout_epochs}] "
                                f"Batch {i+1}/{total_batch}, Batch Loss: {batch_loss:.4f}")

            avg_epoch_loss = epoch_loss / total_batch
            logger.info(f"[Holdout Model][Epoch {epoch+1}/{holdout_epochs}] "
                        f"Average Loss: {avg_epoch_loss:.4f}")

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

    def get_irreducible_loss(self, inputs, targets):
        """Compute the irreducible loss for the current model.
        Returns:
            torch.Tensor: The computed irreducible loss.
        """
        # Ensure holdout model is in evaluation mode for inference
        self.holdout_model.eval()
        with torch.no_grad():
            irreducible_loss = F.cross_entropy(self.holdout_model(inputs), targets, reduction='none')
        return irreducible_loss

    def get_reducible_loss(self, inputs, targets):
        """Compute the reducible loss for the current model using the holdout model.
        Args:
            inputs (torch.Tensor): Input data for which to compute the reducible loss.
            targets (torch.Tensor): Corresponding target labels for the input data.
        Returns:
            torch.Tensor: The computed reducible loss.
        """
        self.model.eval() # Ensure main model is in evaluation mode for inference
        with torch.no_grad():
            total_loss = F.cross_entropy(self.model(inputs), targets, reduction='none')
        irreducible_loss = self.get_irreducible_loss(inputs, targets)
        reducible_loss = total_loss - irreducible_loss

        return reducible_loss

    def selection(self, inputs, targets, selected_num_samples):
        """Select sub-batch with highest reducible loss.
        Args:
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
        Returns:
            torch.Tensor: Indices of the selected samples.
        """
        reducible_loss = self.get_reducible_loss(inputs, targets)
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
        selected_num_samples = min(selected_num_samples, inputs.size(0))
        indices = self.selection(inputs, targets, selected_num_samples)
        inputs = inputs[indices]
        targets = targets[indices]
        indices = indices.to(indexes.device)
        indexes = indexes[indices]

        return inputs, targets, indexes