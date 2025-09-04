from methods.ReweightMethod import ReweightMethod
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
import cvxpy as cp


class RhoLossRW(ReweightMethod):
    """A class for implementing the RhoLoss reweighting method, which reweights samples based on the exponential 
    of the reducible loss scaled by alpha.

    This class inherits from `ReweightMethod` and uses an irreducible loss model (ILmodel) and a target model
    to compute reducible loss for sample selection during training. It supports various alpha scheduling strategies
    for dynamic sample reweighting and handles model training and loading for specific datasets.

    Args:
        config (dict): Configuration dictionary containing method and dataset parameters.
            Expected keys include:
                - 'method_opt': Dictionary with keys 'alpha', 'rho'.
                - 'rho_loss': Dictionary with key 'training_budget'.
                - 'dataset': Dictionary with keys 'name' and 'num_classes'.
                - 'networks': Dictionary with key 'params' containing 'm_type'.
        logger (logging.Logger): Logger instance for logging training and selection information.
    """
    method_name = 'RhoLossRW'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.alpha = config['method_opt']['alpha']
        self.alpha_scheduler = config['method_opt']['alpha_scheduler'] if 'alpha_scheduler' in config['method_opt'] else 'constant'
        self.rho = config['method_opt']['rho']
        self.current_train_indices = np.arange(self.num_train_samples)
        self.reduce_dim = config['method_opt']['reduce_dim'] if 'reduce_dim' in config['method_opt'] else False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setup_holdout_model(config, logger)
        
        self.precompute_losses()


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
        """"Retrieve the holdout model for computing irreducible loss.
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
            holdout_dataloader (DataLoader): DataLoader for holdout set.
            holdout_epochs (int): Number of epochs to train the holdout model.
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
    


    def project_onto_simplex_qp(self, weights, rho = 0.25):
        """
        Projects a vector x onto the probability simplex using quadratic programming.

        Args:
            x (np.ndarray): The input vector to be projected.

        Returns:
            np.ndarray: The projection of x onto the probability simplex.
        """
        n = len(weights)
        simplex_center = cp.Constant(np.ones(n) / n)

        # Define the optimization variable y
        y = cp.Variable(n)
        
        # Define the objective function: 0.5 * ||y - x||^2
        objective = cp.Minimize(0.5 * cp.sum_squares(y - weights))
        
        # Define the constraints
        constraints = [
            cp.sum(y) == 1,  # The elements must sum to one
            cp.sum_squares(y - simplex_center) <= float(rho)/n
            # y >= 1/n - rho/n,           # The elements must be within rho/n of 1/n
            # y <= 1/n + rho/n
        ]
        
        # Define the QP problem and solve it
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Return the optimal value of y
        return y.value


    def get_alpha_per_epoch(self, epoch):
        if self.alpha_scheduler == "constant":
            return self.alpha
        elif self.alpha_scheduler == "decrease_linear":
            min_alpha = self.alpha[0]
            max_alpha = self.alpha[1]
            return max_alpha - (max_alpha - min_alpha) * epoch / self.epochs
        elif self.alpha_scheduler == "decrease_exp":
            min_alpha = self.alpha[0]
            max_alpha = self.alpha[1]
            return max_alpha - (max_alpha - min_alpha) * np.exp(epoch / self.epochs)
        else:
            raise NotImplementedError


    def reducible_loss_weights(self, inputs, targets, indexes, alpha):
        """Select sub-batch with highest reducible loss.
        Args:
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
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

        self.logger.info(f"reducible_loss: {reducible_loss}")
        weights = F.softmax(alpha * reducible_loss, dim=0)
        self.logger.info(f"weights: {weights}")
        weights = weights.cpu().numpy()
        weights = self.project_onto_simplex_qp(weights, self.rho)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        # Return to train mode and return selected indices
        self.logger.info(f"projected_weights: {weights}")
        self.model.train()
        return weights


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
        alpha = self.get_alpha_per_epoch(epoch)

        if i == 0:
            self.logger.info('selecting samples for epoch {}, alpha {}'.format(epoch, alpha))

        # Get projected weights based on the exponential of the reducible loss scaled by alpha
        weights = self.reducible_loss_weights(inputs, targets, indexes, alpha)

        weights = torch.ones_like(weights) / len(weights)
        return weights