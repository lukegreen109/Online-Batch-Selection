import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .SelectionMethod import SelectionMethod
from .method_utils.build_teacher_model import build_teacher_model
from models.BayesNet import KFCALLAWrapper
from transformers import AutoImageProcessor, AutoModelForImageClassification


class Bayesian(SelectionMethod):
    method_name = "Bayesian"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.balance = config["method_opt"]["balance"]
        self.ratio = config["method_opt"]["ratio"]
        self.ratio_scheduler = (
            config["method_opt"]["ratio_scheduler"]
            if "ratio_scheduler" in config["method_opt"]
            else "constant"
        )
        self.warmup_epochs = (
            config["method_opt"]["warmup_epochs"]
            if "warmup_epochs" in config["method_opt"]
            else 0
        )

        self.reduce_dim = (
            config["method_opt"]["reduce_dim"]
            if "reduce_dim" in config["method_opt"]
            else False
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        last_layer_name, _ = list(self.model.named_modules())[-1]

        self.model = KFCALLAWrapper(
            net=self.model,
            num_effective_data=config["num_effective_data"],
            prior_precision=config["prior_precision"],
            n_f_samples=config["n_f_samples"],
            input_dim=config["dataset"]["input_dim"],
            momentum=config["laplace_momentum"],
            last_layer_name=last_layer_name
        )
        self.model = self.model.cuda()

        # See SelectionMethod for data_info; need for teacher classifier
        self.template = self.data_info.get("template", None)
        self.classes = self.data_info.get("classes", None)

        self.setup_teacher_model(config,logger)
        self.test_teacher() # Validate teacher classifier test accuracy
        self.precompute_losses()

        self.alpha = config["alpha"]
        self.adaptive_alpha = config["adaptive_alpha"]


    def setup_teacher_model(self, config, logger):
        """Retrieve the teacher model from config for computing irreducible loss."""
        teacher_config = dict(config)
        teacher_config['classes'] = self.data_info.get('classes')
        teacher_config['template'] = self.data_info.get('template')
        self.teacher_model = build_teacher_model(teacher_config, logger)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()


    def precompute_losses(self):
        """Precompute losses for the training dataset using the teacher model."""
        losses_tensor = torch.zeros(self.train_dset.__len__())

        with torch.no_grad():
            for datas in self.train_loader:
                inputs = datas['input'].to(self.device)
                targets = datas['target'].to(self.device)
                indexes = datas['index']
                outputs = self.teacher_model(inputs)
                loss = - F.cross_entropy(outputs, targets, reduction='none')
                losses_tensor[indexes] = loss.cpu().float()

        # Attach the losses tensor directly to the dataset object
        self.train_dset.teacher_loss_cache = losses_tensor
        self.logger.info(f"Cached teacher losses for {len(losses_tensor)} samples in dataset.")

    def test_teacher(self):
        self.logger.info('=====> Start teacher Validation')
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, datas in enumerate(self.test_loader):
                inputs = datas['input'].cuda()
                targets = datas['target'].cuda()
                outputs = self.teacher_model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = np.mean(all_preds == all_labels)
        self.logger.info(f'=====> teacher Test Accuracy: {acc:.4f}')

    def get_ratio_per_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            self.logger.info("warming up")
            return 1.0
        if self.ratio_scheduler == "constant":
            return self.ratio
        elif self.ratio_scheduler == "increase_linear":
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == "decrease_linear":
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == "increase_exp":
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        elif self.ratio_scheduler == "decrease_exp":
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        else:
            raise NotImplementedError

    def bayesian_selection(self, inputs, targets, indexes, number_to_select):        
        f_samples, outputs, stds, L_U_T_inverse = self.model(
            inputs, selection_pass=True
        )

        first_term = (
            -F.cross_entropy(
                f_samples.flatten(0, 1),
                targets.repeat_interleave(f_samples.shape[1]),
                reduction="none",
            )
            .view(f_samples.shape[0], f_samples.shape[1])
            .mean(1)
        )
        teacher_outputs = self.train_dset.teacher_loss_cache[indexes].to(f_samples.device)
        if self.adaptive_alpha:
            bayes_loss = self.criterion(outputs, targets).item()
            teacher_loss = self.criterion(teacher_outputs, targets).item()
            self.alpha = teacher_loss / (bayes_loss + teacher_loss)

        second_term = teacher_outputs
        third_term = -F.cross_entropy(f_samples.softmax(-1).mean(1).log(), targets, reduction="none")
        select_obj = self.alpha * first_term + (1 - self.alpha) * second_term - third_term
        _, index_selected = torch.topk(select_obj, k=number_to_select, largest=True, sorted=False)
        return index_selected.cpu().numpy()

    def before_batch(self, i, inputs, targets, indexes, epoch):
        ratio = self.get_ratio_per_epoch(epoch)
        if ratio == 1.0:
            if i == 0:
                self.logger.info("using all samples")
            return super().before_batch(i, inputs, targets, indexes, epoch)
        else:
            if i == 0:
                self.logger.info(f"balance: {self.balance}")
                self.logger.info(
                    "selecting samples for epoch {}, ratio {}".format(epoch, ratio)
                )
        number_to_select = int(inputs.shape[0] * ratio)

        self.model.eval()
        with torch.no_grad():
            indices = self.bayesian_selection(inputs, targets, indexes, number_to_select)
        self.model.train()

        inputs = inputs[indices]
        targets = targets[indices]
        indexes = indexes[indices]

        return inputs, targets, indexes