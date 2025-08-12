import numpy as np
import torch
import torch.nn.functional as F

from .SelectionMethod import SelectionMethod
from models.BayesNet import CLIPZeroShotClassifier
from models.BayesNet import KFCALLAWrapper


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

        self.current_train_indices = np.arange(self.num_train_samples)
        self.reduce_dim = (
            config["method_opt"]["reduce_dim"]
            if "reduce_dim" in config["method_opt"]
            else False
        )

        self.model = KFCALLAWrapper(
            net=self.model,
            num_effective_data=config["bayesian"]["num_effective_data"],
            prior_precision=config["bayesian"]["prior_precision"],
            n_f_samples=config["bayesian"]["n_f_samples"],
            momentum=config["bayesian"]["laplace_momentum"],
        )
        self.model = self.model.cuda()

        # See SelectionMethod for data_info; need for clip classifier
        self.template = self.data_info["template"]
        self.classes = self.data_info["classes"]

        self.clip_clf = CLIPZeroShotClassifier(
            self.classes,
            self.template,
            config["dataset"]["name"],
            config["bayesian"]["clip_architecture"],
        )

        self.clip_clf = self.clip_clf.cuda()
        self.clip_clf.eval()
        self.test_clip() # Validate CLIP classifier test accuracy

        assert "tau" in config["bayesian"], (
            "Bayesian tau parameter is not set in config"
        )
        self.tau = config["bayesian"]["tau"]
        assert "adaptive_alpha" in config["bayesian"], (
            "Bayesian adaptive_alpha parameter is not set in config"
        )
        self.alpha = config["bayesian"]["alpha"]
        self.adaptive_alpha = config["bayesian"]["adaptive_alpha"]

    def test_clip(self):
        self.logger.info('=====> Start CLIP Validation')
        self.clip_clf.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, datas in enumerate(self.test_loader):
                inputs = datas['input'].cuda()
                targets = datas['target'].cuda()
                outputs = self.clip_clf(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = np.mean(all_preds == all_labels)
        self.logger.info(f'=====> CLIP Validation Accuracy: {acc:.4f}')

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

    def bayesian_selection(self, inputs, targets, number_to_select):
        f_samples, outputs, stds, L_U_T_inverse = self.model(
            inputs, selection_pass=True
        )
        clip_outputs = self.clip_clf(inputs, tau=self.tau)
        if self.adaptive_alpha:
            bayes_loss = self.criterion(outputs, targets).item()
            clip_loss = self.criterion(clip_outputs, targets).item()
            self.alpha = clip_loss / (bayes_loss + clip_loss)

        first_term = (
            -F.cross_entropy(
                f_samples.flatten(0, 1),
                targets.repeat_interleave(f_samples.shape[1]),
                reduction="none",
            )
            .view(f_samples.shape[0], f_samples.shape[1])
            .mean(1)
        )
        second_term = -F.cross_entropy(clip_outputs, targets, reduction="none")
        third_term = -F.cross_entropy(
            f_samples.softmax(-1).mean(1).log(), targets, reduction="none"
        )
        select_obj = self.alpha * first_term + (1 - self.alpha) * second_term - third_term
        _, index_selected = torch.topk(select_obj, number_to_select)
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
        indices = self.bayesian_selection(inputs, targets, number_to_select)
        inputs = inputs[indices]
        targets = targets[indices]
        indexes = indexes[indices]
        return inputs, targets, indexes