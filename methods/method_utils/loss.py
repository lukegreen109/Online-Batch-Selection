import torch
import torch.nn as nn


def create_criterion(config, logger):
    loss_type = config["training_opt"]["loss_type"]
    loss_params = config["training_opt"]["loss_params"]
    if loss_type == "CrossEntropy":
        criterion = nn.CrossEntropyLoss(**loss_params, reduction="none")
    else:
        raise NotImplementedError

    def criterion_with_reduction_arg(*args, reduction="mean", **kwargs):
        losses = criterion(*args, **kwargs)
        if reduction == "mean":
            return torch.mean(losses)
        elif reduction is None:
            return losses
        else:
            raise ValueError(f"Reduction {reduction} not implemented")

    return criterion_with_reduction_arg


def create_teacher_criterion(config, logger):
    teacher_loss_type = config["rholoss"]["teacher_loss_type"]
    teacher_loss_params = config["rholoss"]["teacher_loss_params"]
    if teacher_loss_type == "CrossEntropy":
        criterion = nn.CrossEntropyLoss(**teacher_loss_params, reduction="none")
    else:
        raise NotImplementedError

    def criterion_with_reduction_arg(*args, reduction="mean", **kwargs):
        losses = criterion(*args, **kwargs)
        if reduction == "mean":
            return torch.mean(losses)
        elif reduction is None:
            return losses
        else:
            raise ValueError(f"Reduction {reduction} not implemented")

    return criterion_with_reduction_arg
