import torch
import torch.nn as nn
import torch.nn.functional as F 

def create_criterion(config, logger):
    loss_type = config['training_opt']['loss_type']
    loss_params = config['training_opt']['loss_params']
    if loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(**loss_params)
    else:
        raise NotImplementedError
    return criterion

def create_teacher_criterion(config, logger):
    teacher_loss_type = config['rholoss']['teacher_loss_type']
    teacher_loss_params = config['rholoss']['teacher_loss_params']
    if teacher_loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(**teacher_loss_params)
    else:
        raise NotImplementedError
    return criterion