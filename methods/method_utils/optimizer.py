import torch.optim as optim

def create_optimizer(model, config):
    training_opt = config['training_opt']
    lr = training_opt['optim_params']['lr']
    if 'weight_decay' in training_opt['optim_params']:
        weight_decay = training_opt['optim_params']['weight_decay']
    if 'betas' in training_opt['optim_params']:
        betas = training_opt['optim_params']['betas']
    if training_opt['optimizer'] == 'SGD':
        momentum = training_opt['optim_params']['momentum']
    else:
        momentum = None
    
    if training_opt['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif training_opt['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif training_opt['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def create_holdout_optimizer(holdout_model, config):
    rholoss = config['rholoss']
    lr = rholoss['holdout_optim_params']['lr']
    if 'weight_decay' in rholoss['holdout_optim_params']:
        weight_decay = rholoss['holdout_optim_params']['weight_decay']
    if 'betas' in rholoss['holdout_optim_params']:
        betas = rholoss['holdout_optim_params']['betas']
    if rholoss['holdout_optimizer'] == 'SGD':
        momentum = rholoss['holdout_optim_params']['momentum']
    else:
        momentum = None
    
    if rholoss['holdout_optimizer'] == 'SGD':
        optimizer = optim.SGD(holdout_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif rholoss['holdout_optimizer'] == 'Adam':
        optimizer = optim.Adam(holdout_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif rholoss['holdout_optimizer'] == 'AdamW':
        optimizer = optim.AdamW(holdout_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def create_scheduler(optimizer, config):
    training_opt = config['training_opt']
    scheduler_params = training_opt['scheduler_params']
    if training_opt['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=scheduler_params['gamma'], step_size=scheduler_params['step_size'])
    elif training_opt['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_opt['num_epochs'], eta_min=scheduler_params['endlr'])
    elif training_opt['scheduler'] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_params['milestones'], gamma=scheduler_params['gamma'])
    elif training_opt['scheduler'] == 'constant':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1) 
    else:
        raise NotImplementedError
    return scheduler