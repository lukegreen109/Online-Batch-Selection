import torch.optim as optim

def create_optimizer(model, config):
    optim_type = config['training_opt']['optimizer']
    optim_params = config['training_opt']['optim_params']

    if optim_type == 'SGD':
        optimizer = optim.SGD(params = model.parameters(), **optim_params)
    elif optim_type == 'Adam':
        optimizer = optim.Adam(params = model.parameters(), **optim_params)
    elif optim_type == 'AdamW':
        optimizer = optim.AdamW(params = model.parameters(), **optim_params)
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