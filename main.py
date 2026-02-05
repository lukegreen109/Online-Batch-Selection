import yaml
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import secrets
from utils import custom_logger,random_str, get_date, re_nest_configs, get_configs
import wandb

import torch.multiprocessing as mp
import methods



def init_seeds(seed):
    print('=====> Using fixed random seed: ' + str(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def main():
    # ============================================================================
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--method',  type=str,
                        default=None,
                        help='batch selection method')
    parser.add_argument('--data',  type=str,
                        default=None,
                        help='dataset name')
    parser.add_argument('--model', type=str,
                        default=None,
                        help='model name')
    parser.add_argument('--optim', type=str,
                        default=None,
                        help='batch size, batch seed, learning rate, optimizer, weight decay')
    parser.add_argument('--save_dir', type=str, 
                        default=None,
                        help='directory to save results')
    parser.add_argument('--log_file', type=str, 
                        default=None, 
                        help='Logger file name.')
    parser.add_argument('--notes', type=str,
                        default=None, 
                        help='Notes for the experiment.')
    parser.add_argument('--wandb_not_upload', action='store_true', 
                        help='Do not upload the result to wandb.')

    args = parser.parse_args()

    # ============================================================================
    # load config file
    print('=====> Loading config files: \n' + args.method + '\n' + args.data + '\n' + args.model + '\n' + args.optim)
    method_config = get_configs(args.method)
    data_config = get_configs(args.data)
    model_config = get_configs(args.model)
    optim_config = get_configs(args.optim)
    config = {**method_config, **data_config, **model_config, **optim_config} # combine into single config
    config['seed'] = args.seed # add seed to config
    print('=====> Config files loaded.')




    if args.log_file is not None:
        config['log_file'] = args.log_file
    

    if args.save_dir is None:
        args.save_dir = './exp/'
        # dataset
        args.save_dir = os.path.join(args.save_dir, config['dataset']['name'])
        # method
        args.save_dir = os.path.join(args.save_dir, config['method'])
        # model
        args.save_dir = args.save_dir + '_' + config['networks']['params']['m_type']
        # bs
        args.save_dir = args.save_dir + '_bs' + str(config['training_opt']['batch_size'])
        # epochs
        args.save_dir = args.save_dir + '_ep' + str(config['training_opt']['num_epochs'])
        # lr
        args.save_dir = args.save_dir + '_lr' + str(config['training_opt']['optim_params']['lr'])
        # optimizer
        args.save_dir = args.save_dir + '_' + config['training_opt']['optimizer']
        # scheduler
        args.save_dir = args.save_dir + '_' + config['training_opt']['scheduler']
        # seed
        args.save_dir = args.save_dir + '_seed' + str(args.seed)
        # ratio
        if 'method_opt' in config:
            if 'ratio' in config['method_opt']:
                args.save_dir = args.save_dir + '_r' + str(config['method_opt']['ratio'])
        # notes
        if args.notes is not None:
            args.save_dir = args.save_dir + '_' + args.notes
    

    # method/save_dir
    save_dir = args.save_dir
    config['save_dir'] = save_dir
    method = config['method']

    if method not in methods.__all__:
        raise ValueError(f'Method {method} is not supported. Please check the methods.py file.')

    # Check if run has already been executed
    if os.path.exists(save_dir):
        print(f'Skip {method} as output already exists.')
    else:
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)


        # wandb_not_upload
        if args.wandb_not_upload:
            os.environ["WANDB_MODE"] = "dryrun"
        else:
            os.environ["WANDB_MODE"] = "run"
        
        if args.log_file is None:
            logger = custom_logger(save_dir)
        else:
            logger = custom_logger(save_dir, args.log_file)

        logger.info('========================= Start Main =========================')


        # save config file
        logger.info('=====> Saving config file')
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info('=====> Config file saved')


        init_seeds(args.seed)
        # logger.info(f'=====> Random seed initialized to {config["seed"]}')
        logger.info(f'=====> Wandb initialized')
        run = wandb.init(config=config,project="Preston_Sandbox")
        re_nest_configs(run.config)
        wandb.define_metric('acc', 'max')
        run.name = method + '_' + config['save_dir'].split('/')[-2]

        wandb_local_path = wandb.run.dir
        # save wandb_local_path to wandb_local_path.txt
        with open(os.path.join(save_dir, 'wandb_local_path.txt'), 'w') as f:
            f.write(wandb_local_path)
            f.close()

        config['num_gpus'] = torch.cuda.device_count()
        logger.info(f'=====> Number of GPUs: {config["num_gpus"]}')

        Method = getattr(methods, method)(config, logger)
        Method.run()

        logger.info('========================= End Main =========================')

        logger.wandb_finish()



if __name__ == '__main__':
    main()