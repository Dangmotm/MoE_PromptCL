import argparse
import torch
import numpy as np
import random
import utils

from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest = 'subparser_name')

    if config == 'cifar100_norgaprompt':
        from configs.cifar100_norgaprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_norgaprompt', help = 'Split-CIFAR100 NoRGa-prompt configs')
    else:
        raise NotImplementedError
    
    get_args_parser(config_parser)
    args = parser.parse_args()
    args.config = config
    return args

def main(args):
    utils.init_distributed_mode(args)
    
    # Save model
    if args.output_dir:
        Path(args.output_dir).mkdir(parents = True, exist_ok = True)
        
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    if 'norgaprompt' in args.config:
        print('Using NoRGa-prompt')
        import trainers.norgaprompt_trainer as norgaprompt_trainer
        norgaprompt_trainer.train(args)
    else: 
        raise NotImplementedError

if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
    