#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

from dataset import *
from trainer import *
from model import *
from preprocessing import ColoredMNIST


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=4321, type=int) 
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--do_train', action='store_false')
parser.add_argument('--learning_rate', default=0.001, type=float) 
parser.add_argument('--drop_rate', default=0, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--bias', action='store_true')
parser.add_argument('--batchnorm', action='store_true')
parser.add_argument('--hidden_dim', default=[64,16], type=int, nargs='+')
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--dataset_path', default='./data', type=str)
parser.add_argument('--split', default=['test'], type=str, nargs='+')
parser.add_argument('--iteration', default=400, type=int)
parser.add_argument('--log_steps', default=50, type=int)
parser.add_argument('--evaluation_steps', default=20, type=int)
parser.add_argument('--metric_list', default=['Accuracy','AUC','F1_macro'], type=str, nargs='+')
parser.add_argument('--eta', default=0.05, type=float)
parser.add_argument('--model_init', default='default', type=str)
parser.add_argument('--trainer', default='ERM', type=str)
parser.add_argument('--reg_name', default=None, type=str)
parser.add_argument('--reg_lambda', default=50, type=float)

args = parser.parse_args()
print (args)

# For reproduction
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
    
download = ColoredMNIST(root=args.dataset_path, env='all_train',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ]))

args.dataset_path = os.path.join(args.dataset_path, 'ColoredMNIST')


def run_exp():
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'device:{device}')

    print ('Loading Dataset')
    dataset = TensorLoader(batch_size=args.batch_size, path=args.dataset_path, split=args.split, workers=0)

    model = MLP(input_dim = dataset.feature_dim, 
            hidden_size = args.hidden_dim, 
            drop_rate = args.drop_rate,
            batchnorm = args.batchnorm,
            bias = args.bias
    )

    def get_parameters(network_list):
        parameter_list_decay = []
        parameter_list_wo_decay = []
        for network in network_list:
            for name, para in network.named_parameters():
                if 'bias' in name:
                    parameter_list_wo_decay.append(para)
                else:
                    parameter_list_decay.append(para)
        return parameter_list_decay, parameter_list_wo_decay

   
    parameter_list_wo_decay, parameter_list_decay = get_parameters([model.model()])
    optimizer = optim.Adam([
                {'params': parameter_list_wo_decay},
                {'params':parameter_list_decay, 'weight_decay': args.weight_decay}
            ], 
            lr=args.learning_rate
    )

    if args.reg_name is None:
        reg_object = None
    else:
        reg_object = loss_register[args.reg_name]()
    
    if args.trainer == 'ERM':
        trainer = trainer_register[args.trainer](device, model, optimizer, dataset, bce_loss(), reg_object, **args.__dict__)
    elif args.trainer == 'groupDRO':
        trainer = trainer_register['ERM'](device, model, optimizer, dataset, groupDRO(bce_loss(), device, n_env=2, eta=args.eta), reg_object, **args.__dict__)    
    elif args.trainer == 'InvRat':
        env_model = MLP(input_dim = dataset.feature_dim + 1, 
            hidden_size = args.hidden_dim, 
            drop_rate = args.drop_rate,
            batchnorm = args.batchnorm,
            bias = args.bias
        )
        env_optimizer = optim.Adam([
                {'params': parameter_list_wo_decay},
                {'params':parameter_list_decay, 'weight_decay': args.weight_decay}
            ], 
            lr=args.learning_rate
        )
        env_loss_fn = bce_loss()
        args.env_optimizer = env_optimizer
        args.env_loss_fn = env_loss_fn
        trainer = trainer_register['InvRat'](device, model, env_model, optimizer, dataset, bce_loss(), reg_object, **args.__dict__)
    else:
        raise NotImplementedError

    if args.do_train:
        trainer.train(args.iteration, args.log_steps, args.evaluation_steps, args.metric_list, **args.__dict__)

    metric_dict_final = {}
    for group in args.split:
        print(f'split:{group}')
        metric_dict = trainer.evaluate(dataset.test_loader[group], args.metric_list, return_loss=False)
        for k,v in metric_dict.items():
            metric_dict_final[k+'_'+group]  = v
    
    print('Test result:', metric_dict_final)
    return metric_dict_final

if __name__ == '__main__':
    metric_dict_list = {}
    for _ in range(3):
        metric_dict = run_exp()
        for k,v in metric_dict.items():
            if k in metric_dict_list:
                metric_dict_list[k].append(v)
            else:
                metric_dict_list[k] = [v]

    metric_result = {}
    for k,v in metric_dict_list.items():
        metric_result[k+'mean'] = np.mean(v)
        metric_result[k+'std'] = np.std(v)
    print ('Summary:', metric_result)


