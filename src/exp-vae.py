#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import json
from pathlib import Path
from torchvision import transforms

from dataset import *
from trainer import *
from model import *
from datasets import ColoredMNIST, SyntheticFolktablesDataset


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=4321, type=int) 
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--do_train', action='store_false')
parser.add_argument('--learning_rate', default=0.001, type=float) 
parser.add_argument('--drop_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--bias', action='store_true')
parser.add_argument('--batchnorm', action='store_true')
parser.add_argument('--hidden_dim', default=[128, 64], type=int, nargs='+')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--dataset_path', default='./data', type=str)
parser.add_argument('--split', default=['test'], type=str, nargs='+')
parser.add_argument('--iteration', default=1000, type=int)
parser.add_argument('--log_steps', default=50, type=int)
parser.add_argument('--evaluation_steps', default=100, type=int)
parser.add_argument('--model_init', default='default', type=str)
parser.add_argument('--reconstruction_weight', default=1.0, type=float)
parser.add_argument('--kl_weight', default=1.0, type=float)
parser.add_argument('--latent_dim', default=5, type=int)
parser.add_argument('--categorization_weight', default=1000.0, type=float)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--save_path', default='./saved_models', type=str)
parser.add_argument('--task', default='ColoredMNIST', type=str, 
                    choices=['ColoredMNIST', 'Folktables', 'SyntheticFolktables'])

args = parser.parse_args()
print(args)

# For reproduction
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

if args.task == 'ColoredMNIST':
    download = ColoredMNIST(root=args.dataset_path, env='all_train',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                  ]))
    args.dataset_path = os.path.join(args.dataset_path, 'ColoredMNIST')
    
elif args.task == 'SyntheticFolktables':
    mean = torch.tensor([1.84915718e+01, 4.02880048e+03, 3.80885041e+01, 1.46956721e+00, 4.32242068e+01])
    std = torch.tensor([3.89205088e+00, 2.69650006e+03, 1.29503584e+01, 4.99072986e-01, 1.50857089e+01])
    transform = lambda x: (x - mean) / std
    download = SyntheticFolktablesDataset(root=args.dataset_path, env='all_train', transform=transform)
    args.dataset_path = os.path.join(args.dataset_path, 'synthetic_folktables')
    
elif args.task == 'Folktables':
    download = FolktablesDataset(root=args.dataset_path, env='all_train')
    args.dataset_path = os.path.join(args.dataset_path, 'folktables')

def run_vae():
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'device: {device}')

    print('Loading Dataset')
    dataset = TensorLoader(batch_size=args.batch_size, path=args.dataset_path, split=args.split, workers=0)

    # Create VAE model
    vae = VAE(
        input_dim=dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        drop_rate=args.drop_rate,
        batchnorm=args.batchnorm
    )

    # Set up optimizer
    optimizer = optim.Adam(
        vae.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Create loss function
    vae_loss = VAELoss(
        reconstruction_weight=args.reconstruction_weight,
        kl_weight=args.kl_weight,
        categorization_weight=args.categorization_weight
    )

    # Create trainer
    trainer = VAETrainer(
        device=device,
        model=vae,
        optimizer=optimizer,
        dataset=dataset,
        loss_fn=vae_loss,
        **args.__dict__
    )

    # Train the VAE
    if args.do_train:
        print('Training VAE...')
        trainer.train(args.iteration, args.log_steps, args.evaluation_steps, **args.__dict__)

    # Save the model if required
    if args.save_model:
        save_dir = Path(args.save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        torch.save(vae.state_dict(), save_dir / f'vae_{args.task}.pt')
        print(f'Model saved to {save_dir / f"vae_{args.task}.pt"}')

    # Generate latent representations and analyze categories
    print('\nAnalyzing latent representations...')
    
    # Training set analysis
    print('Training set:')
    train_stats = trainer.analyze_categories(dataset.training_loader_sequential)
    
    # Test set analysis
    test_results = {}
    for group in args.split:
        print(f'Test split: {group}')
        test_stats = trainer.analyze_categories(dataset.test_loader[group])
        test_results[group] = test_stats
    
    # Save statistics
    if args.save_model:
        stats = {
            'train': train_stats,
            'test': test_results
        }
        with open(save_dir / f'vae_stats_{args.task}.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f'Statistics saved to {save_dir / f"vae_stats_{args.task}.json"}')
    
    # Print summary
    print('\nSummary:')
    print(f'Total categories found in training: {len(train_stats)}')
    for group in test_results:
        print(f'Total categories found in {group}: {len(test_results[group])}')
    
    # Print detailed category information for a few categories
    print('\nSample category details (training):')
    for i, cat_id in enumerate(sorted(train_stats.keys())[:5]):
        cat = train_stats[cat_id]
        binary = ''.join(str(b) for b in cat['binary'])
        print(f"Category {cat_id} (binary: {binary}): {cat['count']} samples, "
              f"label distribution: {cat['label_ratio']}")
              
    return vae, trainer

if __name__ == '__main__':
    vae, trainer = run_vae() 