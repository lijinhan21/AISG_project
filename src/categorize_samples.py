#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from tqdm import tqdm
import random
from torchvision.utils import make_grid
import pandas as pd

from dataset import *
from model import *
from datasets import ColoredMNIST, SyntheticFolktablesDataset

parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--dataset_path', default='./data', type=str)
parser.add_argument('--split', default=['test'], type=str, nargs='+')
parser.add_argument('--model_path', default='./saved_models', type=str)
parser.add_argument('--output_dir', default='./categorized_data', type=str)
parser.add_argument('--hidden_dim', default=[128, 64], type=int, nargs='+')
parser.add_argument('--latent_dim', default=5, type=int)
parser.add_argument('--sample_per_category', default=5, type=int, help='Number of samples to visualize per category')
parser.add_argument('--task', default='ColoredMNIST', type=str, 
                    choices=['ColoredMNIST', 'SyntheticFolktables'])

args = parser.parse_args()
print(args)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
print(f'device: {device}')

# Set paths for task
if args.task == 'ColoredMNIST':
    args.dataset_path = os.path.join(args.dataset_path, 'ColoredMNIST')
elif args.task == 'SyntheticFolktables':
    args.dataset_path = os.path.join(args.dataset_path, 'synthetic_folktables')

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
task_output_dir = output_dir / args.task
task_output_dir.mkdir(exist_ok=True, parents=True)

# Set model file path
model_file = Path(args.model_path) / f'vae_{args.task}.pt'

# Load dataset
print('Loading Dataset')
dataset = TensorLoader(batch_size=args.batch_size, path=args.dataset_path, split=args.split, workers=0)

# Initialize VAE model
vae = VAE(
    input_dim=dataset.feature_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim
)

# Load model if it exists
if model_file.exists():
    print(f'Loading model from {model_file}')
    vae.load_state_dict(torch.load(model_file, map_location=device))
    vae = vae.to(device)
else:
    print(f'Model file {model_file} not found.')
    exit(1)

def process_dataset(dataloader, set_name='train'):
    """
    Process all samples in the dataloader through the VAE and categorize them
    
    Args:
        dataloader: DataLoader object
        set_name: Name of the dataset (train, test, etc.)
        
    Returns:
        all_data: Dictionary with all processed data
    """
    print(f'Processing {set_name} set...')
    vae.eval()
    
    # Store all data
    all_features = []
    all_labels = []
    all_latent = []
    all_categories = []
    all_envs = []
    all_ids = []
    
    with torch.no_grad():
        for bundle_batch in tqdm(dataloader):
            
            # print("bundle_batch:", bundle_batch)
            
            input_batch, label_batch, env_batch, ids = bundle_batch
            input_batch = input_batch.to(device)
            
            # Get latent vectors
            z = vae.get_latent(input_batch)

            # Categorize samples
            categories = categorize_latent_samples(z)

            
            # Store data
            all_features.append(input_batch.cpu())
            all_labels.append(label_batch.cpu())
            all_latent.append(z.cpu())
            all_categories.append(categories.cpu())
            all_envs.append(env_batch.cpu())
            all_ids.append(ids.cpu())
    
    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_latent = torch.cat(all_latent, dim=0)
    all_categories = torch.cat(all_categories, dim=0)
    all_envs = torch.cat(all_envs, dim=0)
    all_ids = torch.cat(all_ids, dim=0)
    
    # Organize data by category
    category_data = {}
    for i in range(32):  # 32 categories (2^5)
        cat_mask = (all_categories == i)
        if cat_mask.sum() > 0:
            indices = torch.nonzero(cat_mask).squeeze()
            category_data[i] = {
                'indices': indices.tolist(),
                'features': all_features[cat_mask],
                'labels': all_labels[cat_mask],
                'latent': all_latent[cat_mask],
                'envs': all_envs[cat_mask],
                'ids': all_ids[cat_mask],
                'count': cat_mask.sum().item(),
                'binary': [(i >> bit) & 1 for bit in range(5)]
            }
    
    # Calculate overall stats
    all_data = {
        'features': all_features,
        'labels': all_labels,
        'latent': all_latent,
        'categories': all_categories,
        'envs': all_envs,
        'ids': all_ids,
        'category_data': category_data
    }
    
    return all_data

def visualize_mnist_samples(all_data, set_name='train'):
    """
    Visualize MNIST samples from each category
    """
    print(f'Visualizing {set_name} MNIST samples...')
    
    # Create a figure with subplots for each category
    n_categories = len(all_data['category_data'])
    n_samples = min(args.sample_per_category, 10)  # Limit to 10 samples per category for visualization
    
    # Calculate grid dimensions
    n_cols = 8
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()
    
    # For each category, select random samples and visualize
    for i, (cat_id, cat_data) in enumerate(sorted(all_data['category_data'].items())):
        if i >= len(axes):
            break
        
        # import pdb; pdb.set_trace()
        # Select random samples
        n_available = len(cat_data['indices'])
        if n_available <= n_samples:
            sample_indices = list(range(n_available))
        else:
            sample_indices = random.sample(range(n_available), n_samples)
        
        # Get samples
        samples = cat_data['features'][sample_indices]
        labels = cat_data['labels'][sample_indices]
        
        # Reshape for MNIST (B, C, H, W)
        # ColoredMNIST is 3-channel
        if samples.shape[1] == 3:
            samples = samples.permute(0, 2, 3, 1)  # (B, H, W, C)
            
            # Create a grid of images
            img_grid = make_grid(samples.permute(0, 3, 1, 2), nrow=5, normalize=True)
            axes[i].imshow(img_grid.permute(1, 2, 0).numpy())
        else:
            # Handle other datasets
            axes[i].text(0.5, 0.5, f"Cat {cat_id}\nN={cat_data['count']}", 
                         ha='center', va='center')
        
        # Set title with binary representation
        binary = ''.join(str(b) for b in cat_data['binary'])
        pos_label_ratio = (cat_data['labels'] == 1).float().mean().item()
        axes[i].set_title(f"Cat {cat_id} ({binary})\nPos%: {pos_label_ratio:.2f}")
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_categories, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(task_output_dir / f'{set_name}_category_samples.png', dpi=300)
    plt.close()

def visualize_tabular_samples(all_data, set_name='train'):
    """
    Visualize tabular data from each category using feature distributions
    """
    print(f'Visualizing {set_name} tabular data...')
    
    # Create a summary dataframe for categories
    cat_summary = []
    for cat_id, cat_data in sorted(all_data['category_data'].items()):
        binary = ''.join(str(b) for b in cat_data['binary'])
        pos_label_ratio = (cat_data['labels'] == 1).float().mean().item()
        cat_summary.append({
            'category': cat_id,
            'binary': binary,
            'count': cat_data['count'],
            'pos_label_ratio': pos_label_ratio
        })
    
    summary_df = pd.DataFrame(cat_summary)
    
    # Save summary to CSV
    summary_df.to_csv(task_output_dir / f'{set_name}_category_summary.csv', index=False)
    
    # Create visualizations for feature distributions by category
    # For tabular data, we'll compare the distribution of features across categories
    
    # Select a subset of categories for clarity (e.g., the 8 most populated)
    top_cats = sorted(all_data['category_data'].items(), 
                      key=lambda x: x[1]['count'], reverse=True)[:8]
    
    # Sample a few features to visualize (first 5 features)
    n_features = min(5, all_data['features'].shape[1])
    
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3*n_features))
    if n_features == 1:
        axes = [axes]
    
    for f_idx in range(n_features):
        for cat_id, cat_data in top_cats:
            # Get feature values for this category
            feature_vals = cat_data['features'][:, f_idx].numpy()
            binary = ''.join(str(b) for b in cat_data['binary'])
            # Plot density
            sns.kdeplot(feature_vals, ax=axes[f_idx], label=f"Cat {cat_id} ({binary})")
        
        axes[f_idx].set_title(f'Feature {f_idx} distribution by category')
        axes[f_idx].legend()
    
    plt.tight_layout()
    plt.savefig(task_output_dir / f'{set_name}_feature_distributions.png', dpi=300)
    plt.close()
    
    # Create a heatmap of mean feature values by category
    mean_features = np.zeros((len(all_data['category_data']), n_features))
    cat_ids = []
    
    for i, (cat_id, cat_data) in enumerate(sorted(all_data['category_data'].items())):
        mean_features[i] = cat_data['features'][:, :n_features].mean(dim=0).numpy()
        cat_ids.append(cat_id)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_features, cmap='coolwarm', center=0,
               xticklabels=[f'F{i}' for i in range(n_features)],
               yticklabels=[f'{cat_id}' for cat_id in cat_ids])
    plt.title(f'Mean Feature Values by Category ({set_name})')
    plt.ylabel('Category ID')
    plt.tight_layout()
    plt.savefig(task_output_dir / f'{set_name}_feature_means_heatmap.png', dpi=300)
    plt.close()

def save_categorized_data(all_data, set_name='train'):
    """
    Save the categorized data to disk
    """
    print(f'Saving {set_name} categorized data...')
    
    # Save the full dataset with category information
    data_dict = {
        'features': all_data['features'].numpy(),
        'labels': all_data['labels'].numpy(),
        'latent': all_data['latent'].numpy(),
        'categories': all_data['categories'].numpy(),
        'envs': all_data['envs'].numpy(),
        'ids': all_data['ids'].numpy()
    }
    
    # Save to numpy file
    np.savez(task_output_dir / f'{set_name}_categorized_data.npz', **data_dict)
    
    # Save summary statistics
    cat_stats = {}
    for cat_id, cat_data in sorted(all_data['category_data'].items()):
        binary = ''.join(str(b) for b in cat_data['binary'])
        pos_label_ratio = (cat_data['labels'] == 1).float().mean().item()
        env_counts = {}
        for env_id in torch.unique(cat_data['envs']):
            env_counts[env_id.item()] = (cat_data['envs'] == env_id).sum().item()
            
        cat_stats[cat_id] = {
            'binary': binary,
            'count': cat_data['count'],
            'pos_label_ratio': pos_label_ratio,
            'env_distribution': env_counts
        }
    
    # Save as JSON
    with open(task_output_dir / f'{set_name}_category_stats.json', 'w') as f:
        json.dump(cat_stats, f, indent=2)
    
    print(f'Data saved to {task_output_dir}')

# Process training data
train_data = process_dataset(dataset.training_loader_sequential, 'train')

# Process test data
# test_data = {}
# for group in args.split:
#     test_data[group] = process_dataset(dataset.test_loader[group], group)

# Visualize samples based on the dataset type
if args.task == 'ColoredMNIST':
    visualize_mnist_samples(train_data, 'train')
    # for group in args.split:
    #     visualize_mnist_samples(test_data[group], group)
else:
    # For tabular data (Folktables or SyntheticFolktables)
    visualize_tabular_samples(train_data, 'train')
    # for group in args.split:
    #     visualize_tabular_samples(test_data[group], group)

# Save the categorized data
save_categorized_data(train_data, 'train')
# for group in args.split:
#     save_categorized_data(test_data[group], group)

print("Categorization and visualization completed!") 