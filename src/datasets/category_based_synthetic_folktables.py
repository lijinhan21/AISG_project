import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import folktables
from folktables import ACSDataSource

from .base import BaseFolktablesDataset

class CategoryBasedSyntheticFolktables(BaseFolktablesDataset):
    """
    Category-based SyntheticFolktables dataset that creates environments from VAE-based categorization.
    Each of the 32 categories from the latent space becomes a separate environment.
    
    Args:
        root: Root directory of dataset where data will be stored
        env: Which environment to load ('train', 'val', 'test', or 'all_train')
        transform: A function/transform that takes in features and returns transformed features
        target_transform: A function/transform that takes in the target and transforms it
    """
    
    def __init__(self, root='./data', env='train', transform=None, target_transform=None):
        super().__init__(root, env, transform, target_transform)
        self.prepare_category_based_synthetic_folktables()
        
        if env in ['train', 'val', 'test']:
            self.data_label_env_tuples = self._load_prepared_data('category_based_synthetic_folktables', env)
        elif env == 'all_train':
            self.data_label_env_tuples = self._load_prepared_data('category_based_synthetic_folktables', 'all_train')
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train, val, test, and all_train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (features, target, env) where target is the income class and env is the category-based environment
        """
        if isinstance(self.data_label_env_tuples, tuple):
            features = self.data_label_env_tuples[0][index]
            target = self.data_label_env_tuples[1][index]
            env = self.data_label_env_tuples[2][index]
        else:
            features, target, env = self.data_label_env_tuples[index]

        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return features, target, env

    def prepare_category_based_synthetic_folktables(self):
        """Prepare the category-based SyntheticFolktables dataset from categorized data."""
        if self._check_prepared_data('category_based_synthetic_folktables', ['train.pt', 'val.pt', 'test.pt']):
            print('Category-based SyntheticFolktables dataset already exists')
            return

        print('Preparing Category-based SyntheticFolktables dataset from categorized data')
        
        # Load categorized training data
        categorized_data_path = os.path.join('./categorized_data/SyntheticFolktables/train_categorized_data.npz')
        if not os.path.exists(categorized_data_path):
            raise FileNotFoundError(f"Categorized data not found at {categorized_data_path}. "
                                   "Please run categorize_samples.py first to generate categorized data.")
        
        # Load the categorized data
        categorized_data = np.load(categorized_data_path)
        features = torch.FloatTensor(categorized_data['features'])
        labels = torch.LongTensor(categorized_data['labels'])
        categories = torch.LongTensor(categorized_data['categories'])
        
        print(f"Loaded {len(features)} samples with {len(torch.unique(categories))} unique categories")
        
        # Use categories as environment labels
        env_labels = categories.clone()
        
        # Split into train and validation sets (80-20 split)
        n_samples = len(features)
        indices = torch.randperm(n_samples)
        train_size = int(0.8 * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_data = (
            features[train_indices],
            labels[train_indices],
            env_labels[train_indices]
        )
        val_data = (
            features[val_indices],
            labels[val_indices],
            env_labels[val_indices]
        )
        
        # For test data, use the original test set without categorization
        original_test_path = os.path.join(self.root, 'synthetic_folktables', 'test.pt')
        if os.path.exists(original_test_path):
            original_test_data = torch.load(original_test_path)
            test_features = original_test_data[0]
            test_labels = original_test_data[1]
            # Assign all test samples to environment -1 to distinguish from training environments
            test_env = torch.full_like(test_labels, -1, dtype=torch.long)
            test_data = (test_features, test_labels, test_env)
        else:
            raise FileNotFoundError(f"Original test data not found at {original_test_path}")

        # Create directory and save datasets
        category_dir = self._create_data_dir('category_based_synthetic_folktables')
        torch.save(train_data, os.path.join(category_dir, 'train.pt'))
        torch.save(val_data, os.path.join(category_dir, 'val.pt'))
        torch.save(test_data, os.path.join(category_dir, 'test.pt'))
        
        print(f"Category-based SyntheticFolktables dataset created with {len(torch.unique(env_labels))} environments")
        print(f"Train: {len(train_data[0])} samples, Val: {len(val_data[0])} samples, Test: {len(test_data[0])} samples")
