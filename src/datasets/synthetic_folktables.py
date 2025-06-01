import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import folktables
from folktables import ACSDataSource

from .base import BaseFolktablesDataset

class SyntheticFolktablesDataset(BaseFolktablesDataset):
    """
    Synthetic Folktables dataset that creates stronger spurious correlations between gender and income.
    
    This dataset manipulates the original Folktables data to create environments with different 
    degrees of spurious correlation between gender (SEX) and income, making it more suitable 
    for demonstrating the advantages of IRM over ERM in handling distribution shifts.
    
    Args:
        root: Root directory of dataset where data will be stored
        env: Which environment to load ('train1', 'train2', 'test', or 'all_train')
        transform: A function/transform that takes in features and returns transformed features
        target_transform: A function/transform that takes in the target and transforms it
    """
    
    def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
        super().__init__(root, env, transform, target_transform)
        self.prepare_synthetic_folktables()
        
        if env in ['train1', 'train2', 'test']:
            self.data_label_tuples = self._load_prepared_data('synthetic_folktables', env)
        elif env == 'all_train':
            self.data_label_tuples = self._load_prepared_data('synthetic_folktables', 'all_train')
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (features, target) where target is the income class
        """
        features, target = self.data_label_tuples[index]

        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return features, target

    def prepare_synthetic_folktables(self):
        """Prepare the Synthetic Folktables dataset with manipulated correlations."""
        if self._check_prepared_data('synthetic_folktables', ['train.pt', 'val.pt', 'test.pt']):
            print('Synthetic Folktables dataset already exists')
            return

        print('Preparing Synthetic Folktables dataset')
        
        # Define the problem
        ACSIncomeNew = folktables.BasicProblem(
            features=[
              'SCHL', 'OCCP', 'WKHP', 'SEX', 'AGEP',
            ],
            target='PINCP',
            target_transform=lambda x: x > 25000,
            group='SEX',
            preprocess=folktables.adult_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )

        # Get data from multiple states
        data_source = ACSDataSource(survey_year='2021', horizon='1-Year', survey='person')
        all_data = data_source.get_data(states=["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"], download=True)
        
        # Convert to numpy arrays
        features, labels, group = ACSIncomeNew.df_to_numpy(all_data)
        
        # Create indices by gender and income
        # SEX: 1 for Male, 2 for Female in ACS data
        male_high_income = np.where((group == 1) & (labels == 1))[0]
        male_low_income = np.where((group == 1) & (labels == 0))[0]
        female_high_income = np.where((group == 2) & (labels == 1))[0]
        female_low_income = np.where((group == 2) & (labels == 0))[0]
        
        print(f"Original distribution:")
        print(f"Male high income: {len(male_high_income)}")
        print(f"Male low income: {len(male_low_income)}")
        print(f"Female high income: {len(female_high_income)}")
        print(f"Female low income: {len(female_low_income)}")
        
        # Create synthetic environments with different sampling probabilities
        
        # Environment 1: Strong correlation (90% male-high, 10% male-low, 10% female-high, 90% female-low)
        env1_size = 20000
        np.random.seed(42)
        
        # Determine sample sizes based on desired ratios
        env1_male_high = int(env1_size * 0.45)  # 45% of data
        env1_male_low = int(env1_size * 0.05)   # 5% of data
        env1_female_high = int(env1_size * 0.05)  # 5% of data
        env1_female_low = env1_size - env1_male_high - env1_male_low - env1_female_high  # 45% of data
        
        # Sample indices for environment 1
        env1_indices = np.concatenate([
            np.random.choice(male_high_income, size=env1_male_high, replace=True),
            np.random.choice(male_low_income, size=env1_male_low, replace=True),
            np.random.choice(female_high_income, size=env1_female_high, replace=True),
            np.random.choice(female_low_income, size=env1_female_low, replace=True)
        ])
        
        # Environment 2: Moderate correlation (70% male-high, 30% male-low, 30% female-high, 70% female-low)
        env2_size = 20000
        
        # Determine sample sizes based on desired ratios
        env2_male_high = int(env2_size * 0.35)  # 35% of data
        env2_male_low = int(env2_size * 0.15)   # 15% of data
        env2_female_high = int(env2_size * 0.15)  # 15% of data
        env2_female_low = env2_size - env2_male_high - env2_male_low - env2_female_high  # 35% of data
        
        # Sample indices for environment 2
        env2_indices = np.concatenate([
            np.random.choice(male_high_income, size=env2_male_high, replace=True),
            np.random.choice(male_low_income, size=env2_male_low, replace=True),
            np.random.choice(female_high_income, size=env2_female_high, replace=True),
            np.random.choice(female_low_income, size=env2_female_low, replace=True)
        ])
        
        # Test Environment: Reversed correlation (10% male-high, 90% male-low, 90% female-high, 10% female-low)
        test_size = 10000
        
        # Determine sample sizes based on desired ratios
        test_male_high = int(test_size * 0.05)  # 5% of data
        test_male_low = int(test_size * 0.45)   # 45% of data
        test_female_high = int(test_size * 0.45)  # 45% of data
        test_female_low = test_size - test_male_high - test_male_low - test_female_high  # 5% of data
        
        # Sample indices for test environment
        test_indices = np.concatenate([
            np.random.choice(male_high_income, size=test_male_high, replace=True),
            np.random.choice(male_low_income, size=test_male_low, replace=True),
            np.random.choice(female_high_income, size=test_female_high, replace=True),
            np.random.choice(female_low_income, size=test_female_low, replace=True)
        ])
        
        # Shuffle indices for each environment
        np.random.shuffle(env1_indices)
        np.random.shuffle(env2_indices)
        np.random.shuffle(test_indices)
        
        # Extract features and labels for each environment
        env1_features = features[env1_indices]
        env1_labels = labels[env1_indices]
        env1_group = group[env1_indices]
        
        env2_features = features[env2_indices]
        env2_labels = labels[env2_indices]
        env2_group = group[env2_indices]
        
        test_features = features[test_indices]
        test_labels = labels[test_indices]
        test_group = group[test_indices]
        
        # Print distribution of each environment for verification
        print(f"\nEnvironment 1 distribution:")
        print(f"Male high income: {np.sum((env1_group == 1) & (env1_labels == 1))}")
        print(f"Male low income: {np.sum((env1_group == 1) & (env1_labels == 0))}")
        print(f"Female high income: {np.sum((env1_group == 2) & (env1_labels == 1))}")
        print(f"Female low income: {np.sum((env1_group == 2) & (env1_labels == 0))}")
        
        print(f"\nEnvironment 2 distribution:")
        print(f"Male high income: {np.sum((env2_group == 1) & (env2_labels == 1))}")
        print(f"Male low income: {np.sum((env2_group == 1) & (env2_labels == 0))}")
        print(f"Female high income: {np.sum((env2_group == 2) & (env2_labels == 1))}")
        print(f"Female low income: {np.sum((env2_group == 2) & (env2_labels == 0))}")
        
        print(f"\nTest environment distribution:")
        print(f"Male high income: {np.sum((test_group == 1) & (test_labels == 1))}")
        print(f"Male low income: {np.sum((test_group == 1) & (test_labels == 0))}")
        print(f"Female high income: {np.sum((test_group == 2) & (test_labels == 1))}")
        print(f"Female low income: {np.sum((test_group == 2) & (test_labels == 0))}")
        
        # Convert to torch tensors
        env1_features = torch.FloatTensor(env1_features)
        env1_labels = torch.LongTensor(env1_labels)
        env1_env = torch.zeros(len(env1_labels), dtype=torch.long)  # Environment 0
        
        env2_features = torch.FloatTensor(env2_features)
        env2_labels = torch.LongTensor(env2_labels)
        env2_env = torch.ones(len(env2_labels), dtype=torch.long)   # Environment 1
        
        test_features = torch.FloatTensor(test_features)
        test_labels = torch.LongTensor(test_labels)
        test_env = torch.ones(len(test_labels), dtype=torch.long) * 2  # Environment 2
        
        # Apply transform if provided
        if self.transform is not None:
            env1_features = self.transform(env1_features)
            env2_features = self.transform(env2_features)
            test_features = self.transform(test_features)
        
        # Create datasets for training environments
        env1_data = (env1_features, env1_labels, env1_env)
        env2_data = (env2_features, env2_labels, env2_env)
        
        # Combine training data for train/val split
        train_features = torch.cat([env1_features, env2_features], dim=0)
        train_labels = torch.cat([env1_labels, env2_labels], dim=0)
        train_env = torch.cat([env1_env, env2_env], dim=0)
        
        # Split into train and validation sets
        train_size = int(0.8 * len(train_features))
        indices = torch.randperm(len(train_features))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_data = (
            train_features[train_indices],
            train_labels[train_indices],
            train_env[train_indices]
        )
        val_data = (
            train_features[val_indices],
            train_labels[val_indices],
            train_env[val_indices]
        )
        test_data = (
            test_features,
            test_labels,
            test_env
        )
        
        # Create directory and save datasets
        synthetic_dir = self._create_data_dir('synthetic_folktables')
        
        # Save environment-specific datasets
        torch.save(list(zip(env1_features, env1_labels)), os.path.join(synthetic_dir, 'train1.pt'))
        torch.save(list(zip(env2_features, env2_labels)), os.path.join(synthetic_dir, 'train2.pt'))
        torch.save(list(zip(test_features, test_labels)), os.path.join(synthetic_dir, 'test.pt'))
        
        # Save the combined datasets
        torch.save(train_data, os.path.join(synthetic_dir, 'train.pt'))
        torch.save(val_data, os.path.join(synthetic_dir, 'val.pt'))
        torch.save(test_data, os.path.join(synthetic_dir, 'test.pt'))
        
        print("Synthetic Folktables dataset created successfully")