# -*- coding: utf-8 -*-

"""
Unbalanced Four-Environment Folktables dataset implementation for IRM experiments.
"""

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import folktables
from folktables import ACSDataSource

from .base import BaseFolktablesDataset

class FourEnvSyntheticFolktables2(BaseFolktablesDataset):
    """
    Unbalanced Four-Environment Folktables dataset that creates environments based on 
    the cross-product of SEX (Male/Female) and Income (Low/High) while preserving 
    the original distribution without balancing.
    
    Environment mapping:
    - Env 0: Male + Low Income (original counts)
    - Env 1: Male + High Income (original counts)
    - Env 2: Female + Low Income (original counts)
    - Env 3: Female + High Income (original counts)
    
    This dataset is designed for testing CategoryReweightedERM method.
    
    Args:
        root: Root directory of dataset where data will be stored
        env: Which environment to load ('train', 'val', 'test', or 'all_train')
        transform: A function/transform that takes in features and returns transformed features
        target_transform: A function/transform that takes in the target and transforms it
    """
    
    def __init__(self, root='./data', env='train', transform=None, target_transform=None):
        super().__init__(root, env, transform, target_transform)
        self.prepare_four_env_folktables()
        
        if env in ['train', 'val', 'test']:
            self.data_label_tuples = self._load_prepared_data('four_env_synthetic_folktables2', env)
        elif env == 'all_train':
            self.data_label_tuples = self._load_prepared_data('four_env_synthetic_folktables2', 'all_train')
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train, val, test, and all_train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (features, target, env) where target is the income class and env is the environment (0-3)
        """
        features, target = self.data_label_tuples[index]

        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return features, target

    def prepare_four_env_folktables(self):
        """Prepare the Unbalanced Four-Environment Folktables dataset."""
        if self._check_prepared_data('four_env_synthetic_folktables2', ['train.pt', 'val.pt', 'test.pt']):
            print('Four-Environment Folktables2 dataset already exists')
            return

        print('Preparing Four-Environment Folktables dataset2')
        
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
        
        # Create environment labels: male_high_income being 0, male_low_income being 1, female_high_income being 2, female_low_income being 3.
        env_labels = np.zeros(len(features), dtype=int)
        env_labels[male_high_income] = 0
        env_labels[male_low_income] = 1
        env_labels[female_high_income] = 2
        env_labels[female_low_income] = 3
        
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
        env1_env = env_labels[env1_indices]
        
        env2_features = features[env2_indices]
        env2_labels = labels[env2_indices]
        env2_group = group[env2_indices]
        env2_env = env_labels[env2_indices]
        
        test_features = features[test_indices]
        test_labels = labels[test_indices]
        test_group = group[test_indices]
        test_env = env_labels[test_indices]
        
        # Print distribution of each environment for verification
        print(f"\nEnvironment 1 distribution:")
        print(f"Male high income: {np.sum(env1_env == 0)}")
        print(f"Male low income: {np.sum(env1_env == 1)}")
        print(f"Female high income: {np.sum(env1_env == 2)}")
        print(f"Female low income: {np.sum(env1_env == 3)}")
        
        print(f"\nEnvironment 2 distribution:")
        print(f"Male high income: {np.sum(env2_env == 0)}")
        print(f"Male low income: {np.sum(env2_env == 1)}")
        print(f"Female high income: {np.sum(env2_env == 2)}")
        print(f"Female low income: {np.sum(env2_env == 3)}")
        
        print(f"\nTest environment distribution:")
        print(f"Male high income: {np.sum((test_group == 1) & (test_labels == 1))}")
        print(f"Male low income: {np.sum((test_group == 1) & (test_labels == 0))}")
        print(f"Female high income: {np.sum((test_group == 2) & (test_labels == 1))}")
        print(f"Female low income: {np.sum((test_group == 2) & (test_labels == 0))}")
        
        # Convert to torch tensors
        env1_features = torch.FloatTensor(env1_features)
        env1_labels = torch.LongTensor(env1_labels)
        # env1_env = torch.zeros(len(env1_labels), dtype=torch.long)  # Environment 0
        env1_env = torch.tensor(env1_env, dtype=torch.long) 
        
        env2_features = torch.FloatTensor(env2_features)
        env2_labels = torch.LongTensor(env2_labels)
        # env2_env = torch.ones(len(env2_labels), dtype=torch.long)   # Environment 1
        env2_env = torch.tensor(env2_env, dtype=torch.long)
        
        test_features = torch.FloatTensor(test_features)
        test_labels = torch.LongTensor(test_labels)
        # test_env = torch.ones(len(test_labels), dtype=torch.long) * 2  # Environment 2
        test_env = torch.tensor(test_env, dtype=torch.long)
        
        # Apply transform if provided
        if self.transform is not None:
            env1_features = self.transform(env1_features)
            env2_features = self.transform(env2_features)
            test_features = self.transform(test_features)
        
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
        synthetic_dir = self._create_data_dir('four_env_synthetic_folktables2')
        
        # # Save environment-specific datasets
        # torch.save(list(zip(env1_features, env1_labels)), os.path.join(synthetic_dir, 'train1.pt'))
        # torch.save(list(zip(env2_features, env2_labels)), os.path.join(synthetic_dir, 'train2.pt'))
        # torch.save(list(zip(test_features, test_labels)), os.path.join(synthetic_dir, 'test.pt'))
        
        # Save the combined datasets
        torch.save(train_data, os.path.join(synthetic_dir, 'train.pt'))
        torch.save(val_data, os.path.join(synthetic_dir, 'val.pt'))
        torch.save(test_data, os.path.join(synthetic_dir, 'test.pt'))
        
        print("Synthetic Folktables dataset created successfully")
        
        
        # ########################################################################3
        # # Define the problem
        # ACSIncomeNew = folktables.BasicProblem(
        #     features=[
        #       'SCHL', 'OCCP', 'WKHP', 'SEX', 'AGEP',
        #     ],
        #     target='PINCP',
        #     target_transform=lambda x: x > 25000,
        #     group='SEX',
        #     preprocess=folktables.adult_filter,
        #     postprocess=lambda x: np.nan_to_num(x, -1),
        # )

        # # Get data from multiple states to have enough samples
        # data_source = ACSDataSource(survey_year='2021', horizon='1-Year', survey='person')
        # train_data = data_source.get_data(states=["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"], download=True)
        
        # # Convert to numpy arrays
        # features, labels, group = ACSIncomeNew.df_to_numpy(train_data)
        
        # # Create indices by gender and income for training data
        # # SEX: 1 for Male, 2 for Female in ACS data
        # male_high_income_idx = np.where((group == 1) & (labels == 1))[0]
        # male_low_income_idx = np.where((group == 1) & (labels == 0))[0]
        # female_high_income_idx = np.where((group == 2) & (labels == 1))[0]
        # female_low_income_idx = np.where((group == 2) & (labels == 0))[0]
        
        # print(f"Original training data distribution:")
        # print(f"Male low income: {len(male_low_income_idx)}")
        # print(f"Male high income: {len(male_high_income_idx)}")
        # print(f"Female low income: {len(female_low_income_idx)}")
        # print(f"Female high income: {len(female_high_income_idx)}")
        
        # male_low_income_idx = np.random.choice(male_low_income_idx, size=4000, replace=False)
        # male_high_income_idx = np.random.choice(male_high_income_idx, size=16000, replace=False)
        # female_low_income_idx = np.random.choice(female_low_income_idx, size=16000, replace=False)
        # female_high_income_idx = np.random.choice(female_high_income_idx, size=4000, replace=False)
        
        # # Create four environments using ALL available data (no balancing)
        # # Env 0: Male + Low Income, Env 1: Male + High Income
        # # Env 2: Female + Low Income, Env 3: Female + High Income
        
        # np.random.seed(42)
        
        # # Use ALL indices for each environment (keeping original distribution)
        # env0_indices = male_low_income_idx
        # env1_indices = male_high_income_idx
        # env2_indices = female_low_income_idx
        # env3_indices = female_high_income_idx
        
        # # Combine all environments for training
        # all_train_indices = np.concatenate([env0_indices, env1_indices, env2_indices, env3_indices])
        
        # # Create environment labels
        # env_labels = np.concatenate([
        #     np.zeros(len(env0_indices)),      # Env 0: Male + Low Income
        #     np.ones(len(env1_indices)),       # Env 1: Male + High Income  
        #     np.full(len(env2_indices), 2),    # Env 2: Female + Low Income
        #     np.full(len(env3_indices), 3)     # Env 3: Female + High Income
        # ])
        
        # # Shuffle the combined training data
        # shuffle_indices = np.random.permutation(len(all_train_indices))
        # all_train_indices = all_train_indices[shuffle_indices]
        # env_labels = env_labels[shuffle_indices]
        
        # # Extract features and labels for training
        # train_features = features[all_train_indices]
        # train_labels = labels[all_train_indices]
        # train_group = group[all_train_indices]
        
        # # Print environment distribution for verification
        # print(f"\nUnbalanced training environment distribution:")
        # for env_id in range(4):
        #     env_mask = (env_labels == env_id)
        #     env_features = train_features[env_mask]
        #     env_group = train_group[env_mask]
        #     env_targets = train_labels[env_mask]
            
        #     male_count = np.sum(env_group == 1)
        #     female_count = np.sum(env_group == 2)
        #     high_income_count = np.sum(env_targets == 1)
        #     low_income_count = np.sum(env_targets == 0)
        #     total_count = len(env_features)
            
        #     print(f"Env {env_id}: Total={total_count}, Male={male_count}, Female={female_count}, "
        #           f"High Income={high_income_count}, Low Income={low_income_count}")
        
        # # Convert training data to torch tensors
        # train_features = torch.FloatTensor(train_features)
        # train_labels = torch.LongTensor(train_labels)
        # env_labels = torch.LongTensor(env_labels)
        
        # # Split training data into train and validation sets while preserving environment distribution
        # unique_envs = torch.unique(env_labels)
        # train_indices_list = []
        # val_indices_list = []
        
        # for env_id in unique_envs:
        #     env_mask = env_labels == env_id
        #     env_indices = torch.where(env_mask)[0]
            
        #     # Split each environment separately to preserve distribution
        #     n_env_samples = len(env_indices)
        #     train_size = int(0.8 * n_env_samples)
            
        #     shuffled_env_indices = env_indices[torch.randperm(n_env_samples)]
        #     train_indices_list.append(shuffled_env_indices[:train_size])
        #     val_indices_list.append(shuffled_env_indices[train_size:])
        
        # # Combine all train and val indices
        # final_train_idx = torch.cat(train_indices_list)
        # final_val_idx = torch.cat(val_indices_list)
        
        # # Shuffle final indices
        # final_train_idx = final_train_idx[torch.randperm(len(final_train_idx))]
        # final_val_idx = final_val_idx[torch.randperm(len(final_val_idx))]
        
        # train_data = (
        #     train_features[final_train_idx],
        #     train_labels[final_train_idx],
        #     env_labels[final_train_idx]
        # )
        # val_data = (
        #     train_features[final_val_idx],
        #     train_labels[final_val_idx],
        #     env_labels[final_val_idx]
        # )
        
        # # Create test data using a different state for distribution shift
        # test_data_source = ACSDataSource(survey_year='2021', horizon='1-Year', survey='person')
        # test_raw_data = test_data_source.get_data(states=["WA", "OR", "NV", "NM", "CO", "UT", "AZ", "HI", "AK", "ID"], download=True)  # Different state for test
        
        # # Convert test data to numpy arrays
        # test_features, test_labels, test_group = ACSIncomeNew.df_to_numpy(test_raw_data)
        
        # # Create test environments with same mapping
        # test_male_high_income_idx = np.where((test_group == 1) & (test_labels == 1))[0]
        # test_male_low_income_idx = np.where((test_group == 1) & (test_labels == 0))[0]
        # test_female_high_income_idx = np.where((test_group == 2) & (test_labels == 1))[0]
        # test_female_low_income_idx = np.where((test_group == 2) & (test_labels == 0))[0]
        
        # test_male_low_income_idx = np.random.choice(test_male_low_income_idx, size=8000, replace=False)
        # test_male_high_income_idx = np.random.choice(test_male_high_income_idx, size=2000, replace=False)
        # test_female_low_income_idx = np.random.choice(test_female_low_income_idx, size=2000, replace=False)
        # test_female_high_income_idx = np.random.choice(test_female_high_income_idx, size=8000, replace=False)

        # # Combine test environments
        # test_all_indices = np.concatenate([
        #     test_male_low_income_idx, test_male_high_income_idx, 
        #     test_female_low_income_idx, test_female_high_income_idx
        # ])
        
        # test_env_labels = np.concatenate([
        #     np.zeros(len(test_male_low_income_idx)),      # Env 0
        #     np.ones(len(test_male_high_income_idx)),       # Env 1
        #     np.full(len(test_female_low_income_idx), 2),   # Env 2
        #     np.full(len(test_female_high_income_idx), 3)   # Env 3
        # ])
        
        # # Shuffle test data
        # test_shuffle_indices = np.random.permutation(len(test_all_indices))
        # test_all_indices = test_all_indices[test_shuffle_indices]
        # test_env_labels = test_env_labels[test_shuffle_indices]
        
        # # Convert test data to tensors
        # test_data = (
        #     torch.FloatTensor(test_features[test_all_indices]),
        #     torch.LongTensor(test_labels[test_all_indices]),
        #     torch.LongTensor(test_env_labels)
        # )
        
        # print(f"\nTest environment distribution:")
        # for env_id in range(4):
        #     env_mask = (test_data[2] == env_id)
        #     env_count = torch.sum(env_mask).item()
        #     env_features = test_data[0][env_mask]
        #     env_targets = test_data[1][env_mask]
        #     high_income_count = torch.sum(env_targets == 1).item()
        #     low_income_count = torch.sum(env_targets == 0).item()
        #     print(f"Env {env_id}: {env_count} samples (High Income: {high_income_count}, Low Income: {low_income_count})")
        
        # # Create directory and save datasets
        # four_env_dir = self._create_data_dir('four_env_synthetic_folktables')
        # torch.save(train_data, os.path.join(four_env_dir, 'train.pt'))
        # torch.save(val_data, os.path.join(four_env_dir, 'val.pt'))
        # torch.save(test_data, os.path.join(four_env_dir, 'test.pt'))
        
        # print(f"Four-Environment Folktables dataset created")
        # print(f"Train: {len(train_data[0])} samples, Val: {len(val_data[0])} samples, Test: {len(test_data[0])} samples")
        
        # # Print final environment distribution for all splits
        # for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        #     env_counts = torch.bincount(split_data[2], minlength=4)
        #     total_samples = len(split_data[0])
        #     percentages = [f"{count.item()}/{total_samples} ({100*count.item()/total_samples:.1f}%)" 
        #                   for count in env_counts]
        #     print(f"{split_name} environment distribution: Env0={percentages[0]}, Env1={percentages[1]}, "
        #           f"Env2={percentages[2]}, Env3={percentages[3]}")
            
        #     # Print label distribution per environment for this split
        #     for env_id in range(4):
        #         env_mask = (split_data[2] == env_id)
        #         if env_mask.sum() > 0:
        #             env_targets = split_data[1][env_mask]
        #             high_income = torch.sum(env_targets == 1).item()
        #             low_income = torch.sum(env_targets == 0).item()
        #             total_env = env_mask.sum().item()
        #             print(f"  {split_name} Env {env_id}: High Income={high_income}/{total_env} ({100*high_income/total_env:.1f}%), "
        #                   f"Low Income={low_income}/{total_env} ({100*low_income/total_env:.1f}%)") 