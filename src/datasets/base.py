# -*- coding: utf-8 -*-

"""
Base classes for dataset implementations.
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class BaseEnvironmentDataset(Dataset):
    """
    Base class for multi-environment datasets used in IRM experiments.
    
    This class provides common functionality for datasets that have multiple
    environments with different data distributions.
    """
    
    def __init__(self, root='./data', env='train', transform=None, target_transform=None):
        """
        Initialize the base dataset.
        
        Args:
            root: Root directory where data will be stored
            env: Environment to load ('train', 'val', 'test', 'all_train')
            transform: Optional transform to apply to inputs
            target_transform: Optional transform to apply to targets
        """
        self.root = root
        self.env = env
        self.transform = transform
        self.target_transform = target_transform
        self.data_label_tuples = None
        
    def __getitem__(self, index):
        """
        Get item by index. Should be implemented by subclasses.
        
        Args:
            index: Index of item to retrieve
            
        Returns:
            Tuple of (input, target) or (input, target, env)
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
        
    def __len__(self):
        """
        Return length of dataset.
        
        Returns:
            Length of dataset
        """
        if isinstance(self.data_label_tuples, tuple):
            return len(self.data_label_tuples[0])
        elif self.data_label_tuples is not None:
            return len(self.data_label_tuples)
        else:
            return 0
            
    def _create_data_dir(self, subdir_name):
        """
        Create data directory if it doesn't exist.
        
        Args:
            subdir_name: Name of subdirectory to create
            
        Returns:
            Path to created directory
        """
        data_dir = os.path.join(self.root, subdir_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir
        
    def _check_prepared_data(self, subdir_name, filenames):
        """
        Check if prepared data files exist.
        
        Args:
            subdir_name: Subdirectory name containing the data
            filenames: List of filenames to check for
            
        Returns:
            True if all files exist, False otherwise
        """
        data_dir = os.path.join(self.root, subdir_name)
        return all(os.path.exists(os.path.join(data_dir, filename)) 
                  for filename in filenames)
                  
    def _load_prepared_data(self, subdir_name, env):
        """
        Load prepared data from disk.
        
        Args:
            subdir_name: Subdirectory containing the data
            env: Environment to load
            
        Returns:
            Loaded data
        """
        if env == 'all_train':
            train_data = torch.load(os.path.join(self.root, subdir_name, 'train.pt'))
            val_data = torch.load(os.path.join(self.root, subdir_name, 'val.pt'))
            # Combine train and val data
            if isinstance(train_data, tuple) and isinstance(val_data, tuple):
                return tuple(torch.cat([train_data[i], val_data[i]], dim=0) 
                           for i in range(len(train_data)))
            else:
                return train_data + val_data
        else:
            return torch.load(os.path.join(self.root, subdir_name, f'{env}.pt'))


class BaseMNISTDataset(BaseEnvironmentDataset, datasets.VisionDataset):
    """
    Base class for MNIST-based datasets.
    """
    
    def __init__(self, root='./data', env='train', transform=None, target_transform=None):
        # Initialize both parent classes
        BaseEnvironmentDataset.__init__(self, root, env, transform, target_transform)
        datasets.VisionDataset.__init__(self, root, transform=transform, 
                                       target_transform=target_transform)


class BaseFolktablesDataset(BaseEnvironmentDataset):
    """
    Base class for Folktables-based datasets.
    """
    
    def __init__(self, root='./data', env='train', transform=None, target_transform=None):
        super().__init__(root, env, transform, target_transform) 