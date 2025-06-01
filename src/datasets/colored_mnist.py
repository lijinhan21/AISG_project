# -*- coding: utf-8 -*-

"""
ColoredMNIST dataset implementations for IRM experiments.
"""

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

from .base import BaseMNISTDataset
from .utils import color_grayscale_arr


class ColoredMNIST(BaseMNISTDataset):
    """
    Colored MNIST dataset for testing IRM. 
    
    Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf
    Creates three environments (two training, one test) by randomly splitting 
    the MNIST dataset and transforming each example with color correlation.
    
    Args:
        root: Root directory of dataset where ColoredMNIST/*.pt will exist
        env: Which environment to load ('train1', 'train2', 'test', or 'all_train')
        transform: A function/transform that takes in an PIL image and returns transformed version
        target_transform: A function/transform that takes in the target and transforms it
    """
    
    def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
        super().__init__(root, env, transform, target_transform)
        self.prepare_colored_mnist()
        
        # Load data based on environment
        if self._check_prepared_data('ColoredMNIST', ['train.pt', 'val.pt', 'test.pt']):
            if env in ['train', 'val', 'test']:
                self.data_label_tuples = self._load_prepared_data('ColoredMNIST', env)
            elif env == 'all_train':
                self.data_label_tuples = self._load_prepared_data('ColoredMNIST', 'all_train')
            else:
                raise RuntimeError(f'{env} env unknown. Valid envs are train, val, test, and all_train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def prepare_colored_mnist(self):
        """Prepare the ColoredMNIST dataset from original MNIST."""
        if self._check_prepared_data('ColoredMNIST', ['train.pt', 'val.pt', 'test.pt']):
            print('Colored MNIST dataset already exists')
            return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        # Set up transform for data preparation
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
        ])

        train_set = []
        test_set = []
        
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with 20% probability
            if np.random.uniform() < 0.2:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability e that depends on the environment
            if idx < 20000:
                # 40% in the first training environment
                if np.random.uniform() < 0.4:
                    color_red = not color_red
            elif idx < 40000:
                # 10% in the second training environment
                if np.random.uniform() < 0.1:
                    color_red = not color_red
            else:
                # 90% in the test environment
                if np.random.uniform() < 0.9:
                    color_red = not color_red

            colored_arr = color_grayscale_arr(im_array, red=color_red)

            if idx < 20000:
                train_set.append((data_transform(Image.fromarray(colored_arr)).reshape(-1), binary_label, 0))
            elif idx < 40000:
                train_set.append((data_transform(Image.fromarray(colored_arr)).reshape(-1), binary_label, 1))
            else:
                test_set.append((data_transform(Image.fromarray(colored_arr)).reshape(-1), binary_label))

        # Split training data into train and validation
        train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42)

        # Convert to tensors
        train_set = list(zip(*train_set))
        train_set[0] = torch.stack(train_set[0])
        train_set[1] = torch.tensor(train_set[1], dtype=torch.long)
        train_set[2] = torch.tensor(train_set[2], dtype=torch.long)

        val_set = list(zip(*val_set))
        val_set[0] = torch.stack(val_set[0])
        val_set[1] = torch.tensor(val_set[1], dtype=torch.long)
        val_set[2] = torch.tensor(val_set[2], dtype=torch.long)

        test_set = list(zip(*test_set))
        test_set[0] = torch.stack(test_set[0])
        test_set[1] = torch.tensor(test_set[1], dtype=torch.long)

        # Create directory and save datasets
        colored_mnist_dir = self._create_data_dir('ColoredMNIST')
        torch.save(train_set, os.path.join(colored_mnist_dir, 'train.pt'))
        torch.save(val_set, os.path.join(colored_mnist_dir, 'val.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))
