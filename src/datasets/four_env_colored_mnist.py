import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

from .base import BaseMNISTDataset
from .utils import color_grayscale_arr

class FourEnvColoredMNIST(BaseMNISTDataset):
    """
    Four-Environment Colored MNIST dataset for testing IRM with 4 distinct environments.
    Creates environments based on the cross-product of Color (Red/Green) and Original Label range (0-4/5-9).
    
    Environment mapping:
    - Env 0: Red + Original Label 0-4
    - Env 1: Red + Original Label 5-9  
    - Env 2: Green + Original Label 0-4
    - Env 3: Green + Original Label 5-9

    Args:
        root: Root directory of dataset where FourEnvColoredMNIST/*.pt will exist
        env: Which environment to load ('train', 'val', 'test', or 'all_train')
        transform: A function/transform that takes in an PIL image and returns transformed version
        target_transform: A function/transform that takes in the target and transforms it
    """
    
    def __init__(self, root='./data', env='train', transform=None, target_transform=None):
        super().__init__(root, env, transform, target_transform)
        self.prepare_four_env_colored_mnist()
        
        if env in ['train', 'val', 'test']:
            self.data_label_env_tuples = self._load_prepared_data('four_env_colored_mnist', env)
        elif env == 'all_train':
            self.data_label_env_tuples = self._load_prepared_data('four_env_colored_mnist', 'all_train')
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train, val, test, and all_train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, env) where target is the binary label and env is the environment (0-3).
        """
        if isinstance(self.data_label_env_tuples, tuple):
            img = self.data_label_env_tuples[0][index]
            target = self.data_label_env_tuples[1][index]
            env = self.data_label_env_tuples[2][index]
        else:
            img, target, env = self.data_label_env_tuples[index]

        # Reshape image from flattened to 3x28x28 if needed
        if len(img.shape) == 1:
            img = img.reshape(3, 28, 28)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, env

    def prepare_four_env_colored_mnist(self):
        """Prepare the Four-Environment Colored MNIST dataset."""
        if self._check_prepared_data('four_env_colored_mnist', ['train.pt', 'val.pt', 'test.pt']):
            print('Four-Environment Colored MNIST dataset already exists')
            return

        print('Preparing Four-Environment Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        # Set up transform for data preparation
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
        ])

        train_set = []
        test_set = []
        
        # first generate color_mnist train set, while saving color info
        # then reassign env label based on color info and original label
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
            
            if color_red:
                environment = 0 if label < 5 else 1
            else:
                environment = 2 if label < 5 else 3

            if idx < 40000:
                train_set.append((data_transform(Image.fromarray(colored_arr)).reshape(-1), binary_label, environment))
        
        # for idx, (im, label) in enumerate(train_mnist):
        #     if idx % 10000 == 0:
        #         print(f'Converting image {idx}/{len(train_mnist)}')
        #     im_array = np.array(im)

        #     # Assign a binary label y to the image based on the digit
        #     binary_label = 0 if label < 5 else 1

        #     # Flip label with 20% probability
        #     if np.random.uniform() < 0.02:
        #         binary_label = binary_label ^ 1

        #     # Color the image either red or green according to its possibly flipped label
        #     color_red = binary_label == 0

        #     # Environment assignment based on original label range and color
        #     original_label_range = 0 if label < 5 else 1  # 0: digits 0-4, 1: digits 5-9
            
        #     # Flip the color with environment-specific probabilities for training data only
        #     if original_label_range == 0:  # Original label 0-4
        #         if np.random.uniform() < 0.1:  # 20% color flip for env 0 and 2
        #             color_red = not color_red
        #     else:  # Original label 5-9  
        #         if np.random.uniform() < 0.2:  # 40% color flip for env 1 and 3
        #             color_red = not color_red

        #     # Determine environment: 0=Red+0-4, 1=Red+5-9, 2=Green+0-4, 3=Green+5-9
        #     if color_red:
        #         environment = 0 if original_label_range == 0 else 1
        #     else:
        #         environment = 2 if original_label_range == 0 else 3

        #     colored_arr = color_grayscale_arr(im_array, red=color_red)
        #     train_set.append((data_transform(Image.fromarray(colored_arr)).reshape(-1), binary_label, environment))

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

        # Use original ColoredMNIST test data instead of creating new test data
        original_test_path = os.path.join(self.root, 'ColoredMNIST', 'test.pt')
        if os.path.exists(original_test_path):
            original_test_data = torch.load(original_test_path)
            test_features = original_test_data[0]
            test_labels = original_test_data[1]
            # Assign all test samples to environment -1 to distinguish from training environments
            test_env = torch.full_like(test_labels, -1, dtype=torch.long)
            test_set = (test_features, test_labels, test_env)
        else:
            raise FileNotFoundError(f"Original ColoredMNIST test data not found at {original_test_path}. "
                                   "Please ensure ColoredMNIST dataset is prepared first.")

        # Create directory and save datasets
        four_env_dir = self._create_data_dir('four_env_colored_mnist')
        torch.save(train_set, os.path.join(four_env_dir, 'train.pt'))
        torch.save(val_set, os.path.join(four_env_dir, 'val.pt'))
        torch.save(test_set, os.path.join(four_env_dir, 'test.pt'))
        
        print(f"Four-Environment Colored MNIST dataset created")
        print(f"Train: {len(train_set[0])} samples, Val: {len(val_set[0])} samples, Test: {len(test_set[0])} samples")
        
        # Print environment distribution
        for split_name, split_data in [('Train', train_set), ('Val', val_set)]:
            env_counts = torch.bincount(split_data[2], minlength=4)
            print(f"{split_name} environment distribution: {env_counts.tolist()}")
        print(f"Test environment: All samples assigned to environment -1 (original ColoredMNIST test data)") 