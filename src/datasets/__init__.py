# -*- coding: utf-8 -*-

"""
Datasets package for AIGS project.
Contains various dataset implementations for IRM experiments.
"""

from .colored_mnist import ColoredMNIST
from .category_based_colored_mnist import CategoryBasedColoredMNIST
from .four_env_colored_mnist import FourEnvColoredMNIST
from .synthetic_folktables import SyntheticFolktablesDataset
from .category_based_synthetic_folktables import CategoryBasedSyntheticFolktables
from .four_env_synthetic_folktables import FourEnvSyntheticFolktables
from .four_env_synthetic_folktables2 import FourEnvSyntheticFolktables2
from .utils import color_grayscale_arr, plot_dataset_digits

__all__ = [
    'ColoredMNIST', 
    'CategoryBasedColoredMNIST', 
    'FourEnvColoredMNIST',
    'FolktablesDataset', 
    'SyntheticFolktablesDataset', 
    'CategoryBasedSyntheticFolktables', 
    'FourEnvSyntheticFolktables',
    'FourEnvSyntheticFolktables2',
    'color_grayscale_arr',
    'plot_dataset_digits'
] 