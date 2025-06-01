# -*- coding: utf-8 -*-

"""
Utility functions for dataset processing and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt


def color_grayscale_arr(arr, red=True):
    """
    Converts grayscale image to either red or green.
    
    Args:
        arr: 2D numpy array representing grayscale image
        red: bool, if True colors red, otherwise green
        
    Returns:
        3D numpy array with RGB channels
    """
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr,
                              np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                              arr,
                              np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr


def plot_dataset_digits(dataset):
    """
    Plot sample digits from a dataset for visualization.
    
    Args:
        dataset: Dataset object with indexable items returning (image, label) tuples
    """
    fig = plt.figure(figsize=(13, 8))
    columns = 6
    rows = 3
    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns * rows):
        img, label = dataset[i]
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Label: " + str(label))  # set title
        plt.imshow(img)

    plt.show()  # finally, render the plot 