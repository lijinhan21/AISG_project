# Exploring the Effect of Environment Partitioning and Data Balancing on Out-of-Distribution Generalization

《人工智能安全于治理技术前沿》大作业：如何构建“好”的学习数据？探究环境划分与数据平衡在分布外泛化任务中的效果

For details of the project, please refer to the [project report](./report.pdf).

## Installation

```
conda create -n aigsp python=3.9
pip install -r requirements.txt
```

## Usage

To run all experiments, you can use the provided script:
```
sh run.sh
```

This includes:
- (1) Training model on two datasets (ColoredMNIST and Synthetic Folktables) using different algorithms (ERM, IRM, REx, InvRat, groupDRO, ChiSquareDRO), and evaluating in different OOD scenarios. 
- (2) Using manually extracted features for environment partitioning.
- (3) Using the proposed data balancing method to balance the training data (ReweightedERM algorithm).
- (4) Using VAE to automatically extract features for environment partitioning and data balancing.

## Coding Structure

Data generation scripts and data loaders are in `src/datasets` directory.
The main training and evaluation script is `src/exp.py`.

All implemented algorithms are in `src/model.py` and `src/trainer.py`.
- `ERM`: `ERM` class in `src/trainer.py` and `erm_bce_loss` class in `src/model.py`.
- `IRM`: `bce_loss` and `IRM` classes in `src/model.py`.
- `REx`: `REx` class in `src/model.py`.
- `InvRat`: `InvRat` class in `src/trainer.py` and `InvRat` class in `src/model.py`.
- `groupDRO`: `groupDRO` class in `src/model.py`.
- `ChiSquareDRO`: `ChiSquareDRO` class in `src/model.py`.
- `ReweightedERM`: `CategoryReweightedERM` class in `src/trainer.py`.
- `VAE`: `VAETrainer` class in `src/trainer.py`, and `VAE` and `VAELoss` class in `src/model.py`.

Plotting training curves and results is done in `src/plot/plot.ipynb`.

---

This repository is based on the codebase of AISG hw2. 