#!/bin/bash
# chmod +x run.sh
seed=1804

# Baselines
sh run_cmnist.sh
sh run_synthetic_folktables.sh

# VAE experiments with 5D latent space and categorization
# For ColoredMNIST
python exp-vae.py --seed $seed --task ColoredMNIST --hidden_dim 128 64 --latent_dim 5 --save_model > logs/VAE_colored_mnist_log.txt
python categorize_samples.py --task ColoredMNIST
python exp.py --seed $seed --trainer ERM --task CategoryBasedColoredMNIST --hidden_dim 64 16
python exp.py --seed $seed --trainer groupDRO --task CategoryBasedColoredMNIST --hidden_dim 64 16
python exp.py --seed $seed --trainer ChiSquareDRO --task CategoryBasedColoredMNIST --hidden_dim 64 16
python exp.py --seed $seed --trainer ERM --task CategoryBasedColoredMNIST --hidden_dim 64 16 --reg_name IRM
python exp.py --seed $seed --trainer ERM --task CategoryBasedColoredMNIST --hidden_dim 64 16 --reg_name REx
python exp.py --seed $seed --trainer InvRat --task CategoryBasedColoredMNIST --hidden_dim 64 16 --reg_name InvRat --reg_lambda 5

# For SyntheticFolktables
python exp-vae.py --seed $seed --task SyntheticFolktables --hidden_dim 128 64 --latent_dim 5 --save_model > logs/VAE_synthetic_folktables_log.txt
python categorize_samples.py --task SyntheticFolktables
python exp.py --seed $seed --trainer ERM --task CategoryBasedSyntheticFolktables --hidden_dim 64 16
python exp.py --seed $seed --trainer groupDRO --task CategoryBasedSyntheticFolktables --hidden_dim 64 16
python exp.py --seed $seed --trainer ChiSquareDRO --task CategoryBasedSyntheticFolktables --hidden_dim 64 16
python exp.py --seed $seed --trainer ERM --task CategoryBasedSyntheticFolktables --hidden_dim 64 16 --reg_name IRM
python exp.py --seed $seed --trainer ERM --task CategoryBasedSyntheticFolktables --hidden_dim 64 16 --reg_name REx
python exp.py --seed $seed --trainer InvRat --task CategoryBasedSyntheticFolktables --hidden_dim 64 16 --reg_name InvRat --reg_lambda 5


# Manually Divided Four Environment experiments
sh run_fourenv_cmnist.sh
sh run_fourenv_synthetic_folktables.sh