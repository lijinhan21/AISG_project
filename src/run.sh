#!/bin/bash
# chmod +x run.sh
seed=1804

# Baselines
python exp.py --seed $seed --trainer ERM --task ColoredMNIST --hidden_dim 64 16 > logs/ERM_log.txt
python exp.py --seed $seed --trainer groupDRO --task ColoredMNIST --hidden_dim 64 16 > logs/groupDRO_log.txt
python exp.py --seed $seed --trainer ChiSquaredDRO --task ColoredMNIST --hidden_dim 64 16 > logs/ChiSquaredDRO_log.txt
python exp.py --seed $seed --trainer ERM --task ColoredMNIST --hidden_dim 64 16 --reg_name IRM > logs/IRM_log.txt
python exp.py --seed $seed --trainer ERM --task ColoredMNIST --hidden_dim 64 16 --reg_name REx > logs/REx_log.txt
python exp.py --seed $seed --trainer InvRat --task ColoredMNIST --hidden_dim 64 16 --reg_name InvRat --reg_lambda 5 > logs/InvRat_log.txt

python exp.py --seed $seed --trainer ERM --task SyntheticFolktables --hidden_dim 8 > logs/ERM_synthetic_folktables.txt
python exp.py --seed $seed --trainer groupDRO --task SyntheticFolktables --hidden_dim 8 > logs/groupDRO_synthetic_folktables.txt
python exp.py --seed $seed --trainer ChiSquaredDRO --task SyntheticFolktables --hidden_dim 8 > logs/ChiSquaredDRO_synthetic_folktables.txt
python exp.py --seed $seed --trainer ERM --task SyntheticFolktables --hidden_dim 8 --reg_name IRM > logs/IRM_synthetic_folktables.txt
python exp.py --seed $seed --trainer ERM --task SyntheticFolktables --hidden_dim 8 --reg_name REx > logs/REx_synthetic_folktables.txt
python exp.py --seed $seed --trainer InvRat --task SyntheticFolktables --hidden_dim 8 --reg_name InvRat --reg_lambda 5 > logs/InvRat_synthetic_folktables.txt

# VAE experiments with 5D latent space and categorization
# For ColoredMNIST
python exp-vae.py --seed $seed --task ColoredMNIST --hidden_dim 128 64 --latent_dim 5 --save_model > logs/VAE_colored_mnist_log.txt

# For SyntheticFolktables
python exp-vae.py --seed $seed --task SyntheticFolktables --hidden_dim 128 64 --latent_dim 5 --save_model > logs/VAE_synthetic_folktables_log.txt

# 
python exp.py --seed $seed --trainer ERM --task FourEnvColoredMNIST --hidden_dim 64 16
python exp.py --seed $seed --trainer CategoryReweightedERM --task FourEnvColoredMNIST --hidden_dim 64 16

python exp.py --seed $seed --trainer ERM --task FourEnvSyntheticFolktables --hidden_dim 8
python exp.py --seed $seed --trainer CategoryReweightedERM --task FourEnvSyntheticFolktables --hidden_dim 8