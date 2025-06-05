seed=1997
logging_folder="logs/FourEnv/CMNIST"
# create logging folder if it doesn't exist
mkdir -p $logging_folder
python exp.py --seed $seed --trainer ERM --task FourEnvColoredMNIST --hidden_dim 64 16 --iteration 1200 > $logging_folder/ERM_log.txt
python exp.py --seed $seed --trainer CategoryReweightedERM --task FourEnvColoredMNIST --hidden_dim 64 16 --iteration 1200 > $logging_folder/CategoryReweightedERM_log.txt
python exp.py --seed $seed --trainer groupDRO --task FourEnvColoredMNIST --hidden_dim 64 16 --iteration 1200 > $logging_folder/groupDRO_log.txt
python exp.py --seed $seed --trainer ChiSquareDRO --task FourEnvColoredMNIST --hidden_dim 64 16 --iteration 1200 > $logging_folder/ChiSquareDRO_log.txt
python exp.py --seed $seed --trainer ERM --reg_name IRM --task FourEnvColoredMNIST --hidden_dim 64 16 --iteration 1200 > $logging_folder/IRM_log.txt
python exp.py --seed $seed --trainer ERM --task FourEnvColoredMNIST --hidden_dim 64 16 --reg_name REx --iteration 1200 > $logging_folder/REx_log.txt
python exp.py --seed $seed --trainer InvRat --task FourEnvColoredMNIST --hidden_dim 64 16 --reg_name InvRat --reg_lambda 5 --iteration 1200 > $logging_folder/InvRat_log.txt