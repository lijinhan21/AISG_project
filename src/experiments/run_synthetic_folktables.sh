seed=2003
logging_folder="logs/TwoEnv/SyntheticFolktables"
mkdir -p $logging_folder

python exp.py --seed $seed --trainer ERM --task SyntheticFolktables --hidden_dim 8 > $logging_folder/ERM_log.txt
python exp.py --seed $seed --trainer groupDRO --task SyntheticFolktables --hidden_dim 8 > $logging_folder/groupDRO_log.txt
python exp.py --seed $seed --trainer ChiSquareDRO --task SyntheticFolktables --hidden_dim 8 > $logging_folder/ChiSquareDRO_log.txt
python exp.py --seed $seed --trainer ERM --task SyntheticFolktables --hidden_dim 8 --reg_name IRM > $logging_folder/IRM_log.txt
python exp.py --seed $seed --trainer ERM --task SyntheticFolktables --hidden_dim 8 --reg_name REx > $logging_folder/REx_log.txt
python exp.py --seed $seed --trainer InvRat --task SyntheticFolktables --hidden_dim 8 --reg_name InvRat --reg_lambda 5 > $logging_folder/InvRat_log.txt