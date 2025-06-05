seed=1997
logging_folder="logs/FourEnv/SyntheticFolktables"
# create logging folder if it doesn't exist
mkdir -p $logging_folder
python exp.py --seed $seed --trainer ERM --task FourEnvSyntheticFolktables2 --hidden_dim 8 > $logging_folder/ERM_log.txt
python exp.py --seed $seed --trainer CategoryReweightedERM --task FourEnvSyntheticFolktables2 --hidden_dim 8 > $logging_folder/CategoryReweightedERM_log.txt
python exp.py --seed $seed --trainer groupDRO --task FourEnvSyntheticFolktables2 --hidden_dim 8  > $logging_folder/groupDRO_log.txt
python exp.py --seed $seed --trainer ChiSquareDRO --task FourEnvSyntheticFolktables2 --hidden_dim 8 > $logging_folder/ChiSquareDRO_log.txt
python exp.py --seed $seed --trainer ERM --reg_name IRM --task FourEnvSyntheticFolktables2 --hidden_dim 8 > $logging_folder/IRM_log.txt
python exp.py --seed $seed --trainer ERM --task FourEnvSyntheticFolktables2 --hidden_dim 8 --reg_name REx > $logging_folder/REx_log.txt
python exp.py --seed $seed --trainer InvRat --task FourEnvSyntheticFolktables2 --hidden_dim 8 --reg_name InvRat --reg_lambda 5 > $logging_folder/InvRat_log.txt