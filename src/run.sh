#!/bin/bash
# chmod +x run.sh
seed=1804
python exp-mnist.py  --seed $seed --trainer ERM > ERM_log.txt
python exp-mnist.py  --seed $seed --reg_name IRM --trainer ERM > IRM_log.txt
python exp-mnist.py  --seed $seed --trainer groupDRO > groupDRO_log.txt
