#!/bin/bash

echo "Check server connection"
sh connection_check.sh

echo "Start head"
sh start_head.sh

echo "Start workers"
parallel-ssh -h workers.txt -P -I < start_worker.sh

echo "Start training"
export PATH="/home/medipixel/anaconda3/bin:$PATH"
source activate opensim-rl
python ../ppo.py
