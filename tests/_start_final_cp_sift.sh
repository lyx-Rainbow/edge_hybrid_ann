#!/bin/bash
set -e
source /home/lyx/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
mkdir -p logs
rm -f logs/final_cp_ext4.done
nohup setsid python run_final_cp_ext4.py --dataset sift1m --method all > logs/final_cp_sift1m.log 2>&1 &
echo $! > logs/final_cp_sift1m.pid
touch logs/final_cp_sift1m.started