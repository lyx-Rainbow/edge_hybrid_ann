#!/bin/bash
set -e
source /home/lyx/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
mkdir -p logs
nohup setsid python run_final_cp_ext4.py --dataset yfcc100m --method all > logs/final_cp_yfcc100m.log 2>&1 &
echo $! > logs/final_cp_yfcc100m.pid
touch logs/final_cp_yfcc100m.started