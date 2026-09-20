#!/bin/bash
set -e
source /home/lyx/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
mkdir -p logs
nohup setsid python run_final_cp_ext4.py --dataset arxiv --method all > logs/final_cp_arxiv.log 2>&1 &
echo $! > logs/final_cp_arxiv.pid
touch logs/final_cp_arxiv.started