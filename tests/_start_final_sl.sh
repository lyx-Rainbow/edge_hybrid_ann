#!/bin/bash
set -e
source /home/lyx/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
mkdir -p logs
rm -f logs/final_sl_ext4.done
nohup setsid python run_final_sl_ext4.py --datasets sift1m gist1m arxiv yfcc100m wit --methods curator diskivf spann prefilter > logs/final_sl_ext4.log 2>&1 &
echo $! > logs/final_sl_ext4.pid
touch logs/final_sl_ext4.started