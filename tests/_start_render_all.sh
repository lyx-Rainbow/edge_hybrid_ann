#!/bin/bash
set -e
source /home/lyx/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
mkdir -p logs
nohup setsid bash render_all_figures.sh > logs/render_all.log 2>&1 &
echo $! > logs/render_all.pid
touch logs/render_all.started