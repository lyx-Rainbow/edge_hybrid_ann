#!/bin/bash
set -e
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
python 5_Plot/fig_final_sl_latency.py
python 5_Plot/fig_final_cp_latency.py
python 5_Plot/fig_final_latency_at_recall.py