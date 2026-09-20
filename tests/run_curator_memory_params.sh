#!/bin/bash
set -e
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
python tests/measure_curator_memory_params.py --datasets sift1m gist1m arxiv yfcc100m wit