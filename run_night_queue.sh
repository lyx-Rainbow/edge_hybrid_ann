#!/bin/bash
# Night queue (run once, idempotent):
#   Curator incremental -> SPANN incremental -> frontier report -> figures
#   -> cleanup of intermediate _sl/_cp JSONs.
# Each phase continues even if the previous one reports a failure; inspect the
# log to see which combos failed.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines || exit 1

echo "=== PHASE Curator ==="
python run_incremental.py --method Curator || echo "PHASE_CURATOR_FAILED"

echo "=== PHASE SPANN ==="
python run_incremental.py --method SPANN || echo "PHASE_SPANN_FAILED"

echo "=== PHASE report ==="
python tests/report_frontier.py || echo "PHASE_REPORT_FAILED"

echo "=== PHASE figures ==="
python 5_Plot/fig_sl_qps_recall_100k.py || echo "FIG_SL_FAILED"
python 5_Plot/fig_cp_qps_recall_100k.py || echo "FIG_CP_FAILED"
python 5_Plot/fig_sl_qps_recall_by_bucket.py || echo "FIG_BUCKET_FAILED"
python 5_Plot/fig_memory_100k.py || echo "FIG_MEMORY_FAILED"

echo "=== PHASE cleanup (intermediate _sl/_cp JSONs) ==="
rm -f 4_Results/Curator/_sl_*.json 4_Results/Curator/_cp_*.json \
      4_Results/DiskIVF/_sl_*.json 4_Results/DiskIVF/_cp_*.json \
      4_Results/SPANN/_sl_*.json 4_Results/SPANN/_cp_*.json \
      4_Results/Pre-Filtering/_sl_*.json 4_Results/Pre-Filtering/_cp_*.json
echo "NIGHT_QUEUE_DONE"
