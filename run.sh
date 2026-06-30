#!/bin/bash
# ============================================================
# One-command experiment runner.
#
# Usage:
#   bash run.sh <index> <dataset> [--sweep]
#
# <index>:   curator | pre-filter | diskivf | spann
# <dataset>: yfcc100m | arxiv | yfcc100m_small | arxiv_small
# --sweep:   run parameter sweep instead of single-run benchmark
#
# Examples:
#   bash run.sh curator yfcc100m              # Curator full benchmark
#   bash run.sh diskivf yfcc100m_small        # DiskIVF micro benchmark
#   bash run.sh spann yfcc100m_small --sweep  # SPANN micro sweep
#   bash run.sh pre-filter arxiv_small --sweep  # PreFiltering micro sweep
# ============================================================
set -e

INDEX="$1"
DATASET="$2"
MODE="${3:-}"

if [ -z "$INDEX" ] || [ -z "$DATASET" ]; then
    echo "Usage: bash run.sh <index> <dataset> [--sweep]"
    echo "  index:   curator | pre-filter | diskivf | spann"
    echo "  dataset: yfcc100m | arxiv | yfcc100m_small | arxiv_small"
    exit 1
fi

# ---- Map index name to paths ----
case "$INDEX" in
    curator)
        RUNNER="Curator/run_curator.py"
        SWEEP_RUNNER="Curator/run_curator_sweep.py"
        CONFIG="3_Config/Curator/curator_${DATASET}.json"
        # Use reduced sweep for full datasets (faster), full sweep for small
        if echo "$DATASET" | grep -q "_small"; then
            SWEEP_CONFIG="3_Config/Curator/sweep.json"
        else
            SWEEP_CONFIG="3_Config/Curator/sweep_full.json"
        fi
        ;;
    pre-filter)
        RUNNER="Pre-Filtering/run_prefiltering.py"
        SWEEP_RUNNER="Pre-Filtering/run_prefiltering_sweep.py"
        CONFIG="3_Config/Pre-Filtering/prefiltering_${DATASET}.json"
        SWEEP_CONFIG="3_Config/Pre-Filtering/sweep.json"
        ;;
    diskivf)
        RUNNER="DiskIVF-PostFiltering/run_diskivf.py"
        SWEEP_RUNNER="DiskIVF-PostFiltering/run_diskivf_sweep.py"
        CONFIG="3_Config/DiskIVF-PostFiltering/diskivf_${DATASET}.json"
        SWEEP_CONFIG="3_Config/DiskIVF-PostFiltering/sweep.json"
        ;;
    spann)
        RUNNER="SPANN-PostFiltering/run_spann.py"
        SWEEP_RUNNER="SPANN-PostFiltering/run_spann_sweep.py"
        CONFIG="3_Config/SPANN-PostFiltering/spann_${DATASET}.json"
        SWEEP_CONFIG="3_Config/SPANN-PostFiltering/sweep.json"
        ;;
    *)
        echo "Unknown index: $INDEX"
        echo "Valid: curator | pre-filter | diskivf | spann"
        exit 1
        ;;
esac

# ---- Check config exists ----
if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG"
    echo "Create it first, or check dataset name."
    exit 1
fi

# ---- Run ----
echo "============================================================"
echo "  Index:   $INDEX"
echo "  Dataset: $DATASET"
echo "  Config:  $CONFIG"
echo "============================================================"

if [ "$MODE" = "--sweep" ]; then
    echo "  Mode:    sweep"
    echo "  Sweep:   $SWEEP_CONFIG"
    echo "============================================================"
    python "$SWEEP_RUNNER" --config "$CONFIG" --sweep "$SWEEP_CONFIG"
else
    echo "  Mode:    single-run benchmark"
    echo "============================================================"
    python "$RUNNER" --config "$CONFIG"
fi
