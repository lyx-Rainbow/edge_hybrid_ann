"""
DiskIVF-PostFiltering experiment orchestration script.

Responsibilities:
  1. Preprocess (train_mds.pkl → train_access.npy) if needed
  2. Generate JSON config file
  3. Call C++ binary (./build/diskivf bench ...) via subprocess
  4. Compute Recall@k metrics
  5. Output summary results to 4_Results/DiskIVF-PostFiltering/

Usage:
    python python/run_experiment.py --dataset arxiv_small
    python python/run_experiment.py --dataset yfcc100m_small --nlist 32 --nprobe 8
    python python/run_experiment.py --dataset arxiv_small --config /tmp/diskivf_config.json
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Project root relative to this script
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent

# Built-in defaults per dataset
BUILTIN_DEFAULTS = {
    "arxiv_small": {
        "dataset": "arxiv_small", "d": 384,
        "nlist": 16, "nprobe": 8, "clus_niter": 20,
        "k": 10, "num_warmup": 10,
    },
    "yfcc100m_small": {
        "dataset": "yfcc100m_small", "d": 192,
        "nlist": 16, "nprobe": 8, "clus_niter": 20,
        "k": 10, "num_warmup": 10,
    },
    "arxiv": {
        "dataset": "arxiv", "d": 384,
        "nlist": 64, "nprobe": 16, "clus_niter": 20,
        "k": 10, "num_warmup": 20,
    },
    "yfcc100m": {
        "dataset": "yfcc100m", "d": 192,
        "nlist": 64, "nprobe": 16, "clus_niter": 20,
        "k": 10, "num_warmup": 20,
    },
}


def load_config(config_path, dataset):
    """Load config from JSON file or use built-in defaults."""
    if config_path:
        with open(config_path) as f:
            return json.load(f)
    if dataset in BUILTIN_DEFAULTS:
        return dict(BUILTIN_DEFAULTS[dataset])
    raise ValueError(f"No defaults for dataset '{dataset}'; use --config")


def compute_recall(pred_ids, gt_ids, k):
    """Compute Recall@k for a single query."""
    valid = set(int(i) for i in gt_ids[:k] if i >= 0)
    if not valid:
        return 1.0
    return len(set(pred_ids[:k]) & valid) / len(valid)


def run_experiment(dataset, config_path=None, nlist=None, nprobe=None,
                   data_dir="1_Data", output_dir="4_Results/DiskIVF-PostFiltering",
                   disk_dir=None, profile=False):
    """Main experiment workflow."""
    # ── Load config ──
    cfg = load_config(config_path, dataset)
    if nlist is not None:
        cfg["nlist"] = nlist
    if nprobe is not None:
        cfg["nprobe"] = nprobe

    k = cfg.get("k", 10)
    gt_dir = Path(data_dir) / "ground_truth" / dataset

    # ── Preprocess: ensure train_access.npy exists ──
    access_path = gt_dir / "train_access.npy"
    if not access_path.exists():
        print("Preprocessing: converting train_mds.pkl → train_access.npy ...")
        with open(gt_dir / "train_mds.pkl", "rb") as f:
            mds = pickle.load(f)
        pairs = []
        for vid, tids in enumerate(mds):
            for tid in tids:
                pairs.append([vid, tid])
        np.save(access_path, np.array(pairs, dtype=np.int32))
        print(f"  Saved {len(pairs)} access pairs")

    # ── Set disk directory ──
    if disk_dir is None:
        disk_dir = str(Path(output_dir) / "disk_data" / dataset)

    # ── Write config JSON ──
    cfg_path = Path(output_dir) / f"config_{dataset}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Paths ──
    binary = PROJ_ROOT / "DiskIVF-PostFiltering" / "build" / "diskivf"
    if not binary.exists():
        print(f"ERROR: C++ binary not found at {binary}")
        print("  Build it first: cd DiskIVF-PostFiltering/build && cmake .. && make -j")
        sys.exit(1)

    # ── Single-label queries ──
    sl_output = Path(output_dir) / f"diskivf_{dataset}_sl.json"
    print(f"\n{'='*60}")
    print(f"DiskIVF-PostFiltering: {dataset} — Single-label queries")
    print(f"{'='*60}")

    sl_cmd = [
        str(binary), "bench",
        "--train_vecs", str(gt_dir / "train_vecs.npy"),
        "--train_access", str(access_path),
        "--queries", str(gt_dir / "query_vecs.npy"),
        "--query_labels", str(gt_dir / "query_labels.npy"),
        "--config", str(cfg_path),
        "--k", str(k),
        "--disk_dir", disk_dir,
        "--output", str(sl_output),
    ]
    if profile:
        sl_cmd.append("--profile")

    print(f"  Running: {' '.join(sl_cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(sl_cmd, capture_output=True, text=True, cwd=str(PROJ_ROOT))
    t_sl = time.perf_counter() - t0
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"ERROR: C++ binary returned {result.returncode}")
        sys.exit(1)

    # ── Compute SL Recall ──
    if sl_output.exists():
        with open(sl_output) as f:
            sl_results = json.load(f)
        gt = np.load(gt_dir / "ground_truth.npy")

        recalls = []
        for q, qr in enumerate(sl_results["queries"]):
            r = compute_recall(qr["labels"], gt[q], k)
            recalls.append(r)

        recalls = np.array(recalls)
        print(f"\n  Single-label Recall@{k}:")
        print(f"    Mean:   {np.mean(recalls):.4f}")
        print(f"    Median: {np.median(recalls):.4f}")
        print(f"    P90:    {np.percentile(recalls, 90):.4f}")
        print(f"    Min:    {np.min(recalls):.4f}")
        print(f"    ==1.0:  {np.sum(recalls == 1.0)}/{len(recalls)}")

    # ── Complex-predicate queries ──
    cp_dir = gt_dir / "complex_predicate"
    if cp_dir.exists():
        print(f"\n{'='*60}")
        print(f"DiskIVF-PostFiltering: {dataset} — Complex-predicate queries")
        print(f"{'='*60}")

        with open(cp_dir / "filters.json") as f:
            filters_info = json.load(f)

        cp_filters = filters_info["filters"]
        cp_vecs = np.load(cp_dir / "query_vecs.npy").astype(np.float32)

        cp_all_recalls = []
        cp_results = {}

        for fi, formula in enumerate(cp_filters):
            safe = formula.replace(" ", "_")
            gt_path = cp_dir / f"gt_{safe}.npy"
            if not gt_path.exists():
                print(f"  [{fi+1}/{len(cp_filters)}] Skip '{formula}': GT not found")
                continue

            cp_gt = np.load(gt_path)
            cp_output = Path(output_dir) / f"diskivf_{dataset}_cp_{safe}.json"

            cp_cmd = [
                str(binary), "bench",
                "--train_vecs", str(gt_dir / "train_vecs.npy"),
                "--train_access", str(access_path),
                "--queries", str(cp_dir / "query_vecs.npy"),
                "--config", str(cfg_path),
                "--k", str(k),
                "--disk_dir", disk_dir,
                "--filter", formula,
                "--output", str(cp_output),
            ]

            t0 = time.perf_counter()
            result = subprocess.run(cp_cmd, capture_output=True, text=True,
                                    cwd=str(PROJ_ROOT))
            elapsed = time.perf_counter() - t0

            if result.returncode != 0:
                print(f"  [{fi+1}/{len(cp_filters)}] ERROR on '{formula}'")
                print(result.stderr[-500:])
                continue

            # Compute Recall
            if cp_output.exists():
                with open(cp_output) as f:
                    cp_res = json.load(f)

                recs = []
                for q, qr in enumerate(cp_res["queries"]):
                    r = compute_recall(qr["labels"], cp_gt[q], k)
                    recs.append(r)
                recs = np.array(recs)

                cp_results[formula] = {
                    "avg_recall": float(np.mean(recs)),
                    "min_recall": float(np.min(recs)),
                    "time_s": float(elapsed),
                }
                cp_all_recalls.extend(recs.tolist())

                if (fi + 1) % 10 == 0 or fi + 1 == len(cp_filters):
                    print(f"  [{fi+1}/{len(cp_filters)}] '{formula}': "
                          f"Recall={np.mean(recs):.4f}, {elapsed:.1f}s")

        if cp_all_recalls:
            cp_all = np.array(cp_all_recalls)
            print(f"\n  CP Recall@{k} (all filters):")
            print(f"    Mean:   {np.mean(cp_all):.4f}")
            print(f"    Min:    {np.min(cp_all):.4f}")
            print(f"    ==1.0:  {np.sum(cp_all == 1.0)}/{len(cp_all)}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Experiment complete: {dataset}")
    print(f"  Results: {output_dir}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument("--nprobe", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default="1_Data")
    parser.add_argument("--output_dir", type=str,
                        default="4_Results/DiskIVF-PostFiltering")
    parser.add_argument("--disk_dir", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    run_experiment(
        args.dataset, args.config, args.nlist, args.nprobe,
        args.data_dir, args.output_dir, args.disk_dir, args.profile,
    )


if __name__ == "__main__":
    main()
