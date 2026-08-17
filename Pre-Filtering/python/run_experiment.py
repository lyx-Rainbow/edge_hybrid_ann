#!/usr/bin/env python3
"""
run_experiment.py — Pre-Filtering experiment orchestration script.

Responsibilities:
  1. Preprocess (train_mds.pkl → train_access.npy) if needed
  2. Call C++ binary (./build/prefiltering bench|search) via subprocess
  3. Compute Recall@k metrics
  4. Output summary results to 4_Results/Pre-Filtering/

Usage:
    python python/run_experiment.py --dataset sift1m_100k
    python python/run_experiment.py --dataset sift1m_100k --filter "AND 0 24"
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

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent

BUILTIN_DEFAULTS = {
    "arxiv_small": {
        "dataset": "arxiv_small", "d": 384, "n_labels": 100,
        "k": 10, "num_warmup": 10,
    },
    "yfcc100m_small": {
        "dataset": "yfcc100m_small", "d": 192, "n_labels": 1000,
        "k": 10, "num_warmup": 10,
    },
    "arxiv": {
        "dataset": "arxiv", "d": 384, "n_labels": 100,
        "k": 10, "num_warmup": 20,
    },
    "yfcc100m": {
        "dataset": "yfcc100m", "d": 192, "n_labels": 1000,
        "k": 10, "num_warmup": 20,
    },
    "sift1m": {
        "dataset": "sift1m", "d": 128, "n_labels": 100,
        "k": 10, "num_warmup": 20,
    },
    "gist1m": {
        "dataset": "gist1m", "d": 960, "n_labels": 100,
        "k": 10, "num_warmup": 20,
    },
    "wit": {
        "dataset": "wit", "d": 384, "n_labels": 1000,
        "k": 10, "num_warmup": 20,
    },
    # ── 100K datasets ──
    "sift1m_100k": {
        "dataset": "sift1m_100k", "d": 128, "n_labels": 100,
        "k": 10, "num_warmup": 20,
    },
    "yfcc100m_100k": {
        "dataset": "yfcc100m_100k", "d": 192, "n_labels": 1000,
        "k": 10, "num_warmup": 20,
    },
    "arxiv_100k": {
        "dataset": "arxiv_100k", "d": 384, "n_labels": 100,
        "k": 10, "num_warmup": 20,
    },
    "gist1m_100k": {
        "dataset": "gist1m_100k", "d": 960, "n_labels": 100,
        "k": 10, "num_warmup": 20,
    },
    "wit_100k": {
        "dataset": "wit_100k", "d": 384, "n_labels": 1000,
        "k": 10, "num_warmup": 20,
    },
}


def load_config(config_path, dataset):
    if config_path:
        with open(config_path) as f:
            return json.load(f)
    if dataset in BUILTIN_DEFAULTS:
        return dict(BUILTIN_DEFAULTS[dataset])
    raise ValueError(f"No defaults for dataset '{dataset}'; use --config")


def compute_recall(pred_ids, gt_ids, k):
    valid = set(int(i) for i in gt_ids[:k] if i >= 0)
    if not valid:
        return 1.0
    return len(set(pred_ids[:k]) & valid) / len(valid)


def main():
    parser = argparse.ArgumentParser(description="Pre-Filtering experiment runner")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="1_Data")
    parser.add_argument("--output_dir", type=str, default="4_Results/Pre-Filtering")
    parser.add_argument("--binary", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--batch_query", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.dataset)
    if args.k is not None:
        cfg["k"] = args.k

    k = cfg["k"]
    gt_dir = Path(args.data_dir) / "ground_truth" / args.dataset

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

    # ── Binary path ──
    if args.binary is None:
        args.binary = str(PROJ_ROOT / "Pre-Filtering" / "build" / "prefiltering")
    if not os.path.exists(args.binary):
        print(f"ERROR: C++ binary not found at {args.binary}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine output filename ──
    if args.filter:
        safe = args.filter.replace(" ", "_")
        results_path = output_dir / f"prefiltering_{args.dataset}_cp_{safe}.json"
    else:
        results_path = output_dir / f"prefiltering_{args.dataset}_sl.json"

    # ── Construct command ──
    cmd = [
        args.binary, "bench",
        "--train_vecs", str(gt_dir / "train_vecs.npy"),
        "--train_access", str(access_path),
        "--queries", str(gt_dir / "query_vecs.npy"),
        "--query_labels", str(gt_dir / "query_labels.npy"),
        "--k", str(k),
        "--output", str(results_path),
    ]
    if args.filter:
        cmd.extend(["--filter", args.filter])
    if args.batch_query:
        cmd.append("--batch-query")
    if args.profile:
        cmd.append("--profile")

    print(f"\n{'='*60}")
    print(f"Pre-Filtering: {args.dataset}")
    print(f"  Mode: {'complex-predicate' if args.filter else 'single-label'}")
    if args.filter:
        print(f"  Filter: {args.filter}")
    print(f"{'='*60}")
    print(f"  Running: {' '.join(cmd)}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJ_ROOT))
    elapsed = time.perf_counter() - t0

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"ERROR: C++ binary returned {result.returncode}")
        sys.exit(1)

    # ── Compute recall ──
    if results_path.exists():
        with open(results_path) as f:
            res = json.load(f)

        if args.filter:
            safe = args.filter.replace(" ", "_")
            gt_path = gt_dir / "complex_predicate" / f"gt_{safe}.npy"
        else:
            gt_path = gt_dir / "ground_truth.npy"

        if gt_path.exists():
            gt = np.load(gt_path)
            recalls = []
            for q, qr in enumerate(res["queries"]):
                r = compute_recall(qr["labels"], gt[q], k)
                recalls.append(r)

            recalls = np.array(recalls)
            print(f"\n  Recall@{k}:")
            print(f"    Mean:   {np.mean(recalls):.4f}")
            print(f"    Median: {np.median(recalls):.4f}")
            print(f"    P90:    {np.percentile(recalls, 90):.4f}")
            print(f"    Min:    {np.min(recalls):.4f}")
            print(f"    ==1.0:  {np.sum(recalls == 1.0)}/{len(recalls)}")
        else:
            print(f"Warning: GT not found at {gt_path}, skipping recall")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
