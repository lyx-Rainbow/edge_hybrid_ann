#!/usr/bin/env python3
"""
run_experiment.py — Orchestrate Curator benchmark: preprocess → run C++ → compute recall.

Usage:
    python run_experiment.py --dataset arxiv_small
    python run_experiment.py --dataset yfcc100m --k 10 --profile
"""
import argparse
import json
import numpy as np
import os
import pickle
import subprocess
import sys
import time

def compute_recall_at_k(gt_labels, result_labels, k):
    """Compute Recall@k for one query."""
    gt_set = set(gt_labels[:k])
    result_set = set(result_labels[:k])
    return len(gt_set & result_set) / k

def main():
    parser = argparse.ArgumentParser(description="Run Curator experiment")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=10, help="Number of results per query")
    parser.add_argument("--curator_bin", type=str, default=None, help="Path to curator binary")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--profile", action="store_true", help="Enable profiling output")
    parser.add_argument("--batch_query", action="store_true", help="Enable inter-query parallelism")
    parser.add_argument("--query_filters", type=str, default=None,
                        help="Per-query filter expressions file (one per line)")
    parser.add_argument("--n_queries", type=int, default=0,
                        help="Limit number of queries (0=all)")
    args = parser.parse_args()

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    curator_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, "1_Data", "ground_truth", args.dataset)
    if args.curator_bin is None:
        # Try build directory first, then look in PATH
        args.curator_bin = os.path.join(curator_root, "build", "curator")
        if not os.path.exists(args.curator_bin):
            args.curator_bin = "curator"  # hope it's on PATH

    os.makedirs(args.data_dir, exist_ok=True)

    # Step 1: Preprocess training data (.pkl -> .npy)
    print("=" * 60)
    print("Step 1: Preprocessing training data")
    print("=" * 60)
    mds_path = os.path.join(args.data_dir, "train_mds.pkl")
    access_path = os.path.join(args.data_dir, "train_access.npy")

    if os.path.exists(mds_path) and not os.path.exists(access_path):
        with open(mds_path, "rb") as f:
            train_mds = pickle.load(f)
        pairs = []
        for vid, tids in enumerate(train_mds):
            for tid in tids:
                pairs.append([vid, tid])
        pairs_arr = np.array(pairs, dtype=np.int32)
        np.save(access_path, pairs_arr)
        print(f"Converted {len(pairs)} access pairs: {mds_path} -> {access_path}")
    elif os.path.exists(access_path):
        print(f"Access pairs already exist: {access_path}")
    else:
        print(f"Warning: No train_mds.pkl or train_access.npy found at {args.data_dir}")

    # Step 2: Run C++ curator bench
    print("\n" + "=" * 60)
    print("Step 2: Running C++ Curator benchmark")
    print("=" * 60)

    train_vecs = os.path.join(args.data_dir, "train_vecs.npy")
    query_vecs = os.path.join(args.data_dir, "query_vecs.npy")
    query_labels = os.path.join(args.data_dir, "query_labels.npy")
    results_json = os.path.join(args.data_dir, "results.json")

    if not os.path.exists(query_labels):
        # Create default query labels (all unfiltered)
        qv = np.load(query_vecs)
        nq = qv.shape[0]
        np.save(query_labels, np.full(nq, -1, dtype=np.int32))
        print(f"Created default query labels: {nq} queries (all unfiltered)")

    cmd = [
        args.curator_bin, "bench",
        "--train_vecs", train_vecs,
        "--train_access", access_path if os.path.exists(access_path) else "",
        "--queries", query_vecs,
        "--query_labels", query_labels,
        "--k", str(args.k),
        "--output", results_json,
    ]

    # Remove empty --train_access if no access data
    if not os.path.exists(access_path):
        cmd = [a for a in cmd if a != "--train_access" and a != access_path]
        cmd = [a for a in cmd if a != ""]

    if args.config and os.path.exists(args.config):
        cmd += ["--config", args.config]
    if args.query_filters and os.path.exists(args.query_filters):
        cmd += ["--query_filters", args.query_filters]
    if args.batch_query:
        cmd.append("--batch-query")
    if args.profile:
        cmd.append("--profile")

    print(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: curator returned code {result.returncode}")
        print(result.stderr)
        sys.exit(1)
    print(f"C++ benchmark completed in {elapsed:.1f}s")

    # Step 3: Compute recall
    print("\n" + "=" * 60)
    print("Step 3: Computing recall metrics")
    print("=" * 60)

    if not os.path.exists(results_json):
        print(f"ERROR: results.json not found at {results_json}")
        sys.exit(1)

    with open(results_json) as f:
        results = json.load(f)

    gt_path = os.path.join(args.data_dir, "ground_truth.npy")
    if os.path.exists(gt_path):
        gt_labels = np.load(gt_path)
        n_queries = min(len(results["queries"]), gt_labels.shape[0])
        recalls = []
        for q in range(n_queries):
            res_labels = results["queries"][q]["labels"]
            r = compute_recall_at_k(gt_labels[q], res_labels, args.k)
            recalls.append(r)

        recalls = np.array(recalls)
        print(f"Recall@{args.k}:")
        print(f"  Mean:   {np.mean(recalls):.4f}")
        print(f"  Median: {np.median(recalls):.4f}")
        print(f"  P90:    {np.percentile(recalls, 90):.4f}")
        print(f"  P99:    {np.percentile(recalls, 99):.4f}")
        print(f"  Min:    {np.min(recalls):.4f}")
    else:
        print(f"Warning: ground_truth.npy not found at {gt_path}, skipping recall")

    # Step 4: Summary report
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    config = results.get("config", {})
    print(f"  Dataset:       {args.dataset}")
    print(f"  d:             {config.get('d', '?')}")
    print(f"  nlist:         {config.get('nlist', '?')}")
    print(f"  Build time:    {results.get('build_time_s', 0):.1f}s")
    print(f"  Memory:        {results.get('memory_bytes', 0) / (1024**2):.1f} MB")
    print(f"  Queries:       {len(results.get('queries', []))}")
    print(f"  Total C++ time: {elapsed:.1f}s")
    print("\nExperiment complete.")

if __name__ == "__main__":
    main()
