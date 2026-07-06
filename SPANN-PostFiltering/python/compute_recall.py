#!/usr/bin/env python3
"""
compute_recall.py — Compute Recall@k from SPANN-PostFiltering C++ output JSON

Usage:
  python python/compute_recall.py --results results.json --ground_truth gt.npy --k 10
  python python/compute_recall.py --results results.json --dataset arxiv_small --data_dir 1_Data --k 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compute_recall(pred_ids, gt_ids, k):
    """Compute recall@k between predicted and ground-truth result lists."""
    valid = set(int(i) for i in gt_ids[:k] if i >= 0)
    if not valid:
        return 1.0
    return len(set(pred_ids[:k]) & valid) / len(valid)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Recall@k from C++ results JSON")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to C++ results JSON file")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Path to ground_truth.npy (overrides --dataset)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for GT lookup")
    parser.add_argument("--data_dir", type=str, default="1_Data",
                        help="Root data directory")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of results (default: 10)")
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        results = json.load(f)

    # Load ground truth
    if args.ground_truth:
        gt = np.load(args.ground_truth)
    elif args.dataset:
        gt_path = Path(args.data_dir) / "ground_truth" / args.dataset / "ground_truth.npy"
        gt = np.load(gt_path)
    else:
        print("Error: need --ground_truth or --dataset")
        sys.exit(1)

    k = args.k

    # Compute recall per query
    recalls = []
    for q, query_result in enumerate(results["queries"]):
        r = compute_recall(query_result["labels"], gt[q], k)
        recalls.append(r)

    recalls = np.array(recalls)

    print(f"\nRecall@{k} Results:")
    print(f"  Index:     {results.get('index', 'unknown')}")
    print(f"  Queries:   {len(recalls)}")
    print(f"  Recall@{k}:")
    print(f"    Mean:   {np.mean(recalls):.4f}")
    print(f"    Median: {np.median(recalls):.4f}")
    print(f"    Std:    {np.std(recalls):.4f}")
    print(f"    P90:    {np.percentile(recalls, 90):.4f}")
    print(f"    P99:    {np.percentile(recalls, 99):.4f}")
    print(f"    Min:    {np.min(recalls):.4f}")
    print(f"    ==1.0:  {np.sum(recalls == 1.0)}/{len(recalls)}")
    empty = sum(1 for qr in results["queries"]
                if all(x == -1 for x in qr["labels"]))
    print(f"    Empty:  {empty}")
    if "build_time_s" in results:
        print(f"  Build:    {results['build_time_s']:.2f} s")
    if "memory_bytes" in results:
        print(f"  Memory:   {results['memory_bytes'] / 1024**2:.1f} MB")
    print()


if __name__ == "__main__":
    main()
