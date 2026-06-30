"""
Subset queries for full datasets (arxiv, yfcc100m) from 1000→500 SL queries
and 100×100→50×40 CP queries.

SL queries are downsampled from existing queries with proportional bucket
stratification. CP queries are subsetted from existing data.

Usage:
    python 1_Data/subset_full_queries.py --dataset yfcc100m
    python 1_Data/subset_full_queries.py --dataset arxiv
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def subset_sl_queries(gt_dir, n_queries=500, seed=42):
    """Downsample existing SL queries with proportional bucket coverage."""
    query_info_path = gt_dir / "query_info.json"

    with open(query_info_path) as f:
        old_info = json.load(f)

    old_n = len(old_info)
    if old_n <= n_queries:
        print(f"  Already {old_n} queries (target {n_queries}), nothing to do")
        return old_n

    # Group by bucket
    bucket_indices = {}
    for i, entry in enumerate(old_info):
        b = entry["bucket"]
        bucket_indices.setdefault(b, []).append(i)

    print(f"  Existing {old_n} queries across {len(bucket_indices)} buckets:")
    for b, idxs in sorted(bucket_indices.items()):
        print(f"    {b}: {len(idxs)} queries")

    # Proportional allocation
    rng = np.random.RandomState(seed)
    allocated = {}
    remaining = n_queries

    # Exclude "overflow" from proportional allocation
    normal_buckets = {b: idxs for b, idxs in bucket_indices.items()
                      if "overflow" not in b.lower()}
    overflow_buckets = {b: idxs for b, idxs in bucket_indices.items()
                        if "overflow" in b.lower()}

    total_normal = sum(len(idxs) for idxs in normal_buckets.values())

    for b, idxs in normal_buckets.items():
        n_bucket = max(1, int(n_queries * len(idxs) / total_normal))
        n_bucket = min(n_bucket, len(idxs), remaining)
        allocated[b] = n_bucket
        remaining -= n_bucket

    # Fill remaining with overflow or redistribute
    for b, idxs in overflow_buckets.items():
        if remaining <= 0:
            break
        n_bucket = min(len(idxs), remaining)
        allocated[b] = n_bucket
        remaining -= n_bucket

    # If still remaining, add to largest buckets
    if remaining > 0:
        for b in sorted(normal_buckets.keys(), key=lambda x: len(normal_buckets[x]), reverse=True):
            if remaining <= 0:
                break
            idxs = normal_buckets[b]
            extra = min(len(idxs) - allocated.get(b, 0), remaining)
            if extra > 0:
                allocated[b] = allocated.get(b, 0) + extra
                remaining -= extra

    # Pick indices
    keep_indices = []
    bucket_kept = {}
    for b, idxs in bucket_indices.items():
        n_pick = allocated.get(b, 0)
        if n_pick > 0:
            pick = sorted(rng.choice(idxs, n_pick, replace=False).tolist())
            keep_indices.extend(pick)
            bucket_kept[b] = n_pick

    keep_indices = sorted(set(keep_indices))
    if len(keep_indices) > n_queries:
        keep_indices = sorted(rng.choice(keep_indices, n_queries, replace=False).tolist())
    n_kept = len(keep_indices)

    print(f"\n  Selected {n_kept} queries:")
    for b, n in sorted(bucket_kept.items()):
        print(f"    {b}: {n} queries")

    # Load and slice arrays
    query_vecs = np.load(gt_dir / "query_vecs.npy").astype(np.float32)
    query_labels = np.load(gt_dir / "query_labels.npy").astype(np.int32)
    gt = np.load(gt_dir / "ground_truth.npy")

    new_vecs = query_vecs[keep_indices]
    new_labels = query_labels[keep_indices]
    new_info = [old_info[i] for i in keep_indices]
    new_gt = gt[keep_indices]

    # Save
    np.save(gt_dir / "query_vecs.npy", new_vecs)
    np.save(gt_dir / "query_labels.npy", new_labels)
    np.save(gt_dir / "ground_truth.npy", new_gt)
    with open(gt_dir / "query_info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    return n_kept


def subset_cp_data(gt_dir, n_filters=50, n_queries=40):
    """Subset existing CP data to first n_filters x n_queries."""
    cp_dir = Path(gt_dir) / "complex_predicate"
    if not cp_dir.exists():
        print("  No CP data found, skipping")
        return False

    with open(cp_dir / "filters.json") as f:
        filters_info = json.load(f)

    all_filters = sorted(filters_info["filters"])
    old_n_queries = filters_info.get("n_queries", 100)

    if len(all_filters) <= n_filters and old_n_queries <= n_queries:
        print(f"  Already {len(all_filters)} filters x {old_n_queries} queries, nothing to do")
        return True

    keep_filters = all_filters[:n_filters]
    old_selectivities = filters_info.get("selectivities", {})

    # Save new filters.json
    new_selectivities = {}
    for formula in keep_filters:
        if formula in old_selectivities:
            new_selectivities[formula] = old_selectivities[formula]

    with open(cp_dir / "filters.json", "w") as f:
        json.dump({
            "n_filters": len(keep_filters),
            "n_queries": n_queries,
            "filters": keep_filters,
            "selectivities": new_selectivities,
        }, f, indent=2)

    # Slice query vectors
    old_query_vecs = np.load(cp_dir / "query_vecs.npy").astype(np.float32)
    old_query_indices = np.load(cp_dir / "query_indices.npy")
    np.save(cp_dir / "query_vecs.npy", old_query_vecs[:n_queries])
    np.save(cp_dir / "query_indices.npy", old_query_indices[:n_queries])

    # Slice GT files, delete unused
    for formula in keep_filters:
        safe = formula.replace(" ", "_")
        old_gt = np.load(cp_dir / f"gt_{safe}.npy")
        np.save(cp_dir / f"gt_{safe}.npy", old_gt[:n_queries])

    for formula in all_filters:
        if formula not in set(keep_filters):
            safe = formula.replace(" ", "_")
            fpath = cp_dir / f"gt_{safe}.npy"
            if fpath.exists():
                fpath.unlink()

    print(f"  CP subset: {len(keep_filters)} filters x {n_queries} queries (was {len(all_filters)}x{old_n_queries})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["arxiv", "yfcc100m"])
    parser.add_argument("--data_dir", type=str, default="1_Data")
    parser.add_argument("--n_sl_query", type=int, default=500)
    parser.add_argument("--n_cp_filter", type=int, default=50)
    parser.add_argument("--n_cp_query", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Subsetting queries for: {args.dataset}")
    print(f"  SL target: {args.n_sl_query} queries")
    print(f"  CP target: {args.n_cp_filter} filters x {args.n_cp_query} queries")
    print(f"{'='*60}")

    gt_dir = Path(args.data_dir) / "ground_truth" / args.dataset

    # ---- Subset SL ----
    print("\n--- Single-label queries ---")
    n_kept = subset_sl_queries(gt_dir, n_queries=args.n_sl_query, seed=args.seed)

    # ---- Subset CP ----
    print("\n--- Complex-predicate queries ---")
    cp_ok = subset_cp_data(gt_dir, n_filters=args.n_cp_filter, n_queries=args.n_cp_query)

    # ---- Update metadata ----
    meta_path = gt_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["n_queries"] = n_kept
        if cp_ok:
            meta["n_filters"] = args.n_cp_filter
            meta["n_cp_queries"] = args.n_cp_query
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ---- Verify ----
    print(f"\n{'='*60}")
    print("Verification:")
    qv = np.load(gt_dir / "query_vecs.npy")
    ql = np.load(gt_dir / "query_labels.npy")
    gt = np.load(gt_dir / "ground_truth.npy")
    with open(gt_dir / "query_info.json") as f:
        qi = json.load(f)
    print(f"  query_vecs:     {qv.shape}")
    print(f"  query_labels:   {ql.shape}")
    print(f"  ground_truth:   {gt.shape}")
    print(f"  query_info:     {len(qi)} entries")

    buckets = Counter(e["bucket"] for e in qi)
    for b in sorted(buckets):
        print(f"    {b}: {buckets[b]}")

    if cp_ok:
        cp_dir = gt_dir / "complex_predicate"
        with open(cp_dir / "filters.json") as f:
            fi = json.load(f)
        cp_qv = np.load(cp_dir / "query_vecs.npy")
        print(f"  CP filters:     {len(fi['filters'])}")
        print(f"  CP query_vecs:  {cp_qv.shape}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
