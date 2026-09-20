#!/usr/bin/env python3
"""
Assign each single-label query to one of five percentile-based selectivity
buckets: 1p / 25p / 50p / 75p / 99p.

The percentile targets are computed from the per-label selectivity
distribution of the training set. Each query is assigned to the nearest
target (in percentile-rank space) rather than to equal-width numeric bins,
so the buckets correspond to the requested selectivity quantiles.

Usage:
    python 1_Data/make_percentile_buckets.py --dataset sift1m
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
PERCENTILES = [1, 25, 50, 75, 99]


def load_train_mds(dataset):
    p = PROJ_ROOT / "1_Data/ground_truth" / dataset / "train_mds.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--write", action="store_true",
                        help="write query_info.json (default is dry run)")
    args = parser.parse_args()

    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / args.dataset
    qinfo_path = gt_dir / "query_info.json"
    query_info = json.load(open(qinfo_path))
    mds = load_train_mds(args.dataset)
    n_train = len(mds)

    cnt = Counter()
    for md in mds:
        for lab in md:
            cnt[lab] += 1
    label_sel = {int(lab): v / n_train for lab, v in cnt.items()}

    sel_values = np.array(sorted(label_sel.values()))
    targets = np.percentile(sel_values, PERCENTILES)
    print(f"Dataset: {args.dataset}, labels={len(sel_values)}, queries={len(query_info)}")
    for p, t in zip(PERCENTILES, targets):
        print(f"  {p}p target selectivity = {t:.6f}")

    # For each query, find nearest percentile target.
    counts = Counter()
    new_info = []
    for q in query_info:
        lab = q.get("label")
        sel = q.get("selectivity")
        if lab is None or sel is None:
            q["orig_bucket"] = q.get("bucket")
            q["bucket"] = "unknown"
            new_info.append(q)
            counts["unknown"] += 1
            continue
        ranks = np.abs(targets - sel)
        idx = int(np.argmin(ranks))
        bucket = f"{PERCENTILES[idx]}p"
        q["orig_bucket"] = q.get("bucket")
        q["bucket"] = bucket
        q["selectivity_percentile"] = PERCENTILES[idx]
        new_info.append(q)
        counts[bucket] += 1

    print("Bucket counts:", dict(counts))

    if args.write:
        # Keep backup of original bucket field by rewriting from existing file
        # before overwrite (already have orig_bucket per query).
        qinfo_path.write_text(json.dumps(new_info, indent=2))
        print(f"Written {qinfo_path}")
    else:
        print("Dry run (use --write to apply)")


if __name__ == "__main__":
    main()
