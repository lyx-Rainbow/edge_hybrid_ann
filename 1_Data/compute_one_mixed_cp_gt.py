#!/usr/bin/env python3
"""
Compute ground truth for one mixed AND+OR complex predicate on a full dataset.

This is intentionally limited to ONE filter (plus one OR/AND if desired) so
we avoid the expensive full 50-100-filter CP GT computation.

Usage:
    python 1_Data/compute_one_mixed_cp_gt.py --dataset arxiv --formula "AND 0 OR 1 2" --n_queries 100
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_ann_dataset import compute_complex_predicate_gt  # noqa: E402

PROJ_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--formula", default="AND 0 OR 1 2")
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds_dir = PROJ_ROOT / "1_Data/ground_truth" / args.dataset
    cp_dir = ds_dir / "complex_predicate"
    cp_dir.mkdir(parents=True, exist_ok=True)

    train_vecs = np.load(ds_dir / "train_vecs.npy").astype(np.float32)
    with open(ds_dir / "train_mds.pkl", "rb") as f:
        train_mds = pickle.load(f)

    # Use existing CP query vectors if present; otherwise sample from SL queries.
    if (cp_dir / "query_vecs.npy").exists():
        query_vecs = np.load(cp_dir / "query_vecs.npy").astype(np.float32)
        if len(query_vecs) < args.n_queries:
            print(f"WARNING: only {len(query_vecs)} CP queries available", flush=True)
    else:
        query_vecs = np.load(ds_dir / "query_vecs.npy").astype(np.float32)
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(query_vecs), min(args.n_queries, len(query_vecs)), replace=False)
        query_vecs = query_vecs[idx]

    filters = [args.formula]
    t0 = time.time()
    cp_gt, selectivities = compute_complex_predicate_gt(
        train_vecs, train_mds, query_vecs, filters, k=args.k)
    print(f"Computed in {time.time() - t0:.1f}s", flush=True)
    print(f"Selectivity: {selectivities[args.formula]}", flush=True)

    safe = args.formula.replace(" ", "_")
    np.save(cp_dir / f"gt_{safe}.npy", cp_gt[args.formula])
    np.save(cp_dir / "query_vecs.npy", query_vecs.astype(np.float32))

    # Merge into filters.json if it exists, otherwise create minimal one.
    filters_path = cp_dir / "filters.json"
    if filters_path.exists():
        info = json.load(open(filters_path))
    else:
        info = {"n_filters": 0, "n_queries": len(query_vecs),
                "filters": [], "selectivities": {}}
    if args.formula not in info["filters"]:
        info["filters"].append(args.formula)
    info["selectivities"][args.formula] = float(selectivities[args.formula])
    info["n_queries"] = len(query_vecs)
    info["n_filters"] = len(info["filters"])
    with open(filters_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("Saved", cp_dir / f"gt_{safe}.npy")


if __name__ == "__main__":
    main()
