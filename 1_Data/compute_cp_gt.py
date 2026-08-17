"""
Compute complex-predicate ground truth for an already-prepared dataset.

This is a standalone script that reads an existing dataset directory (e.g.
``1_Data/ground_truth/wit_100k/``) and computes complex-predicate GT inside a
``complex_predicate/`` subdirectory.

Useful when:
- CP GT is too slow to run inline (WIT ~2h, GIST ~3h)
- You want to re-generate CP GT with different filter parameters
- You want to add CP GT to an existing dataset that was created without it

Usage:
    python 1_Data/compute_cp_gt.py --dataset_dir 1_Data/ground_truth/wit_100k \
        --n_filters 50 --n_cp_queries 100 --k 10 --seed 42
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Make 1_Data/ importable
_script_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, _script_dir)

from prepare_ann_dataset import (  # noqa: E402
    generate_random_filters,
    compute_complex_predicate_gt,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute complex-predicate GT for an existing dataset"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="Path to dataset directory (contains train_vecs.npy etc.)",
    )
    parser.add_argument("--n_filters", type=int, default=50)
    parser.add_argument("--n_cp_queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--query_pool", type=str, default="query_vecs.npy",
        help="Filename of query vectors to sample CP queries from "
             "(default: query_vecs.npy = SL queries; pass a .npy path)",
    )
    parser.add_argument(
        "--train_vecs_name", type=str, default="train_vecs.npy",
    )
    parser.add_argument(
        "--train_mds_name", type=str, default="train_mds.pkl",
    )
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir)
    if not ds_dir.is_dir():
        print(f"ERROR: dataset directory not found: {ds_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Complex-Predicate GT Computation")
    print(f"  Dataset:  {ds_dir}")
    print(f"  Filters:  {args.n_filters}")
    print(f"  Queries:  {args.n_cp_queries}")
    print(f"  k:        {args.k}")
    print(f"  Seed:     {args.seed}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load train data
    # ------------------------------------------------------------------
    train_path = ds_dir / args.train_vecs_name
    mds_path = ds_dir / args.train_mds_name
    print(f"\n  Loading training vectors: {train_path}")
    t0 = time.perf_counter()
    train_vecs = np.load(train_path).astype(np.float32)
    print(f"  Loaded {train_vecs.shape[0]:,} x {train_vecs.shape[1]} "
          f"in {time.perf_counter() - t0:.1f}s")

    print(f"  Loading training labels: {mds_path}")
    t0 = time.perf_counter()
    with open(mds_path, "rb") as f:
        train_mds = pickle.load(f)
    print(f"  Loaded {len(train_mds):,} label lists "
          f"in {time.perf_counter() - t0:.1f}s")

    # Determine labels
    all_labels = sorted(set().union(*train_mds))
    print(f"  Active labels: {len(all_labels)}")

    # ------------------------------------------------------------------
    # 2. Load / select query vectors
    # ------------------------------------------------------------------
    query_path = ds_dir / args.query_pool
    if query_path.exists():
        print(f"\n  Loading query vectors: {query_path}")
        query_vecs_all = np.load(query_path).astype(np.float32)
    else:
        print(f"\n  Query file {query_path} not found, using train subset")
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(train_vecs), args.n_cp_queries, replace=False)
        query_vecs_all = train_vecs[idx]

    cp_rng = np.random.RandomState(args.seed + 1)
    n_cp = min(args.n_cp_queries, len(query_vecs_all))
    cp_idx = cp_rng.choice(len(query_vecs_all), n_cp, replace=False)
    cp_query_vecs = query_vecs_all[cp_idx]
    print(f"  Selected {n_cp} CP queries")

    # ------------------------------------------------------------------
    # 3. Generate filters (with selectivity awareness when possible)
    # ------------------------------------------------------------------
    # Compute per-label selectivity for better AND filter coverage
    n_train = len(train_vecs)
    label_sels = {}
    for lab in all_labels:
        count = sum(1 for md in train_mds if lab in md)
        label_sels[lab] = count / n_train
    filters = generate_random_filters(
        all_labels, n_filters=args.n_filters, seed=args.seed,
        label_selectivities=label_sels,
    )
    print(f"  Generated {len(filters)} filters")
    n_high = sum(1 for v in label_sels.values() if v >= 0.10)
    print(f"  High-sel labels (>=10%): {n_high}/{len(all_labels)}")

    # ------------------------------------------------------------------
    # 4. Compute CP GT
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Computing CP GT: {len(filters)} filters x {n_cp} queries (k={args.k})")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    cp_gt, cp_selectivities = compute_complex_predicate_gt(
        train_vecs, train_mds, cp_query_vecs, filters, k=args.k,
    )
    t_cp = time.perf_counter() - t0
    print(f"\n  CP GT done in {t_cp:.1f}s ({t_cp / 60:.1f} min)")

    # Selectivity stats by filter type
    and_simple = [v for k, v in cp_selectivities.items()
                  if k.startswith("AND ") and "NOT" not in k]
    or_simple = [v for k, v in cp_selectivities.items()
                 if k.startswith("OR ") and k.count("OR") == 1]
    not_simple = [v for k, v in cp_selectivities.items()
                   if k.startswith("NOT ")]

    print(f"\n  Filter selectivity summary:")
    if and_simple:
        print(f"    AND:  min={min(and_simple):.4%}, max={max(and_simple):.4%}, "
              f"n={len(and_simple)}")
        n_gt_1pct = sum(1 for s in and_simple if s > 0.01)
        print(f"          AND > 1%: {n_gt_1pct}/{len(and_simple)}")
    if or_simple:
        print(f"    OR:   min={min(or_simple):.4%}, max={max(or_simple):.4%}, "
              f"n={len(or_simple)}")
    if not_simple:
        print(f"    NOT:  min={min(not_simple):.4%}, max={max(not_simple):.4%}, "
              f"n={len(not_simple)}")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    cp_dir = ds_dir / "complex_predicate"
    cp_dir.mkdir(parents=True, exist_ok=True)

    for formula, gt in cp_gt.items():
        safe_name = formula.replace(" ", "_")
        np.save(cp_dir / f"gt_{safe_name}.npy", gt)

    with open(cp_dir / "filters.json", "w") as f:
        json.dump({
            "n_filters": len(filters),
            "n_queries": n_cp,
            "filters": sorted(filters),
            "selectivities": {k: float(v) for k, v in cp_selectivities.items()},
        }, f, indent=2)

    np.save(cp_dir / "query_vecs.npy", cp_query_vecs.astype(np.float32))
    np.save(cp_dir / "query_indices.npy", cp_idx.astype(np.int32))

    print(f"\n{'='*60}")
    print(f"Saved to: {cp_dir.resolve()}")
    print(f"  {len(filters)} gt_*.npy files")
    print(f"  filters.json")
    print(f"  query_vecs.npy  ({cp_query_vecs.shape})")
    print(f"  query_indices.npy  ({cp_idx.shape})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
