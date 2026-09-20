#!/usr/bin/env python3
"""Recompute complex-predicate ground truth after a label/access repair.

Only datasets whose labels changed need this.  The script reads the current
complex_predicate/filters.json, recomputes exact filtered top-k for every listed
formula, updates selectivity metadata, and backs up old gt_*.npy files.
"""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "1_Data"))
from prepare_ann_dataset import compute_complex_predicate_gt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / args.dataset
    cp_dir = gt_dir / "complex_predicate"
    filters_path = cp_dir / "filters.json"
    if not filters_path.exists():
        print(f"no complex_predicate/filters.json for {args.dataset}")
        return

    filters_info = json.load(open(filters_path))
    filters = filters_info.get("filters", [])
    if not filters:
        print("no filters to recompute")
        return

    mds = pickle.load(open(gt_dir / "train_mds.pkl", "rb"))
    train_vecs = np.load(gt_dir / "train_vecs.npy", mmap_mode="r")
    query_vecs = np.load(cp_dir / "query_vecs.npy")
    print(f"dataset={args.dataset} filters={filters} queries={len(query_vecs)}")

    results, selectivities = compute_complex_predicate_gt(
        train_vecs, mds, query_vecs, filters, k=args.k)

    for formula in filters:
        safe = formula.replace(" ", "_")
        path = cp_dir / f"gt_{safe}.npy"
        if path.exists():
            backup = path.with_name(path.name + ".bak_before_cp_refresh")
            if not backup.exists():
                shutil.copy2(path, backup)
        np.save(path, results[formula])
        print(f"saved {path} sel={selectivities[formula]:.6f}")

    backup = filters_path.with_name(filters_path.name + ".bak_before_cp_refresh")
    if not backup.exists():
        shutil.copy2(filters_path, backup)
    filters_info["selectivities"] = {
        str(k): float(v) for k, v in selectivities.items()}
    with open(filters_path, "w") as f:
        json.dump(filters_info, f, indent=2)
    print("updated", filters_path)


if __name__ == "__main__":
    main()
