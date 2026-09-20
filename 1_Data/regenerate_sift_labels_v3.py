#!/usr/bin/env python3
"""
Regenerate sift1m labels with a smoother, wider selectivity distribution.

Target:
  1p  ≈ 0.05%
  25p ≈ 0.40%
  50p ≈ 1.50%
  75p ≈ 4.00%
  99p ≈ 30.00%

Each vector gets exactly 3 labels, and label counts are assigned by a
predefined 100-label distribution so the quantiles are controlled directly.
"""
import json
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "1_Data"))

from prepare_ann_dataset import (
    build_inverted_index,
    compute_single_label_gt,
    compute_complex_predicate_gt,
)

DATASET = "sift1m"
GT_DIR = PROJ_ROOT / "1_Data/ground_truth" / DATASET
CP_FORMULA = "OR 6 3"
K = 10
SEED = 1234
N_LABELS = 100
# Counts for N=1,000,000 (sum = 3,000,000 = 3 labels/vector)
COUNTS = (
    [500] * 10 +
    [4000] * 30 +
    [15000] * 30 +
    [40000] * 20 +
    [128125] * 8 +
    [300000] * 2
)


def main():
    train_vecs = np.load(GT_DIR / "train_vecs.npy").astype(np.float32)
    n, d = train_vecs.shape
    assert n == 1_000_000
    assert sum(COUNTS) == 3 * n

    for name in ["train_mds.pkl", "train_access.npy", "ground_truth.npy"]:
        p = GT_DIR / name
        if p.exists():
            bak = p.with_suffix(p.suffix + ".bak2")
            shutil.copy2(p, bak)
            print(f"Backed up {name} -> {bak.name}", flush=True)

    rng = np.random.RandomState(SEED)
    pool = np.repeat(np.arange(N_LABELS, dtype=np.int32), COUNTS)
    rng.shuffle(pool)
    mds = [sorted(set(pool[i * 3:(i + 1) * 3].tolist())) for i in range(n)]
    print("Generated mds", flush=True)

    label_counts = np.bincount(pool, minlength=N_LABELS)
    sels = label_counts / n
    print("Counts sum", label_counts.sum(), flush=True)
    print("Quantiles:", np.percentile(sels, [1, 25, 50, 75, 99]), flush=True)

    with open(GT_DIR / "train_mds.pkl", "wb") as f:
        pickle.dump(mds, f)
    print("Saved train_mds.pkl", flush=True)

    pairs = []
    for vid, md in enumerate(mds):
        for t in md:
            pairs.append((vid, t))
    access = np.array(pairs, dtype=np.int32)
    np.save(GT_DIR / "train_access.npy", access)
    print(f"Saved train_access.npy ({len(access)} pairs)", flush=True)

    with open(GT_DIR / "all_labels.json", "w") as f:
        json.dump(list(range(N_LABELS)), f)

    # Query info update
    query_info = json.load(open(GT_DIR / "query_info.json"))
    for q in query_info:
        lab = q.get("label")
        q["selectivity"] = float(label_counts[lab] / n)
        q.pop("bucket", None)
        q.pop("orig_bucket", None)
        q.pop("selectivity_percentile", None)

    targets = np.percentile(sels, [1, 25, 50, 75, 99])
    names = ["1p", "25p", "50p", "75p", "99p"]
    counts = Counter()
    for q in query_info:
        sel = q["selectivity"]
        idx = int(np.argmin(np.abs(targets - sel)))
        q["bucket"] = names[idx]
        q["selectivity_percentile"] = [1, 25, 50, 75, 99][idx]
        counts[q["bucket"]] += 1
    with open(GT_DIR / "query_info.json", "w") as f:
        json.dump(query_info, f, indent=2)
    print("Bucket counts:", dict(counts), flush=True)

    # Recompute SL GT
    label_to_indices = build_inverted_index(mds, N_LABELS)
    query_vecs = np.load(GT_DIR / "query_vecs.npy").astype(np.float32)
    query_labels = np.load(GT_DIR / "query_labels.npy").astype(np.int32)
    t0 = time.time()
    sl_gt = compute_single_label_gt(train_vecs, query_vecs, query_labels,
                                    label_to_indices, k=K)
    np.save(GT_DIR / "ground_truth.npy", sl_gt)
    print(f"SL GT done in {time.time()-t0:.1f}s", flush=True)

    # CP GT
    cp_dir = GT_DIR / "complex_predicate"
    cp_dir.mkdir(exist_ok=True)
    cp_query_vecs = np.load(cp_dir / "query_vecs.npy").astype(np.float32)
    t0 = time.time()
    cp_gt, cp_sels = compute_complex_predicate_gt(
        train_vecs, mds, cp_query_vecs, [CP_FORMULA], k=K)
    safe = CP_FORMULA.replace(" ", "_")
    np.save(cp_dir / f"gt_{safe}.npy", cp_gt[CP_FORMULA])
    with open(cp_dir / "filters.json", "w") as f:
        json.dump({"n_filters": 1, "n_queries": len(cp_query_vecs),
                   "filters": [CP_FORMULA],
                   "selectivities": {CP_FORMULA: float(cp_sels[CP_FORMULA])}},
                  f, indent=2)
    print(f"CP GT done in {time.time()-t0:.1f}s sel={cp_sels[CP_FORMULA]:.4f}", flush=True)

    print("Done", flush=True)


if __name__ == "__main__":
    main()
