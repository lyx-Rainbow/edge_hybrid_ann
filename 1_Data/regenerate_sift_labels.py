#!/usr/bin/env python3
"""
Regenerate sift1m synthetic labels so that:
  - 1p selectivity < 1%
  - 99p selectivity around 30%

Then recompute single-label and one complex-predicate ground truth.
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

from synthesize_labels import synthesize_labels_random
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


def main():
    train_vecs = np.load(GT_DIR / "train_vecs.npy").astype(np.float32)
    n, d = train_vecs.shape
    print(f"Loaded {n} vectors d={d}", flush=True)

    # 1. Backup old files.
    for name in ["train_mds.pkl", "train_access.npy", "ground_truth.npy"]:
        p = GT_DIR / name
        if p.exists():
            bak = p.with_suffix(p.suffix + ".bak")
            shutil.copy2(p, bak)
            print(f"Backed up {name} -> {bak.name}", flush=True)

    # 2. Generate hierarchical labels with wide selectivity range.
    mds = synthesize_labels_random(
        n_vectors=n,
        n_labels=100,
        avg_labels_per_vec=3,
        seed=SEED,
        distribution="hierarchical",
        n_coarse=10,
        n_fine=90,
        coarse_sel_min=0.20,
        coarse_sel_max=0.35,
        fine_sel_min=0.0005,
        fine_sel_max=0.01,
        zipf_alpha=2.0,
        coarse_labels_per_vec=1,
        fine_labels_per_vec=2,
    )
    label_counts = Counter()
    for md in mds:
        for lab in md:
            label_counts[lab] += 1
    sels = np.array([label_counts.get(i, 0) / n for i in range(100)])
    print(f"Selectivity percentiles: "
          f"1p={np.percentile(sels, 1):.5f}, "
          f"25p={np.percentile(sels, 25):.5f}, "
          f"50p={np.percentile(sels, 50):.5f}, "
          f"75p={np.percentile(sels, 75):.5f}, "
          f"99p={np.percentile(sels, 99):.5f}", flush=True)

    with open(GT_DIR / "train_mds.pkl", "wb") as f:
        pickle.dump(mds, f)
    print("Saved train_mds.pkl", flush=True)

    # 3. Rebuild train_access.npy.
    pairs = []
    for vid, md in enumerate(mds):
        for t in md:
            pairs.append((vid, t))
    access = np.array(pairs, dtype=np.int32)
    np.save(GT_DIR / "train_access.npy", access)
    print(f"Saved train_access.npy ({len(access)} pairs)", flush=True)
    all_labels = sorted(set().union(*mds))
    with open(GT_DIR / "all_labels.json", "w") as f:
        json.dump(all_labels, f)
    print(f"Saved all_labels.json ({len(all_labels)})", flush=True)

    # 4. Rebuild query_info with new per-label selectivities and 5 percentile buckets.
    query_info = json.load(open(GT_DIR / "query_info.json"))
    for q in query_info:
        lab = q.get("label")
        q["selectivity"] = float(label_counts.get(lab, 0) / n)
        q.pop("bucket", None)
        q.pop("orig_bucket", None)
        q.pop("selectivity_percentile", None)
    # Save temporary, then use the same percentile assignment script logic.
    with open(GT_DIR / "query_info.json", "w") as f:
        json.dump(query_info, f, indent=2)

    # Compute percentile targets from label selectivities and assign nearest.
    labels = sorted(label_counts.keys())
    label_sels = np.array([label_counts.get(l, 0) / n for l in labels])
    targets = np.percentile(label_sels, [1, 25, 50, 75, 99])
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

    # 5. Recompute single-label ground truth.
    label_to_indices = build_inverted_index(mds, 100)
    query_vecs = np.load(GT_DIR / "query_vecs.npy").astype(np.float32)
    query_labels = np.load(GT_DIR / "query_labels.npy").astype(np.int32)
    t0 = time.time()
    sl_gt = compute_single_label_gt(
        train_vecs, query_vecs, query_labels, label_to_indices, k=K)
    np.save(GT_DIR / "ground_truth.npy", sl_gt)
    print(f"Saved ground_truth.npy in {time.time()-t0:.1f}s", flush=True)

    # 6. Recompute complex-predicate GT for the selected OR filter.
    cp_dir = GT_DIR / "complex_predicate"
    cp_dir.mkdir(exist_ok=True)
    cp_query_vecs = np.load(cp_dir / "query_vecs.npy").astype(np.float32)
    t0 = time.time()
    cp_gt, cp_sels = compute_complex_predicate_gt(
        train_vecs, mds, cp_query_vecs, [CP_FORMULA], k=K)
    safe = CP_FORMULA.replace(" ", "_")
    np.save(cp_dir / f"gt_{safe}.npy", cp_gt[CP_FORMULA])
    filters_path = cp_dir / "filters.json"
    filters_info = {"n_filters": 1, "n_queries": len(cp_query_vecs),
                    "filters": [CP_FORMULA], "selectivities": {CP_FORMULA: float(cp_sels[CP_FORMULA])}}
    with open(filters_path, "w") as f:
        json.dump(filters_info, f, indent=2)
    print(f"Saved CP GT for {CP_FORMULA} sel={cp_sels[CP_FORMULA]:.4f} in {time.time()-t0:.1f}s", flush=True)

    # 7. Update metadata.
    meta_path = GT_DIR / "metadata.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        meta["avg_labels_per_vec"] = float(np.mean([len(m) for m in mds]))
        meta["label_method"] = "hierarchical_random"
        meta["single_label_gt_time_s"] = 0
        meta["complex_predicate_gt_time_s"] = 0
        json.dump(meta, open(meta_path, "w"), indent=2)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
