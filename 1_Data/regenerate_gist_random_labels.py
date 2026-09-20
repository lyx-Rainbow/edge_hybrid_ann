#!/usr/bin/env python3
"""Regenerate gist1m labels with geometry-independent random assignment.

gist1m currently uses kmeans labels, which correlate selectivity with cluster
geometry and invalidate the post-filter selectivity trend assumption.  This
script generates hierarchical random labels with a sift1m-like selectivity
distribution and can refresh label-dependent metadata / ground truth without
rebuilding the label-independent parts of the vector index.

Dry run by default.  Use --apply to write.
"""
import argparse
import json
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "1_Data"))

from repair_duplicate_labels import (  # noqa: E402
    K,
    access_from_labels,
    assign_buckets,
    backup_once,
    patch_diskivf_index,
    recompute_sl_gt,
    write_spann_metadata,
)

N_LABELS = 100
SEED = 1234

# Controlled selectivity profile copied from the sift1m v3 regeneration:
# 10 x 0.05%, 30 x 0.4%, 30 x 1.5%, 20 x 4%, 8 x 12.8%, 2 x 30%.
# Sum = 3 * 1,000,000 = 3 labels per vector at N = 1M.
COUNTS_1M = (
    [500] * 10
    + [4000] * 30
    + [15000] * 30
    + [40000] * 20
    + [128125] * 8
    + [300000] * 2
)


def scaled_counts(n_vectors):
    """Scale the 1M reference counts to roughly 3 * n_vectors labels."""
    base = np.asarray(COUNTS_1M, dtype=np.float64) * (n_vectors / 1_000_000.0)
    counts = np.floor(base).astype(np.int64)
    remainder = 3 * n_vectors - int(counts.sum())
    if remainder > 0:
        frac = base - counts
        order = np.argsort(-frac)
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    return counts


def generate_unique_random_labels(n_vectors, counts, seed=SEED, block=100_000):
    """Sample 3 distinct labels per vector with probabilities proportional to counts.

    The Gumbel top-k trick keeps each vector's labels unique while preserving the
    target selectivity distribution approximately.
    """
    rng = np.random.RandomState(seed)
    counts = counts.astype(np.float64)
    log_w = np.log(np.maximum(counts, 1e-12))
    labels = []
    for start in range(0, n_vectors, block):
        stop = min(start + block, n_vectors)
        u = rng.random_sample((stop - start, len(counts)))
        score = log_w[np.newaxis, :] - np.log(-np.log(np.maximum(u, 1e-12)))
        local = np.argpartition(-score, 3, axis=1)[:, :3]
        for row in local:
            labels.append(sorted(int(x) for x in row))
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gist1m")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--recompute-gt", action="store_true")
    parser.add_argument("--spann-index-dir", default=None)
    parser.add_argument("--diskivf-index-dir", default=None)
    args = parser.parse_args()

    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / args.dataset
    meta_path = gt_dir / "metadata.json"
    meta = json.load(open(meta_path))
    n = int(np.load(gt_dir / "train_vecs.npy", mmap_mode="r").shape[0])
    print(f"dataset={args.dataset} N={n}")

    # Independent labels with the controlled sift1m-like selectivity profile.
    counts_arr = scaled_counts(n)
    labels = generate_unique_random_labels(n, counts_arr, seed=SEED)
    counts = Counter()
    for lab_list in labels:
        for lab in lab_list:
            counts[int(lab)] += 1
    sels = np.asarray([counts.get(i, 0) / n for i in range(N_LABELS)])
    print("selectivity quantiles: "
          f"1p={np.percentile(sels, 1):.5f} "
          f"25p={np.percentile(sels, 25):.5f} "
          f"50p={np.percentile(sels, 50):.5f} "
          f"75p={np.percentile(sels, 75):.5f} "
          f"99p={np.percentile(sels, 99):.5f}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to write.")
        return

    access_path = gt_dir / "train_access.npy"
    mds_path = gt_dir / "train_mds.pkl"
    qinfo_path = gt_dir / "query_info.json"
    all_labels_path = gt_dir / "all_labels.json"
    for path in [access_path, mds_path, qinfo_path, all_labels_path, meta_path]:
        backup_once(path)

    access = access_from_labels(labels)
    with open(mds_path, "wb") as f:
        pickle.dump(labels, f)
    np.save(access_path, access)
    with open(all_labels_path, "w") as f:
        json.dump(sorted(counts), f)

    query_info = json.load(open(qinfo_path))
    changed = assign_buckets(query_info, counts, n)
    with open(qinfo_path, "w") as f:
        json.dump(query_info, f, indent=2)
    print(f"bucket changes={changed}")

    meta.update({
        "label_method": "hierarchical_random",
        "n_labels": len(counts),
        "train_size": n,
        "avg_labels_per_vec": float(np.mean([len(x) for x in labels])),
        "single_label_gt_time_s": 0,
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if args.spann_index_dir:
        write_spann_metadata(args.spann_index_dir, labels)
    if args.diskivf_index_dir:
        patch_diskivf_index(args.diskivf_index_dir, labels)
    if args.recompute_gt:
        recompute_sl_gt(gt_dir, access, labels, k=K)
    print("gist1m random-label regeneration complete")


if __name__ == "__main__":
    main()
