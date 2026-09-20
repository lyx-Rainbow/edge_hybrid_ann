#!/usr/bin/env python3
"""
Regenerate WIT synthetic labels with a sift1m-like selectivity distribution.

Current WIT labels are 1000 K-means labels with max selectivity ~4%.
This script replaces them with 100 synthetic labels whose per-label
selectivity quantiles closely match sift1m:

  1p    ~ 0.05%
  25p   ~ 0.40%
  50p   ~ 1.50%
  75p   ~ 4.00%
  99p   ~ 30.00%

Each vector gets 3 labels.  The script also re-selects the 1000 SL queries
from the WIT query pool and recomputes SL ground truth.
"""
import json
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "1_Data"))

from prepare_wit_dataset import (
    build_inverted_index,
    select_single_label_queries,
)

DATASET = "wit"
GT_DIR = PROJ_ROOT / "1_Data/ground_truth" / DATASET
CACHE_DIR = GT_DIR / ".cache"
SPLIT = CACHE_DIR / ".cache_split.npz"
ALL_VECS_CACHE = CACHE_DIR / ".cache_all_vecs.npy"

K = 10
SEED = 1234
N_LABELS = 300
N_TRAIN = 2_966_420
N_POOL = 741_606
# Zipf-like skew (alpha=0.85): max selectivity ~31%, long low-selectivity tail.
ALPHA = 0.85


def make_counts_for_n(counts, n_total, seed):
    """Return a shuffled pool of label ids summing to 3*n_total."""
    assert sum(counts) == 3 * n_total, (sum(counts), 3 * n_total)
    rng = np.random.RandomState(seed)
    pool = np.repeat(np.arange(N_LABELS, dtype=np.int32), counts)
    rng.shuffle(pool)
    return pool


def make_mds_from_pool(pool):
    n = len(pool) // 3
    return [sorted(set(pool[i * 3:(i + 1) * 3].tolist())) for i in range(n)]


def make_zipf_counts(n_labels, n_total, alpha, seed):
    """Return integer label counts summing to 3*n_total with a Zipf-like skew."""
    rng = np.random.RandomState(seed)
    rank = rng.permutation(n_labels)  # randomize which labels get high counts
    weights = 1.0 / (rank + 1) ** alpha
    counts = (weights / weights.sum() * (3 * n_total)).astype(np.int64)
    counts[-1] += 3 * n_total - int(counts.sum())
    assert int(counts.sum()) == 3 * n_total
    return counts


def compute_single_label_gt_memory_safe(
    train_vecs, query_vecs, query_labels, label_to_indices, k=10, batch=5000,
):
    """Compute SL GT one query at a time to keep peak memory bounded."""
    n_queries = len(query_labels)
    gt = np.full((n_queries, k), -1, dtype=np.int32)

    for qi, lab in enumerate(query_labels):
        candidates = label_to_indices.get(int(lab), [])
        if len(candidates) == 0:
            continue
        candidates = np.asarray(candidates, dtype=np.int32)
        qvec = query_vecs[qi]

        best_d = np.full(k, np.inf, dtype=np.float32)
        best_i = np.full(k, -1, dtype=np.int32)

        for start in range(0, len(candidates), batch):
            cand = candidates[start:start + batch]
            vecs = train_vecs[cand]           # [b, d]
            dists = np.sum((vecs - qvec) ** 2, axis=1)  # [b]

            merged_d = np.concatenate([best_d, dists])
            merged_i = np.concatenate([best_i, cand])
            keep = min(k, len(merged_d))
            sel = np.argpartition(merged_d, keep - 1)[:keep]
            best_d = merged_d[sel]
            best_i = merged_i[sel]
            del vecs, dists, merged_d, merged_i

        order = np.argsort(best_d)
        gt[qi, :len(best_i)] = best_i[order]

        if (qi + 1) % 200 == 0:
            print(f"    GT: {qi + 1}/{n_queries} queries processed...",
                  flush=True)

    return gt


def main():
    # Backup existing SL-related files.
    for name in ["train_mds.pkl", "train_access.npy", "ground_truth.npy",
                 "query_vecs.npy", "query_labels.npy", "query_info.json",
                 "all_labels.json", "metadata.json"]:
        p = GT_DIR / name
        if p.exists():
            bak = p.with_name(p.name + ".bak_wit_redo")
            if not bak.exists():
                shutil.copy2(p, bak)
                print(f"Backed up {name} -> {bak.name}", flush=True)

    # Generate train labels with the desired skewed distribution.
    train_counts = make_zipf_counts(N_LABELS, N_TRAIN, ALPHA, seed=SEED)
    train_pool = make_counts_for_n(train_counts, N_TRAIN, seed=SEED)
    train_mds = make_mds_from_pool(train_pool)
    print(f"Generated train labels: {len(train_mds):,} vectors, "
          f"{N_LABELS} labels", flush=True)

    # Generate pool labels with the same selectivity shape (for query selection).
    pool_counts = (train_counts * N_POOL // N_TRAIN).astype(np.int64)
    pool_counts = np.maximum(pool_counts, 1)
    pool_counts[-1] += 3 * N_POOL - int(pool_counts.sum())
    assert int(pool_counts.sum()) == 3 * N_POOL
    pool_pool = make_counts_for_n(pool_counts, N_POOL, seed=SEED + 1)
    pool_mds = make_mds_from_pool(pool_pool)
    print(f"Generated pool labels: {len(pool_mds):,} vectors", flush=True)

    # Load cached vector order / split.
    split = np.load(SPLIT)
    train_idx = split["train_idx"]
    pool_idx = split["pool_idx"]
    assert len(train_idx) == N_TRAIN
    assert len(pool_idx) == N_POOL

    # Save train labels + access pairs.
    with open(GT_DIR / "train_mds.pkl", "wb") as f:
        pickle.dump(train_mds, f)

    pairs = []
    for vid, md in enumerate(train_mds):
        for t in md:
            pairs.append((vid, t))
    access = np.array(pairs, dtype=np.int32)
    np.save(GT_DIR / "train_access.npy", access)
    print(f"Saved train_access.npy ({len(access):,} pairs)", flush=True)

    with open(GT_DIR / "all_labels.json", "w") as f:
        json.dump(list(range(N_LABELS)), f)

    # Build inverted index and select stratified SL queries from the pool.
    label_to_indices = build_inverted_index(train_mds, N_LABELS)
    train_counts = np.bincount(
        np.concatenate([np.array(x) for x in train_mds]), minlength=N_LABELS
    )
    train_sels = train_counts / N_TRAIN

    all_vecs_mmap = np.load(ALL_VECS_CACHE, mmap_mode="r")
    pool_vecs = all_vecs_mmap[pool_idx].copy()  # ~1.1 GB
    del all_vecs_mmap

    query_vecs, query_labels_raw, query_info_raw = select_single_label_queries(
        pool_vecs, pool_mds, train_mds, label_to_indices,
        n_queries=1000, seed=42,
    )
    print(f"Selected {len(query_labels_raw)} queries", flush=True)

    # Reassign named percentile buckets using train label selectivities.
    targets = np.percentile(train_sels, [1, 25, 50, 75, 99])
    names = ["1p", "25p", "50p", "75p", "99p"]
    query_info = []
    counts = Counter()
    for q, lab in zip(query_info_raw, query_labels_raw):
        sel = float(train_sels[lab])
        idx = int(np.argmin(np.abs(targets - sel)))
        ent = {
            "query_idx": int(q["query_idx"]),
            "label": int(lab),
            "selectivity": sel,
            "bucket": names[idx],
            "selectivity_percentile": [1, 25, 50, 75, 99][idx],
        }
        query_info.append(ent)
        counts[ent["bucket"]] += 1
    with open(GT_DIR / "query_info.json", "w") as f:
        json.dump(query_info, f, indent=2)
    print("Bucket counts:", dict(counts), flush=True)

    np.save(GT_DIR / "query_vecs.npy", query_vecs.astype(np.float32))
    np.save(GT_DIR / "query_labels.npy", np.array(query_labels_raw, dtype=np.int32))
    print("Saved query vectors/labels", flush=True)

    # Recompute SL ground truth one query at a time (memory-safe).
    train_vecs = np.load(GT_DIR / "train_vecs.npy", mmap_mode="r")
    gt = compute_single_label_gt_memory_safe(
        train_vecs, query_vecs, np.array(query_labels_raw, dtype=np.int32),
        label_to_indices, k=K,
    )
    np.save(GT_DIR / "ground_truth.npy", gt.astype(np.int32))
    print(f"Saved ground_truth.npy: {gt.shape}", flush=True)

    # Update metadata.
    meta_path = GT_DIR / "metadata.json"
    meta = json.load(open(meta_path))
    meta["n_labels"] = N_LABELS
    meta["label_method"] = "synthetic-random"
    meta["distribution"] = "sift1m-like"
    meta["avg_labels_per_vec"] = 3
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("Updated metadata.json", flush=True)

    print("Done", flush=True)


if __name__ == "__main__":
    main()
