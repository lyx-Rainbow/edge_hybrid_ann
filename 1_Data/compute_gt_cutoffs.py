#!/usr/bin/env python3
"""Compute exact label-filtered top-k distance cutoffs for single-label GT.

The original single-label ground truth stores only the top-k vector IDs.  When
many vectors are at the same distance (especially exact duplicates), different
implementations legitimately return different tied subsets, so strict
ID-intersection recall cannot reach 1.0 even for an exact scan.

This script stores the exact top-k distance rows and the k-th distance
(``ground_truth_cutoff.npy``) per query.  Recall can then count a returned
result as correct if it has the query label and its distance is no larger than
the cutoff (+ a small tolerance), which makes the metric well-defined under
distance ties.

Usage:
    python 1_Data/compute_gt_cutoffs.py --datasets wit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GT_ROOT = ROOT / "1_Data/ground_truth"


def label_to_vids(access_pairs: np.ndarray):
    order = np.argsort(access_pairs[:, 1], kind="stable")
    labels = access_pairs[order, 1]
    vids = access_pairs[order, 0]
    uniq, starts = np.unique(labels, return_index=True)
    mapping = {}
    for index, label in enumerate(uniq):
        start = int(starts[index])
        end = int(starts[index + 1]) if index + 1 < len(starts) else len(labels)
        mapping[int(label)] = np.unique(vids[start:end].astype(np.int64))
    return mapping


def compute(dataset: str, k: int = 10) -> None:
    base = GT_ROOT / dataset
    gt = np.load(base / "ground_truth.npy")
    qlabels = np.load(base / "query_labels.npy").astype(np.int64)
    qvecs = np.load(base / "query_vecs.npy").astype(np.float32)
    train_vecs = np.load(base / "train_vecs.npy", mmap_mode="r")
    access = np.load(base / "train_access.npy")

    n_queries = len(qlabels)
    mapping = label_to_vids(access)
    distances = np.full((n_queries, k), np.nan, dtype=np.float32)
    cutoff = np.full(n_queries, np.nan, dtype=np.float32)
    qnorm = np.sum(qvecs * qvecs, axis=1)

    labels_with_queries = sorted(set(int(x) for x in qlabels))
    for label in labels_with_queries:
        candidates = mapping.get(label)
        if candidates is None or len(candidates) == 0:
            continue
        qidx = np.where(qlabels == label)[0]
        cand_vecs = np.asarray(train_vecs[candidates], dtype=np.float32)
        q_mat = qvecs[qidx]
        cnorm = np.sum(cand_vecs * cand_vecs, axis=1)
        # Squared L2 via BLAS: ||c||^2 + ||q||^2 - 2 c.q
        dist = (cnorm[:, None] + qnorm[qidx][None, :]
                - 2.0 * (cand_vecs @ q_mat.T))
        np.maximum(dist, 0.0, out=dist)
        actual_k = min(k, len(candidates))
        selected = np.argpartition(dist, actual_k - 1, axis=0)[:actual_k, :]
        for column, qi in enumerate(qidx):
            values = np.sort(dist[selected[:, column], column])
            distances[qi, :actual_k] = values
            cutoff[qi] = values[-1]
        print(f"  {dataset}: label {label} "
              f"({len(qidx)} queries, {len(candidates)} candidates)", flush=True)

    np.save(base / f"ground_truth_distances_top{k}.npy", distances)
    np.save(base / "ground_truth_cutoff.npy", cutoff)
    valid = np.isfinite(cutoff)
    print(f"{dataset}: wrote cutoffs for {valid.sum()}/{n_queries} queries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["wit"])
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    for dataset in args.datasets:
        compute(dataset, k=args.k)


if __name__ == "__main__":
    main()
