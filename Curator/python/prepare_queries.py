#!/usr/bin/env python3
"""
prepare_queries.py — Generate query files and compute ground truth for Curator.

Usage:
    python prepare_queries.py --dataset arxiv
    python prepare_queries.py --dataset yfcc100m --n_queries 100 --selectivity 0.1
"""
import argparse
import numpy as np
import os
import sys
import time

def compute_ground_truth(queries, database, k=100, batch_size=1000):
    """Brute-force L2 nearest neighbor search for ground truth."""
    nq, d = queries.shape
    ndb = database.shape[0]
    gt_labels = np.zeros((nq, k), dtype=np.int32)
    gt_dists = np.zeros((nq, k), dtype=np.float32)

    for start in range(0, nq, batch_size):
        end = min(start + batch_size, nq)
        q_batch = queries[start:end]
        dists = np.zeros((end - start, ndb), dtype=np.float32)
        for i in range(end - start):
            diff = database - q_batch[i]
            dists[i] = np.sum(diff ** 2, axis=1)
        idx = np.argpartition(dists, k, axis=1)[:, :k]
        batch_dists = np.take_along_axis(dists, idx, axis=1)
        sort_idx = np.argsort(batch_dists, axis=1)
        gt_labels[start:end] = np.take_along_axis(idx, sort_idx, axis=1)
        gt_dists[start:end] = np.take_along_axis(batch_dists, sort_idx, axis=1)

        if (end % (10 * batch_size)) == 0:
            print(f"  GT progress: {end}/{nq}")

    return gt_labels, gt_dists

def main():
    parser = argparse.ArgumentParser(description="Prepare query data for Curator")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_queries", type=int, default=100, help="Number of queries to generate")
    parser.add_argument("--selectivity", type=float, default=0.1,
                        help="Selectivity for query tenant labels (0.0-1.0)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, "1_Data", "ground_truth", args.dataset)
    if args.output_dir is None:
        args.output_dir = args.data_dir

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    train_path = os.path.join(args.data_dir, "train_vecs.npy")
    query_path = os.path.join(args.data_dir, "query_vecs.npy")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found")
        sys.exit(1)

    train_vecs = np.load(train_path)
    ntrain, d = train_vecs.shape
    print(f"Database: {ntrain} x {d}")

    # Load or generate queries
    if os.path.exists(query_path):
        query_vecs = np.load(query_path)
        print(f"Loaded {query_vecs.shape[0]} existing queries")
    else:
        np.random.seed(42)
        idx = np.random.choice(ntrain, min(args.n_queries, ntrain), replace=False)
        query_vecs = train_vecs[idx].copy()
        np.save(query_path, query_vecs)
        print(f"Generated {len(idx)} queries -> {query_path}")

    n_queries = query_vecs.shape[0]

    # Generate query labels (tenant IDs for filtered search)
    query_labels_path = os.path.join(args.output_dir, "query_labels.npy")
    query_labels = np.full(n_queries, -1, dtype=np.int32)
    if args.selectivity > 0:
        n_labeled = int(n_queries * args.selectivity)
        query_labels[:n_labeled] = np.random.randint(0, 100, n_labeled, dtype=np.int32)

    np.save(query_labels_path, query_labels)
    print(f"Query labels saved to {query_labels_path}")

    # Compute ground truth
    gt_path = os.path.join(args.output_dir, "ground_truth.npy")
    if not os.path.exists(gt_path):
        print(f"Computing ground truth for {n_queries} queries (k=100)...")
        t0 = time.time()
        gt_labels, gt_dists = compute_ground_truth(query_vecs, train_vecs, k=100)
        np.save(gt_path, gt_labels)
        print(f"Ground truth saved to {gt_path} ({time.time() - t0:.1f}s)")

    print("Query preparation complete.")

if __name__ == "__main__":
    main()
