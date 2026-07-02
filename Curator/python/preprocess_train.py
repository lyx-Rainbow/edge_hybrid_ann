#!/usr/bin/env python3
"""
preprocess_train.py — Convert training data from .pkl to .npy for C++ consumption.

Usage:
    python preprocess_train.py --dataset arxiv
    python preprocess_train.py --dataset yfcc100m --data_dir ../../1_Data/ground_truth
"""
import argparse
import pickle
import numpy as np
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Preprocess training data for Curator")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., arxiv, yfcc100m)")
    parser.add_argument("--data_dir", type=str, default=None, help="Base data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for .npy files")
    args = parser.parse_args()

    # Determine paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, "1_Data", "ground_truth", args.dataset)
    if args.output_dir is None:
        args.output_dir = args.data_dir

    os.makedirs(args.output_dir, exist_ok=True)

    # Load training vectors
    train_path = os.path.join(args.data_dir, "train_vecs.npy")
    if not os.path.exists(train_path):
        print(f"Error: train_vecs.npy not found at {train_path}")
        sys.exit(1)

    train_vecs = np.load(train_path)
    ntrain, d = train_vecs.shape
    print(f"Training vectors: {ntrain} x {d}")

    # Load train_mds.pkl and convert to COO pairs [vid, tid] per vector
    mds_path = os.path.join(args.data_dir, "train_mds.pkl")
    if os.path.exists(mds_path):
        with open(mds_path, "rb") as f:
            train_mds = pickle.load(f)
        # train_mds: list of lists, train_mds[vid] = [tid1, tid2, ...]
        pairs = []
        for vid, tids in enumerate(train_mds):
            for tid in tids:
                pairs.append([vid, tid])
        pairs_arr = np.array(pairs, dtype=np.int32)
        out_path = os.path.join(args.output_dir, "train_access.npy")
        np.save(out_path, pairs_arr)
        print(f"Access pairs: {len(pairs)} rows → {out_path}")
    else:
        print(f"Warning: {mds_path} not found, skipping access pairs generation")

    # Verify output
    vecs_out = os.path.join(args.output_dir, "train_vecs.npy")
    if train_path != vecs_out:
        np.save(vecs_out, train_vecs)
        print(f"Training vectors saved to {vecs_out}")

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
