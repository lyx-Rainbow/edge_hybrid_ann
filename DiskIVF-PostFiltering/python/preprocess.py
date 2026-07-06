"""
DiskIVF-PostFiltering offline preprocessing.

Converts train_mds.pkl → train_access.npy (vid, tid pairs).
This is the same as the Curator and Pre-Filtering preprocess step.

Usage:
    python python/preprocess.py --dataset arxiv_small
    python python/preprocess.py --dataset yfcc100m_small
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def preprocess(dataset, data_dir="1_Data"):
    """Convert train_mds.pkl to train_access.npy."""
    gt_dir = Path(data_dir) / "ground_truth" / dataset
    mds_path = gt_dir / "train_mds.pkl"
    out_path = gt_dir / "train_access.npy"

    if not mds_path.exists():
        print(f"ERROR: {mds_path} not found")
        sys.exit(1)

    with open(mds_path, "rb") as f:
        mds = pickle.load(f)

    pairs = []
    for vid, tids in enumerate(mds):
        for tid in tids:
            pairs.append([vid, tid])

    pairs_arr = np.array(pairs, dtype=np.int32)
    np.save(out_path, pairs_arr)
    print(f"Saved {len(pairs)} access pairs to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. arxiv_small)")
    parser.add_argument("--data_dir", type=str, default="1_Data")
    args = parser.parse_args()
    preprocess(args.dataset, args.data_dir)


if __name__ == "__main__":
    main()
