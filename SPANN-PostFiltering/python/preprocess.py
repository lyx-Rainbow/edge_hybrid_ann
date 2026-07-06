#!/usr/bin/env python3
"""
preprocess.py — SPANN-PostFiltering 数据预处理

职责:
  1. train_mds.pkl → train_access.npy（与其他索引相同）
  2. train_mds.pkl → metadata.bin（vid_to_labels 二进制格式，供 C++ 直接读取）
  3. 生成 JSON 配置文件

用法:
  python preprocess.py --dataset arxiv_small
  python preprocess.py --dataset yfcc100m --output_dir /tmp/spann_prep
"""

import argparse
import json
import os
import pickle
import struct
import sys
from pathlib import Path

import numpy as np


def convert_mds_to_access(train_mds: list, output_path: str) -> int:
    """Convert train_mds (list of label lists) to access pairs .npy."""
    pairs = []
    for vid, tids in enumerate(train_mds):
        for tid in tids:
            pairs.append([vid, tid])

    pairs_arr = np.array(pairs, dtype=np.int32)
    np.save(output_path, pairs_arr)
    print(f"  train_access.npy: {len(pairs)} access pairs saved to {output_path}")
    return len(pairs)


def convert_mds_to_metadata_bin(train_mds: list, output_path: str):
    """Convert train_mds to binary metadata format for C++ PostFiltering.

    Format:
      [4 bytes] n_vectors (uint32)
      For each vector i:
        [4 bytes] n_labels_i (uint32)
        [n_labels_i × 4 bytes] labels (int32 array, sorted)
    """
    n = len(train_mds)

    with open(output_path, "wb") as f:
        # Header: n_vectors
        f.write(struct.pack("<I", n))

        for labels in train_mds:
            # Sort labels to enable binary_search in C++
            sorted_labels = sorted(labels)
            nl = len(sorted_labels)
            f.write(struct.pack("<I", nl))
            if nl > 0:
                # Pack all labels as int32 little-endian
                f.write(struct.pack(f"<{nl}i", *sorted_labels))

    file_size = os.path.getsize(output_path)
    total_labels = sum(len(md) for md in train_mds)
    print(f"  metadata.bin: {n} vectors, {total_labels} total labels, "
          f"{file_size / 1024:.1f} KB saved to {output_path}")


def generate_default_config(dataset: str, output_path: str):
    """Generate a default SPANN config JSON based on dataset."""
    defaults = {
        "arxiv_small": {
            "dataset": "arxiv_small", "d": 384,
            "dist_method": "L2", "num_threads": 1,
            "max_check": 4096, "hash_exp": 8,
            "k": 10, "num_warmup": 10,
            "overfetch_factor": 50, "overfetch_adaptive": True,
            "batch_query": False,
        },
        "yfcc100m_small": {
            "dataset": "yfcc100m_small", "d": 192,
            "dist_method": "L2", "num_threads": 1,
            "max_check": 4096, "hash_exp": 8,
            "k": 10, "num_warmup": 10,
            "overfetch_factor": 50, "overfetch_adaptive": True,
            "batch_query": False,
        },
        "arxiv": {
            "dataset": "arxiv", "d": 384,
            "dist_method": "L2", "num_threads": 1,
            "max_check": 8192, "hash_exp": 8,
            "k": 10, "num_warmup": 20,
            "overfetch_factor": 50, "overfetch_adaptive": True,
            "batch_query": False,
        },
        "yfcc100m": {
            "dataset": "yfcc100m", "d": 192,
            "dist_method": "L2", "num_threads": 1,
            "max_check": 8192, "hash_exp": 8,
            "k": 10, "num_warmup": 20,
            "overfetch_factor": 50, "overfetch_adaptive": True,
            "batch_query": False,
        },
    }

    if dataset not in defaults:
        print(f"Warning: no defaults for dataset '{dataset}', using arxiv_small defaults")
        cfg = defaults["arxiv_small"].copy()
        cfg["dataset"] = dataset
    else:
        cfg = defaults[dataset].copy()

    with open(output_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Default config saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SPANN-PostFiltering data preprocessing")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., arxiv_small, yfcc100m)")
    parser.add_argument("--data_dir", type=str, default="1_Data",
                        help="Root data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data_dir/ground_truth/<dataset>/)")
    args = parser.parse_args()

    dataset = args.dataset
    data_root = Path(args.data_dir)
    gt_dir = data_root / "ground_truth" / dataset

    if not gt_dir.exists():
        print(f"Error: dataset directory not found: {gt_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else gt_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SPANN-PostFiltering Preprocessing: {dataset}")
    print(f"  Data dir:  {gt_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}")

    # Step 1: Load train_mds.pkl
    mds_path = gt_dir / "train_mds.pkl"
    print(f"\nLoading train_mds.pkl from {mds_path} ...")
    with open(mds_path, "rb") as f:
        train_mds = pickle.load(f)
    print(f"  Loaded {len(train_mds)} vectors with labels")

    # Step 2: Generate train_access.npy
    print("\nGenerating train_access.npy ...")
    access_path = output_dir / "train_access.npy"
    convert_mds_to_access(train_mds, str(access_path))

    # Step 3: Generate metadata.bin (C++ PostFiltering compatible)
    print("\nGenerating metadata.bin ...")
    metadata_path = output_dir / "metadata.bin"
    convert_mds_to_metadata_bin(train_mds, str(metadata_path))

    # Step 4: Generate default config
    print("\nGenerating default config ...")
    config_path = output_dir / "spann_config.json"
    generate_default_config(dataset, str(config_path))

    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"  Outputs:")
    print(f"    {access_path}")
    print(f"    {metadata_path}")
    print(f"    {config_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
