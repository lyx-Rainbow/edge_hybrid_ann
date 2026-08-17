#!/usr/bin/env python3
"""
run_experiment.py — SPANN-PostFiltering 实验编排脚本

职责:
  1. 数据预处理（.pkl → train_access.npy + metadata.bin）
  2. 生成 JSON 配置文件
  3. subprocess 调用 C++ 二进制（./build/spann_pf bench ...）
  4. 读取 C++ 输出的 results.json，计算 Recall@k 指标
  5. 输出汇总结果到 4_Results/SPANN-PostFiltering/

用法:
  python python/run_experiment.py --dataset arxiv_small
  python python/run_experiment.py --dataset arxiv_small --config /tmp/spann_config.json --k 10 --profile
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── 内置默认配置 ──
BUILTIN_DEFAULTS = {
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
    "sift1m_small": {
        "dataset": "sift1m_small", "d": 128,
        "dist_method": "L2", "num_threads": 1,
        "max_check": 2048, "hash_exp": 6,
        "k": 10, "num_warmup": 5,
        "overfetch_factor": 50, "overfetch_adaptive": True,
        "batch_query": False,
    },
    "gist1m_small": {
        "dataset": "gist1m_small", "d": 960,
        "dist_method": "L2", "num_threads": 1,
        "max_check": 2048, "hash_exp": 6,
        "k": 10, "num_warmup": 5,
        "overfetch_factor": 50, "overfetch_adaptive": True,
        "batch_query": False,
    },
    "wit": {
        "dataset": "wit", "d": 384,
        "dist_method": "L2", "num_threads": 1,
        "max_check": 8192, "hash_exp": 8,
        "k": 10, "num_warmup": 20,
        "overfetch_factor": 50, "overfetch_adaptive": True,
        "batch_query": False,
    },
    "wit_small": {
        "dataset": "wit_small", "d": 384,
        "dist_method": "L2", "num_threads": 1,
        "max_check": 4096, "hash_exp": 8,
        "k": 10, "num_warmup": 10,
        "overfetch_factor": 50, "overfetch_adaptive": True,
        "batch_query": False,
    },
    # ── 100K datasets ──
    "sift1m_100k": {
        "dataset": "sift1m_100k", "d": 128,
        "dist_method": "L2", "num_threads": 1,
        "max_check": 8192, "hash_exp": 8,
        "k": 10, "num_warmup": 20,
        "overfetch_factor": 50, "overfetch_adaptive": True,
        "batch_query": False,
    },
    "yfcc100m_100k": {
        "dataset": "yfcc100m_100k", "d": 192,
        "dist_method": "L2", "num_threads": 1,
        "max_check": 8192, "hash_exp": 8,
        "k": 10, "num_warmup": 20,
        "overfetch_factor": 50, "overfetch_adaptive": True,
        "batch_query": False,
    },
    "arxiv_100k": {
        "dataset": "arxiv_100k", "d": 384,
        "dist_method": "L2", "num_threads": 1,
        "max_check": 8192, "hash_exp": 8,
        "k": 10, "num_warmup": 20,
        "overfetch_factor": 50, "overfetch_adaptive": True,
        "batch_query": False,
    },
    "gist1m_100k": {
        "dataset": "gist1m_100k", "d": 960,
        "dist_method": "L2", "num_threads": 1,
        "max_check": 8192, "hash_exp": 8,
        "k": 10, "num_warmup": 20,
        "overfetch_factor": 50, "overfetch_adaptive": True,
        "batch_query": False,
    },
    "wit_100k": {
        "dataset": "wit_100k", "d": 384,
        "dist_method": "L2", "num_threads": 1,
        "max_check": 8192, "hash_exp": 8,
        "k": 10, "num_warmup": 20,
        "overfetch_factor": 50, "overfetch_adaptive": True,
        "batch_query": False,
    },
}


def load_config(config_path, dataset):
    """Load config from file, or use builtin defaults."""
    if config_path:
        with open(config_path) as f:
            return json.load(f)
    if dataset in BUILTIN_DEFAULTS:
        return BUILTIN_DEFAULTS[dataset].copy()
    # Fallback
    print(f"Warning: no defaults for '{dataset}', using arxiv_small defaults")
    cfg = BUILTIN_DEFAULTS["arxiv_small"].copy()
    cfg["dataset"] = dataset
    return cfg


def load_data(dataset, data_dir):
    """Load training/query data and ground truth."""
    gt_dir = Path(data_dir) / "ground_truth" / dataset
    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        import pickle
        train_mds = pickle.load(f)
    query_vecs = np.load(gt_dir / "query_vecs.npy").astype(np.float32)
    query_labels = np.load(gt_dir / "query_labels.npy").astype(np.int32)
    with open(gt_dir / "query_info.json") as f:
        query_info = json.load(f)
    gt = np.load(gt_dir / "ground_truth.npy")
    return train_vecs, train_mds, query_vecs, query_labels, query_info, gt


def load_cp_data(dataset, data_dir):
    """Load complex-predicate query data."""
    cp_dir = Path(data_dir) / "ground_truth" / dataset / "complex_predicate"
    query_vecs = np.load(cp_dir / "query_vecs.npy").astype(np.float32)
    with open(cp_dir / "filters.json") as f:
        filters_info = json.load(f)
    gt_dict = {}
    for formula in filters_info["filters"]:
        safe = formula.replace(" ", "_")
        gt_dict[formula] = np.load(cp_dir / f"gt_{safe}.npy")
    return query_vecs, filters_info, gt_dict


def compute_recall(pred_ids, gt_ids, k):
    """Compute recall@k between predicted and ground-truth result lists."""
    valid = set(int(i) for i in gt_ids[:k] if i >= 0)
    if not valid:
        return 1.0
    return len(set(pred_ids[:k]) & valid) / len(valid)


def preprocess_if_needed(dataset, data_dir):
    """Generate train_access.npy and metadata.bin if they don't exist."""
    gt_dir = Path(data_dir) / "ground_truth" / dataset
    access_path = gt_dir / "train_access.npy"
    metadata_path = gt_dir / "metadata.bin"

    if access_path.exists() and metadata_path.exists():
        print(f"  Preprocessed files already exist, skipping...")
        return str(access_path), str(metadata_path)

    print(f"  Preprocessing: generating train_access.npy and metadata.bin...")
    from preprocess import convert_mds_to_access, convert_mds_to_metadata_bin

    with open(gt_dir / "train_mds.pkl", "rb") as f:
        import pickle
        train_mds = pickle.load(f)

    convert_mds_to_access(train_mds, str(access_path))
    convert_mds_to_metadata_bin(train_mds, str(metadata_path))

    return str(access_path), str(metadata_path)


def main():
    parser = argparse.ArgumentParser(
        description="SPANN-PostFiltering experiment runner")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file path")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., arxiv_small)")
    parser.add_argument("--data_dir", type=str, default="1_Data",
                        help="Root data directory")
    parser.add_argument("--output_dir", type=str, default="4_Results/SPANN-PostFiltering")
    parser.add_argument("--binary", type=str, default="./build/spann_pf",
                        help="Path to spann_pf executable")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of results per query (overrides config)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling output")
    parser.add_argument("--batch_query", action="store_true",
                        help="Enable inter-query OpenMP parallelism")
    parser.add_argument("--filter", type=str, default=None,
                        help="Complex predicate filter (overrides query_labels)")
    args = parser.parse_args()

    # ── Load config ──
    if args.config:
        cfg = load_config(args.config, None)
        dataset = cfg.get("dataset", args.dataset)
    elif args.dataset:
        cfg = load_config(None, args.dataset)
        dataset = args.dataset
    else:
        parser.error("Need --config or --dataset")

    if args.k is not None:
        cfg["k"] = args.k
    if args.batch_query:
        cfg["batch_query"] = True
        cfg["num_threads"] = 1

    k = cfg["k"]
    num_warmup = cfg.get("num_warmup", 20)

    print(f"\n{'='*60}")
    print(f"SPANN-PostFiltering Experiment: {dataset}")
    print(f"  k={k}, dist={cfg.get('dist_method', 'L2')}, "
          f"max_check={cfg.get('max_check', 8192)}")
    print(f"{'='*60}")

    # ── Load data ──
    print("\nLoading data...")
    train_vecs, train_mds, query_vecs, query_labels, query_info, sl_gt = \
        load_data(dataset, args.data_dir)
    print(f"  Train: {train_vecs.shape[0]:,} x {train_vecs.shape[1]}")
    print(f"  SL queries: {len(query_vecs)}")

    # ── Preprocess ──
    access_path, metadata_path = preprocess_if_needed(dataset, args.data_dir)

    # ── Generate config file ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"spann_{dataset}_cfg.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Construct CLI command ──
    index_dir = str(output_dir / f"spann_index_{dataset}")
    results_path = str(output_dir / f"spann_{dataset}_results.json")

    cmd = [
        args.binary, "bench",
        "--train_vecs", str(Path(args.data_dir) / "ground_truth" / dataset / "train_vecs.npy"),
        "--train_access", access_path,
        "--queries", str(Path(args.data_dir) / "ground_truth" / dataset / "query_vecs.npy"),
        "--query_labels", str(Path(args.data_dir) / "ground_truth" / dataset / "query_labels.npy"),
        "--config", str(config_path),
        "--k", str(k),
        "--index_dir", index_dir,
        "--output", results_path,
    ]
    if args.profile:
        cmd.append("--profile")
    if args.batch_query:
        cmd.append("--batch-query")
    if args.filter:
        cmd.extend(["--filter", args.filter])

    print(f"\n{'='*60}")
    print("Running C++ benchmark...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"Error: spann_pf exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nC++ benchmark completed in {elapsed:.1f}s")

    # ── Load results and compute recall ──
    if not os.path.exists(results_path):
        print(f"Error: results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        cpp_results = json.load(f)

    # Compute single-label recall
    recalls = []
    for q, qr in enumerate(cpp_results["queries"]):
        r = compute_recall(qr["labels"], sl_gt[q], k)
        recalls.append(r)

    recalls_arr = np.array(recalls)
    print(f"\n{'='*60}")
    print(f"Single-label Recall@{k}")
    print(f"  Mean:   {np.mean(recalls_arr):.4f}")
    print(f"  Median: {np.median(recalls_arr):.4f}")
    print(f"  P90:    {np.percentile(recalls_arr, 90):.4f}")
    print(f"  P99:    {np.percentile(recalls_arr, 99):.4f}")
    print(f"  Min:    {np.min(recalls_arr):.4f}")
    print(f"  ==1.0:  {np.sum(recalls_arr == 1.0)}/{len(recalls_arr)}")
    empty = sum(1 for qr in cpp_results["queries"]
                if all(x == -1 for x in qr["labels"]))
    print(f"  Empty:  {empty}")
    print(f"  Build:  {cpp_results['build_time_s']:.2f} s")
    print(f"  Memory: {cpp_results['memory_bytes'] / 1024**2:.1f} MB")
    print(f"{'='*60}\n")

    # ── Complex-predicate queries (only if no --filter override) ──
    if args.filter is None:
        cp_dir = Path(args.data_dir) / "ground_truth" / dataset / "complex_predicate"
        if cp_dir.exists():
            print(f"\n{'='*60}")
            print(f"Complex-predicate queries (k={k})")
            print(f"{'='*60}")

            cp_query_vecs, filters_info, cp_gt_dict = load_cp_data(dataset, args.data_dir)
            cp_filters = filters_info["filters"]
            cp_selectivities = filters_info.get("selectivities", {})
            n_cp = len(cp_query_vecs)
            print(f"  {len(cp_filters)} filters x {n_cp} queries")

            cp_all_recalls = []
            for fi, formula in enumerate(cp_filters):
                gt = cp_gt_dict[formula]

                # Run C++ benchmark for this filter
                cp_results_path = str(output_dir / f"spann_{dataset}_cp_{fi}_results.json")
                cp_cmd = [
                    args.binary, "bench",
                    "--train_vecs", str(Path(args.data_dir) / "ground_truth" / dataset / "train_vecs.npy"),
                    "--train_access", access_path,
                    "--queries", str(cp_dir / "query_vecs.npy"),
                    "--config", str(config_path),
                    "--k", str(k),
                    "--index_dir", index_dir,
                    "--output", cp_results_path,
                    "--filter", formula,
                ]

                t_cp = time.perf_counter()
                cp_result = subprocess.run(cp_cmd, capture_output=True, text=True)
                cp_elapsed = time.perf_counter() - t_cp

                if cp_result.returncode != 0:
                    print(f"  [{fi+1}/{len(cp_filters)}] FAILED: {formula}")
                    continue

                with open(cp_results_path) as f:
                    cp_res = json.load(f)

                cp_recs = []
                for q, qr in enumerate(cp_res["queries"]):
                    r = compute_recall(qr["labels"], gt[q], k)
                    cp_recs.append(r)

                sel = cp_selectivities.get(formula, 0)
                cp_all_recalls.extend(cp_recs)
                print(f"  [{fi+1}/{len(cp_filters)}] {formula}: "
                      f"recall={np.mean(cp_recs):.4f}, "
                      f"selectivity={sel:.4f}, "
                      f"time={cp_elapsed:.1f}s")

            if cp_all_recalls:
                print(f"\n  Overall CP Recall@{k}: {np.mean(cp_all_recalls):.4f}")
            print(f"{'='*60}\n")

    print("Experiment complete!")


if __name__ == "__main__":
    main()
