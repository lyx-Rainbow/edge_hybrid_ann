"""
Curator build-parameter sweep: test different build configs, measure
index memory / RSS / build time only (no queries).

Usage:
    python Curator/run_curator_build_sweep.py \
        --config 3_Config/Curator/curator_yfcc100m_small.json \
        --sweep 3_Config/Curator/build_sweep.json
"""

import argparse
import copy
import gc
import json
import pickle
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))
from curator import Curator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_Utils"))
from memory_utils import getCurrentRSS  # noqa: E402


def load_data(dataset, data_dir):
    gt_dir = Path(data_dir) / "ground_truth" / dataset
    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        train_mds = pickle.load(f)
    return train_vecs, train_mds


def build_index(train_vecs, train_mds, build_cfg, search_cfg, d):
    index = Curator(
        d=d,
        nlist=build_cfg["nlist"],
        bf_capacity=build_cfg.get("bf_capacity", 1000),
        bf_error_rate=build_cfg.get("bf_error_rate", 0.001),
        max_sl_size=build_cfg.get("max_sl_size", 128),
        clus_niter=build_cfg.get("clus_niter", 20),
        max_leaf_size=build_cfg.get("max_leaf_size", 128),
        nprobe=search_cfg.get("nprobe", 1200),
        prune_thres=search_cfg.get("prune_thres", 1.6),
        variance_boost=search_cfg.get("variance_boost", 0.2),
        search_ef=search_cfg.get("search_ef", 128),
        beam_size=search_cfg.get("beam_size", 1),
        use_temp_index_caching=search_cfg.get("use_temp_index_caching", True),
        pq_M=build_cfg["pq_M"],
        pq_nbits=build_cfg["pq_nbits"],
        pq_enabled=build_cfg.get("pq_enabled", True),
        pq_use_adc_rerank=build_cfg.get("pq_use_adc_rerank", False),
        pq_rerank_topk_factor=build_cfg.get("pq_rerank_topk_factor", 4),
        use_flash_storage=build_cfg.get("use_flash_storage", False),
        disk_cache_prefix="4_Results/Curator/disk_data_build_sweep",
    )

    t0 = time.perf_counter()
    index.train(train_vecs)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(len(train_vecs)):
        index.create(train_vecs[i], label=i)
    t_add = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i, labels in enumerate(train_mds):
        for lab in labels:
            index.grant_access(i, int(lab))
    t_access = time.perf_counter() - t0

    t0 = time.perf_counter()
    index.flush()
    t_flush = time.perf_counter() - t0

    return index, {
        "train_s": float(t_train),
        "add_vectors_s": float(t_add),
        "grant_access_s": float(t_access),
        "flush_s": float(t_flush),
    }


def expand_sweep(sweep_cfg):
    keys = [k for k in sweep_cfg if not k.startswith("_")]
    if not keys:
        return [{}]
    values = [sweep_cfg[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sweep",  type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="1_Data")
    parser.add_argument("--output_dir", type=str, default="4_Results/Curator")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.sweep) as f:
        sweep_cfg = json.load(f)

    dataset = args.dataset or cfg["dataset"]
    combos = expand_sweep(sweep_cfg)

    print(f"\n{'='*60}")
    print(f"Curator Build Sweep: {dataset}")
    print(f"  Combinations: {len(combos)}")
    for i, c in enumerate(combos):
        print(f"    [{i}] {c}")
    print(f"{'='*60}")

    # Load data once
    print("\nLoading data...")
    train_vecs, train_mds = load_data(dataset, args.data_dir)
    print(f"  Train: {train_vecs.shape[0]:,} x {train_vecs.shape[1]}")
    d = train_vecs.shape[1]

    results = []
    for i, combo in enumerate(combos):
        print(f"\n  [{i + 1}/{len(combos)}] {combo}")

        # Free previous index before building new one
        if i > 0:
            del index
            gc.collect()

        build_cfg = copy.deepcopy(cfg["build"])
        build_cfg.update(combo)

        rss_before = getCurrentRSS() / (1024 * 1024)
        t0 = time.perf_counter()
        index, times = build_index(train_vecs, train_mds, build_cfg, cfg.get("search", {}), d)
        t_build = time.perf_counter() - t0
        rss_after = getCurrentRSS() / (1024 * 1024)
        idx_mem = index.get_index_memory_bytes() / (1024 * 1024)

        entry = {
            "params": combo,
            "build_time_s": float(t_build),
            "index_memory_mb": float(idx_mem),
            "rss_before_build_mb": float(rss_before),
            "rss_after_build_mb": float(rss_after),
            "rss_delta_mb": float(rss_after - rss_before),
            "times": times,
        }
        results.append(entry)
        print(f"    mem={idx_mem:.1f}MB  build={t_build:.1f}s  "
              f"rss_delta={entry['rss_delta_mb']:.1f}MB")

    # Summary
    print(f"\n{'='*60}")
    print("Summary (sorted by index_memory_mb)")
    print(f"{'='*60}")
    results_sorted = sorted(results, key=lambda e: e["index_memory_mb"])
    header = f"  {'#':>3s}  {'pq_M':>5s} {'pq_nbits':>8s}  {'mem(MB)':>9s}  {'build(s)':>9s}  {'rss_delta':>9s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for i, e in enumerate(results_sorted):
        p = e["params"]
        print(f"  {i:3d}  "
              f"{p.get('pq_M', '-'):>5}  "
              f"{p.get('pq_nbits', '-'):>8}  "
              f"{e['index_memory_mb']:9.1f}  "
              f"{e['build_time_s']:9.1f}  "
              f"{e['rss_delta_mb']:9.1f}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": dataset,
        "base_config": cfg,
        "sweep_config": sweep_cfg,
        "n_combinations": len(combos),
        "results": results,
    }
    out_path = out_dir / f"build_sweep_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
