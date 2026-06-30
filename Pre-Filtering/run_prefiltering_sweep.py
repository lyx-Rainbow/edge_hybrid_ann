"""
Pre-Filtering parameter sweep (single-label queries).

PreFiltering has no tunable search params (brute-force). This script
exists for API consistency — it runs one combination and reports results.

Usage:
    python Pre-Filtering/run_prefiltering_sweep.py --dataset yfcc100m_small \
        --sweep config/Pre-Filtering/sweep.json
"""

import argparse
import copy
import json
import pickle
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_Utils"))
from memory_utils import getPeakRSS, getCurrentRSS, QueryRSSSampler  # noqa: E402

# ---- Import index and defaults from the runner ----
from run_prefiltering import (  # noqa: E402
    PreFilteringIndex,
    compute_recall,
    load_data,
    load_config as load_idx_config,
)


def expand_sweep(sweep_cfg: dict) -> list[dict]:
    keys = list(sweep_cfg.keys())
    # Filter out comment keys
    keys = [k for k in keys if not k.startswith("_")]
    if not keys:
        return [{}]
    values = [sweep_cfg[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def load_cp_data(dataset, data_dir):
    cp_dir = Path(data_dir) / "ground_truth" / dataset / "complex_predicate"
    cp_query_vecs = np.load(cp_dir / "query_vecs.npy").astype(np.float32)
    with open(cp_dir / "filters.json") as f:
        fi = json.load(f)
    cp_filters = fi["filters"]
    cp_gt = {}
    for formula in cp_filters:
        safe = formula.replace(" ", "_")
        cp_gt[formula] = np.load(cp_dir / f"gt_{safe}.npy")
    return cp_query_vecs, cp_filters, cp_gt

def run_cp_benchmark(index, filters, cp_query_vecs, cp_gt, k):
    n_q = len(cp_query_vecs)
    per_filter, all_lats, all_recs = {}, [], []
    for formula in filters:
        gt = cp_gt[formula]; lats, recs = [], []
        for qi in range(n_q):
            t0 = time.perf_counter()
            res = index.query_with_complex_predicate(cp_query_vecs[qi], k, formula)
            ms = (time.perf_counter() - t0) * 1000
            lats.append(ms)
            valid = set(int(i) for i in gt[qi][:k] if i >= 0)
            recs.append(len(set(res[:k]) & valid) / len(valid) if valid else 1.0)
        per_filter[formula] = {
            "avg_latency_ms": float(np.mean(lats)),
            "avg_recall": float(np.mean(recs)),
            "qps": float(1000.0 / np.mean(lats)) if np.mean(lats) > 0 else 0,
        }
        all_lats.extend(lats); all_recs.extend(recs)
    return {
        "n_filters": len(filters), "n_queries": n_q,
        "avg_latency_ms": float(np.mean(all_lats)),
        "avg_recall": float(np.mean(all_recs)),
        "qps": float(1000.0 / np.mean(all_lats)) if all_lats else 0,
        "per_filter": per_filter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["arxiv", "arxiv_small", "yfcc100m", "yfcc100m_small"])
    parser.add_argument("--sweep", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="1_Data")
    parser.add_argument("--output_dir", type=str, default="4_Results/Pre-Filtering")
    args = parser.parse_args()

    if args.config:
        cfg = load_idx_config(args.config, None)
        dataset = cfg.get("dataset", args.dataset)
    elif args.dataset:
        cfg = load_idx_config(None, args.dataset)
        dataset = args.dataset
    else:
        parser.error("Need --config or --dataset")

    with open(args.sweep) as f:
        sweep_cfg = json.load(f)

    k = cfg["k"]
    num_warmup = cfg.get("num_warmup", 20)
    combos = expand_sweep(sweep_cfg)
    print(f"\n{'='*60}\nPre-Filtering Sweep: {dataset}\n{'='*60}")
    print(f"  Combinations: {len(combos)}")

    # Load
    train_vecs, train_mds, query_vecs, query_labels, query_info, sl_gt = \
        load_data(dataset, args.data_dir)
    print(f"  Train: {train_vecs.shape[0]:,} x {train_vecs.shape[1]}")
    print(f"  Queries: {len(query_vecs)}, k={k}")

    # Build
    print(f"\nBuilding index...")
    index = PreFilteringIndex()
    t0 = time.perf_counter()
    index.build(train_vecs, train_mds)
    t_build = time.perf_counter() - t0
    idx_mem = index.get_index_memory_bytes() / (1024 * 1024)
    rss_build = getCurrentRSS() / (1024 * 1024)
    peak_build = getPeakRSS() / (1024 * 1024)
    print(f"  Build: {t_build:.1f}s, mem: {idx_mem:.1f} MB, "
          f"RSS: {rss_build:.1f} MB, peak: {peak_build:.1f} MB")

    cp_enabled = True
    try:
        cp_query_vecs, cp_filters, cp_gt = load_cp_data(dataset, args.data_dir)
        print(f"  CP queries: {len(cp_query_vecs)}, filters: {len(cp_filters)}")
    except Exception as e:
        print(f"  CP data not available ({e}), skipping")
        cp_enabled = False

    # Sweep
    sweep_results = []
    for i, combo in enumerate(combos):
        print(f"\n  [{i + 1}/{len(combos)}] {combo}")
        for _ in range(min(num_warmup, len(query_vecs))):
            index.query(query_vecs[0], k, int(query_labels[0]))

        lats, recs = [], []
        bucket_lats, bucket_recs = {}, {}
        _rss = QueryRSSSampler()
        for qi in range(len(query_vecs)):
            t0 = time.perf_counter()
            res = index.query(query_vecs[qi], k, int(query_labels[qi]))
            ms = (time.perf_counter() - t0) * 1000
            lats.append(ms)
            _rss.sample()
            rec = compute_recall(res, sl_gt[qi], k)
            recs.append(rec)
            bucket = query_info[qi]["bucket"]
            if "overflow" not in bucket.lower():
                bucket_lats.setdefault(bucket, []).append(ms)
                bucket_recs.setdefault(bucket, []).append(rec)

        per_bucket = {}
        for b in sorted(bucket_lats):
            per_bucket[b] = {
                "count": len(bucket_lats[b]),
                "avg_latency_ms": float(np.mean(bucket_lats[b])),
                "avg_recall": float(np.mean(bucket_recs[b])),
            }

        entry = {
            "params": combo,
            "latency_ms": float(np.mean(lats)),
            "p50_ms": float(np.percentile(lats, 50)),
            "p95_ms": float(np.percentile(lats, 95)),
            "qps": float(1000.0 / np.mean(lats)),
            "recall": float(np.mean(recs)),
            "per_bucket": per_bucket,
            "rss_peak_query_mb": _rss.peak_mb,
        }

        if cp_enabled:
            cp_result = run_cp_benchmark(
                index, cp_filters, cp_query_vecs, cp_gt, k=cfg["k"])
            entry["complex_predicate"] = cp_result
        sweep_results.append(entry)
        print(f"    lat={entry['latency_ms']:.2f}ms  QPS={entry['qps']:.0f}  recall@{k}={entry['recall']:.4f}")

    # Summary
    print(f"\n{'='*60}\nSweep Summary\n{'='*60}")
    print(f"  {'#':>3s}  {'nprobe':>7s}  {'lat(ms)':>8s}  {'QPS':>8s}  {'recall':>8s}")
    for i, e in enumerate(sweep_results):
        print(f"  {i:3d}  {e['params'].get('nprobe', '-'):>7}  "
              f"{e['latency_ms']:8.2f}  {e['qps']:8.0f}  {e['recall']:8.4f}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": dataset, "index": "PreFiltering", "k": k,
        "build_time_s": float(t_build),
        "index_memory_mb": float(idx_mem),
        "rss_after_build_mb": float(rss_build),
        "peak_rss_build_mb": float(peak_build),
        "n_combinations": len(combos),
        "sweep_results": sweep_results,
    }
    out_path = out_dir / f"sweep_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
