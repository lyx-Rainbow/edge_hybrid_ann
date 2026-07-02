"""
Parameter sweep runner for Curator benchmark.

Runs Curator with multiple search-parameter combinations on a single
index build, collecting latency / recall / memory for each config.

Usage:
    python run_curator_sweep.py --config config/curator_yfcc100m.json \
        --sweep config/sweep_example.json
"""

import argparse
import copy
import json
import pickle
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))
from curator import Curator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_Utils"))
from memory_utils import getCurrentRSS, QueryRSSSampler  # noqa: E402
from predicate import compute_qualified_indices  # noqa: E402


# ---------------------------------------------------------------------------
# CP query helpers
# ---------------------------------------------------------------------------

def load_cp_data(dataset: str, data_dir: str):
    """Load complex-predicate queries, filters, and ground truth."""
    cp_dir = Path(data_dir) / "ground_truth" / dataset / "complex_predicate"
    cp_query_vecs = np.load(cp_dir / "query_vecs.npy").astype(np.float32)
    with open(cp_dir / "filters.json") as f:
        filters_info = json.load(f)
    cp_filters = filters_info["filters"]
    cp_selectivities = filters_info.get("selectivities", {})
    cp_gt = {}
    for formula in cp_filters:
        safe = formula.replace(" ", "_")
        cp_gt[formula] = np.load(cp_dir / f"gt_{safe}.npy")
    return cp_query_vecs, cp_filters, cp_selectivities, cp_gt


def build_cp_filters(index: Curator, train_mds: list, filters: list):
    """Precompute qualified labels and build index_filter for each filter."""
    print(f"  Building complex-predicate filters ({len(filters)} total)...")
    t0 = time.perf_counter()
    for formula in filters:
        qualified = compute_qualified_indices(formula, train_mds)
        index.index_filter(predicate=formula, qualified_labels=qualified)
    print(f"  Filters built in {time.perf_counter() - t0:.1f}s")


def run_cp_benchmark(
    index: Curator,
    filters: list,
    cp_query_vecs: np.ndarray,
    cp_gt: dict,
    k: int,
) -> dict:
    """Run CP queries for all filters. Returns per-filter results dict."""
    n_q = len(cp_query_vecs)
    per_filter = {}
    all_lats, all_recs = [], []

    for formula in filters:
        gt = cp_gt[formula]
        lats, recs = [], []
        for qi in range(n_q):
            t0 = time.perf_counter()
            res = index.query_with_complex_predicate(cp_query_vecs[qi], k, formula)
            ms = (time.perf_counter() - t0) * 1000
            lats.append(ms)
            valid = set(int(i) for i in gt[qi][:k] if i >= 0)
            if valid:
                recs.append(len(set(res[:k]) & valid) / len(valid))
            else:
                recs.append(1.0)
        per_filter[formula] = {
            "avg_latency_ms": float(np.mean(lats)),
            "avg_recall": float(np.mean(recs)),
            "qps": float(1000.0 / np.mean(lats)) if np.mean(lats) > 0 else 0,
        }
        all_lats.extend(lats)
        all_recs.extend(recs)

    return {
        "n_filters": len(filters),
        "n_queries": n_q,
        "avg_latency_ms": float(np.mean(all_lats)),
        "avg_recall": float(np.mean(all_recs)),
        "qps": float(1000.0 / np.mean(all_lats)) if all_lats else 0,
        "per_filter": per_filter,
    }


# ---------------------------------------------------------------------------
# Reused from run_curator.py
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_data(dataset: str, data_dir: str):
    gt_dir = Path(data_dir) / "ground_truth" / dataset
    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        train_mds = pickle.load(f)
    query_vecs = np.load(gt_dir / "query_vecs.npy").astype(np.float32)
    query_labels = np.load(gt_dir / "query_labels.npy").astype(np.int32)
    with open(gt_dir / "query_info.json", "r") as f:
        query_info = json.load(f)
    ground_truth = np.load(gt_dir / "ground_truth.npy")
    return train_vecs, train_mds, query_vecs, query_labels, query_info, ground_truth


def compute_recall(pred_ids: list[int], gt_ids: np.ndarray, k: int) -> float:
    valid_gt = set(int(i) for i in gt_ids[:k] if i >= 0)
    if not valid_gt:
        return 1.0
    pred_set = set(pred_ids[:k])
    return len(pred_set & valid_gt) / len(valid_gt)


def build_index(train_vecs: np.ndarray, train_mds: list, cfg: dict) -> Curator:
    build_cfg = cfg["build"]
    search_cfg = cfg["search"]
    d = cfg["d"]

    index = Curator(
        d=d,
        nlist=build_cfg["nlist"],
        bf_capacity=build_cfg["bf_capacity"],
        bf_error_rate=build_cfg["bf_error_rate"],
        max_sl_size=build_cfg["max_sl_size"],
        clus_niter=build_cfg["clus_niter"],
        max_leaf_size=build_cfg["max_leaf_size"],
        variance_boost=search_cfg["variance_boost"],
        search_ef=search_cfg["search_ef"],
        beam_size=search_cfg["beam_size"],
        use_temp_index_caching=search_cfg["use_temp_index_caching"],
        pq_M=build_cfg["pq_M"],
        pq_nbits=build_cfg["pq_nbits"],
        pq_enabled=build_cfg["pq_enabled"],
        pq_use_adc_rerank=build_cfg["pq_use_adc_rerank"],
        pq_rerank_topk_factor=build_cfg["pq_rerank_topk_factor"],
        use_flash_storage=build_cfg["use_flash_storage"],
        disk_cache_prefix="4_Results/Curator/disk_data_sweep",
    )

    print(f"  PQ: M={build_cfg['pq_M']}, nbits={build_cfg['pq_nbits']}, "
          f"sub-vector dim={d // build_cfg['pq_M']}")

    print("  Training (K-means clustering)...")
    t0 = time.perf_counter()
    index.train(train_vecs)
    print(f"  Training done in {time.perf_counter() - t0:.1f}s")

    print(f"  Adding {len(train_vecs)} vectors...")
    t0 = time.perf_counter()
    for i in range(len(train_vecs)):
        index.create(train_vecs[i], label=i)
        if (i + 1) % 100000 == 0:
            print(f"    {i + 1}/{len(train_vecs)} vectors added ({time.perf_counter() - t0:.1f}s)")
    print(f"  All vectors added in {time.perf_counter() - t0:.1f}s")

    print("  Granting tenant access...")
    t0 = time.perf_counter()
    for i, labels in enumerate(train_mds):
        for lab in labels:
            index.grant_access(i, int(lab))
    print(f"  Access granted in {time.perf_counter() - t0:.1f}s")

    print("  Flushing...")
    t0 = time.perf_counter()
    index.flush()
    print(f"  Flush done in {time.perf_counter() - t0:.1f}s")

    return index


def run_benchmark(
    index: Curator,
    search_params: dict,
    query_vecs: np.ndarray,
    query_labels: np.ndarray,   # int32, [n_queries]
    query_info: list,           # per-query metadata with "bucket" field
    ground_truth: np.ndarray,
    k: int,
    num_warmup: int,
):
    """Run benchmark with given search params. Returns results dict with per_bucket."""
    index.search_params = search_params

    n_queries = len(query_vecs)

    # Warmup
    for i in range(min(num_warmup, n_queries)):
        index.query(query_vecs[i], k=k, tenant_id=int(query_labels[i]))

    latencies = []
    recalls = []
    empty_results = 0
    bucket_lats = {}
    bucket_recs = {}
    rss_sampler = QueryRSSSampler()

    for i in range(n_queries):
        t0 = time.perf_counter()
        result_ids = index.query(query_vecs[i], k=k, tenant_id=int(query_labels[i]))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        latencies.append(elapsed_ms)
        rss_sampler.sample()
        recall = compute_recall(result_ids, ground_truth[i], k)
        recalls.append(recall)

        if len(result_ids) == 0 or all(r == -1 for r in result_ids):
            empty_results += 1

        bucket = query_info[i]["bucket"]
        if "overflow" not in bucket.lower():
            bucket_lats.setdefault(bucket, []).append(elapsed_ms)
            bucket_recs.setdefault(bucket, []).append(recall)

    per_bucket = {}
    for b in sorted(bucket_lats):
        per_bucket[b] = {
            "count": len(bucket_lats[b]),
            "avg_latency_ms": float(np.mean(bucket_lats[b])),
            "avg_recall": float(np.mean(bucket_recs[b])),
        }

    return {
        "n_queries": n_queries,
        "k": k,
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "qps": float(1000.0 / np.mean(latencies)),
        "avg_recall": float(np.mean(recalls)),
        "min_recall": float(np.min(recalls)),
        "empty_results": empty_results,
        "per_bucket": per_bucket,
        "rss_peak_query_mb": rss_sampler.peak_mb,
    }


def expand_sweep(sweep_cfg: dict) -> list[dict]:
    """Expand a sweep config into a list of individual search-parameter dicts.

    sweep_cfg looks like:
    {
      "search_ef": [64, 128, 256],
      "beam_size": [1, 2, 4]
    }
    """
    keys = list(sweep_cfg.keys())
    values = [sweep_cfg[k] for k in keys]
    combos = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Curator parameter sweep benchmark"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Base config JSON (e.g. config/curator_yfcc100m.json)")
    parser.add_argument("--sweep", type=str, required=True,
                        help="Sweep config JSON (parameter -> list of values)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["arxiv", "arxiv_small", "yfcc100m", "yfcc100m_small",
                                "sift1m", "sift1m_small", "gist1m", "gist1m_small"],
                        help="Dataset name (uses built-in defaults; overridden by --config)")
    parser.add_argument("--data_dir", type=str, default="1_Data")
    parser.add_argument("--output_dir", type=str, default="4_Results/Curator")
    args = parser.parse_args()

    # Load configs — prefer --config, fall back to --dataset defaults
    if args.config:
        cfg = load_config(args.config)
    elif args.dataset:
        # Build a minimal config from built-in defaults (same as run_curator.py)
        from run_curator import BUILTIN_DEFAULTS as CURATOR_DEFAULTS  # noqa: E402
        if args.dataset not in CURATOR_DEFAULTS:
            parser.error(f"No built-in defaults for dataset '{args.dataset}'")
        cfg = CURATOR_DEFAULTS[args.dataset]
    else:
        parser.error("Either --config or --dataset must be provided")

    sweep_cfg = load_config(args.sweep)
    dataset = cfg["dataset"]
    bench_cfg = cfg["benchmark"]

    print(f"\n{'='*60}")
    print(f"Curator Sweep: {dataset}")
    print(f"{'='*60}")
    print(f"  Base config: {args.config}")

    # Expand parameter combinations
    combos = expand_sweep(sweep_cfg)
    print(f"  Sweep combinations: {len(combos)}")
    for i, combo in enumerate(combos):
        print(f"    [{i}] {combo}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n  Loading data...")
    train_vecs, train_mds, query_vecs, query_labels, query_info, ground_truth = \
        load_data(dataset, args.data_dir)
    print(f"  Train: {train_vecs.shape[0]:,} x {train_vecs.shape[1]}")
    print(f"  Queries: {len(query_vecs)}, k={bench_cfg['k']}")

    # ------------------------------------------------------------------
    # 2. Build index once
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Building Curator index (shared across all sweep points)")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    index = build_index(train_vecs, train_mds, cfg)
    t_build = time.perf_counter() - t0

    index_mem = index.get_index_memory_bytes() / (1024 * 1024)
    rss_after_build = getCurrentRSS() / (1024 * 1024)
    
    print(f"\n  Build time:  {t_build:.1f}s ({t_build / 60:.1f} min)")
    print(f"  Index memory: {index_mem:.1f} MB")
    print(f"  RSS after build: {rss_after_build:.1f} MB")
    print(f"  Peak RSS build:  {rss_after_build:.1f} MB")

    # ------------------------------------------------------------------
    # 2b. Load CP data and pre-build filters
    # ------------------------------------------------------------------
    cp_enabled = True
    try:
        cp_query_vecs, cp_filters, cp_selectivities, cp_gt = \
            load_cp_data(dataset, args.data_dir)
        build_cp_filters(index, train_mds, cp_filters)
        print(f"  CP queries: {len(cp_query_vecs)}, filters: {len(cp_filters)}")
    except Exception as e:
        print(f"  CP data not available ({e}), skipping CP queries")
        cp_enabled = False

    # Free Python-side data no longer needed (C++ has its own copies & flash storage)
    del train_vecs, train_mds
    import gc; gc.collect()
    rss_after_free = getCurrentRSS() / (1024 * 1024)
    print(f"  RSS after freeing train data: {rss_after_free:.1f} MB")

    # ------------------------------------------------------------------
    # 3. Sweep
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Sweeping {len(combos)} parameter combinations")
    print(f"{'='*60}")

    sweep_results = []
    for i, combo in enumerate(combos):
        print(f"\n  [{i + 1}/{len(combos)}] {combo}")

        # Merge combo with base search params
        params = copy.deepcopy(cfg["search"])
        params.update(combo)

        result = run_benchmark(
            index, params,
            query_vecs, query_labels, query_info, ground_truth,
            k=bench_cfg["k"],
            num_warmup=bench_cfg["num_warmup"],
        )
        entry = {
            "params": params,
            "latency_ms": result["avg_latency_ms"],
            "p50_ms": result["p50_latency_ms"],
            "p95_ms": result["p95_latency_ms"],
            "p99_ms": result["p99_latency_ms"],
            "qps": result["qps"],
            "recall": result["avg_recall"],
            "min_recall": result["min_recall"],
            "empty_results": result["empty_results"],
            "per_bucket": result["per_bucket"],
            "rss_peak_query_mb": result["rss_peak_query_mb"],
        }

        if cp_enabled:
            cp_result = run_cp_benchmark(
                index, cp_filters, cp_query_vecs, cp_gt, k=bench_cfg["k"],
            )
            entry["complex_predicate"] = cp_result
            print(f"    SL: lat={entry['latency_ms']:.2f}ms  "
                  f"QPS={entry['qps']:.0f}  recall@{bench_cfg['k']}={entry['recall']:.4f}")
            print(f"    CP: lat={cp_result['avg_latency_ms']:.2f}ms  "
                  f"QPS={cp_result['qps']:.0f}  recall@{bench_cfg['k']}={cp_result['avg_recall']:.4f}")
        else:
            print(f"    latency={entry['latency_ms']:.2f}ms  "
                  f"QPS={entry['qps']:.0f}  recall@{bench_cfg['k']}={entry['recall']:.4f}")

        sweep_results.append(entry)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Sweep Summary")
    print(f"{'='*60}")
    header = (f"  {'#':>3s}  {'search_ef':>10s} "
              f"{'boost':>6s} {'beam':>5s}  "
              f"{'lat(ms)':>8s} {'QPS':>8s} {'recall':>8s}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for i, entry in enumerate(sweep_results):
        p = entry["params"]
        print(f"  {i:3d}  "
              f"{p.get('search_ef', '-'):>10} "
              f"{p.get('variance_boost', '-'):>6} "
              f"{p.get('beam_size', '-'):>5}  "
              f"{entry['latency_ms']:8.2f} "
              f"{entry['qps']:8.0f} "
              f"{entry['recall']:8.4f}")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "dataset": dataset,
        "base_config": cfg,
        "sweep_config": sweep_cfg,
        "build_time_s": float(t_build),
        "index_memory_mb": float(index_mem),
        "rss_after_build_mb": float(rss_after_build),        "n_combinations": len(combos),
        "sweep_results": sweep_results,
    }

    out_path = output_dir / f"sweep_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {out_path.resolve()}")
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
