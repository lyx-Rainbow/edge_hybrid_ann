"""
Run Curator index benchmark on arxiv / yfcc100m datasets.

Loads the pre-computed train split and queries, builds a Curator index,
then runs filtered queries across different selectivity buckets.

Usage:
    python run_curator.py --config config/curator_yfcc100m.json
    python run_curator.py --dataset yfcc100m           # fallback: built-in defaults
    python run_curator.py --dataset arxiv --nlist 64   # override specific params
"""

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))
from curator import Curator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_Utils"))
from memory_utils import getPeakRSS, getCurrentRSS  # noqa: E402
from memory_profiler import MemoryProfiler  # noqa: E402
from query_profiler import QueryProfiler  # noqa: E402


# ---------------------------------------------------------------------------
# Built-in defaults (used when no --config is provided)
# ---------------------------------------------------------------------------

BUILTIN_DEFAULTS = {
    "arxiv": {
        "d": 384,
        "n_labels": 100,
        "build": {
            "nlist": 64,
            "bf_capacity": 1000,
            "bf_error_rate": 0.001,
            "max_sl_size": 128,
            "clus_niter": 20,
            "max_leaf_size": 128,
            "pq_M": 24,
            "pq_nbits": 8,
            "pq_enabled": True,
            "pq_use_adc_rerank": False,
            "pq_rerank_topk_factor": 4,
            "use_flash_storage": True,
        },
        "search": {
            "variance_boost": 0.2,
            "search_ef": 128,
            "beam_size": 1,
            "use_temp_index_caching": True,
        },
        "benchmark": {
            "k": 10,
            "n_queries": 1000,
            "num_warmup": 20,
        },
    },
    "yfcc100m": {
        "d": 192,
        "n_labels": 1000,
        "build": {
            "nlist": 32,
            "bf_capacity": 1000,
            "bf_error_rate": 0.001,
            "max_sl_size": 128,
            "clus_niter": 20,
            "max_leaf_size": 128,
            "pq_M": 16,
            "pq_nbits": 8,
            "pq_enabled": True,
            "pq_use_adc_rerank": False,
            "pq_rerank_topk_factor": 4,
            "use_flash_storage": True,
        },
        "search": {
            "variance_boost": 0.2,
            "search_ef": 128,
            "beam_size": 1,
            "use_temp_index_caching": True,
        },
        "benchmark": {
            "k": 10,
            "n_queries": 1000,
            "num_warmup": 20,
        },
    },
    "yfcc100m_small": {
        "d": 192,
        "n_labels": 200,
        "build": {
            "nlist": 32, "bf_capacity": 1000, "bf_error_rate": 0.001,
            "max_sl_size": 256, "clus_niter": 20, "max_leaf_size": 128,
            "pq_M": 64, "pq_nbits": 8, "pq_enabled": True,
            "pq_use_adc_rerank": True, "pq_rerank_topk_factor": 4,
            "use_flash_storage": True,
        },
        "search": {
            "variance_boost": 0.2, "search_ef": 1024, "beam_size": 4,
            "use_temp_index_caching": True,
        },
        "benchmark": {"k": 10, "n_queries": 500, "num_warmup": 10},
    },
    "arxiv_small": {
        "d": 384,
        "n_labels": 100,
        "build": {
            "nlist": 16, "bf_capacity": 1000, "bf_error_rate": 0.001,
            "max_sl_size": 128, "clus_niter": 10, "max_leaf_size": 128,
            "pq_M": 24, "pq_nbits": 8, "pq_enabled": True,
            "pq_use_adc_rerank": False, "pq_rerank_topk_factor": 4,
            "use_flash_storage": True,
        },
        "search": {
            "variance_boost": 0.4, "search_ef": 1024, "beam_size": 4,
            "use_temp_index_caching": True,
        },
        "benchmark": {"k": 10, "n_queries": 500, "num_warmup": 10},
    },
    "sift1m": {
        "d": 128,
        "n_labels": 100,
        "build": {
            "nlist": 64, "bf_capacity": 1000, "bf_error_rate": 0.001,
            "max_sl_size": 128, "clus_niter": 20, "max_leaf_size": 128,
            "pq_M": 16, "pq_nbits": 8, "pq_enabled": True,
            "pq_use_adc_rerank": True, "pq_rerank_topk_factor": 4,
            "use_flash_storage": True,
        },
        "search": {
            "variance_boost": 0.2, "search_ef": 128, "beam_size": 1,
            "use_temp_index_caching": True,
        },
        "benchmark": {"k": 10, "n_queries": 1000, "num_warmup": 20},
    },
    "gist1m": {
        "d": 960,
        "n_labels": 100,
        "build": {
            "nlist": 32, "bf_capacity": 1000, "bf_error_rate": 0.001,
            "max_sl_size": 128, "clus_niter": 20, "max_leaf_size": 128,
            "pq_M": 32, "pq_nbits": 8, "pq_enabled": True,
            "pq_use_adc_rerank": False, "pq_rerank_topk_factor": 4,
            "use_flash_storage": True,
        },
        "search": {
            "variance_boost": 0.2, "search_ef": 128, "beam_size": 1,
            "use_temp_index_caching": True,
        },
        "benchmark": {"k": 10, "n_queries": 1000, "num_warmup": 20},
    },
    "sift1m_small": {
        "d": 128,
        "n_labels": 50,
        "build": {
            "nlist": 32, "bf_capacity": 1000, "bf_error_rate": 0.001,
            "max_sl_size": 256, "clus_niter": 20, "max_leaf_size": 128,
            "pq_M": 16, "pq_nbits": 8, "pq_enabled": True,
            "pq_use_adc_rerank": True, "pq_rerank_topk_factor": 4,
            "use_flash_storage": True,
        },
        "search": {
            "variance_boost": 0.2, "search_ef": 1024, "beam_size": 4,
            "use_temp_index_caching": True,
        },
        "benchmark": {"k": 10, "n_queries": 200, "num_warmup": 10},
    },
    "gist1m_small": {
        "d": 960,
        "n_labels": 50,
        "build": {
            "nlist": 16, "bf_capacity": 1000, "bf_error_rate": 0.001,
            "max_sl_size": 128, "clus_niter": 20, "max_leaf_size": 128,
            "pq_M": 32, "pq_nbits": 8, "pq_enabled": True,
            "pq_use_adc_rerank": False, "pq_rerank_topk_factor": 4,
            "use_flash_storage": True,
        },
        "search": {
            "variance_boost": 0.4, "search_ef": 1024, "beam_size": 4,
            "use_temp_index_caching": True,
        },
        "benchmark": {"k": 10, "n_queries": 200, "num_warmup": 10},
    },
}


def load_config(config_path: str | None, dataset: str) -> dict:
    """Load config from JSON file, falling back to built-in defaults."""
    if config_path is not None:
        with open(config_path, "r") as f:
            return json.load(f)
    # Auto-load from standard config location
    auto_path = Path("3_Config/Curator") / f"curator_{dataset}.json"
    if auto_path.exists():
        with open(auto_path, "r") as f:
            return json.load(f)
    if dataset in BUILTIN_DEFAULTS:
        return BUILTIN_DEFAULTS[dataset]
    raise ValueError(f"No config file and no built-in defaults for dataset '{dataset}'")


def load_data(dataset: str, data_dir: str):
    """Load train vectors, train metadata, query vectors, query labels, GT."""
    gt_dir = Path(data_dir) / "ground_truth" / dataset

    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        train_mds = pickle.load(f)
    query_vecs = np.load(gt_dir / "query_vecs.npy").astype(np.float32)
    query_labels = np.load(gt_dir / "query_labels.npy").astype(np.int32)
    with open(gt_dir / "query_info.json", "r") as f:
        query_info = json.load(f)
    ground_truth = np.load(gt_dir / "ground_truth.npy")  # [n_queries, k], int32

    return train_vecs, train_mds, query_vecs, query_labels, query_info, ground_truth


def compute_recall(pred_ids: list[int], gt_ids: np.ndarray, k: int) -> float:
    """Compute recall@k against ground truth. GT entries with -1 are ignored."""
    valid_gt = set(int(i) for i in gt_ids[:k] if i >= 0)
    if not valid_gt:
        return 1.0
    pred_set = set(pred_ids[:k])
    return len(pred_set & valid_gt) / len(valid_gt)


def build_curator_index(
    train_vecs: np.ndarray,
    train_mds: list,
    cfg: dict,
) -> tuple[Curator, dict]:
    """Create, train, populate, and finalize a Curator index.

    Returns (index, memory_snapshots) where memory_snapshots records
    peak RSS at key build stages.
    """
    build_cfg = cfg["build"]
    search_cfg = cfg["search"]
    d = cfg["d"]

    memory = {}

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
        disk_cache_prefix="4_Results/Curator/disk_data",
    )

    print(f"  PQ: M={build_cfg['pq_M']}, nbits={build_cfg['pq_nbits']}, "
          f"sub-vector dim={d // build_cfg['pq_M']}")

    # Train
    print("  Training (K-means clustering)...")
    t0 = time.perf_counter()
    index.train(train_vecs)
    t_train = time.perf_counter() - t0
    print(f"  Training done in {t_train:.1f}s")

    # Add vectors
    print(f"  Adding {len(train_vecs)} vectors...")
    t0 = time.perf_counter()
    for i in range(len(train_vecs)):
        index.create(train_vecs[i], label=i)
        if (i + 1) % 100000 == 0:
            print(f"    {i + 1}/{len(train_vecs)} vectors added ({time.perf_counter() - t0:.1f}s)")
    t_add = time.perf_counter() - t0
    print(f"  All vectors added in {t_add:.1f}s")

    # Grant access
    print("  Granting tenant access...")
    t0 = time.perf_counter()
    for i, labels in enumerate(train_mds):
        for lab in labels:
            index.grant_access(i, int(lab))
    t_access = time.perf_counter() - t0
    print(f"  Access granted in {t_access:.1f}s")

    # Flush (finalize PQ + flash)
    print("  Flushing (finalizing index)...")
    t0 = time.perf_counter()
    index.flush()
    t_flush = time.perf_counter() - t0
    print(f"  Flush done in {t_flush:.1f}s")

    # Snapshot: after build
    memory["rss_after_build_mb"] = getCurrentRSS() / (1024 * 1024)
    memory["index_memory_mb"] = index.get_index_memory_bytes() / (1024 * 1024)
    memory["build_times"] = {
        "train_s": float(t_train),
        "add_vectors_s": float(t_add),
        "grant_access_s": float(t_access),
        "flush_s": float(t_flush),
    }

    return index, memory


def build_curator_index_profiled(
    train_vecs: np.ndarray,
    train_mds: list,
    cfg: dict,
) -> tuple[Curator, dict]:
    """Like build_curator_index but captures per-phase memory breakdowns."""
    build_cfg = cfg["build"]
    search_cfg = cfg["search"]
    d = cfg["d"]

    memory = {}
    phases = {}

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
        disk_cache_prefix="4_Results/Curator/disk_data",
    )

    profiler = MemoryProfiler(index)

    snap_before = profiler.get_snapshot()

    # Train
    print("  Training (K-means clustering)...")
    t0 = time.perf_counter()
    index.train(train_vecs)
    phases["train_s"] = float(time.perf_counter() - t0)
    snap_after_train = profiler.get_snapshot()

    # Add vectors
    print(f"  Adding {len(train_vecs)} vectors...")
    t0 = time.perf_counter()
    for i in range(len(train_vecs)):
        index.create(train_vecs[i], label=i)
    phases["add_vectors_s"] = float(time.perf_counter() - t0)
    snap_after_add = profiler.get_snapshot()

    # Grant access
    print("  Granting tenant access...")
    t0 = time.perf_counter()
    for i, labels in enumerate(train_mds):
        for lab in labels:
            index.grant_access(i, int(lab))
    phases["grant_access_s"] = float(time.perf_counter() - t0)
    snap_after_grant = profiler.get_snapshot()

    # Flush
    print("  Flushing (finalizing index)...")
    t0 = time.perf_counter()
    index.flush()
    phases["flush_s"] = float(time.perf_counter() - t0)
    snap_after_flush = profiler.get_snapshot()

    memory["rss_after_build_mb"] = getCurrentRSS() / (1024 * 1024)
    memory["index_memory_mb"] = index.get_index_memory_bytes() / (1024 * 1024)
    memory["build_times"] = phases
    memory["memory_snapshots"] = {
        "before_train": snap_before,
        "after_train": snap_after_train,
        "after_add": snap_after_add,
        "after_grant": snap_after_grant,
        "after_flush": snap_after_flush,
    }
    # Phase-by-phase deltas
    memory["memory_deltas"] = profiler.compare_snapshots(snap_before, snap_after_flush)

    return index, memory


def run_queries(
    index: Curator,
    query_vecs: np.ndarray,
    query_labels: np.ndarray,   # int32, [n_queries] — one label per query
    query_info: list,
    ground_truth: np.ndarray,
    k: int = 10,
    num_warmup: int = 20,
    profile: bool = False,
):
    """Run single-label filtered queries, matching single-label GT."""
    n_queries = len(query_vecs)
    query_profiler = QueryProfiler(index) if profile else None

    # Warmup
    print(f"  Warming up ({num_warmup} queries)...")
    for i in range(min(num_warmup, n_queries)):
        index.query(query_vecs[i], k=k, tenant_id=int(query_labels[i]))

    # Benchmark
    print(f"  Benchmarking {n_queries} queries...")
    latencies = []
    recalls = []
    profile_samples = [] if profile else None  # sampled profiles
    bucket_latencies = {}
    bucket_recalls = {}
    empty_results = 0

    rss_peak_query = getCurrentRSS() / (1024 * 1024)

    for i in range(n_queries):
        q_vec = query_vecs[i]
        tenant_id = int(query_labels[i])

        if profile:
            result_ids, step_profile = index.query_with_profile(
                q_vec, k=k, tenant_id=tenant_id
            )
            elapsed_ms = step_profile.get("total_search_time_ms", 0.0)
            # Sample profiles: one every 20 queries to keep output size manageable
            if i % 20 == 0:
                profile_samples.append({
                    "query_idx": i,
                    "bucket": query_info[i].get("bucket", "unknown"),
                    "selectivity": query_info[i].get("selectivity", 0.0),
                    **step_profile,
                })
        else:
            t0 = time.perf_counter()
            result_ids = index.query(q_vec, k=k, tenant_id=tenant_id)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        latencies.append(elapsed_ms)
        rss_peak_query = max(rss_peak_query, getCurrentRSS() / (1024 * 1024))

        recall = compute_recall(result_ids, ground_truth[i], k)
        recalls.append(recall)

        if len(result_ids) == 0 or all(r == -1 for r in result_ids):
            empty_results += 1

        bucket = query_info[i]["bucket"]
        bucket_latencies.setdefault(bucket, []).append(elapsed_ms)
        bucket_recalls.setdefault(bucket, []).append(recall)

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n_queries} queries done")

    results = {
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
        "rss_peak_query_mb": float(rss_peak_query),
        "per_bucket": {},
    }

    if profile_samples is not None:
        results["profile_samples"] = profile_samples

    for bucket in sorted(bucket_latencies.keys()):
        lats = bucket_latencies[bucket]
        recs = bucket_recalls[bucket]
        results["per_bucket"][bucket] = {
            "count": len(lats),
            "avg_latency_ms": float(np.mean(lats)),
            "p50_latency_ms": float(np.percentile(lats, 50)),
            "qps": float(1000.0 / np.mean(lats)) if np.mean(lats) > 0 else 0.0,
            "avg_recall": float(np.mean(recs)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Curator index benchmark on arxiv/yfcc100m"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file (e.g. config/curator_yfcc100m.json)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["arxiv", "arxiv_small", "yfcc100m", "yfcc100m_small",
                 "sift1m", "sift1m_small", "gist1m", "gist1m_small"],
        help="Dataset name (only used when --config is not provided)",
    )
    parser.add_argument("--data_dir", type=str, default="1_Data")
    parser.add_argument("--output_dir", type=str, default="4_Results/Curator")
    # CLI overrides (take precedence over config values)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument("--n_queries", type=int, default=None)
    parser.add_argument(
        "--profile", action="store_true", default=False,
        help="Enable detailed memory breakdown and query step profiling",
    )
    args = parser.parse_args()

    # Determine dataset name
    if args.config:
        cfg = load_config(args.config, dataset=None)
        dataset = cfg["dataset"]
    elif args.dataset:
        cfg = load_config(None, dataset=args.dataset)
        dataset = args.dataset
    else:
        parser.error("Either --config or --dataset must be provided")

    # Apply CLI overrides
    if args.k is not None:
        cfg["benchmark"]["k"] = args.k
    if args.nlist is not None:
        cfg["build"]["nlist"] = args.nlist
    if args.n_queries is not None:
        cfg["benchmark"]["n_queries"] = args.n_queries
    bench_cfg = cfg["benchmark"]

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Curator Benchmark: {dataset}")
    print(f"{'='*60}")
    print(f"  Config: d={cfg['d']}, nlist={cfg['build']['nlist']}, "
          f"n_labels={cfg['n_labels']}")

    print("\n  Loading data...")
    t0 = time.perf_counter()
    train_vecs, train_mds, query_vecs, query_labels, query_info, ground_truth = \
        load_data(dataset, args.data_dir)
    t_load = time.perf_counter() - t0

    # Initial memory snapshot
    rss_after_load_mb = getCurrentRSS() / (1024 * 1024)

    print(f"  Train: {train_vecs.shape[0]:,} x {train_vecs.shape[1]}")
    print(f"  Queries: {len(query_vecs)}")
    print(f"  Avg labels/train: {sum(len(m) for m in train_mds) / len(train_mds):.1f}")
    print(f"  Data load time: {t_load:.1f}s, RSS after load: {rss_after_load_mb:.1f} MB")

    bucket_counts = Counter(e["bucket"] for e in query_info)
    print("\n  Query selectivity buckets:")
    for bname in sorted(bucket_counts.keys()):
        b_entries = [e for e in query_info if e["bucket"] == bname]
        b_sels = [e["selectivity"] for e in b_entries]
        print(f"    {bname:30s} n={len(b_entries):4d}  "
              f"mean_sel={np.mean(b_sels):.4f}")

    # ------------------------------------------------------------------
    # 2. Build index
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Building Curator index")
    print(f"{'='*60}")

    t_build_start = time.perf_counter()
    if args.profile:
        index, build_memory = build_curator_index_profiled(train_vecs, train_mds, cfg)
    else:
        index, build_memory = build_curator_index(train_vecs, train_mds, cfg)
    t_build_total = time.perf_counter() - t_build_start

    print(f"\n  Build time:  {t_build_total:.1f}s ({t_build_total / 60:.1f} min)")
    print(f"  Index memory (self-reported): {build_memory['index_memory_mb']:.1f} MB")
    print(f"  RSS after build:     {build_memory['rss_after_build_mb']:.1f} MB")

    # ------------------------------------------------------------------
    # 3. Run queries
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Running single-label queries (k={bench_cfg['k']})")
    print(f"{'='*60}")

    results = run_queries(
        index, query_vecs, query_labels, query_info, ground_truth,
        k=bench_cfg["k"], num_warmup=bench_cfg["num_warmup"],
        profile=args.profile,
    )

    # ------------------------------------------------------------------
    # 4. Print results
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Avg latency:   {results['avg_latency_ms']:.2f} ms")
    print(f"  P50 latency:   {results['p50_latency_ms']:.2f} ms")
    print(f"  P95 latency:   {results['p95_latency_ms']:.2f} ms")
    print(f"  P99 latency:   {results['p99_latency_ms']:.2f} ms")
    print(f"  QPS:           {results['qps']:.1f}")
    print(f"  Avg recall@{bench_cfg['k']}: {results['avg_recall']:.4f}")
    print(f"  Min recall@{bench_cfg['k']}: {results['min_recall']:.4f}")
    print(f"  Empty results: {results['empty_results']}/{results['n_queries']}")
    print(f"  RSS peak query:   {results['rss_peak_query_mb']:.1f} MB")

    print(f"\n  Per selectivity bucket:")
    print(f"  {'Bucket':<32s} {'Count':>5s} {'Lat(ms)':>9s} {'QPS':>8s} {'Recall':>8s}")
    print(f"  {'-'*32} {'-'*5} {'-'*9} {'-'*8} {'-'*8}")
    for bname in sorted(results["per_bucket"].keys()):
        b = results["per_bucket"][bname]
        print(f"  {bname:<32s} {b['count']:5d} {b['avg_latency_ms']:9.2f} "
              f"{b['qps']:8.1f} {b['avg_recall']:8.4f}")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Final memory snapshot (if profiling enabled)
    final_breakdown = None
    if args.profile:
        mem_prof = MemoryProfiler(index)
        final_breakdown = mem_prof.get_snapshot()

    output = {
        "dataset": dataset,
        "d": cfg["d"],
        "nlist": cfg["build"]["nlist"],
        "n_labels": cfg["n_labels"],
        "k": bench_cfg["k"],
        "train_size": int(len(train_vecs)),
        "n_queries": len(query_vecs),
        "build_time_s": float(t_build_total),
        "data_load_time_s": float(t_load),
        "memory": {
            "rss_after_load_mb": float(rss_after_load_mb),
            **build_memory,
        },
        "index_params": cfg["build"],
        "search_params": cfg["search"],
        "results": results,
    }

    if final_breakdown is not None:
        output["final_memory_breakdown"] = final_breakdown

    out_path = output_dir / f"curator_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {out_path.resolve()}")
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
