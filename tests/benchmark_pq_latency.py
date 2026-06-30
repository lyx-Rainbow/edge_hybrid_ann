"""
PQ External Storage Latency Benchmark
对比 persist_pq_codes=True (外存/mmaps) vs False (内存) 的查询延迟
使用 Goal3 的 profiling 基础设施获取各步骤耗时分解

Usage:
    python tests/benchmark_pq_latency.py --dataset yfcc100m_small --n_queries 500
    python tests/benchmark_pq_latency.py --dataset arxiv_small --n_queries 200
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Curator" / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "2_Utils"))
from curator import Curator
from query_profiler import QueryProfiler


def stats(vals):
    arr = np.array([v for v in vals if v is not None and v >= 0], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def build_index(train_vecs, train_mds, cfg, disk_prefix, persist):
    """构建索引并返回 Curator 实例."""
    d = cfg["d"]
    b = cfg["build"]
    s = cfg["search"]

    # 清理旧文件
    os.makedirs(disk_prefix, exist_ok=True)
    for fname in ["pq_codes.bin", "vectors.bin"]:
        p = os.path.join(disk_prefix, fname)
        if os.path.exists(p):
            os.remove(p)

    idx = Curator(
        d=d, nlist=b["nlist"],
        bf_capacity=b["bf_capacity"], bf_error_rate=b["bf_error_rate"],
        max_sl_size=b["max_sl_size"], clus_niter=b["clus_niter"],
        max_leaf_size=b["max_leaf_size"],
        variance_boost=s["variance_boost"], search_ef=s["search_ef"],
        beam_size=s["beam_size"], use_temp_index_caching=s["use_temp_index_caching"],
        pq_M=b["pq_M"], pq_nbits=b["pq_nbits"], pq_enabled=b["pq_enabled"],
        pq_use_adc_rerank=b["pq_use_adc_rerank"],
        pq_rerank_topk_factor=b["pq_rerank_topk_factor"],
        use_flash_storage=b["use_flash_storage"],
        disk_cache_prefix=disk_prefix,
        persist_pq_codes=persist,
    )
    idx.train(train_vecs)
    for i in range(len(train_vecs)):
        idx.create(train_vecs[i], label=i)
    for i, labels in enumerate(train_mds):
        for lab in labels:
            idx.grant_access(i, int(lab))
    idx.flush()
    return idx


def main():
    parser = argparse.ArgumentParser(description="PQ External Storage Latency Benchmark")
    parser.add_argument("--dataset", default="yfcc100m_small",
                        choices=["yfcc100m", "arxiv", "yfcc100m_small", "arxiv_small"])
    parser.add_argument("--data_dir", default="1_Data")
    parser.add_argument("--n_queries", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num_warmup", type=int, default=30)
    parser.add_argument("--disk_prefix", default="4_Results/Curator/disk_data")
    args = parser.parse_args()

    print("=" * 70)
    print("PQ External Storage — Query Latency Benchmark")
    print(f"Dataset: {args.dataset}, k={args.k}, warmup={args.num_warmup}, queries={args.n_queries}")
    print("=" * 70)

    # ── 加载数据 ──
    gt_dir = Path(args.data_dir) / "ground_truth" / args.dataset
    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        train_mds = pickle.load(f)
    query_vecs = np.load(gt_dir / "query_vecs.npy").astype(np.float32)
    query_labels = np.load(gt_dir / "query_labels.npy").astype(np.int32)
    try:
        with open(gt_dir / "query_info.json") as f:
            query_info = json.load(f)
    except FileNotFoundError:
        query_info = [{}] * len(query_vecs)

    config_path = PROJECT_ROOT / f"3_Config/Curator/curator_{args.dataset}.json"
    with open(config_path) as f:
        cfg = json.load(f)

    d = cfg["d"]
    b = cfg["build"]
    s = cfg["search"]

    print(f"  Vectors: {len(train_vecs):,} x {d}")
    print(f"  PQ: M={b['pq_M']}, nbits={b['pq_nbits']}")
    print(f"  ADC rerank: {b['pq_use_adc_rerank']} (topk_factor={b['pq_rerank_topk_factor']})")
    print(f"  search_ef={s['search_ef']}, beam_size={s['beam_size']}")

    # ── 构建两个索引 ──
    print("\n" + "-" * 50)
    print("Building persist_pq_codes=True  (disk / mmap)...")
    t0 = time.perf_counter()
    idx_disk = build_index(train_vecs, train_mds, cfg, args.disk_prefix, persist=True)
    t_build_disk = time.perf_counter() - t0
    print(f"  Build time: {t_build_disk:.1f}s")

    print("\nBuilding persist_pq_codes=False (memory)...")
    t0 = time.perf_counter()
    idx_mem = build_index(train_vecs, train_mds, cfg, args.disk_prefix, persist=False)
    t_build_mem = time.perf_counter() - t0
    print(f"  Build time: {t_build_mem:.1f}s")

    # 释放 Python 侧大数据
    del train_vecs, train_mds
    gc.collect()

    # ── 预热 ──
    print(f"\nWarmup ({args.num_warmup} queries each)...")
    n_avail = len(query_vecs)
    for i in range(args.num_warmup):
        idx = i % n_avail
        tid = int(query_labels[idx])
        idx_disk.query(query_vecs[idx], args.k, tid)
        idx_mem.query(query_vecs[idx], args.k, tid)

    # ── Benchmark ──
    print(f"Benchmarking ({args.n_queries} queries each)...")
    n_q = min(args.n_queries, n_avail)

    wall_disk = []
    wall_mem = []
    profiler_disk = QueryProfiler(idx_disk)
    profiler_mem = QueryProfiler(idx_mem)
    cpp_disk = []
    cpp_mem = []

    for i in range(n_q):
        tid = int(query_labels[i])
        x = query_vecs[i]

        # Wall-clock: disk mode
        t0 = time.perf_counter()
        idx_disk.query(x, args.k, tid)
        wall_disk.append((time.perf_counter() - t0) * 1000)

        # Wall-clock: memory mode
        t0 = time.perf_counter()
        idx_mem.query(x, args.k, tid)
        wall_mem.append((time.perf_counter() - t0) * 1000)

        # C++ profiling: disk mode
        cpp_disk.append(profiler_disk.profile_single(x, args.k, tid))

        # C++ profiling: memory mode
        cpp_mem.append(profiler_mem.profile_single(x, args.k, tid))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_q}...")

    # ── 聚合统计 ──
    steps = [
        ("total_search_time_ms",        "Total (C++ internal)"),
        ("beam_search_time_ms",         "  Beam Search"),
        ("pq_table_build_time_ms",      "  PQ Table Build"),
        ("frontier_search_time_ms",     "  Frontier Search"),
        ("pq_distance_compute_time_ms", "  PQ Distance Compute"),
        ("candidate_merge_time_ms",     "  Candidate Merge"),
        ("rerank_time_ms",              "  ADC Rerank"),
        ("exact_distance_compute_time_ms", "  Exact Distance"),
    ]

    print("\n" + "=" * 70)
    print("LATENCY COMPARISON: disk (mmap) vs memory")
    print("=" * 70)

    header = f"{'Step':<30s} {'Mode':<8s} {'Mean':>8s} {'P50':>8s} {'P95':>8s} {'P99':>8s} {'Delta':>8s}"
    sep = f"{'':->30s} {'':->8s} {'':->8s} {'':->8s} {'':->8s} {'':->8s} {'':->8s}"
    print(f"\n{header}\n{sep}")

    for field, label in steps:
        s_disk = stats([p.get(field, 0) or 0 for p in cpp_disk])
        s_mem  = stats([p.get(field, 0) or 0 for p in cpp_mem])
        delta  = s_disk["mean"] - s_mem["mean"]
        delta_str = f"{delta:+.3f}" if abs(delta) >= 0.0005 else "  ~0"
        print(f"{label:<30s} {'disk':<8s} {s_disk['mean']:8.3f} {s_disk['p50']:8.3f} {s_disk['p95']:8.3f} {s_disk['p99']:8.3f} {delta_str:>8s}")
        print(f"{'':30s} {'memory':<8s} {s_mem['mean']:8.3f} {s_mem['p50']:8.3f} {s_mem['p95']:8.3f} {s_mem['p99']:8.3f}")

    # ── Wall-clock ──
    s_wall_disk = stats(wall_disk)
    s_wall_mem  = stats(wall_mem)
    delta_wall  = s_wall_disk["mean"] - s_wall_mem["mean"]
    print(f"\n{'Wall-clock (Python)':<30s} {'disk':<8s} {s_wall_disk['mean']:8.3f} {s_wall_disk['p50']:8.3f} {s_wall_disk['p95']:8.3f} {s_wall_disk['p99']:8.3f} {delta_wall:+.3f}")
    print(f"{'':30s} {'memory':<8s} {s_wall_mem['mean']:8.3f} {s_wall_mem['p50']:8.3f} {s_wall_mem['p95']:8.3f} {s_wall_mem['p99']:8.3f}")

    # ── 辅助统计 ──
    print(f"\n{'─'*70}")
    print("Ancillary Stats")
    print(f"{'─'*70}")

    for lbl, profs in [("disk", cpp_disk), ("memory", cpp_mem)]:
        nodes = [p.get("frontier_nodes_popped", 0) or 0 for p in profs]
        sls   = [p.get("frontier_shortlists_scanned", 0) or 0 for p in profs]
        rerank = [p.get("rerank_count", 0) or 0 for p in profs]
        children = [p.get("frontier_children_expanded", 0) or 0 for p in profs]
        print(f"  [{lbl}]")
        print(f"    rerank_count:              mean={np.mean(rerank):.1f}")
        print(f"    frontier_nodes_popped:     mean={np.mean(nodes):.1f}")
        print(f"    frontier_shortlists_scanned: mean={np.mean(sls):.1f}")
        print(f"    frontier_children_expanded:  mean={np.mean(children):.1f}")

    # ── 内存对比 ──
    print(f"\n{'─'*70}")
    print("Memory Comparison")
    print(f"{'─'*70}")
    bd_disk = idx_disk.get_memory_breakdown()
    bd_mem  = idx_mem.get_memory_breakdown()
    mem_disk = idx_disk.get_index_memory_bytes() / (1024 * 1024)
    mem_mem  = idx_mem.get_index_memory_bytes() / (1024 * 1024)
    pq_theory = len(query_labels) * 0  # placeholder, actual value below
    print(f"  Index memory (disk):      {mem_disk:.2f} MB")
    print(f"  Index memory (memory):    {mem_mem:.2f} MB")
    print(f"  Memory saved:             {mem_mem - mem_disk:.2f} MB")
    print(f"  PQ codes bytes (disk):    {bd_disk['pq']['codes_bytes'] / (1024*1024):.4f} MB")
    print(f"  PQ codes bytes (memory):  {bd_mem['pq']['codes_bytes'] / (1024*1024):.4f} MB")
    print(f"  pq_codes_on_disk():       {idx_disk.index.pq_codes_on_disk()}")
    print(f"  pq_codes_in_memory():     {idx_disk.index.pq_codes_in_memory()}")

    # ── 延迟增幅 ──
    print(f"\n{'─'*70}")
    print("LATENCY OVERHEAD (disk vs memory)")
    print(f"{'─'*70}")
    total_disk = stats([p.get("total_search_time_ms", 0) or 0 for p in cpp_disk])
    total_mem  = stats([p.get("total_search_time_ms", 0) or 0 for p in cpp_mem])
    overhead_mean = (total_disk["mean"] - total_mem["mean"]) / max(total_mem["mean"], 0.001) * 100
    overhead_p50  = (total_disk["p50"] - total_mem["p50"]) / max(total_mem["p50"], 0.001) * 100
    overhead_p95  = (total_disk["p95"] - total_mem["p95"]) / max(total_mem["p95"], 0.001) * 100
    overhead_p99  = (total_disk["p99"] - total_mem["p99"]) / max(total_mem["p99"], 0.001) * 100

    print(f"  C++ internal total search time:")
    print(f"    Mean overhead:   {overhead_mean:+.2f}%  ({total_disk['mean']:.3f} vs {total_mem['mean']:.3f} ms)")
    print(f"    P50  overhead:   {overhead_p50:+.2f}%  ({total_disk['p50']:.3f} vs {total_mem['p50']:.3f} ms)")
    print(f"    P95  overhead:   {overhead_p95:+.2f}%  ({total_disk['p95']:.3f} vs {total_mem['p95']:.3f} ms)")
    print(f"    P99  overhead:   {overhead_p99:+.2f}%  ({total_disk['p99']:.3f} vs {total_mem['p99']:.3f} ms)")

    wovh_p50 = (s_wall_disk["p50"] - s_wall_mem["p50"]) / max(s_wall_mem["p50"], 0.001) * 100
    wovh_p95 = (s_wall_disk["p95"] - s_wall_mem["p95"]) / max(s_wall_mem["p95"], 0.001) * 100
    print(f"\n  Wall-clock (incl. Python overhead):")
    print(f"    P50  overhead:   {wovh_p50:+.2f}%  ({s_wall_disk['p50']:.3f} vs {s_wall_mem['p50']:.3f} ms)")
    print(f"    P95  overhead:   {wovh_p95:+.2f}%  ({s_wall_disk['p95']:.3f} vs {s_wall_mem['p95']:.3f} ms)")

    # ── PQ Distance Compute 详细对比（这是直接受影响的部分） ──
    pqdist_disk = stats([p.get("pq_distance_compute_time_ms", 0) or 0 for p in cpp_disk])
    pqdist_mem  = stats([p.get("pq_distance_compute_time_ms", 0) or 0 for p in cpp_mem])
    pq_ovh = (pqdist_disk["mean"] - pqdist_mem["mean"]) / max(pqdist_mem["mean"], 0.001) * 100
    print(f"\n  PQ Distance Compute (most affected by external storage):")
    print(f"    Mean overhead:   {pq_ovh:+.2f}%  ({pqdist_disk['mean']:.3f} vs {pqdist_mem['mean']:.3f} ms)")
    print(f"    P50  overhead:   {(pqdist_disk['p50']-pqdist_mem['p50'])/max(pqdist_mem['p50'],0.001)*100:+.2f}%")

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
