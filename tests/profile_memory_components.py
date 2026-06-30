"""
Profile Curator index memory by component (build-time + post-build breakdown).

Usage:
    python tests/profile_memory_components.py --dataset yfcc100m_small
    python tests/profile_memory_components.py --dataset arxiv_small
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_Utils"))
sys.path.insert(0, str(PROJECT_ROOT / "Curator" / "python"))
from memory_utils import getCurrentRSS  # noqa: E402
from memory_profiler import MemoryProfiler  # noqa: E402


def mb() -> float:
    return getCurrentRSS() / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="Profile Curator memory components"
    )
    parser.add_argument("--dataset", required=True,
                        choices=["yfcc100m", "arxiv", "yfcc100m_small", "arxiv_small"])
    parser.add_argument("--data_dir", default="1_Data")
    parser.add_argument("--output_dir", default="4_Results/memory_components")
    args = parser.parse_args()

    try:
        from curator import Curator  # noqa: E402
    except ImportError as e:
        print(f"Curator not available: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Curator Memory Component Profiler: {args.dataset}")
    print(f"{'='*60}")

    # ---- load config ----
    config_path = PROJECT_ROOT / f"3_Config/Curator/curator_{args.dataset}.json"
    if not config_path.exists():
        print(f"Config not found: {config_path}, using built-in defaults")
        config_path = None

    import json as _json
    if config_path:
        with open(config_path) as f:
            cfg = _json.load(f)
    else:
        from run_curator import BUILTIN_DEFAULTS as _def
        cfg = _def[args.dataset]

    # ---- load data ----
    import pickle as _pkl
    gt_dir = Path(args.data_dir) / "ground_truth" / args.dataset
    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        train_mds = _pkl.load(f)

    d = cfg["d"]
    n = len(train_vecs)
    build_cfg = cfg["build"]
    search_cfg = cfg.get("search", {})
    print(f"  Vectors: {n:,} x {d}")
    print(f"  Labels:  {cfg['n_labels']}")

    # ---- build index ----
    snapshots = {}
    rss_before = mb()

    index = Curator(
        d=d, nlist=build_cfg["nlist"],
        bf_capacity=build_cfg.get("bf_capacity", 1000),
        bf_error_rate=build_cfg.get("bf_error_rate", 0.001),
        max_sl_size=build_cfg.get("max_sl_size", 128),
        clus_niter=build_cfg.get("clus_niter", 20),
        max_leaf_size=build_cfg.get("max_leaf_size", 128),
        variance_boost=search_cfg.get("variance_boost", 0.2),
        search_ef=search_cfg.get("search_ef", 128),
        beam_size=search_cfg.get("beam_size", 1),
        use_temp_index_caching=search_cfg.get("use_temp_index_caching", True),
        pq_M=build_cfg["pq_M"], pq_nbits=build_cfg["pq_nbits"],
        pq_enabled=build_cfg.get("pq_enabled", True),
        pq_use_adc_rerank=build_cfg.get("pq_use_adc_rerank", False),
        pq_rerank_topk_factor=build_cfg.get("pq_rerank_topk_factor", 4),
        use_flash_storage=build_cfg.get("use_flash_storage", False),
        disk_cache_prefix="4_Results/Curator/disk_data_memprof",
    )

    profiler = MemoryProfiler(index)
    snapshots["01_before_train"] = profiler.get_snapshot()
    snapshots["01_before_train"]["label"] = "Before train"
    gc.collect()

    # Phase 1: Train
    print("  Training...")
    t0 = time.perf_counter()
    index.train(train_vecs)
    print(f"    done in {time.perf_counter() - t0:.1f}s")
    snapshots["02_after_train"] = profiler.get_snapshot()
    snapshots["02_after_train"]["label"] = "After train"
    gc.collect()

    # Phase 2: Add vectors
    print(f"  Adding {n} vectors...")
    t0 = time.perf_counter()
    for i in range(n):
        index.create(train_vecs[i], label=i)
        if (i + 1) % 100000 == 0:
            print(f"    {i+1}/{n} ({time.perf_counter() - t0:.1f}s)")
    print(f"    done in {time.perf_counter() - t0:.1f}s")
    snapshots["03_after_add"] = profiler.get_snapshot()
    snapshots["03_after_add"]["label"] = "After add vectors"
    gc.collect()

    # Phase 3: Grant access
    print("  Granting access...")
    t0 = time.perf_counter()
    total_grants = 0
    for i, labels in enumerate(train_mds):
        for lab in labels:
            index.grant_access(i, int(lab))
            total_grants += 1
    print(f"    {total_grants} grants in {time.perf_counter() - t0:.1f}s")
    snapshots["04_after_grant"] = profiler.get_snapshot()
    snapshots["04_after_grant"]["label"] = "After grant access"
    gc.collect()

    # Phase 4: Flush
    print("  Flushing...")
    t0 = time.perf_counter()
    index.flush()
    print(f"    done in {time.perf_counter() - t0:.1f}s")
    snapshots["05_after_flush"] = profiler.get_snapshot()
    snapshots["05_after_flush"]["label"] = "After flush"

    rss_after = mb()

    # ---- print breakdown ----
    print(f"\n{'='*60}")
    print("Final Memory Breakdown")
    print(f"{'='*60}")
    profiler.print_breakdown()

    # ---- compare snapshots ----
    print(f"\n{'='*60}")
    print("Phase-by-phase delta (MB)")
    print(f"{'='*60}")

    phases = [
        "01_before_train", "02_after_train", "03_after_add",
        "04_after_grant", "05_after_flush",
    ]
    for i in range(1, len(phases)):
        before = snapshots[phases[i - 1]]
        after = snapshots[phases[i]]
        print(f"\n  {before['label']} → {after['label']}:")
        profiler.print_compare(before, after)

    # ---- save ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert snapshots to serializable form (remove non-JSON fields)
    serializable = {}
    for key, snap in snapshots.items():
        serializable[key] = {
            "label": snap.get("label", key),
            "rss_mb": snap["rss_mb"],
            "index_total_mb": snap["index_total_mb"],
            "overhead_mb": snap["overhead_mb"],
            "breakdown": snap["breakdown"],
        }

    out = {
        "dataset": args.dataset,
        "n_vectors": n,
        "d": d,
        "rss_before_mb": float(rss_before),
        "rss_after_mb": float(rss_after),
        "rss_delta_mb": float(rss_after - rss_before),
        "snapshots": serializable,
    }

    out_path = out_dir / f"memory_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Saved: {out_path.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
