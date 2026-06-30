"""
Profile Curator build-time breakdown per dataset.
Isolated: uses /tmp for flash storage, writes to 4_Results/build_time_correct/.

Usage:
    python tests/diagnose_build_time.py --dataset yfcc100m
    python tests/diagnose_build_time.py --dataset arxiv
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
from memory_utils import getCurrentRSS  # noqa: E402


def mb():
    return getCurrentRSS() / (1024 * 1024)


def load_data(dataset, data_dir):
    import pickle
    gt_dir = Path(data_dir) / "ground_truth" / dataset
    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        train_mds = pickle.load(f)
    return train_vecs, train_mds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["yfcc100m", "arxiv"])
    parser.add_argument("--data_dir", default="1_Data")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT / "Curator" / "python"))
    try:
        from curator import Curator  # noqa: E402
    except ImportError as e:
        print(f"Curator not available: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Curator Build Profile: {args.dataset}")
    print(f"{'='*60}")

    # ---- Load config ----
    config_path = PROJECT_ROOT / "3_Config/Curator" / f"curator_{args.dataset}.json"
    with open(config_path) as f:
        cfg = json.load(f)

    # ---- Load data ----
    train_vecs, train_mds = load_data(args.dataset, args.data_dir)
    d = train_vecs.shape[1]
    n = len(train_vecs)
    build_cfg, search_cfg = cfg["build"], cfg.get("search", {})
    print(f"  Vectors: {n:,} x {d}")
    print(f"  Labels:  {cfg['n_labels']}")

    # ---- Build index ----
    index = Curator(
        d=d, nlist=build_cfg["nlist"],
        bf_capacity=build_cfg.get("bf_capacity", 1000),
        bf_error_rate=build_cfg.get("bf_error_rate", 0.001),
        max_sl_size=build_cfg.get("max_sl_size", 256),
        clus_niter=build_cfg.get("clus_niter", 20),
        max_leaf_size=build_cfg.get("max_leaf_size", 128),
        variance_boost=search_cfg.get("variance_boost", 0.2),
        search_ef=search_cfg.get("search_ef", 128),
        beam_size=search_cfg.get("beam_size", 4),
        use_temp_index_caching=search_cfg.get("use_temp_index_caching", True),
        pq_M=build_cfg["pq_M"], pq_nbits=build_cfg["pq_nbits"],
        pq_enabled=build_cfg.get("pq_enabled", True),
        pq_use_adc_rerank=build_cfg.get("pq_use_adc_rerank", False),
        pq_rerank_topk_factor=build_cfg.get("pq_rerank_topk_factor", 4),
        use_flash_storage=build_cfg.get("use_flash_storage", True),
        disk_cache_prefix=f"/tmp/curator_bt_{args.dataset}",
    )

    phases = {}
    rss_before_all = mb()

    # Phase 1: Hierarchical K-means clustering
    gc.collect()
    t0 = time.perf_counter()
    index.train(train_vecs)
    phases["train"] = float(time.perf_counter() - t0)
    print(f"  train (hierarchical K-means):  {phases['train']:.1f}s")

    # Phase 2: Add vectors to tree
    gc.collect()
    t0 = time.perf_counter()
    for i in range(n):
        index.create(train_vecs[i], label=i)
    phases["create"] = float(time.perf_counter() - t0)
    print(f"  create ({n} vectors):          {phases['create']:.1f}s")

    # Phase 3: Grant tenant access
    gc.collect()
    t0 = time.perf_counter()
    for i, labels in enumerate(train_mds):
        for lab in labels:
            index.grant_access(i, int(lab))
    phases["grant_access"] = float(time.perf_counter() - t0)
    print(f"  grant_access:                  {phases['grant_access']:.1f}s")

    # Phase 4a: PQ training + encoding
    gc.collect()
    t0 = time.perf_counter()
    index.index.train_pq_codebook()
    phases["pq_train_encode"] = float(time.perf_counter() - t0)
    print(f"  PQ (train + encode):           {phases['pq_train_encode']:.1f}s")

    # Phase 4b: Flash storage write
    gc.collect()
    t0 = time.perf_counter()
    index.index.finalize_flash_storage()
    phases["flash_write"] = float(time.perf_counter() - t0)
    print(f"  Flash write:                   {phases['flash_write']:.1f}s")

    rss_after_all = mb()
    idx_mem = index.get_index_memory_bytes() / (1024 * 1024)

    # Cleanup
    del train_vecs, train_mds, index
    gc.collect()

    total = sum(phases.values())

    # ---- Save ----
    out_dir = PROJECT_ROOT / "4_Results" / "build_time_correct"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "dataset": args.dataset,
        "index": "Curator",
        "n_vectors": n, "d": d,
        "build": build_cfg,
        "phases": phases,
        "total_build_s": float(total),
        "index_memory_mb": float(idx_mem),
        "rss_before_build_mb": float(rss_before_all),
        "rss_after_build_mb": float(rss_after_all),
    }

    out_path = out_dir / f"curator_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- Output ----
    print(f"\n{'='*60}")
    print(f"Curator Build Breakdown: {args.dataset}")
    print(f"{'='*60}")
    print(f"  {'Phase':<30s} {'Time':>8s} {'%':>7s}")
    print(f"  {'─'*30} {'─'*8} {'─'*7}")
    for name, t in phases.items():
        pct = t / total * 100
        print(f"  {name:<30s} {t:7.1f}s {pct:6.1f}%")
    print(f"  {'─'*30} {'─'*8} {'─'*7}")
    print(f"  {'Total':<30s} {total:7.1f}s")
    print(f"\n  Index memory: {idx_mem:.0f} MB")
    print(f"  RSS: {rss_before_all:.0f} → {rss_after_all:.0f} MB")
    print(f"  Saved: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
