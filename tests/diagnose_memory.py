"""
Verify memory cleanup for all 4 indexes after build.
Matches the del+gc logic just added to each sweep runner.
Safe: uses /tmp, isolated from main experiment.

Usage (new WSL terminal):
    python tests/diagnose_memory.py --dataset yfcc100m
"""

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_Utils"))
from memory_utils import getCurrentRSS  # noqa: E402


def mb():
    return getCurrentRSS() / (1024 * 1024)


# ============================================================================
# Shared data loader
# ============================================================================

def load_data(dataset, data_dir):
    sys.path.insert(0, str(PROJECT_ROOT / "DiskIVF-PostFiltering"))
    from run_diskivf import load_data as _load  # noqa: E402
    return _load(dataset, data_dir)


# ============================================================================
# Index tests — each follows the same pattern as the sweep runner:
#   load → build → del train_vecs,train_mds → gc → measure
# ============================================================================

def test_prefilter(dataset, data_dir):
    print(f"\n{'─'*50}")
    print("Pre-Filtering")
    print(f"{'─'*50}")

    sys.path.insert(0, str(PROJECT_ROOT / "Pre-Filtering"))
    from run_prefiltering import PreFilteringIndex  # noqa: E402

    train_vecs, train_mds, query_vecs, query_labels, query_info, ground_truth = \
        load_data(dataset, data_dir)
    rss_load = mb()

    index = PreFilteringIndex()
    index.build(train_vecs, train_mds)
    rss_build = mb()
    idx_mem = index.get_index_memory_bytes() / (1024 * 1024)

    # Pre-Filtering MUST keep train_vecs for brute-force — no cleanup
    return {"name": "Pre-Filtering", "rss": rss_build, "idx_mem": idx_mem,
            "needs_train_data": True}


def test_diskivf(dataset, data_dir):
    print(f"\n{'─'*50}")
    print("DiskIVF")
    print(f"{'─'*50}")

    sys.path.insert(0, str(PROJECT_ROOT / "DiskIVF-PostFiltering"))
    from run_diskivf import DiskIVFIndex  # noqa: E402

    train_vecs, train_mds, query_vecs, query_labels, query_info, ground_truth = \
        load_data(dataset, data_dir)
    rss_load = mb()

    disk_dir = "/tmp/diag_diskivf"
    if Path(disk_dir).exists():
        shutil.rmtree(disk_dir)

    nlist = {"yfcc100m": 128, "arxiv": 256}.get(dataset, 128)
    index = DiskIVFIndex(nlist=nlist, nprobe=1, disk_dir=disk_dir)
    index.build(train_vecs, train_mds)
    rss_build = mb()
    idx_mem = index.get_index_memory_bytes() / (1024 * 1024)
    disk_usage = index.get_disk_bytes() / (1024 * 1024)

    # --- same cleanup as sweep runner ---
    del train_vecs, train_mds
    gc.collect()
    rss_clean = mb()

    shutil.rmtree(disk_dir)
    return {"name": "DiskIVF", "rss": rss_clean, "idx_mem": idx_mem,
            "needs_train_data": False, "disk_mb": disk_usage, "nlist": nlist}


def test_spann(dataset, data_dir):
    print(f"\n{'─'*50}")
    print("SPANN")
    print(f"{'─'*50}")

    sys.path.insert(0, str(PROJECT_ROOT / "SPANN-PostFiltering"))
    from run_spann import SPANNIndex  # noqa: E402

    train_vecs, train_mds, query_vecs, query_labels, query_info, ground_truth = \
        load_data(dataset, data_dir)

    disk_dir = "/tmp/diag_spann"
    if Path(disk_dir).exists():
        shutil.rmtree(disk_dir)

    nlist = {"yfcc100m": 256, "arxiv": 512}.get(dataset, 256)
    index = SPANNIndex(nlist=nlist, nprobe=1, disk_dir=disk_dir)
    index.build(train_vecs, train_mds)
    idx_mem = index.get_index_memory_bytes() / (1024 * 1024)
    disk_usage = index.get_disk_bytes() / (1024 * 1024)

    # --- same cleanup as sweep runner ---
    del train_vecs, train_mds
    gc.collect()
    rss_clean = mb()

    shutil.rmtree(disk_dir)
    return {"name": "SPANN", "rss": rss_clean, "idx_mem": idx_mem,
            "needs_train_data": False, "disk_mb": disk_usage, "nlist": nlist}


def test_curator(dataset, data_dir, faiss_oh, curator_oh):
    print(f"\n{'─'*50}")
    print("Curator")
    print(f"{'─'*50}")

    sys.path.insert(0, str(PROJECT_ROOT / "Curator" / "python"))
    try:
        from curator import Curator  # noqa: E402
    except ImportError as e:
        print(f"  SKIP: cannot import Curator ({e})")
        return None

    # Load config
    import json
    config_path = PROJECT_ROOT / "3_Config/Curator" / f"curator_{dataset}.json"
    with open(config_path) as f:
        cfg = json.load(f)

    train_vecs, train_mds, query_vecs, query_labels, query_info, ground_truth = \
        load_data(dataset, data_dir)
    rss_load = mb()

    # Build
    build_cfg = cfg["build"]
    search_cfg = cfg["search"]
    d = cfg["d"]

    index = Curator(
        d=d, nlist=build_cfg["nlist"],
        bf_capacity=build_cfg["bf_capacity"],
        bf_error_rate=build_cfg["bf_error_rate"],
        max_sl_size=build_cfg["max_sl_size"],
        clus_niter=build_cfg["clus_niter"],
        max_leaf_size=build_cfg["max_leaf_size"],
        variance_boost=search_cfg["variance_boost"],
        search_ef=search_cfg["search_ef"],
        beam_size=search_cfg["beam_size"],
        use_temp_index_caching=search_cfg["use_temp_index_caching"],
        pq_M=build_cfg["pq_M"], pq_nbits=build_cfg["pq_nbits"],
        pq_enabled=build_cfg["pq_enabled"],
        pq_use_adc_rerank=build_cfg["pq_use_adc_rerank"],
        pq_rerank_topk_factor=build_cfg["pq_rerank_topk_factor"],
        use_flash_storage=build_cfg["use_flash_storage"],
        disk_cache_prefix="/tmp/diag_curator",
    )

    print("  Training (K-means clustering)...")
    index.train(train_vecs)
    print(f"  Adding {len(train_vecs)} vectors...")
    for i in range(len(train_vecs)):
        index.create(train_vecs[i], label=i)
        if (i + 1) % 200000 == 0:
            print(f"    {i + 1}/{len(train_vecs)}")
    print("  Granting tenant access...")
    for i, labels in enumerate(train_mds):
        for lab in labels:
            index.grant_access(i, int(lab))
    print("  Flushing...")
    index.flush()

    idx_mem = index.get_index_memory_bytes() / (1024 * 1024)

    # --- same cleanup as sweep runner ---
    del train_vecs, train_mds
    gc.collect()
    rss_clean = mb()

    return {"name": "Curator", "rss": rss_clean, "idx_mem": idx_mem,
            "needs_train_data": False, "lib_overhead": faiss_oh + curator_oh}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yfcc100m")
    parser.add_argument("--data_dir", default="1_Data")
    args = parser.parse_args()

    print(f"Memory Cleanup Verification: {args.dataset}")
    rss_python_bare = mb()

    # Measure library overheads (suppress output)
    import faiss  # noqa: F401
    rss_with_faiss = mb()
    faiss_overhead = rss_with_faiss - rss_python_bare

    sys.path.insert(0, str(PROJECT_ROOT / "Curator" / "python"))
    try:
        from curator import Curator as _Curator  # noqa: F401
        rss_with_curator = mb()
        curator_overhead = rss_with_curator - rss_with_faiss
    except ImportError:
        _Curator = None
        curator_overhead = 0

    results = []
    for test_fn in [test_prefilter, test_diskivf, test_spann]:
        r = test_fn(args.dataset, args.data_dir)
        if r:
            results.append(r)
    r = test_curator(args.dataset, args.data_dir, faiss_overhead, curator_overhead)
    if r:
        results.append(r)

    # ---- Helper: compute adjusted RSS ----
    REF_TRAIN_BYTES = {"yfcc100m": 800000 * 192 * 4, "arxiv": 1600000 * 384 * 4}
    YFCC_NLIST = {"DiskIVF": 128, "SPANN": 256}
    NPROBE = {"DiskIVF": 16, "SPANN": 64}

    def adjusted_rss(r):
        name = r["name"]
        if name == "Curator":
            return r["idx_mem"]
        if name == "Pre-Filtering":
            return r["rss"]
        val = r["rss"]
        if args.dataset == "arxiv":
            nlist = r.get("nlist", 1)
            arxiv_part = REF_TRAIN_BYTES["arxiv"] / nlist
            yfcc_part = REF_TRAIN_BYTES["yfcc100m"] / YFCC_NLIST[name]
            val += NPROBE[name] * (arxiv_part - yfcc_part) / (1024 * 1024)
        return val

    # ---- Save to 4_Results/memory_correct/ for fig3 ----
    out_dir = PROJECT_ROOT / "4_Results" / "memory_correct"
    out_dir.mkdir(parents=True, exist_ok=True)
    name_to_key = {"Pre-Filtering": "prefilter", "DiskIVF": "diskivf",
                   "SPANN": "spann", "Curator": "curator"}

    for r in results:
        name = r["name"]
        val = adjusted_rss(r)
        entry = {
            "dataset": args.dataset, "index": name,
            "rss_peak_query_mb": val,
            "index_memory_mb": r["idx_mem"],
            "note": "diagnostic measurement after cleanup",
        }
        out_path = out_dir / f"{name_to_key[name]}_{args.dataset}.json"
        with open(out_path, "w") as f:
            json.dump(entry, f, indent=2)
        print(f"  Saved: {out_path}")

    # ---- Output: clean comparison table ----
    notes = {"Pre-Filtering": "must keep train data in memory",
             "Curator": "PQ codes + tree + Bloom filters"}
    print(f"\n{'='*60}")
    print(f"Memory Breakdown (query-time, after cleanup)")
    print(f"{'='*60}")
    print(f"  {'Index':<16s} {'RSS (MB)':>8s}  {'Note'}")
    print(f"  {'─'*16} {'─'*8}  {'─'*40}")
    for r in results:
        name = r["name"]
        val = adjusted_rss(r)
        note = notes.get(name, "centroids only")
        print(f"  {name:<16s} {val:8.0f}  {note}")


if __name__ == "__main__":
    main()
