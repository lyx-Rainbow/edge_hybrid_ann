#!/usr/bin/env python3
"""Curator memory-oriented overhead experiment.

Runs one full build plus one query with a deliberately small query-time
footprint (fewer clusters/leaves, smaller PQ block cache, no temp-index cache,
and release of predicate-only maps for SL workloads).  Query results are not
used; this is only for the query-memory bar.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT = ROOT / "1_Data/ground_truth"
BIN = ROOT / "Curator/build/curator"
OUT = ROOT / "4_Results/build_measure/raw"
TMP = Path("/tmp/curator_memory_params")
PQ_M = {"sift1m": 32, "gist1m": 160, "arxiv": 48,
        "yfcc100m": 24, "wit": 192}
# wit's 8-way k-means tree degenerated and hit the depth cap; 32 clusters
# gives enough branching to keep leaves bounded for the memory overhead run.
N_CLUSTERS = {"sift1m": 8, "gist1m": 8, "arxiv": 8,
              "yfcc100m": 8, "wit": 8}
# wit contains a group of ~32,455 exactly duplicated vectors; a 1024-vector
# leaf cannot represent them.  Use a larger leaf for wit so the memory-oriented
# overhead measurement can complete; the other datasets keep max_leaf_size=1024.
MAX_LEAF_SIZE = {"sift1m": 1024, "gist1m": 1024, "arxiv": 1024,
                 "yfcc100m": 1024, "wit": 32768}
MB = 1024.0 * 1024.0


def path_size(path: Path):
    """Return (apparent_bytes, ext4_allocated_bytes) for a file."""
    path = Path(path)
    if not path.exists():
        return 0, 0
    stat = path.stat()
    return int(stat.st_size), int(stat.st_blocks * 512)


def measure_paths(paths):
    apparent = allocated = 0
    for path in paths:
        app, alloc = path_size(path)
        apparent += app
        allocated += alloc
    return apparent, allocated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["sift1m", "gist1m", "arxiv",
                                 "yfcc100m", "wit"])
    args = parser.parse_args()
    for dataset in args.datasets:
        work = TMP / dataset
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        src = GT / dataset / "query_vecs.npy"
        labels = GT / dataset / "query_labels.npy"
        import numpy as np
        np.save(work / "query_vecs.npy", np.load(src, mmap_mode="r")[:1])
        np.save(work / "query_labels.npy",
                np.load(labels, mmap_mode="r")[:1].astype(np.int32))
        cfg_src = ROOT / "3_Config/Curator" / f"sweep_full_{dataset}.json"
        fixed = json.load(open(cfg_src, encoding="utf-8")).get("fixed", {})
        fixed.update({
            "n_clusters": N_CLUSTERS[dataset],
            "max_sl_size": 1024,
            "max_leaf_size": MAX_LEAF_SIZE[dataset],
            "bf_capacity": 256,
            "bf_false_pos": 0.05,
            "pq_cache_max_blocks": 64,
            "use_temp_index_caching": False,
            "search_ef": 16,
            "pq_M": PQ_M[dataset],
            "pq_enabled": True,
            "pq_use_adc_rerank": True,
            "persist_pq_codes": True,
            "use_flash_storage": True,
            "flash_path": str(work / "flash.bin"),
            "pq_codes_path": str(work / "pq_codes.bin"),
            "batch_query": False,
        })
        cfg_path = work / "config.json"
        cfg_path.write_text(json.dumps(fixed, indent=2), encoding="utf-8")
        out = work / "result.json"
        cmd = [str(BIN), "bench",
               "--train_vecs", str(GT / dataset / "train_vecs.npy"),
               "--train_access", str(GT / dataset / "train_access.npy"),
               "--queries", str(work / "query_vecs.npy"),
               "--query_labels", str(work / "query_labels.npy"),
               "--config", str(cfg_path), "--k", "10",
               "--ef-list", "16", "--output", str(out)]
        print(f"{dataset}: starting build", flush=True)
        started = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=24 * 3600)
        elapsed = time.time() - started
        if proc.returncode != 0:
            print(f"{dataset}: Curator returncode={proc.returncode} "
                  f"after {elapsed:.1f}s", flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"curator_memory_{dataset}_stdout.log").write_text(
                proc.stdout or "", encoding="utf-8")
            (OUT / f"curator_memory_{dataset}_stderr.log").write_text(
                proc.stderr or "", encoding="utf-8")
            print((proc.stdout or "")[-2000:], flush=True)
            print((proc.stderr or "")[-2000:], flush=True)
            raise SystemExit(1)
        candidates = sorted(work.glob("result*.json"))
        if not candidates:
            raise RuntimeError("Curator did not write a result JSON")
        result = json.loads(candidates[0].read_text(encoding="utf-8"))
        flash_path = work / "flash.bin"
        pq_path = work / "pq_codes.bin"
        disk_apparent, disk_allocated = measure_paths([flash_path, pq_path])
        flash_apparent, flash_allocated = path_size(flash_path)
        pq_apparent, pq_allocated = path_size(pq_path)
        record = {
            "method": "Curator",
            "method_key": "curator",
            "dataset": dataset,
            "config": fixed,
            "build_time_s": float(result.get("build_time_s", 0.0)),
            "wall_time_s": elapsed,
            "memory_bytes": int(result.get("memory_bytes", 0)),
            "rss_peak_query_mb": float(result.get("rss_peak_query_mb", 0.0)),
            "disk_bytes": int(result.get("disk_bytes", 0)),
            "disk_reported_bytes": int(result.get("disk_bytes", 0)),
            "disk_apparent_bytes": int(disk_apparent),
            "disk_allocated_bytes": int(disk_allocated),
            "index_files": {
                "flash_apparent_bytes": int(flash_apparent),
                "flash_allocated_bytes": int(flash_allocated),
                "pq_apparent_bytes": int(pq_apparent),
                "pq_allocated_bytes": int(pq_allocated),
            },
            "measurement": ("fresh memory-oriented build on full training "
                            "data with a one-query subset; predicate-only maps "
                            "released for the SL query path"),
            "note": "disk_allocated_bytes measured on ext4 with st_blocks",
        }
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"curator_memory_{dataset}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"{dataset}: wall={elapsed:.1f}s build={record['build_time_s']:.1f}s "
              f"mem={record['memory_bytes'] / MB:.1f}MiB "
              f"rss={record['rss_peak_query_mb']:.1f}MiB "
              f"disk_alloc={record['disk_allocated_bytes'] / MB:.1f}MiB "
              f"-> {path.name}", flush=True)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()