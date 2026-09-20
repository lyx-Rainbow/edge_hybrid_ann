#!/usr/bin/env python3
"""Build-measurement runner for the three full-scale bar figures.

The script performs two kinds of measurements:

1. Fresh builds for methods where existing build metadata is not reliable
   (Curator, PreFilter).  A one-query subset is used so the expensive builds
   are not followed by unnecessary query sweeps.
2. Collection of already measured build results for DiskIVF and SPANN.

Raw per-(method,dataset) measurements are written to
``4_Results/build_measure/raw`` and a chart-ready summary is written to
``4_Results/build_measure/index_metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GT_ROOT = ROOT / "1_Data" / "ground_truth"
RESULT_ROOT = ROOT / "4_Results"
OUT_DIR = RESULT_ROOT / "build_measure"
RAW_DIR = OUT_DIR / "raw"
TMP_ROOT = Path("/tmp/build_measure")

DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]
METHODS = ["curator", "diskivf", "spann", "prefilter"]
CURATOR_PQ_M = {
    "sift1m": 32,
    "gist1m": 480,
    "arxiv": 192,
    "yfcc100m": 96,
    "wit": 192,
}

# Query-memory measurement point for DiskIVF.  Its search path streams one
# cluster at a time, so peak RSS is dominated by the largest cluster rather
# than by nprobe; 128 is a high nprobe available for every full dataset.
DISKIVF_HIGH_NPROBE = {dataset: 128 for dataset in DATASETS}
# Query-memory measurement concurrency.  The DiskIVF search path loads one
# cluster at a time per query; batch queries allow up to this many clusters to
# be resident simultaneously.
DISKIVF_BATCH_QUERIES = 8
DISKIVF_BATCH_THREADS = 8

CURATOR_BIN = ROOT / "Curator" / "build" / "curator"
PREFILTER_BIN = ROOT / "Pre-Filtering" / "build" / "prefiltering"

MB = 1024.0 * 1024.0


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def dataset_shape(dataset: str):
    path = GT_ROOT / dataset / "train_vecs.npy"
    shape = np.load(path, mmap_mode="r").shape
    return int(shape[0]), int(shape[1])


def make_one_query_subset(dataset: str, work_dir: Path) -> None:
    """Create one-query .npy files so builds never trigger full searches."""
    work_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = GT_ROOT / dataset
    query_vecs = np.load(gt_dir / "query_vecs.npy", mmap_mode="r")
    query_labels = np.load(gt_dir / "query_labels.npy", mmap_mode="r")
    np.save(work_dir / "query_vecs.npy", np.ascontiguousarray(query_vecs[:1]))
    labels = np.ascontiguousarray(query_labels[:1].astype(np.int32))
    np.save(work_dir / "query_labels.npy", labels)


def path_size(path: Path):
    if not path.exists():
        return 0, 0
    total_apparent = 0
    total_allocated = 0
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    for item in files:
        stat = item.stat()
        total_apparent += stat.st_size
        total_allocated += stat.st_blocks * 512
    return total_apparent, total_allocated


def measure_paths(paths) -> tuple[int, int]:
    apparent = allocated = 0
    for path in paths:
        app, alloc = path_size(Path(path))
        apparent += app
        allocated += alloc
    return apparent, allocated


def run_command(cmd, timeout: int, log_tail_chars: int = 3000):
    log("RUN " + " ".join(str(c) for c in cmd))
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        log(f"TIMEOUT after {elapsed:.1f}s")
        return None, elapsed, str(exc)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        log(f"FAIL rc={proc.returncode} after {elapsed:.1f}s")
        log("STDOUT tail: " + (proc.stdout or "")[-log_tail_chars:])
        log("STDERR tail: " + (proc.stderr or "")[-log_tail_chars:])
        return None, elapsed, proc.stderr or proc.stdout or ""
    log("rc=0 wall_time=%.1fs" % elapsed)
    return proc, elapsed, proc.stdout or ""


def curator_raw_path(dataset: str) -> Path:
    return RAW_DIR / f"curator_{dataset}.json"


def prefilter_raw_path(dataset: str) -> Path:
    return RAW_DIR / f"prefilter_{dataset}.json"


def run_curator(dataset: str, force: bool = False):
    raw_path = curator_raw_path(dataset)
    if raw_path.exists() and not force:
        log(f"skip existing {raw_path.name}")
        return
    n, d = dataset_shape(dataset)
    pq_m = CURATOR_PQ_M[dataset]
    work_dir = TMP_ROOT / f"curator_{dataset}"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    make_one_query_subset(dataset, work_dir)

    config_path = ROOT / "3_Config" / "Curator" / f"sweep_full_{'sift' if dataset == 'sift1m' else dataset}.json"
    config = load_json(config_path)
    fixed = dict(config.get("fixed", {}))
    fixed.update(
        {
            "pq_M": pq_m,
            "pq_nbits": 8,
            "pq_enabled": True,
            "pq_use_adc_rerank": True,
            "persist_pq_codes": True,
            "use_flash_storage": True,
            "flash_path": str(work_dir / "flash.bin"),
            "pq_codes_path": str(work_dir / "pq_codes.bin"),
            "search_ef": 16,
            "batch_query": False,
        }
    )
    cfg_path = work_dir / "config.json"
    write_json(cfg_path, fixed)
    output_path = work_dir / "result.json"
    cmd = [
        str(CURATOR_BIN), "bench",
        "--train_vecs", str(GT_ROOT / dataset / "train_vecs.npy"),
        "--train_access", str(GT_ROOT / dataset / "train_access.npy"),
        "--queries", str(work_dir / "query_vecs.npy"),
        "--query_labels", str(work_dir / "query_labels.npy"),
        "--config", str(cfg_path),
        "--k", "10",
        "--ef-list", "16",
        "--output", str(output_path),
    ]
    proc, wall_time, _ = run_command(cmd, timeout=12 * 3600)
    if proc is None:
        return

    result_candidates = sorted(work_dir.glob("result*.json"))
    if not result_candidates:
        log(f"no Curator result JSON in {work_dir}")
        return
    result = load_json(result_candidates[0])
    flash_path = work_dir / "flash.bin"
    pq_path = work_dir / "pq_codes.bin"
    apparent, allocated = measure_paths([flash_path, pq_path])
    record = {
        "method": "Curator",
        "method_key": "curator",
        "dataset": dataset,
        "config": fixed,
        "build_time_s": float(result.get("build_time_s", 0.0)),
        "wall_time_s": float(wall_time),
        "memory_bytes": int(result.get("memory_bytes", 0)),
        "rss_peak_query_mb": float(result.get("rss_peak_query_mb", 0.0)),
        "disk_reported_bytes": int(result.get("disk_bytes", 0)),
        "disk_apparent_bytes": int(apparent),
        "disk_allocated_bytes": int(allocated),
        "raw_vector_bytes": int(n * d * 4),
        "pq_code_bytes": int(n * pq_m),
        "index_files": {
            "flash_apparent_bytes": int(path_size(flash_path)[0]),
            "flash_allocated_bytes": int(path_size(flash_path)[1]),
            "pq_apparent_bytes": int(path_size(pq_path)[0]),
            "pq_allocated_bytes": int(path_size(pq_path)[1]),
        },
        "measurement": "fresh build on full training data with one-query subset",
        "note": "disk_allocated_bytes measured on ext4; sparse holes are not counted",
    }
    write_json(raw_path, record)
    log(
        "saved %s build=%.1fs mem=%.1fMB disk_alloc=%.1fMB"
        % (raw_path.name, record["build_time_s"], record["memory_bytes"] / MB, allocated / MB)
    )
    shutil.rmtree(work_dir, ignore_errors=True)


def run_prefilter(dataset: str, force: bool = False):
    raw_path = prefilter_raw_path(dataset)
    if raw_path.exists() and not force:
        log(f"skip existing {raw_path.name}")
        return
    n, d = dataset_shape(dataset)
    work_dir = TMP_ROOT / f"prefilter_{dataset}"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    make_one_query_subset(dataset, work_dir)

    output_path = work_dir / "result.json"
    cmd = [
        str(PREFILTER_BIN), "bench",
        "--train_vecs", str(GT_ROOT / dataset / "train_vecs.npy"),
        "--train_access", str(GT_ROOT / dataset / "train_access.npy"),
        "--queries", str(work_dir / "query_vecs.npy"),
        "--query_labels", str(work_dir / "query_labels.npy"),
        "--k", "10",
        "--output", str(output_path),
    ]
    proc, wall_time, _ = run_command(cmd, timeout=12 * 3600)
    if proc is None:
        return
    result = load_json(output_path)
    raw_vector_bytes = int(n * d * 4)
    memory_bytes = int(result.get("memory_bytes", 0))
    record = {
        "method": "PreFilter",
        "method_key": "prefilter",
        "dataset": dataset,
        "config": {"d": d, "k": 10, "mode": "in-memory exact scan"},
        "build_time_s": float(result.get("build_time_s", 0.0)),
        "wall_time_s": float(wall_time),
        "memory_bytes": memory_bytes,
        "rss_peak_query_mb": float(result.get("rss_peak_query_mb", 0.0)),
        "disk_reported_bytes": int(result.get("disk_bytes", 0)),
        "disk_apparent_bytes": 0,
        "disk_allocated_bytes": 0,
        "raw_vector_bytes": raw_vector_bytes,
        "vector_index_memory_bytes": raw_vector_bytes,
        "access_metadata_bytes": max(memory_bytes - raw_vector_bytes, 0),
        "measurement": "fresh in-memory build with query RSS sampled after releasing the build-only NPY buffer",
        "note": "no external index files; raw vectors are loaded in memory; the build-only NPY buffer is released before the query phase",
    }
    write_json(raw_path, record)
    log(
        "saved %s build=%.2fs mem=%.1fMB rss=%.1fMB"
        % (raw_path.name, record["build_time_s"], memory_bytes / MB, record["rss_peak_query_mb"])
    )
    shutil.rmtree(work_dir, ignore_errors=True)


def measure_diskivf(dataset: str):
    build_path = RESULT_ROOT / "DiskIVF" / f"_diskivf_{dataset}_build_sqrt.json"
    if not build_path.exists():
        build_path = RESULT_ROOT / "DiskIVF" / f"_diskivf_{dataset}_build.json"
    if not build_path.exists():
        log(f"missing DiskIVF build JSON for {dataset}")
        return None
    build = load_json(build_path)
    n, d = dataset_shape(dataset)
    index_dir = Path("/home/lyx") / f"diskivf_{dataset}_build_sqrt"
    if not index_dir.exists():
        index_dir = RESULT_ROOT / "DiskIVF-PostFiltering" / "disk_data" / f"{dataset}_build_sqrt"
    apparent, allocated = measure_paths([index_dir])
    if apparent == 0:
        apparent = int(build.get("disk_bytes", 0))
        allocated = apparent
    record = {
        "method": "DiskIVF",
        "method_key": "diskivf",
        "dataset": dataset,
        "config": build.get("config", {}),
        "build_time_s": float(build.get("build_time_s", 0.0)),
        "memory_bytes": int(build.get("memory_bytes", 0)),
        "rss_peak_query_mb": float(build.get("rss_peak_query_mb", 0.0)),
        "disk_reported_bytes": int(build.get("disk_bytes", 0)),
        "disk_apparent_bytes": int(apparent),
        "disk_allocated_bytes": int(allocated),
        "raw_vector_bytes": int(n * d * 4),
        "measurement": "existing full build result + external directory measurement",
        "note": str(build_path),
    }
    nprobe = DISKIVF_HIGH_NPROBE.get(dataset, 128)
    preload_path = RAW_DIR / (
        f"diskivf_query_preload_{dataset}_nprobe{nprobe}_q1.json")
    batch_path = RAW_DIR / (
        f"diskivf_query_batch_{dataset}_nprobe{nprobe}"
        f"_q{DISKIVF_BATCH_QUERIES}.json")
    single_path = RAW_DIR / f"diskivf_query_{dataset}_nprobe{nprobe}.json"

    # Memory-oriented overhead experiment: all nprobe clusters are loaded
    # before scanning (query results unchanged; peak RSS measured externally).
    if preload_path.exists():
        query = load_json(preload_path)
        record["rss_peak_query_mb"] = float(
            query.get("rss_peak_query_mb", record["rss_peak_query_mb"]))
        record["query_rss_nprobe"] = int(query.get("nprobe", nprobe))
        record["query_rss_query_count"] = int(query.get("query_count", 1))
        record["query_rss_preload_clusters"] = bool(
            query.get("preload_clusters", True))
        record["query_rss_source"] = str(preload_path)
        record["measurement"] = (
            "existing full build result + external directory measurement; "
            "query RSS from one-query overhead experiment with all nprobe "
            "clusters preloaded")
        return record
    if batch_path.exists():
        query = load_json(batch_path)
        record["rss_peak_query_mb"] = float(
            query.get("rss_peak_query_mb", record["rss_peak_query_mb"]))
        record["query_rss_nprobe"] = int(query.get("nprobe", nprobe))
        record["query_rss_query_count"] = int(
            query.get("query_count", DISKIVF_BATCH_QUERIES))
        record["query_rss_omp_threads"] = int(
            query.get("omp_num_threads", DISKIVF_BATCH_THREADS))
        record["query_rss_source"] = str(batch_path)
        record["measurement"] = (
            "existing full build result + external directory measurement; "
            "query RSS from 8-query batch high-nprobe search-mode measurement")
    elif single_path.exists():
        query = load_json(single_path)
        record["rss_peak_query_mb"] = float(
            query.get("rss_peak_query_mb", record["rss_peak_query_mb"]))
        record["query_rss_nprobe"] = int(query.get("nprobe", nprobe))
        record["query_rss_query_count"] = int(query.get("query_count", 1))
        record["query_rss_source"] = str(single_path)
        record["measurement"] = (
            "existing full build result + external directory measurement; "
            "query RSS from one-query high-nprobe search-mode measurement")
    else:
        record["query_rss_nprobe"] = None
    return record


def measure_spann(dataset: str):
    build_path = RESULT_ROOT / "SPANN" / f"_spann_{dataset}_build.json"
    if not build_path.exists():
        log(f"missing SPANN build JSON for {dataset}")
        return None
    build = load_json(build_path)
    n, d = dataset_shape(dataset)
    index_dir = Path("/home/lyx/spann_index") / dataset
    if not index_dir.exists():
        index_dir = RESULT_ROOT / "SPANN-PostFiltering" / "spann_index" / dataset
    apparent, allocated = measure_paths([index_dir])
    if apparent == 0:
        apparent = int(build.get("disk_bytes", 0))
        allocated = apparent
    return {
        "method": "SPANN",
        "method_key": "spann",
        "dataset": dataset,
        "config": build.get("config", {}),
        "build_time_s": float(build.get("build_time_s", 0.0)),
        "memory_bytes": int(build.get("memory_bytes", 0)),
        "rss_peak_query_mb": float(build.get("rss_peak_query_mb", 0.0)),
        "disk_reported_bytes": int(build.get("disk_bytes", 0)),
        "disk_apparent_bytes": int(apparent),
        "disk_allocated_bytes": int(allocated),
        "raw_vector_bytes": int(n * d * 4),
        "measurement": "existing full build result + external directory measurement",
        "note": str(build_path),
    }


def max_sweep_rss(dataset: str, method_dir: str, pq_m=None):
    path = RESULT_ROOT / method_dir / f"sweep_{dataset}.json"
    if not path.exists():
        return 0.0
    data = load_json(path)
    values = []
    for row in data.get("sweep_results", []):
        params = row.get("params", {})
        if pq_m is not None and params.get("pq_M") not in (None, pq_m):
            continue
        value = row.get("rss_peak_query_mb") or 0.0
        if value:
            values.append(float(value))
    return max(values) if values else 0.0


def build_summary(raw_records: dict):
    metrics = {
        "query_memory_mb": {ds: {} for ds in DATASETS},
        "build_time_s": {ds: {} for ds in DATASETS},
        "index_volume_mb": {ds: {} for ds in DATASETS},
        "volume_breakdown_mb": {ds: {} for ds in DATASETS},
        "sources": {ds: {} for ds in DATASETS},
    }
    for ds in DATASETS:
        for method in METHODS:
            record = raw_records.get((method, ds))
            if not record:
                continue
            # Query-time peak RSS.
            curator_memory_data = None
            if method == "curator":
                memory_record = RAW_DIR / f"curator_memory_{ds}.json"
                if memory_record.exists():
                    curator_memory_data = load_json(memory_record)
                    rss = float(curator_memory_data.get("rss_peak_query_mb", 0.0))
                else:
                    rss = max_sweep_rss(ds, "Curator", pq_m=CURATOR_PQ_M[ds])
                    if rss <= 0:
                        rss = float(record.get("rss_peak_query_mb", 0.0))
            elif method == "spann":
                rss = max_sweep_rss(ds, "SPANN")
                rss = max(rss, float(record.get("rss_peak_query_mb", 0.0)))
            else:
                rss = float(record.get("rss_peak_query_mb", 0.0))
            metrics["query_memory_mb"][ds][method] = rss

            metrics["build_time_s"][ds][method] = float(record.get("build_time_s", 0.0))

            # The volume figure reuses exactly the same query-memory values as
            # the memory figure for its in-memory segment, so the two figures
            # always report a consistent memory component.  External storage is
            # still the measured on-disk index footprint.
            memory_mb = rss
            if method == "prefilter":
                external = 0
            elif method == "curator" and curator_memory_data is not None:
                # Prefer the external storage measured on the same
                # memory-oriented Curator build; fall back to the raw
                # benchmark build for old records that lack the field.
                external = int(curator_memory_data.get(
                    "disk_allocated_bytes",
                    record.get("disk_allocated_bytes",
                               record.get("disk_reported_bytes", 0))))
            else:
                external = int(record.get("disk_allocated_bytes",
                                         record.get("disk_reported_bytes", 0)))
            metrics["index_volume_mb"][ds][method] = memory_mb + external / MB
            metrics["volume_breakdown_mb"][ds][method] = {
                "memory": memory_mb,
                "external": external / MB,
            }
            if method == "curator" and curator_memory_data is not None:
                metrics["sources"][ds][method] = curator_memory_data.get(
                    "measurement", "")
            else:
                metrics["sources"][ds][method] = record.get("measurement", "")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--methods", nargs="+", default=["curator", "prefilter"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.collect_only:
        for ds in args.datasets:
            if "curator" in args.methods:
                log(f"===== Curator {ds} =====")
                run_curator(ds, force=args.force)
            if "prefilter" in args.methods:
                log(f"===== PreFilter {ds} =====")
                run_prefilter(ds, force=args.force)

    raw_records = {}
    for ds in args.datasets:
        for method in METHODS:
            if method == "diskivf":
                record = measure_diskivf(ds)
            elif method == "spann":
                record = measure_spann(ds)
            else:
                path = RAW_DIR / f"{method}_{ds}.json"
                record = load_json(path) if path.exists() else None
            if record:
                raw_path = RAW_DIR / f"{method}_{ds}.json"
                write_json(raw_path, record)
                raw_records[(method, ds)] = record
            else:
                log(f"no record for {method}/{ds}")

    metrics = build_summary(raw_records)
    metrics["definitions"] = {
        "query_memory_mb": (
            "Query-phase peak process RSS in MB measured after build-only raw NPY buffers are released. "
            "PreFilter: in-memory exact scan. "
            "Curator: one-query memory-oriented build (n_clusters=8, "
            "pq_cache_max_blocks=64, SL predicate-only maps released); max_leaf_size=1024 "
            "for sift1m/gist1m/arxiv/yfcc100m, while wit uses max_leaf_size=32768 "
            "because it contains ~32,455 exactly duplicated vectors that cannot be "
            "split into a 1024-vector leaf. "
            "DiskIVF: one-query overhead experiment at nprobe=128 with all nprobe "
            "clusters preloaded; peak RSS measured externally with /usr/bin/time -v. "
            "SPANN: maximum RSS across existing full-query sweep points."
        ),
        "build_time_s": (
            "Index-build-only time in seconds. Curator and PreFilter are fresh "
            "full-training-data builds followed by a one-query subset; DiskIVF/SPANN "
            "are taken from their existing full-scale build JSON files."
        ),
        "index_volume_mb": (
            "Query-phase peak process memory (the same query_memory_mb values plotted "
            "in fig_full_memory) plus external index storage. Curator memory and "
            "external storage are both from the memory-oriented overhead build "
            "(n_clusters=8, pq_cache_max_blocks=64; max_leaf_size=1024 except wit, "
            "which uses 32768 for its ~32,455 exact duplicate vectors); its external "
            "storage is the allocated ext4 byte count measured with st_blocks so "
            "sparse holes are not counted. DiskIVF/SPANN external storage is the "
            "allocated directory size of their full build. PreFilter has no external "
            "index files."
        ),
        "volume_breakdown_mb": (
            "Stacked components of index_volume_mb used by fig_full_volume: memory is "
            "the query_memory_mb value (identical to fig_full_memory); external is the "
            "measured on-disk index storage. Curator uses the same memory-oriented "
            "build for both components (wit: max_leaf_size=32768 due to duplicate "
            "vectors)."
        ),
    }
    write_json(OUT_DIR / "index_metrics.json", metrics)
    log(f"wrote {OUT_DIR / 'index_metrics.json'}")


if __name__ == "__main__":
    main()