#!/usr/bin/env python3
"""Fill missing Curator memory-optimized records with calibrated estimates.

If a memory-oriented full build has been measured for at least one dataset, we
scale Curator's reported source-level index memory (``memory_bytes``) and the
residual non-index overhead to the remaining datasets.  This is only used when
the corresponding ``curator_memory_<dataset>.json`` experiment is unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "4_Results/build_measure/raw"
DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    measured = {}
    old = {}
    for dataset in DATASETS:
        old_path = RAW / f"curator_{dataset}.json"
        new_path = RAW / f"curator_memory_{dataset}.json"
        if old_path.exists():
            old[dataset] = load(old_path)
        if new_path.exists():
            measured[dataset] = load(new_path)

    if not measured:
        raise SystemExit("no measured memory-oriented Curator records exist")

    memory_ratios = []
    overhead_ratios = []
    for dataset, new in measured.items():
        reference = old.get(dataset)
        if not reference:
            continue
        old_mem = float(reference.get("memory_bytes", 0.0)) / (1024.0 ** 2)
        old_rss = float(reference.get("rss_peak_query_mb", 0.0))
        new_mem = float(new.get("memory_bytes", 0.0)) / (1024.0 ** 2)
        new_rss = float(new.get("rss_peak_query_mb", 0.0))
        if old_mem > 0 and new_mem > 0 and old_rss > old_mem:
            memory_ratios.append(new_mem / old_mem)
            overhead_ratios.append((new_rss - new_mem) / (old_rss - old_mem))

    if not memory_ratios or not overhead_ratios:
        raise SystemExit("cannot calibrate estimates from measured records")

    memory_ratio = sum(memory_ratios) / len(memory_ratios)
    overhead_ratio = sum(overhead_ratios) / len(overhead_ratios)

    for dataset in DATASETS:
        if dataset in measured:
            continue
        reference = old.get(dataset)
        if not reference:
            raise SystemExit(f"missing reference record for {dataset}")
        old_mem = float(reference.get("memory_bytes", 0.0)) / (1024.0 ** 2)
        old_rss = float(reference.get("rss_peak_query_mb", 0.0))
        estimate_mem = old_mem * memory_ratio
        estimate_rss = (estimate_mem
                        + (old_rss - old_mem) * overhead_ratio)
        record = {
            "method": "Curator",
            "method_key": "curator",
            "dataset": dataset,
            "estimated": True,
            "memory_bytes": int(round(estimate_mem * (1024.0 ** 2))),
            "rss_peak_query_mb": float(estimate_rss),
            "config_model": (
                "n_clusters=8, max_leaf_size=1024, pq_cache_max_blocks=64, "
                "SL predicate-only maps released"),
            "measurement": (
                "estimated from measured memory-oriented Curator experiments "
                "and source-level memory_bytes scaling"),
            "calibration": {
                "memory_ratio": memory_ratio,
                "overhead_ratio": overhead_ratio,
                "measured_datasets": sorted(measured),
            },
        }
        path = RAW / f"curator_memory_{dataset}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"{dataset}: estimated rss={estimate_rss:.1f} MiB -> {path.name}")


if __name__ == "__main__":
    main()