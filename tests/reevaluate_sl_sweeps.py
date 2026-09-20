#!/usr/bin/env python3
"""Refresh SL sweep Recall values from retained raw search JSONs.

Useful after ground_truth.npy is repaired without changing query labels:
QPS can be reused from raw per-query timings, while Recall must be recomputed
against the new ground truth.
"""
import argparse
import glob
import json
import re
import shutil
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))
import run_100k_sweep as R  # noqa: E402

K = 10


def diskivf_raw(dataset, params):
    npb = int(params.get("nprobe", 0))
    path = PROJ_ROOT / "4_Results/DiskIVF" / f"_sl_diskivf_{dataset}_nprobe{npb}_sqrt.json"
    return path if path.exists() else None


def spann_raw_index(dataset):
    pattern = re.compile(
        rf"^_sl_spann_{re.escape(dataset)}_(?:l2_)?mc(\d+)_of([0-9.]+)_sir(\d+)\.json$")
    out = {}
    for path in (PROJ_ROOT / "4_Results/SPANN").glob(f"_sl_spann_{dataset}*.json"):
        m = pattern.match(path.name)
        if not m:
            continue
        key = (int(m.group(1)), round(float(m.group(2)), 6), int(m.group(3)))
        out[key] = path
    return out


def curator_raw_index(dataset):
    pattern = re.compile(rf"^_sl_curator_{re.escape(dataset)}_pq(\d+)_ef(\d+)\.json$")
    out = {}
    for path in (PROJ_ROOT / "4_Results/Curator").glob(f"_sl_curator_{dataset}_pq*_ef*.json"):
        m = pattern.match(path.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = path
    return out


def reload_entry(raw_path, dataset, params, query_info, selected_buckets):
    with open(raw_path) as f:
        res = json.load(f)
    return R._parse_sl_from_json(
        res, dataset, K, 1.0, params, query_info, selected_buckets,
        memory_bytes=res.get("memory_bytes", 0),
        rss_peak_query_mb=res.get("rss_peak_query_mb", 0),
        disk_bytes=res.get("disk_bytes", 0),
    )


def refresh_dataset(dataset, method):
    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / dataset
    query_info = json.load(open(gt_dir / "query_info.json"))
    selected_buckets = R.select_3_buckets(gt_dir / "query_info.json")
    sweep_path = PROJ_ROOT / "4_Results" / method / f"sweep_{dataset}.json"
    if not sweep_path.exists():
        print(f"skip missing {sweep_path}")
        return
    data = json.load(open(sweep_path))
    spann_raws = spann_raw_index(dataset) if method == "SPANN" else {}
    curator_raws = curator_raw_index(dataset) if method == "Curator" else {}
    refreshed = 0
    missing = 0
    for i, entry in enumerate(data.get("sweep_results", [])):
        params = entry.get("params", {})
        if method == "DiskIVF":
            raw = diskivf_raw(dataset, params)
        elif method == "Curator":
            key = (int(params.get("pq_M", 0)), int(params.get("search_ef", 0)))
            raw = curator_raws.get(key)
        else:
            key = (int(params.get("max_check", 0)),
                   round(float(params.get("overfetch_factor", 0)), 6),
                   int(params.get("search_internal_result_num", 0)))
            raw = spann_raws.get(key)
        if raw is None:
            missing += 1
            continue
        try:
            data["sweep_results"][i] = reload_entry(
                raw, dataset, params, query_info, selected_buckets)
            refreshed += 1
        except Exception as exc:  # keep old entry on parse failure
            print(f"  failed {raw.name}: {exc}")
            missing += 1
    backup = sweep_path.with_name(sweep_path.name + ".bak_before_recall_refresh")
    if not backup.exists():
        shutil.copy2(sweep_path, backup)
    with open(sweep_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"{dataset} {method}: refreshed={refreshed} missing={missing} -> {sweep_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+",
                        default=["DiskIVF", "SPANN", "Curator"],
                        choices=["DiskIVF", "SPANN", "Curator"])
    args = parser.parse_args()
    for dataset in args.datasets:
        for method in args.methods:
            refresh_dataset(dataset, method)


if __name__ == "__main__":
    main()
