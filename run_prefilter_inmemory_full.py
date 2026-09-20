#!/usr/bin/env python3
"""Run the exact in-memory PreFilter baseline on a full dataset and write a sweep summary.

This is the algorithm-level PreFilter result used for QPS-vs-selectivity figures.
The external-scan variant is kept separate in sweep_<dataset>.json.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
import run_100k_sweep as R

K = 10
TIMEOUT = 7200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    ds = args.dataset
    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / ds
    bin_path = R.BINARIES["Pre-Filtering"]
    out_dir = PROJ_ROOT / "4_Results/Pre-Filtering"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / f"_sl_prefiltering_{ds}_inmem.json"
    sweep_out = out_dir / f"sweep_{ds}_inmem.json"

    query_info = json.load(open(gt_dir / "query_info.json"))
    selected_buckets = R.select_3_buckets(gt_dir / "query_info.json")
    cmd = [
        str(bin_path), "bench",
        "--train_vecs", str(gt_dir / "train_vecs.npy"),
        "--train_access", str(gt_dir / "train_access.npy"),
        "--queries", str(gt_dir / "query_vecs.npy"),
        "--query_labels", str(gt_dir / "query_labels.npy"),
        "--k", str(K), "--output", str(raw_out),
    ]
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stderr[-2000:], flush=True)
        raise SystemExit(1)
    res = json.load(open(raw_out))
    sl = R._parse_sl_from_json(
        res, ds, K, elapsed, {}, query_info, selected_buckets,
        memory_bytes=res.get("memory_bytes", 0),
        rss_peak_query_mb=res.get("rss_peak_query_mb", 0),
        disk_bytes=res.get("disk_bytes", 0),
    )
    data = {
        "dataset": ds,
        "method": "Pre-Filtering",
        "mode": "in-memory",
        "n_combinations": 1,
        "sweep_results": [sl],
        "index_memory_mb": res.get("memory_bytes", 0) / (1024.0 ** 2),
        "rss_peak_query_mb": res.get("rss_peak_query_mb", 0),
        "build_time_s": res.get("build_time_s", 0),
        "disk_bytes": 0,
        "note": "exact in-memory brute-force PreFilter baseline",
    }
    with open(sweep_out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"SL R={sl['avg_recall']:.4f} QPS={sl['qps']:.2f}")
    print("Saved", sweep_out)


if __name__ == "__main__":
    main()
