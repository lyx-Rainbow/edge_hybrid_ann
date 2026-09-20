#!/usr/bin/env python3
"""Generic real external-memory PreFilter runner (full queries, real disk chunks)."""
import argparse
import json
import subprocess
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
import run_100k_sweep as R

K = 10
TIMEOUT = 7200
CHUNK = 4096

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ds = args.dataset
    GT_DIR = PROJ_ROOT / "1_Data/ground_truth" / ds
    BIN = R.BINARIES["Pre-Filtering"]
    OUT_DIR = PROJ_ROOT / "4_Results/Pre-Filtering"
    VEC_FILE = Path("/home/lyx") / f"{ds}.pfvec"
    query_info = json.load(open(GT_DIR / "query_info.json"))
    selected_buckets = R.select_3_buckets(GT_DIR / "query_info.json")
    out = OUT_DIR / f"_sl_prefiltering_{ds}_real.json"
    cmd = [str(BIN), "bench",
           "--train_vecs", str(GT_DIR / "train_vecs.npy"),
           "--train_access", str(GT_DIR / "train_access.npy"),
           "--queries", str(GT_DIR / "query_vecs.npy"),
           "--query_labels", str(GT_DIR / "query_labels.npy"),
           "--k", str(K), "--external-scan", "--scan-chunk", str(CHUNK),
           "--vector-file", str(VEC_FILE), "--output", str(out)]
    print(" ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True); raise SystemExit(1)
    res = json.load(open(out))
    sl = R._parse_sl_from_json(res, ds, K, dt, {}, query_info, selected_buckets,
        memory_bytes=res.get("memory_bytes", 0),
        rss_peak_query_mb=res.get("rss_peak_query_mb", 0),
        disk_bytes=res.get("disk_bytes", 0))
    data = {
        "dataset": ds, "method": "Pre-Filtering", "n_combinations": 1,
        "sweep_results": [sl],
        "index_memory_mb": res.get("memory_bytes", 0) / (1024.0 ** 2),
        "rss_peak_query_mb": res.get("rss_peak_query_mb", 0),
        "build_time_s": res.get("build_time_s", 0),
        "disk_bytes": res.get("disk_bytes", 0),
        "note": "real external chunked disk scan"
    }
    with open(OUT_DIR / f"sweep_{ds}.json", "w") as f:
        json.dump(data, f, indent=2)
    print("SL", sl.get("avg_recall"), sl.get("qps"), flush=True)

if __name__ == "__main__":
    main()


