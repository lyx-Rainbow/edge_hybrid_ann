#!/usr/bin/env python3
"""Generic DiskIVF full-scale search-mode runner with nlist ≈ sqrt(N)."""
import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent
import run_100k_sweep as R

K = 10
NPROBES = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 128]
TIMEOUT = 7200

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ext4", action="store_true",
                        help="copy the label-independent index to WSL ext4 before search")
    args = parser.parse_args()
    ds = args.dataset
    nprobes = [1, 2, 4, 8] if args.quick else NPROBES

    GT_DIR = PROJ_ROOT / "1_Data/ground_truth" / ds
    BIN = R.BINARIES["DiskIVF"]
    OUT_DIR = PROJ_ROOT / "4_Results/DiskIVF"
    SWEEP = OUT_DIR / f"sweep_{ds}.json"
    DISK_DIR = PROJ_ROOT / "4_Results/DiskIVF-PostFiltering/disk_data" / f"{ds}_build_sqrt"
    if args.ext4:
        ext4_dir = Path("/home/lyx") / f"diskivf_{ds}_build_sqrt"
        if not ext4_dir.exists():
            print(f"Copying DiskIVF index to {ext4_dir} ...", flush=True)
            shutil.copytree(DISK_DIR, ext4_dir)
        DISK_DIR = ext4_dir
    # nlist ≈ sqrt(number of training vectors)
    train_shape = np.load(GT_DIR / "train_vecs.npy", mmap_mode="r").shape
    NLIST = int(round(math.sqrt(train_shape[0])))
    print(f"Dataset {ds}: N={train_shape[0]}, nlist={NLIST}", flush=True)

    def run_cmd(cmd, timeout=TIMEOUT):
        print(" ".join(str(c) for c in cmd), flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        if r.returncode != 0:
            print(r.stderr[-1500:], flush=True)
            raise RuntimeError(f"command failed rc={r.returncode}: {cmd}")
        return dt

    if not DISK_DIR.exists() or not any(DISK_DIR.iterdir()):
        print("Building DiskIVF index once ...", flush=True)
        DISK_DIR.mkdir(parents=True, exist_ok=True)
        build_out = OUT_DIR / f"_diskivf_{ds}_build_sqrt.json"
        cmd = [str(BIN), "bench",
               "--train_vecs", str(GT_DIR / "train_vecs.npy"),
               "--train_access", str(GT_DIR / "train_access.npy"),
               "--queries", str(GT_DIR / "query_vecs.npy"),
               "--query_labels", str(GT_DIR / "query_labels.npy"),
               "--nlist", str(NLIST), "--nprobe", "1",
               "--disk_dir", str(DISK_DIR), "--k", str(K),
               "--output", str(build_out)]
        run_cmd(cmd, timeout=TIMEOUT)
    else:
        print("Reusing existing sqrt-nlist DiskIVF index", flush=True)

    query_info = json.load(open(GT_DIR / "query_info.json"))
    selected_buckets = R.select_3_buckets(GT_DIR / "query_info.json")
    data = json.load(open(SWEEP)) if SWEEP.exists() else {"dataset": ds, "method": "DiskIVF", "sweep_results": [], "n_combinations": 0}
    # Keep old nlist=32 entries but avoid mixing in new table; use separate file if needed.
    # For now, we write into the same sweep; entries include nlist.
    existing = {(r["params"]["nprobe"], r["params"].get("nlist")) for r in data["sweep_results"] if r.get("qps", 0) > 0}
    missing = [n for n in nprobes if (n, NLIST) not in existing and not any(True for e in existing if e[0] == n)]
    # Simpler: if any entry with same nprobe but old nlist exists, re-run for new nlist.
    existing_by_probe = {r["params"]["nprobe"] for r in data["sweep_results"] if r.get("qps", 0) > 0}
    missing = [n for n in nprobes if n not in existing_by_probe or any(r["params"].get("nlist") != NLIST for r in data["sweep_results"] if r.get("params", {}).get("nprobe") == n and r.get("qps", 0) > 0)]
    print("missing", missing, flush=True)

    for nprobe in missing:
        out = OUT_DIR / f"_sl_diskivf_{ds}_nprobe{nprobe}_sqrt.json"
        cmd = [str(BIN), "search",
               "--disk_dir", str(DISK_DIR),
               "--queries", str(GT_DIR / "query_vecs.npy"),
               "--query_labels", str(GT_DIR / "query_labels.npy"),
               "--nprobe", str(nprobe), "--k", str(K), "--batch-query",
               "--output", str(out)]
        try:
            el = run_cmd(cmd, timeout=TIMEOUT)
        except Exception as e:
            print(f"FAILED nprobe={nprobe}: {e}", flush=True)
            continue
        res = json.load(open(out))
        sl = R._parse_sl_from_json(res, ds, K, el, {"nprobe": nprobe, "nlist": NLIST},
            query_info, selected_buckets,
            memory_bytes=res.get("memory_bytes", 0),
            rss_peak_query_mb=res.get("rss_peak_query_mb", 0),
            disk_bytes=res.get("disk_bytes", 0))
        if sl.get("qps", 0) <= 0:
            print(f"qps=0 skip {nprobe}", flush=True); continue
        # Remove old same-nprobe entries with old nlist to avoid duplicates.
        data["sweep_results"] = [r for r in data["sweep_results"] if not (r.get("params", {}).get("nprobe") == nprobe and r.get("qps", 0) > 0)]
        data["sweep_results"].append(sl)
        data["n_combinations"] = len(data["sweep_results"])
        with open(SWEEP, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  nprobe={nprobe} R={sl['avg_recall']:.4f} Q={sl['qps']:.2f}", flush=True)
    print("Done", flush=True)

if __name__ == "__main__":
    main()
