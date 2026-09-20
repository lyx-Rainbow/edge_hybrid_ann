#!/usr/bin/env python3
"""Generic SPANN full-scale search-mode runner: build once, then sweep search params."""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
import run_100k_sweep as R

K = 10
COMBOS = [
    (2048, 10, 0), (4096, 25, 0), (8192, 10, 0), (8192, 50, 0),
    (8192, 100, 0), (8192, 200, 0), (16384, 100, 0), (16384, 200, 0),
    (32768, 500, 0), (32768, 1000, 0), (32768, 2000, 0), (8192, 500, 0),
    # SIR-boosted points for low-selectivity / natural-label datasets.
    (8192, 500, 100), (16384, 500, 200), (32768, 1000, 500),
    (32768, 2000, 1000),
]
QUICK = [(8192, 50, 0), (16384, 100, 0)]
TIMEOUT = 21600
FIXED = {"dist_method": "L2", "num_threads": 1, "max_check": 8192,
         "hash_exp": 10, "bkt_kmeans_k": 64, "k": K,
         "overfetch_factor": 50, "overfetch_adaptive": False,
         "batch_query": False,
         "search_internal_result_num": 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ds = args.dataset
    combos = QUICK if args.quick else COMBOS
    GT_DIR = PROJ_ROOT / "1_Data/ground_truth" / ds
    BIN = R.BINARIES["SPANN"]
    INDEX_DIR = PROJ_ROOT / "4_Results/SPANN-PostFiltering/spann_index" / ds
    OUT_DIR = PROJ_ROOT / "4_Results/SPANN"
    SWEEP = OUT_DIR / f"sweep_{ds}.json"

    def run_cmd(cmd, timeout=TIMEOUT):
        print(" ".join(str(c) for c in cmd), flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        if r.returncode != 0:
            print(r.stderr[-1000:], flush=True)
            raise RuntimeError(f"command failed rc={r.returncode}: {cmd}")
        return dt

    build_out = OUT_DIR / f"_spann_{ds}_build.json"
    if not (INDEX_DIR / "indexloader.ini").exists():
        cfg_path = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp")
        json.dump(FIXED, cfg_path); cfg_path.close()
        cmd = [str(BIN), "bench",
               "--train_vecs", str(GT_DIR / "train_vecs.npy"),
               "--train_access", str(GT_DIR / "train_access.npy"),
               "--queries", str(GT_DIR / "query_vecs.npy"),
               "--query_labels", str(GT_DIR / "query_labels.npy"),
               "--config", cfg_path.name, "--index_dir", str(INDEX_DIR),
               "--k", str(K), "--output", str(build_out)]
        print("Building SPANN ...", flush=True)
        run_cmd(cmd, timeout=TIMEOUT)
        print("Build done", flush=True)
    else:
        print("Reusing SPANN index", flush=True)

    query_info = json.load(open(GT_DIR / "query_info.json"))
    selected_buckets = R.select_3_buckets(GT_DIR / "query_info.json")
    data = json.load(open(SWEEP)) if SWEEP.exists() else {"dataset": ds, "method": "SPANN", "sweep_results": [], "n_combinations": 0}
    existing = {(r["params"].get("max_check"), round(r["params"].get("overfetch_factor", 0), 1), r["params"].get("search_internal_result_num", 0))
                for r in data["sweep_results"] if r.get("qps", 0) > 0}
    missing = [c for c in combos if c not in existing]
    print("missing", missing, flush=True)
    for mc, of, sir in missing:
        cfg = dict(FIXED)
        cfg["search_internal_result_num"] = sir
        cfg_path = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp")
        json.dump(cfg, cfg_path); cfg_path.close()
        out = OUT_DIR / f"_sl_spann_{ds}_mc{mc}_of{of}_sir{sir}.json"
        cmd = [str(BIN), "search",
               "--index_dir", str(INDEX_DIR),
               "--queries", str(GT_DIR / "query_vecs.npy"),
               "--query_labels", str(GT_DIR / "query_labels.npy"),
               "--config", cfg_path.name,
               "--max_check", str(mc), "--overfetch_factor", str(of),
               "--k", str(K), "--output", str(out)]
        try:
            el = run_cmd(cmd, timeout=TIMEOUT)
        except Exception as e:
            print(f"FAILED {mc},{of}: {e}", flush=True); continue
        res = json.load(open(out))
        sl = R._parse_sl_from_json(res, ds, K, el,
            {"max_check": mc, "overfetch_factor": of, "search_internal_result_num": sir},
            query_info, selected_buckets,
            memory_bytes=res.get("memory_bytes", 0),
            rss_peak_query_mb=res.get("rss_peak_query_mb", 0),
            disk_bytes=res.get("disk_bytes", 0))
        if sl.get("qps", 0) <= 0:
            print("qps=0 skip", flush=True); continue
        data["sweep_results"].append(sl)
        data["n_combinations"] = len(data["sweep_results"])
        with open(SWEEP, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  mc={mc} of={of} R={sl['avg_recall']:.4f} Q={sl['qps']:.2f}", flush=True)
    print("Done", flush=True)

if __name__ == "__main__":
    main()
