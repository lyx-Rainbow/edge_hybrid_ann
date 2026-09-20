#!/usr/bin/env python3
"""Generic Curator full-scale runner using build-once multi-search_ef mode."""
import argparse
import json
import os
import subprocess
import tempfile
import time
from itertools import product
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
import run_100k_sweep as R

K = 10
TIMEOUT = 7200

def build_multi_cmd(ds, pq_m, efs, cfg_path, out_base):
    GT_DIR = PROJ_ROOT / "1_Data/ground_truth" / ds
    BIN = R.BINARIES["Curator"]
    return [
        str(BIN), "bench",
        "--train_vecs", str(GT_DIR / "train_vecs.npy"),
        "--train_access", str(GT_DIR / "train_access.npy"),
        "--queries", str(GT_DIR / "query_vecs.npy"),
        "--query_labels", str(GT_DIR / "query_labels.npy"),
        "--k", str(K),
        "--config", cfg_path,
        "--ef-list", ",".join(str(e) for e in efs),
        "--output", str(out_base),
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ds = args.dataset
    cfg_path = PROJ_ROOT / "3_Config/Curator" / f"sweep_full_{ds}.json"
    cfg = json.load(open(cfg_path))
    pqs = cfg["build_params"]["pq_M"][ds]
    efs = cfg["search_params"]["search_ef"]
    if args.quick:
        pqs = pqs[:1]
        efs = [16, 64, 256]

    GT_DIR = PROJ_ROOT / "1_Data/ground_truth" / ds
    OUT_DIR = PROJ_ROOT / "4_Results/Curator"
    SWEEP = OUT_DIR / f"sweep_{ds}.json"
    FIXED = cfg["fixed"]
    query_info = json.load(open(GT_DIR / "query_info.json"))
    selected_buckets = R.select_3_buckets(GT_DIR / "query_info.json")
    data = json.load(open(SWEEP)) if SWEEP.exists() else {"dataset": ds, "method": "Curator", "sweep_results": [], "n_combinations": 0}
    existing = {(r["params"]["pq_M"], r["params"]["search_ef"], r["params"].get("nprobe", 8))
                for r in data["sweep_results"] if r.get("qps", 0) > 0}

    for pq_m in pqs:
        missing_efs = [ef for ef in efs if (pq_m, ef, FIXED.get("nprobe", 8)) not in existing]
        if not missing_efs:
            continue
        print(f"Building pq_M={pq_m}, efs={missing_efs}", flush=True)
        base_cfg = dict(FIXED)
        base_cfg.update({"pq_M": pq_m, "search_ef": missing_efs[0],
                         "flash_path": str(OUT_DIR / f"disk_data_{ds}_pq{pq_m}")})
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp")
        json.dump(base_cfg, tmp); tmp.close()
        out_base = OUT_DIR / f"_sl_curator_{ds}_pq{pq_m}.json"
        cmd = build_multi_cmd(ds, pq_m, missing_efs, tmp.name, out_base)
        print(" ".join(str(c) for c in cmd), flush=True)
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT pq={pq_m}", flush=True)
            os.unlink(tmp.name); continue
        dt = time.time() - t0
        if r.returncode != 0:
            print("FAIL", r.stderr[-2000:], flush=True)
            os.unlink(tmp.name); continue
        for ef in missing_efs:
            out = OUT_DIR / f"_sl_curator_{ds}_pq{pq_m}_ef{ef}.json"
            if not out.exists():
                print(f"missing output {out}", flush=True); continue
            res = json.load(open(out))
            sl = R._parse_sl_from_json(res, ds, K, dt, {"pq_M": pq_m, "search_ef": ef, "nprobe": FIXED.get("nprobe", 8)},
                query_info, selected_buckets,
                memory_bytes=res.get("memory_bytes", 0),
                rss_peak_query_mb=res.get("rss_peak_query_mb", 0),
                disk_bytes=res.get("disk_bytes", 0))
            if sl.get("qps", 0) <= 0:
                print(f"qps=0 skip pq={pq_m} ef={ef}", flush=True); continue
            data["sweep_results"].append(sl)
            data["n_combinations"] = len(data["sweep_results"])
            with open(SWEEP, "w") as f:
                json.dump(data, f, indent=2)
            pb = sl.get("per_bucket", {})
            print(f"  pq={pq_m} ef={ef} R={sl['avg_recall']:.4f} Q={sl['qps']:.2f} "
                  f"1p={pb.get('1p',{}).get('avg_recall',0):.3f}", flush=True)
        os.unlink(tmp.name)
        try:
            os.remove(str(OUT_DIR / f"disk_data_{ds}_pq{pq_m}"))
        except OSError:
            pass
    print("Done", flush=True)

if __name__ == "__main__":
    main()
