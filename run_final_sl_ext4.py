#!/usr/bin/env python3
"""Run the final single-label sweeps with all indexes on WSL ext4.

The parameter plan in 3_Config/final_sl_ext4_plan.json was chosen from the
previous measured QPS-recall points so that every method/dataset curve can
produce at least five visible vertices between Recall@10 = 0.5 and the
method's feasible upper range.  Curator builds once per (dataset, pq_M) and
searches an ef list in-process; DiskIVF and SPANN reuse ext4-resident indexes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_100k_sweep as R  # noqa: E402

K = 10
PLAN_PATH = ROOT / "3_Config/final_sl_ext4_plan.json"
SUPPLEMENT_PATH = ROOT / "3_Config/final_sl_smoothing_supplement.json"
RAW_ROOT = ROOT / "4_Results/_final_ext4_raw"
METHOD_DIRS = {
    "curator": ROOT / "4_Results/Curator",
    "diskivf": ROOT / "4_Results/DiskIVF",
    "spann": ROOT / "4_Results/SPANN",
    "prefilter": ROOT / "4_Results/Pre-Filtering",
}
DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]
CURATOR_CONFIG_KEYS = {
    "sift1m": "sift1m",
    "gist1m": "gist1m",
    "arxiv": "arxiv",
    "yfcc100m": "yfcc100m",
    "wit": "wit",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def load_plan_with_supplement():
    """Load the main SL plan plus targeted smoothing combos, if present.

    The supplement only adds SPANN parameter combinations; no existing point is
    removed or re-run because the runner reuses every raw output it finds.
    """
    plan = load_json(PLAN_PATH)
    if not SUPPLEMENT_PATH.exists():
        return plan
    supplement = load_json(SUPPLEMENT_PATH)
    for dataset, combos in supplement.items():
        if dataset not in plan:
            continue
        current = plan[dataset].setdefault("spann", {}).setdefault("combos", [])
        seen = {tuple(item) for item in current}
        for combo in combos:
            key = tuple(combo)
            if key not in seen:
                current.append(list(combo))
                seen.add(key)
    return plan


def save_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
    os.replace(tmp, path)


def backup_once(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_name(path.name + ".bak_before_ext4_unify")
    if not backup.exists():
        shutil.copy2(path, backup)


def run_command(cmd, timeout=86400, label=""):
    log(("RUN " + " ".join(str(c) for c in cmd)) if not label else f"RUN {label}")
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        print(proc.stdout[-4000:], flush=True)
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"command failed rc={proc.returncode}: {cmd[0]}")
    log(f"  rc=0 wall={elapsed:.1f}s")
    return elapsed


def query_context(ds):
    gt_dir = ROOT / "1_Data/ground_truth" / ds
    query_info = load_json(gt_dir / "query_info.json")
    selected_buckets = R.select_3_buckets(gt_dir / "query_info.json")
    return gt_dir, query_info, selected_buckets


def write_sweep(method_key, ds, entries, extra=None):
    out_dir = METHOD_DIRS[method_key]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sweep_{ds}.json"
    backup_once(path)
    data = {
        "dataset": ds,
        "method": {"curator": "Curator", "diskivf": "DiskIVF",
                   "spann": "SPANN", "prefilter": "Pre-Filtering"}[method_key],
        "index_storage": "wsl_ext4" if method_key != "prefilter" else "in_memory",
        "generated_by": "run_final_sl_ext4.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_combinations": len(entries),
        "sweep_results": entries,
    }
    if extra:
        data.update(extra)
    save_json_atomic(path, data)
    log(f"  wrote {path} ({len(entries)} entries)")


# ---------------------------------------------------------------------------
# Curator
# ---------------------------------------------------------------------------

def set_spann_search_cap(index_dir: Path, cap: int) -> None:
    """Set SPTAG SearchInternalResultNum before index load.

    SPTAG only initializes its SSD extra-searcher buffers correctly when this
    mutable parameter is present in indexloader.ini at load time; overriding it
    after load is not equivalent in the current SPTAG build.
    """
    ini = Path(index_dir) / "indexloader.ini"
    if not ini.exists():
        raise FileNotFoundError(ini)
    lines = ini.read_text(encoding="utf-8").splitlines()
    out = []
    found = False
    for line in lines:
        if line.startswith("SearchInternalResultNum="):
            out.append(f"SearchInternalResultNum={int(cap)}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"SearchInternalResultNum={int(cap)}")
    ini.write_text("\n".join(out) + "\n", encoding="utf-8")


def run_curator(ds, plan, reuse_raw=True):
    builds = plan["curator"]["builds"]
    gt_dir, query_info, selected_buckets = query_context(ds)
    binary = R.BINARIES["Curator"]
    out_dir = METHOD_DIRS["curator"]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = RAW_ROOT / "curator" / ds
    raw_dir.mkdir(parents=True, exist_ok=True)
    cfg_src = ROOT / "3_Config/Curator" / f"sweep_full_{CURATOR_CONFIG_KEYS[ds]}.json"
    if not cfg_src.exists():
        raise FileNotFoundError(cfg_src)
    base_config = load_json(cfg_src)
    entries = []
    for build in builds:
        pq_m = int(build["pq_M"])
        efs = [int(x) for x in build["search_ef"]]
        base_out = raw_dir / f"curator_{ds}_pq{pq_m}.json"
        existing = {ef: base_out.with_name(base_out.stem + f"_ef{ef}.json")
                    for ef in efs}
        missing = [ef for ef in efs if not existing[ef].exists()]
        if missing or not reuse_raw:
            flash_dir = Path(f"/home/lyx/curator_index/{ds}/pq{pq_m}")
            flash_dir.mkdir(parents=True, exist_ok=True)
            cfg = dict(base_config.get("fixed", {}))
            cfg.update({
                "pq_M": pq_m,
                "search_ef": efs[0],
                "flash_path": str(flash_dir / "flash.bin"),
                "pq_codes_path": str(flash_dir / "pq_codes.bin"),
                "persist_pq_codes": True,
                "use_flash_storage": True,
                "batch_query": False,
            })
            cfg_path = raw_dir / f"config_pq{pq_m}.json"
            save_json_atomic(cfg_path, cfg)
            cmd = [
                str(binary), "bench",
                "--train_vecs", str(gt_dir / "train_vecs.npy"),
                "--train_access", str(gt_dir / "train_access.npy"),
                "--queries", str(gt_dir / "query_vecs.npy"),
                "--query_labels", str(gt_dir / "query_labels.npy"),
                "--k", str(K),
                "--config", str(cfg_path),
                "--ef-list", ",".join(str(ef) for ef in (missing or efs)),
                "--output", str(base_out),
            ]
            run_command(cmd, timeout=86400,
                        label=f"Curator {ds} pq_M={pq_m} build + {len(missing or efs)} efs")
        for ef in efs:
            out_path = existing[ef]
            if not out_path.exists():
                log(f"  missing Curator raw output for ef={ef}, skipping")
                continue
            res = load_json(out_path)
            params = {"pq_M": pq_m, "search_ef": ef}
            sl = R._parse_sl_from_json(
                res, ds, K, res.get("build_time_s", 0) + 1.0, params,
                query_info, selected_buckets,
                memory_bytes=res.get("memory_bytes", 0),
                rss_peak_query_mb=res.get("rss_peak_query_mb", 0),
                disk_bytes=res.get("disk_bytes", 0))
            if sl.get("qps", 0) <= 0:
                log(f"  skip qps=0 pq={pq_m} ef={ef}")
                continue
            sl["index_storage"] = "wsl_ext4"
            sl["flash_path"] = str(Path(f"/home/lyx/curator_index/{ds}/pq{pq_m}"))
            entries.append(sl)
            log(f"  Curator pq={pq_m} ef={ef}: R={sl['avg_recall']:.4f} Q={sl['qps']:.2f}")
    entries.sort(key=lambda r: (r["params"].get("pq_M", 0), r["params"].get("search_ef", 0)))
    write_sweep("curator", ds, entries,
                extra={"index_dir": f"/home/lyx/curator_index/{ds}",
                       "builds": builds})


# ---------------------------------------------------------------------------
# DiskIVF
# ---------------------------------------------------------------------------
def run_diskivf(ds, plan, reuse_raw=True):
    nprobes = [int(x) for x in plan["diskivf"]["nprobe"]]
    gt_dir, query_info, selected_buckets = query_context(ds)
    binary = R.BINARIES["DiskIVF"]
    index_dir = Path(f"/home/lyx/diskivf_{ds}_build_sqrt")
    if not index_dir.exists():
        raise FileNotFoundError(index_dir)
    meta = np.fromfile(index_dir / "metadata.bin", dtype=np.int32)
    if meta.size >= 3:
        nlist = int(meta[2])
    else:
        nlist = 0
    raw_dir = RAW_ROOT / "diskivf" / ds
    raw_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for nprobe in nprobes:
        out_path = raw_dir / f"diskivf_{ds}_nprobe{nprobe}.json"
        if not reuse_raw or not out_path.exists():
            cmd = [
                str(binary), "search",
                "--disk_dir", str(index_dir),
                "--queries", str(gt_dir / "query_vecs.npy"),
                "--query_labels", str(gt_dir / "query_labels.npy"),
                "--nprobe", str(nprobe), "--k", str(K),
                "--batch-query", "--output", str(out_path),
            ]
            run_command(cmd, timeout=7200, label=f"DiskIVF {ds} nprobe={nprobe}")
        if not out_path.exists():
            continue
        res = load_json(out_path)
        params = {"nprobe": nprobe, "nlist": nlist}
        elapsed = res.get("build_time_s", 0) + 1.0
        sl = R._parse_sl_from_json(
            res, ds, K, elapsed, params, query_info, selected_buckets,
            memory_bytes=res.get("memory_bytes", 0),
            rss_peak_query_mb=res.get("rss_peak_query_mb", 0),
            disk_bytes=res.get("disk_bytes", 0))
        if sl.get("qps", 0) <= 0:
            continue
        sl["index_storage"] = "wsl_ext4"
        entries.append(sl)
        log(f"  DiskIVF nprobe={nprobe}: R={sl['avg_recall']:.4f} Q={sl['qps']:.2f}")
    entries.sort(key=lambda r: r["params"]["nprobe"])
    write_sweep("diskivf", ds, entries,
                extra={"index_dir": str(index_dir), "nlist": nlist})


# ---------------------------------------------------------------------------
# SPANN
# ---------------------------------------------------------------------------
def run_spann(ds, plan, reuse_raw=True):
    combos = [(int(mc), float(of), int(sir)) for mc, of, sir in plan["spann"]["combos"]]
    gt_dir, query_info, selected_buckets = query_context(ds)
    binary = R.BINARIES["SPANN"]
    index_dir = Path(f"/home/lyx/spann_index/{ds}")
    if not index_dir.exists():
        raise FileNotFoundError(index_dir)
    cfg_src = ROOT / "3_Config/SPANN-PostFiltering" / f"sweep_full_{ds}.json"
    if not cfg_src.exists():
        # The legacy full sift1m SPANN sweep shared the generic fixed config.
        cfg_src = ROOT / "3_Config/SPANN-PostFiltering/sweep_full_arxiv.json"
    fixed = dict(load_json(cfg_src).get("fixed", {}))
    # SPTAG must be told L2 before index load; relying on the wrapper's
    # post-load override is not sufficient for the SSD extra searcher.
    fixed["dist_method"] = "L2"
    fixed["batch_query"] = False
    raw_dir = RAW_ROOT / "spann" / ds
    raw_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for mc, of, sir in combos:
        tag = f"mc{int(mc)}_of{str(of).replace('.', 'p')}_sir{sir}"
        out_path = raw_dir / f"spann_{ds}_{tag}.json"
        # SPTAG must see the cap in indexloader.ini before LoadIndex().
        cap = 64 if int(sir) <= 0 else int(sir)
        set_spann_search_cap(index_dir, cap)
        cfg = dict(fixed)
        cfg.update({"search_internal_result_num": 0, "max_check": mc,
                    "overfetch_factor": of, "k": K})
        cfg_path = raw_dir / f"config_{ds}_{tag}.json"
        save_json_atomic(cfg_path, cfg)
        if not reuse_raw or not out_path.exists():
            cmd = [
                str(binary), "search",
                "--index_dir", str(index_dir),
                "--queries", str(gt_dir / "query_vecs.npy"),
                "--query_labels", str(gt_dir / "query_labels.npy"),
                "--config", str(cfg_path),
                "--max_check", str(mc), "--overfetch_factor", str(of),
                "--k", str(K), "--output", str(out_path),
            ]
            run_command(cmd, timeout=43200,
                        label=f"SPANN {ds} mc={mc} of={of} sir={sir}")
        if not out_path.exists():
            continue
        res = load_json(out_path)
        params = {"max_check": mc, "overfetch_factor": of,
                  "search_internal_result_num": sir}
        sl = R._parse_sl_from_json(
            res, ds, K, res.get("build_time_s", 0) + 1.0, params,
            query_info, selected_buckets,
            memory_bytes=res.get("memory_bytes", 0),
            rss_peak_query_mb=res.get("rss_peak_query_mb", 0),
            disk_bytes=res.get("disk_bytes", 0))
        if sl.get("qps", 0) <= 0:
            continue
        sl["index_storage"] = "wsl_ext4"
        entries.append(sl)
        log(f"  SPANN mc={mc} of={of} sir={sir}: R={sl['avg_recall']:.4f} Q={sl['qps']:.2f}")
    entries.sort(key=lambda r: (r["params"]["max_check"],
                                r["params"]["overfetch_factor"],
                                r["params"]["search_internal_result_num"]))
    write_sweep("spann", ds, entries,
                extra={"index_dir": str(index_dir)})


# ---------------------------------------------------------------------------
# PreFilter (exact in-memory baseline)
# ---------------------------------------------------------------------------
def run_prefilter(ds, plan, reuse_raw=True):
    gt_dir, query_info, selected_buckets = query_context(ds)
    binary = R.BINARIES["Pre-Filtering"]
    out_path = METHOD_DIRS["prefilter"] / f"_sl_prefiltering_{ds}_inmem.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not reuse_raw or not out_path.exists():
        cmd = [
            str(binary), "bench",
            "--train_vecs", str(gt_dir / "train_vecs.npy"),
            "--train_access", str(gt_dir / "train_access.npy"),
            "--queries", str(gt_dir / "query_vecs.npy"),
            "--query_labels", str(gt_dir / "query_labels.npy"),
            "--k", str(K), "--output", str(out_path),
        ]
        run_command(cmd, timeout=7200, label=f"PreFilter-inmem {ds}")
    res = load_json(out_path)
    sl = R._parse_sl_from_json(
        res, ds, K, res.get("build_time_s", 0) + 1.0, {},
        query_info, selected_buckets,
        memory_bytes=res.get("memory_bytes", 0),
        rss_peak_query_mb=res.get("rss_peak_query_mb", 0),
        disk_bytes=res.get("disk_bytes", 0))
    sl["index_storage"] = "in_memory"
    data = {
        "dataset": ds, "method": "Pre-Filtering", "mode": "in-memory",
        "index_storage": "in_memory",
        "generated_by": "run_final_sl_ext4.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_combinations": 1,
        "sweep_results": [sl],
    }
    save_json_atomic(METHOD_DIRS["prefilter"] / f"sweep_{ds}_inmem.json", data)
    log(f"  PreFilter-inmem: R={sl['avg_recall']:.4f} Q={sl['qps']:.2f}")


RUNNERS = {
    "curator": run_curator,
    "diskivf": run_diskivf,
    "spann": run_spann,
    "prefilter": run_prefilter,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--methods", nargs="+", default=["curator", "diskivf", "spann", "prefilter"])
    parser.add_argument("--no-reuse", action="store_true",
                        help="ignore existing _final_ext4_raw files")
    args = parser.parse_args()
    plan = load_plan_with_supplement()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    reuse = not args.no_reuse
    for ds in args.datasets:
        if ds not in plan:
            continue
        for method in args.methods:
            log("=" * 70)
            log(f"{method} / {ds}")
            try:
                RUNNERS[method](ds, plan[ds], reuse_raw=reuse)
            except Exception as exc:
                log(f"FAILED {method}/{ds}: {exc}")
                import traceback
                traceback.print_exc()
                raise
    log("ALL DONE")


if __name__ == "__main__":
    main()