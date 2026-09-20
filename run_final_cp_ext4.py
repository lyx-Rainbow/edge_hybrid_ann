#!/usr/bin/env python3
"""Final complex-predicate sweeps with all indexes unified on WSL ext4.

The plan is in 3_Config/final_cp_ext4_plan.json.  Curator is built once per
dataset/pq_M and searched with an ef list (new CLI support in Curator main);
DiskIVF/SPANN indexes are reused from /home/lyx ext4 copies; PreFilter is the
real external-scan baseline already stored on ext4.
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
PLAN_PATH = ROOT / "3_Config/final_cp_ext4_plan.json"
RAW_ROOT = ROOT / "4_Results/_final_ext4_raw/cp"
METHOD_DIRS = {
    "curator": ROOT / "4_Results/Curator",
    "diskivf": ROOT / "4_Results/DiskIVF",
    "spann": ROOT / "4_Results/SPANN",
    "prefilter": ROOT / "4_Results/Pre-Filtering",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


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
    log(f"RUN {label}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stdout[-4000:], flush=True)
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"command failed rc={proc.returncode}: {cmd[0]}")
    log("  done")


def paths_for(ds, plan):
    gt_dir = ROOT / "1_Data/ground_truth" / ds
    gt = np.load(gt_dir / "complex_predicate" / plan["gt"])
    query = gt_dir / "complex_predicate/query_vecs.npy"
    return gt_dir, gt, query


# ---------------------------------------------------------------------------
# Curator
# ---------------------------------------------------------------------------
def curator_raw_paths(ds, plan):
    pq = int(plan["curator_pq_M"])
    base = RAW_ROOT / f"curator_{ds}_pq{pq}.json"
    return pq, base, {int(ef): base.with_name(base.stem + f"_ef{ef}.json")
                      for ef in plan["curator_search_ef"]}


def run_curator(ds, plan, reuse=True):
    pq, base, outputs = curator_raw_paths(ds, plan)
    missing = [ef for ef, path in outputs.items() if not path.exists()]
    if missing or not reuse:
        gt_dir, gt, query = paths_for(ds, plan)
        cfg_src = ROOT / plan["curator_config"]
        fixed = dict(load_json(cfg_src).get("fixed", {}))
        flash_dir = Path(f"/home/lyx/curator_cp/{ds}/pq{pq}")
        flash_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(fixed)
        cfg.update({
            "pq_M": pq,
            "search_ef": missing[0] if missing else plan["curator_search_ef"][0],
            "flash_path": str(flash_dir / "flash.bin"),
            "pq_codes_path": str(flash_dir / "pq_codes.bin"),
            "persist_pq_codes": True,
            "use_flash_storage": True,
            "batch_query": False,
        })
        cfg_path = RAW_ROOT / f"curator_{ds}_pq{pq}_config.json"
        save_json_atomic(cfg_path, cfg)
        filters_file = R._write_cp_filters_file([plan["formula"]], rpn=True)
        try:
            cmd = [
                str(R.BINARIES["Curator"]), "bench",
                "--train_vecs", str(gt_dir / "train_vecs.npy"),
                "--train_access", str(gt_dir / "train_access.npy"),
                "--queries", str(query),
                "--config", str(cfg_path),
                "--k", str(K),
                "--filters_file", filters_file,
                "--ef-list", ",".join(str(ef) for ef in (missing or plan["curator_search_ef"])),
                "--output", str(base),
            ]
            run_command(cmd, timeout=86400,
                        label=f"Curator-CP {ds} pq_M={pq} ef-list={missing or len(plan['curator_search_ef'])}")
        finally:
            try:
                os.unlink(filters_file)
            except OSError:
                pass
    return pq, outputs


def parse_curator(ds, plan, pq, outputs):
    _, gt, _ = paths_for(ds, plan)
    rpn = R.polish_to_rpn(plan["formula"])
    entries = []
    for ef, path in sorted(outputs.items()):
        if not path.exists():
            continue
        res = load_json(path)
        cp = R._parse_cp_from_json(res, {rpn: gt}, K)
        if not cp:
            continue
        info = (cp.get("per_filter") or {}).get(rpn)
        if not info:
            continue
        cp = dict(cp)
        cp["per_filter"] = {plan["formula"]: info}
        cp["avg_recall"] = info["avg_recall"]
        cp["qps"] = info["qps"]
        cp["avg_latency_ms"] = info["avg_latency_ms"]
        entries.append({
            "params": {"pq_M": pq, "search_ef": ef},
            "complex_predicate": cp,
            "build_time_s": res.get("build_time_s", 0),
            "memory_bytes": res.get("memory_bytes", 0),
            "disk_bytes": res.get("disk_bytes", 0),
            "rss_peak_query_mb": res.get("rss_peak_query_mb", 0),
            "index_storage": "wsl_ext4",
        })
    return entries


# ---------------------------------------------------------------------------
# DiskIVF / SPANN / PreFilter
# ---------------------------------------------------------------------------
def _output_path(kind, ds, tag):
    return RAW_ROOT / f"{kind}_{ds}_{tag}.json"


def run_diskivf(ds, plan, reuse=True):
    _, _, query = paths_for(ds, plan)
    index = Path(f"/home/lyx/diskivf_{ds}_build_sqrt")
    filters_file = R._write_cp_filters_file([plan["formula"]], rpn=False)
    try:
        for nprobe in plan["diskivf_nprobe"]:
            out = _output_path("diskivf", ds, f"nprobe{nprobe}")
            if out.exists() and reuse:
                continue
            cmd = [
                str(R.BINARIES["DiskIVF"]), "search",
                "--disk_dir", str(index),
                "--queries", str(query),
                "--nprobe", str(nprobe), "--k", str(K),
                "--batch-query", "--filters_file", filters_file,
                "--output", str(out),
            ]
            run_command(cmd, timeout=7200, label=f"DiskIVF-CP {ds} nprobe={nprobe}")
    finally:
        try:
            os.unlink(filters_file)
        except OSError:
            pass



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


def run_spann(ds, plan, reuse=True):
    _, _, query = paths_for(ds, plan)
    index = Path(f"/home/lyx/spann_index/{ds}")
    filters_file = R._write_cp_filters_file([plan["formula"]], rpn=False)
    cfg_src = ROOT / "3_Config/SPANN-PostFiltering" / f"sweep_full_{ds}.json"
    if not cfg_src.exists():
        # The legacy full sift1m SPANN sweep shared the generic fixed config.
        cfg_src = ROOT / "3_Config/SPANN-PostFiltering/sweep_full_arxiv.json"
    fixed = dict(load_json(cfg_src).get("fixed", {}))
    # SPTAG must be told L2 before index load; relying on the wrapper's
    # post-load override is not sufficient for the SSD extra searcher.
    fixed["dist_method"] = "L2"
    fixed["batch_query"] = False
    try:
        for mc, of, sir in plan["spann_combos"]:
            tag = f"mc{int(mc)}_of{str(of).replace('.', 'p')}_sir{sir}"
            out = _output_path("spann", ds, tag)
            if out.exists() and reuse:
                continue
            # SPTAG must see the cap in indexloader.ini before LoadIndex().
            cap = 64 if int(sir) <= 0 else int(sir)
            set_spann_search_cap(index, cap)
            cfg = dict(fixed)
            cfg.update({"search_internal_result_num": 0, "max_check": mc,
                        "overfetch_factor": of, "k": K})
            cfg_path = RAW_ROOT / f"spann_{ds}_{tag}_config.json"
            save_json_atomic(cfg_path, cfg)
            cmd = [
                str(R.BINARIES["SPANN"]), "search",
                "--index_dir", str(index),
                "--queries", str(query),
                "--config", str(cfg_path),
                "--max_check", str(mc), "--overfetch_factor", str(of),
                "--k", str(K), "--filters_file", filters_file,
                "--output", str(out),
            ]
            run_command(cmd, timeout=43200, label=f"SPANN-CP {ds} {tag}")
    finally:
        try:
            os.unlink(filters_file)
        except OSError:
            pass


def run_prefilter(ds, plan, reuse=True):
    _, _, query = paths_for(ds, plan)
    out = _output_path("prefilter", ds, "ext4")
    if out.exists() and reuse:
        return
    filters_file = R._write_cp_filters_file([plan["formula"]], rpn=False)
    try:
        cmd = [
            str(R.BINARIES["Pre-Filtering"]), "bench",
            "--train_vecs", str(ROOT / "1_Data/ground_truth" / ds / "train_vecs.npy"),
            "--train_access", str(ROOT / "1_Data/ground_truth" / ds / "train_access.npy"),
            "--queries", str(query),
            "--k", str(K), "--external-scan", "--scan-chunk", "4096",
            "--vector-file", f"/home/lyx/{ds}.pfvec",
            "--filters_file", filters_file,
            "--output", str(out),
        ]
        run_command(cmd, timeout=14400, label=f"PreFilter-CP {ds} external")
    finally:
        try:
            os.unlink(filters_file)
        except OSError:
            pass


def _cp_entry(res, params, formula, gt):
    cp = R._parse_cp_from_json(res, {formula: gt}, K)
    if not cp:
        return None
    return {
        "params": params,
        "complex_predicate": cp,
        "build_time_s": res.get("build_time_s", 0),
        "memory_bytes": res.get("memory_bytes", 0),
        "disk_bytes": res.get("disk_bytes", 0),
        "rss_peak_query_mb": res.get("rss_peak_query_mb", 0),
        "index_storage": "wsl_ext4",
    }


def assemble(ds, plan):
    _, gt, _ = paths_for(ds, plan)
    pq = int(plan["curator_pq_M"])

    curator_entries = []
    for ef in plan["curator_search_ef"]:
        path = RAW_ROOT / f"curator_{ds}_pq{pq}.json"
        path = path.with_name(path.stem + f"_ef{ef}.json")
        if path.exists():
            curator_entries.extend(parse_curator(ds, plan, pq, {int(ef): path}))

    disk_entries = []
    for nprobe in plan["diskivf_nprobe"]:
        path = _output_path("diskivf", ds, f"nprobe{nprobe}")
        if path.exists():
            entry = _cp_entry(load_json(path), {"nprobe": nprobe}, plan["formula"], gt)
            if entry:
                disk_entries.append(entry)

    spann_entries = []
    for mc, of, sir in plan["spann_combos"]:
        tag = f"mc{int(mc)}_of{str(of).replace('.', 'p')}_sir{sir}"
        path = _output_path("spann", ds, tag)
        if path.exists():
            entry = _cp_entry(load_json(path),
                              {"max_check": mc, "overfetch_factor": of,
                               "search_internal_result_num": sir},
                              plan["formula"], gt)
            if entry:
                spann_entries.append(entry)

    pre_entries = []
    pre = _output_path("prefilter", ds, "ext4")
    if pre.exists():
        entry = _cp_entry(load_json(pre), {}, plan["formula"], gt)
        if entry:
            entry["index_storage"] = "wsl_ext4_external_scan"
            pre_entries.append(entry)

    per_method = {
        "curator": curator_entries,
        "diskivf": disk_entries,
        "spann": spann_entries,
        "prefilter": pre_entries,
    }
    labels = {"curator": "Curator", "diskivf": "DiskIVF",
              "spann": "SPANN", "prefilter": "Pre-Filtering"}
    for method_key, entries in per_method.items():
        out_path = METHOD_DIRS[method_key] / f"sweep_{ds}_cp.json"
        backup_once(out_path)
        data = {
            "dataset": ds,
            "method": labels[method_key],
            "filter": plan["formula"],
            "predicate_type": plan["ptype"],
            "index_storage": "wsl_ext4",
            "generated_by": "run_final_cp_ext4.py",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_combinations": len(entries),
            "sweep_results": entries,
        }
        save_json_atomic(out_path, data)
        log(f"assembled {out_path} ({len(entries)} entries)")
    return per_method

RUNNERS = {
    "curator": run_curator,
    "diskivf": run_diskivf,
    "spann": run_spann,
    "prefilter": run_prefilter,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["sift1m", "yfcc100m", "arxiv"])
    parser.add_argument("--method", default="all",
                        choices=["all", "curator", "diskivf", "spann", "prefilter"])
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    plan = load_json(PLAN_PATH)[args.dataset]
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    methods = (["curator", "diskivf", "spann", "prefilter"]
               if args.method == "all" else [args.method])
    for method in methods:
        log("=" * 70)
        log(f"CP {method} / {args.dataset}")
        try:
            RUNNERS[method](args.dataset, plan, reuse=not args.no_reuse)
        except Exception as exc:
            log(f"FAILED {method}/{args.dataset}: {exc}")
            raise
        if method == "curator":
            # Curator's in-process ef-list writes raw files directly; parse them now.
            pq, _, outputs = curator_raw_paths(args.dataset, plan)
            assemble(args.dataset, plan)
        else:
            assemble(args.dataset, plan)
    assemble(args.dataset, plan)
    log("ALL DONE")


if __name__ == "__main__":
    main()