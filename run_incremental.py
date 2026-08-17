#!/usr/bin/env python3
"""
Incremental sweep filler: run ONLY new parameter combos that extend the
Pareto frontier of the existing sweep JSONs, then merge results back in.

Design rule (derived from the existing data): every new combo is picked to
land on the Pareto frontier line (extend the low-recall/high-QPS end, the
high-recall/low-QPS end, or fill a big gap), so the plotted line gets more
points and longer recall coverage.

Usage:
    python run_incremental.py --method DiskIVF
    python run_incremental.py --method Curator
    python run_incremental.py --method SPANN
"""
import argparse, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))
import run_100k_sweep as R

TIMEOUT = 1800

# ---------------------------------------------------------------------------
# New-combo manifests (derived from current frontier gaps in sweep JSONs).
#   Curator:  (pq_M, search_ef)   — left end uses the pq_M that already gives
#             the best QPS at ef=64; right end uses the best-recall pq_M.
#   DiskIVF:  (nprobe, mode)      — mode "both" = SL+CP, "sl" = SL only (CP
#             already exists for those combos). nprobe=16 of gist1m_100k is
#             omitted: ~192 MB/query x 1000 queries is not feasible on NTFS.
#   SPANN:    (max_check, overfetch_factor)
# ---------------------------------------------------------------------------
CURATOR_NEW = {
    "sift1m_100k":   [(32, 16),  (128, 2048)],
    "yfcc100m_100k": [(24, 16),  (64, 2048)],
    "arxiv_100k":    [(128, 16), (128, 2048)],
    # gist1m/wit: multiple right-end combos to push recall as high as possible
    "gist1m_100k":   [(96, 16),  (240, 2048), (240, 4096), (320, 4096)],
    "wit_100k":      [(96, 16),  (96, 2048),  (128, 4096)],
}

# (nprobe, mode, timeout_s). nprobe=32 => exact recall 1.0 (right end >= 0.95).
# Timeouts sized from the measured ~25 MB/s effective throughput of this batch:
# 77 MB/query x 1020 queries ~= 3160 s, 96 MB/query x 1020 ~= 3920 s.
DISKIVF_NEW = {
    "sift1m_100k":   [(3, "both", 1800), (6, "both", 1800)],
    "yfcc100m_100k": [(3, "both", 1800), (6, "both", 1800), (8, "sl", 1800),
                      (16, "sl", 1800), (32, "both", 4500)],
    "arxiv_100k":    [(3, "both", 1800), (6, "both", 1800), (16, "both", 4500)],
    "gist1m_100k":   [(3, "both", 1800), (6, "both", 1800), (8, "both", 5400)],
    "wit_100k":      [(3, "both", 1800), (6, "both", 1800), (8, "sl", 1800),
                      (16, "both", 4500)],
}

# Larger overfetch on the right end to push recall up (SPANN post-filtering
# recall rises with k' = k * overfetch_factor). Round-2 additions (max_check
# 32768 + of 2000/5000) target labels with few vectors.
# Round-3 additions are 3-tuples (max_check, overfetch_factor, hash_exp):
# overfetch turned out to be capped by SPTAG's head approximation, so we also
# sweep HashTableExponent (finer BKTree head partitioning => better recall).
SPANN_NEW = {
    "sift1m_100k":   [(16384, 200)],
    "yfcc100m_100k": [(16384, 200), (32768, 2000), (32768, 5000),
                      (16384, 200, 9), (16384, 200, 10),
                      (16384, 200, 8, 64), (16384, 200, 8, 128)],
    "arxiv_100k":    [(1024, 10), (16384, 200)],
    "gist1m_100k":   [(16384, 200), (16384, 400), (32768, 2000),
                      (16384, 200, 9), (16384, 200, 10),
                      (16384, 200, 8, 64), (16384, 200, 8, 128)],
    "wit_100k":      [(16384, 200), (16384, 1000), (32768, 2000), (32768, 5000),
                      (32768, 2000, 9), (32768, 2000, 10),
                      (32768, 2000, 8, 64), (32768, 2000, 8, 128),
                      (16384, 200, 8, 0, 20000), (16384, 200, 8, 0, 50000)],
}

METHOD_DIRS = {"Curator": "Curator", "DiskIVF": "DiskIVF", "SPANN": "SPANN"}
DATASETS = ["sift1m_100k", "yfcc100m_100k", "arxiv_100k", "gist1m_100k", "wit_100k"]
K = 10


def load_cp_shared(dataset):
    """Return (cp_selected, cp_gt_map, rpn_to_polish, query_info, selected_buckets)."""
    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / dataset
    cp_dir = gt_dir / "complex_predicate"
    cp_selected = R.select_representative_filters(cp_dir / "filters.json")
    cp_gt_map = {}
    rpn_to_polish = {}
    for formula in cp_selected:
        safe = formula.replace(" ", "_")
        cp_gt_map[formula] = np.load(cp_dir / f"gt_{safe}.npy")
        rpn_to_polish[R.polish_to_rpn(formula)] = formula
    query_info = json.load(open(gt_dir / "query_info.json"))
    selected_buckets = R.select_3_buckets(gt_dir / "query_info.json")
    return cp_selected, cp_gt_map, rpn_to_polish, query_info, selected_buckets


def backup(path):
    bak = path.with_name(path.name + ".bak_incr")
    if path.exists() and not bak.exists():
        import shutil
        shutil.copy2(path, bak)
        print(f"  backup -> {bak.name}", flush=True)


def find_entry(sweep_results, params):
    for r in sweep_results:
        if r.get("params") == params:
            return r
    return None


def merge_entry(entry, sl=None, cp=None, sl_res=None, cp_res=None):
    """Merge newly parsed SL/CP stats into an entry dict (in place)."""
    if sl is not None:
        for k, v in sl.items():
            entry[k] = v
    if cp is not None:
        entry["complex_predicate"] = cp
    # If SL never succeeded (existing gap), take memory/build from the CP call
    if sl is None and cp is not None and cp_res is not None:
        entry.setdefault("build_time_s", cp_res.get("build_time_s", 0))
        entry.setdefault("memory_bytes", cp_res.get("memory_bytes", 0))
        mb = cp_res.get("memory_bytes", 0)
        entry.setdefault("memory_mb", float(mb) / (1024.0 ** 2) if mb else 0)
        entry.setdefault("rss_peak_query_mb",
                         float(cp_res.get("rss_peak_query_mb", 0)))
    if sl is not None and cp is not None and cp_res is not None:
        entry.setdefault("build_time_s", cp_res.get("build_time_s", 0))


def run_curator_incremental(dataset):
    subdir = METHOD_DIRS["Curator"]
    out_dir = PROJ_ROOT / "4_Results" / subdir
    path = out_dir / f"sweep_{dataset}.json"
    data = json.load(open(path))
    backup(path)

    sweep_cfg = json.load(open(PROJ_ROOT / "3_Config/Curator/sweep_100k.json"))
    fixed = sweep_cfg["fixed"]
    cp_sel, cp_gt, rpn2pol, qinfo, buckets = load_cp_shared(dataset)

    for pq_m, ef in CURATOR_NEW[dataset]:
        params = {"pq_M": pq_m, "search_ef": ef}
        if find_entry(data["sweep_results"], params) is not None:
            print(f"  skip existing {params}", flush=True)
            continue
        tag = f"pq{pq_m}_ef{ef}"
        print(f"  [Curator {dataset}] {params} ...", flush=True)

        base_cfg = dict(fixed)
        base_cfg["pq_M"] = pq_m
        base_cfg["search_ef"] = ef
        base_cfg["flash_path"] = str(out_dir / f"disk_data_{dataset}_{tag}")

        tmp_cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                              delete=False, dir="/tmp")
        json.dump(base_cfg, tmp_cfg)
        tmp_cfg.close()
        extra_args = ["--config", tmp_cfg.name]

        entry = {"params": params}
        ok, el, res = R.run_cpp_bench(
            "Curator", dataset, extra_args,
            out_dir / f"_sl_{tag}.json", timeout=TIMEOUT)
        sl = None
        if ok and res:
            sl = R._parse_sl_from_json(
                res, dataset, K, el, params, qinfo, buckets,
                memory_bytes=res.get("memory_bytes", 0),
                rss_peak_query_mb=res.get("rss_peak_query_mb", 0))
        else:
            print("    SL_FAILED", flush=True)

        cp = None
        cp_res = None
        filters_file = R._write_cp_filters_file(cp_sel, rpn=True)
        ok2, el2, cp_res = R.run_cpp_bench(
            "Curator", dataset, extra_args,
            out_dir / f"_cp_{tag}.json", timeout=TIMEOUT,
            query_labels_opt=False, filters_file_path=filters_file,
            query_vecs_name="complex_predicate/query_vecs.npy")
        os.unlink(filters_file)
        if ok2 and cp_res:
            cp = R._parse_cp_from_json(cp_res, cp_gt, K, rpn2pol)
        else:
            print("    CP_FAILED", flush=True)

        merge_entry(entry, sl=sl, cp=cp, cp_res=cp_res)
        data["sweep_results"].append(entry)
        os.unlink(tmp_cfg.name)
        print(f"    SL R@10={entry.get('avg_recall')} QPS={entry.get('qps')}  "
              f"CP R@10={entry.get('complex_predicate', {}).get('avg_recall')}", flush=True)

    data["n_combinations"] = len(data["sweep_results"])
    json.dump(data, open(path, "w"), indent=2)
    print(f"  saved {path.name}", flush=True)


def run_diskivf_incremental(dataset):
    subdir = METHOD_DIRS["DiskIVF"]
    out_dir = PROJ_ROOT / "4_Results" / subdir
    path = out_dir / f"sweep_{dataset}.json"
    data = json.load(open(path))
    backup(path)

    cp_sel, cp_gt, rpn2pol, qinfo, buckets = load_cp_shared(dataset)
    # Same path convention as run_100k_sweep.py:435
    disk_dir_base = PROJ_ROOT / "4_Results/DiskIVF-PostFiltering/disk_data" / dataset

    for nprobe, mode, timeout_s in DISKIVF_NEW[dataset]:
        params = {"nprobe": nprobe, "nlist": 32}
        existing = find_entry(data["sweep_results"], params)
        if existing is not None:
            has_sl = existing.get("qps") is not None
            has_cp = existing.get("complex_predicate") is not None
            need_sl = (mode in ("both", "sl")) and not has_sl
            need_cp = (mode == "both") and not has_cp
            if not need_sl and not need_cp:
                print(f"  skip existing {params}", flush=True)
                continue
            entry = existing
        else:
            entry = {"params": params}
            data["sweep_results"].append(entry)
            need_sl = mode in ("both", "sl")
            need_cp = mode == "both"

        disk_dir = f"{disk_dir_base}_nprobe{nprobe}"
        # The C++ mkdir() only creates the leaf dir and ignores failure;
        # the parent chain may be missing (git-clean removes disk_data/*),
        # so create the full chain here.
        os.makedirs(disk_dir, exist_ok=True)
        extra_args = ["--nlist", "32", "--nprobe", str(nprobe),
                      "--disk_dir", disk_dir]
        print(f"  [DiskIVF {dataset}] nprobe={nprobe} mode={mode} "
              f"(need_sl={need_sl}, need_cp={need_cp}) ...", flush=True)

        if need_sl:
            ok, el, res = R.run_cpp_bench(
                "DiskIVF", dataset, extra_args,
                out_dir / f"_sl_diskivf_nprobe{nprobe}.json", timeout=timeout_s)
            if ok and res:
                sl = R._parse_sl_from_json(
                    res, dataset, K, el, params, qinfo, buckets,
                    memory_bytes=res.get("memory_bytes", 0),
                    rss_peak_query_mb=res.get("rss_peak_query_mb", 0))
                merge_entry(entry, sl, None)
            else:
                print("    SL_FAILED", flush=True)

        if need_cp:
            filters_file = R._write_cp_filters_file(cp_sel, rpn=False)
            ok2, el2, cp_res = R.run_cpp_bench(
                "DiskIVF", dataset, extra_args,
                out_dir / f"_cp_diskivf_nprobe{nprobe}.json", timeout=timeout_s,
                query_labels_opt=False, filters_file_path=filters_file,
                query_vecs_name="complex_predicate/query_vecs.npy")
            os.unlink(filters_file)
            if ok2 and cp_res:
                cp = R._parse_cp_from_json(cp_res, cp_gt, K)
                merge_entry(entry, None, cp, cp_res)
            else:
                print("    CP_FAILED", flush=True)

        print(f"    SL R@10={entry.get('avg_recall')} QPS={entry.get('qps')}  "
              f"CP R@10={entry.get('complex_predicate', {}).get('avg_recall')}", flush=True)

    data["n_combinations"] = len(data["sweep_results"])
    json.dump(data, open(path, "w"), indent=2)
    print(f"  saved {path.name}", flush=True)


def run_spann_incremental(dataset):
    subdir = METHOD_DIRS["SPANN"]
    out_dir = PROJ_ROOT / "4_Results" / subdir
    path = out_dir / f"sweep_{dataset}.json"
    data = json.load(open(path))
    backup(path)

    sweep_cfg = json.load(open(PROJ_ROOT / "3_Config/SPANN-PostFiltering/sweep_100k.json"))
    spann_fixed = sweep_cfg["fixed"]
    cp_sel, cp_gt, rpn2pol, qinfo, buckets = load_cp_shared(dataset)

    tmp_cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False, dir="/tmp")
    json.dump(spann_fixed, tmp_cfg)
    tmp_cfg.close()

    for item in SPANN_NEW[dataset]:
        if len(item) == 5:
            mc, of, hash_exp, bkt_k, sir = item
        elif len(item) == 4:
            mc, of, hash_exp, bkt_k = item
            sir = None
        elif len(item) == 3:
            mc, of, hash_exp = item
            bkt_k, sir = None, None
        else:
            mc, of, hash_exp, bkt_k, sir = item[0], item[1], None, None, None
        if bkt_k == 0:
            bkt_k = None
        if sir == 0:
            sir = None

        params = {"max_check": mc, "overfetch_factor": of}
        if hash_exp is not None:
            params["hash_exp"] = hash_exp
        if bkt_k is not None:
            params["bkt_kmeans_k"] = bkt_k
        if sir is not None:
            params["search_internal_result_num"] = sir

        if find_entry(data["sweep_results"], params) is not None:
            print(f"  skip existing {params}", flush=True)
            continue

        # Per-combo config: override build parameters when given
        cfg_path = tmp_cfg.name
        combo_tmp = None
        overrides = {}
        if hash_exp is not None and spann_fixed.get("hash_exp") != hash_exp:
            overrides["hash_exp"] = hash_exp
        if bkt_k is not None and spann_fixed.get("bkt_kmeans_k", 0) != bkt_k:
            overrides["bkt_kmeans_k"] = bkt_k
        if sir is not None and spann_fixed.get("search_internal_result_num", 0) != sir:
            overrides["search_internal_result_num"] = sir
        if overrides:
            combo_cfg = dict(spann_fixed)
            combo_cfg.update(overrides)
            combo_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                                    delete=False, dir="/tmp")
            json.dump(combo_cfg, combo_tmp)
            combo_tmp.close()
            cfg_path = combo_tmp.name

        he_tag = f"_he{hash_exp}" if hash_exp is not None else ""
        bk_tag = f"_bk{bkt_k}" if bkt_k is not None else ""
        sir_tag = f"_sir{sir}" if sir is not None else ""
        # Same path convention as run_100k_sweep.py:530
        idx_dir = str(PROJ_ROOT / "4_Results/SPANN-PostFiltering/spann_index"
                      / f"{dataset}_mc{mc}_of{of}{he_tag}{bk_tag}{sir_tag}")
        os.makedirs(idx_dir, exist_ok=True)
        extra_args = ["--index_dir", idx_dir, "--config", cfg_path,
                      "--max_check", str(mc), "--overfetch_factor", str(of)]
        print(f"  [SPANN {dataset}] {params} ...", flush=True)

        entry = {"params": params}
        ok, el, res = R.run_cpp_bench(
            "SPANN", dataset, extra_args,
            out_dir / f"_sl_spann_mc{mc}_of{of}{he_tag}{bk_tag}{sir_tag}.json",
            timeout=TIMEOUT)
        sl = None
        if ok and res:
            sl = R._parse_sl_from_json(
                res, dataset, K, el, params, qinfo, buckets,
                memory_bytes=res.get("memory_bytes", 0),
                rss_peak_query_mb=res.get("rss_peak_query_mb", 0))
        else:
            print("    SL_FAILED", flush=True)

        cp = None
        cp_res = None
        filters_file = R._write_cp_filters_file(cp_sel, rpn=False)
        ok2, el2, cp_res = R.run_cpp_bench(
            "SPANN", dataset, extra_args,
            out_dir / f"_cp_spann_mc{mc}_of{of}{he_tag}{bk_tag}{sir_tag}.json",
            timeout=TIMEOUT,
            query_labels_opt=False, filters_file_path=filters_file,
            query_vecs_name="complex_predicate/query_vecs.npy")
        os.unlink(filters_file)
        if combo_tmp is not None:
            os.unlink(combo_tmp.name)
        if ok2 and cp_res:
            cp = R._parse_cp_from_json(cp_res, cp_gt, K)
        else:
            print("    CP_FAILED", flush=True)

        merge_entry(entry, sl=sl, cp=cp, cp_res=cp_res)
        data["sweep_results"].append(entry)
        print(f"    SL R@10={entry.get('avg_recall')} QPS={entry.get('qps')}  "
              f"CP R@10={entry.get('complex_predicate', {}).get('avg_recall')}", flush=True)

    os.unlink(tmp_cfg.name)
    data["n_combinations"] = len(data["sweep_results"])
    json.dump(data, open(path, "w"), indent=2)
    print(f"  saved {path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["Curator", "DiskIVF", "SPANN"])
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    for ds in datasets:
        print(f"\n===== {args.method} {ds} =====", flush=True)
        if args.method == "Curator":
            run_curator_incremental(ds)
        elif args.method == "DiskIVF":
            run_diskivf_incremental(ds)
        elif args.method == "SPANN":
            run_spann_incremental(ds)
    print("\nAll incremental sweeps complete!", flush=True)


if __name__ == "__main__":
    main()
