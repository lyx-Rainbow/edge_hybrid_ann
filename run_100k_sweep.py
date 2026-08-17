#!/usr/bin/env python3
"""
Unified sweep runner for 100K benchmark experiments.

Supports 4 methods (all via C++ CLI):
  - Curator:   C++ CLI bench (rebuild per combo: pq_M × search_ef)
  - DiskIVF:   C++ CLI bench (rebuild per nprobe)
  - SPANN:     C++ CLI bench (rebuild per combo: max_check × overfetch_factor)
  - Pre-Filtering: C++ CLI bench (single run)

When both --run_sl and --run_cp are specified, each combo runs SL first,
then CP (two C++ calls), and the results are merged into one entry.

Usage:
    python run_100k_sweep.py --dataset sift1m_100k --method Curator
    python run_100k_sweep.py --dataset sift1m_100k --method all --run_sl --run_cp
"""
import argparse, json, os, subprocess, sys, time, tempfile
from itertools import product
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent

# ===========================================================================
# Recall computation
# ===========================================================================
def compute_recall(pred_ids, gt_ids, k):
    valid = set(int(i) for i in gt_ids[:k] if i >= 0)
    if not valid:
        return 1.0
    return len(set(pred_ids[:k]) & valid) / len(valid)


def load_gt(dataset, filter_expr=None):
    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / dataset
    if filter_expr:
        safe = filter_expr.replace(" ", "_")
        return np.load(gt_dir / "complex_predicate" / f"gt_{safe}.npy")
    return np.load(gt_dir / "ground_truth.npy")


# ===========================================================================
# Filter classification and representative selection
# ===========================================================================
def classify_filter(formula: str) -> str:
    t = formula.split()
    if not t:
        return "UNKNOWN"
    if t[0] == "NOT":
        return "NOT"
    if t[0] == "AND":
        if "NOT" in t:
            return "AND_NOT"
        return "AND"
    if t[0] == "OR":
        return "OR"
    return "UNKNOWN"


def select_representative_filters(filters_json_path, max_total=6):
    with open(filters_json_path) as f:
        info = json.load(f)
    filters = info["filters"]
    selectivities = info.get("selectivities", {})

    seen = set()
    deduped = []
    for formula in filters:
        if formula not in seen:
            seen.add(formula)
            deduped.append(formula)

    groups = {"AND": [], "AND_NOT": [], "OR": [], "NOT": []}
    for formula in deduped:
        cat = classify_filter(formula)
        if cat in groups:
            sel = selectivities.get(formula, 0.5)
            groups[cat].append((formula, sel))

    for cat in groups:
        groups[cat].sort(key=lambda x: x[1])

    selected = []
    if len(groups["AND"]) >= 2:
        selected.extend([groups["AND"][0][0], groups["AND"][-1][0]])
    elif groups["AND"]:
        selected.append(groups["AND"][0][0])
    if len(groups["AND_NOT"]) >= 2:
        selected.extend([groups["AND_NOT"][0][0], groups["AND_NOT"][-1][0]])
    elif groups["AND_NOT"]:
        selected.append(groups["AND_NOT"][0][0])
    if groups["OR"]:
        mid = len(groups["OR"]) // 2
        selected.append(groups["OR"][mid][0])
    if groups["NOT"]:
        mid = len(groups["NOT"]) // 2
        selected.append(groups["NOT"][mid][0])

    return selected[:max_total]


def select_3_buckets(query_info_path):
    with open(query_info_path) as f:
        query_info = json.load(f)
    buckets = sorted(set(q["bucket"] for q in query_info
                         if "overflow" not in q["bucket"].lower()))
    N = len(buckets)
    if N <= 3:
        return buckets
    idx = sorted(set([0, N // 2, N - 1]))
    return [buckets[i] for i in idx]


def polish_to_rpn(formula: str) -> str:
    """Convert Polish (prefix) notation to Reverse Polish (postfix) for Curator."""
    tokens = formula.split()
    result, _ = _polish_to_rpn_rec(tokens, 0)
    return " ".join(result)


def _polish_to_rpn_rec(tokens, i):
    if i >= len(tokens):
        return [], i
    token = tokens[i]
    if token in ("AND", "OR"):
        left, i = _polish_to_rpn_rec(tokens, i + 1)
        right, i = _polish_to_rpn_rec(tokens, i)
        return left + right + [token], i
    elif token == "NOT":
        operand, i = _polish_to_rpn_rec(tokens, i + 1)
        return operand + [token], i
    else:
        return [token], i + 1


# ===========================================================================
# C++ CLI helpers
# ===========================================================================
BINARIES = {
    "Curator": PROJ_ROOT / "Curator/build/curator",
    "DiskIVF": PROJ_ROOT / "DiskIVF-PostFiltering/build/diskivf",
    "SPANN": PROJ_ROOT / "SPANN-PostFiltering/build/spann_pf",
    "Pre-Filtering": PROJ_ROOT / "Pre-Filtering/build/prefiltering",
}


def run_cpp_bench(method, dataset, extra_args, output_path,
                  timeout=600, query_vecs_name="query_vecs.npy",
                  query_labels_opt=True, filters_file_path=None):
    """Run C++ bench command.

    query_labels_opt: if True (default), pass --query_labels for SL mode.
                      if False, omit (CP mode ignores labels).
    """
    binary = BINARIES[method]
    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / dataset

    cmd = [str(binary), "bench",
           "--train_vecs", str(gt_dir / "train_vecs.npy"),
           "--train_access", str(gt_dir / "train_access.npy"),
           "--queries", str(gt_dir / query_vecs_name),
           "--k", "10",
           "--output", str(output_path)]

    if query_labels_opt:
        cmd.extend(["--query_labels", str(gt_dir / "query_labels.npy")])

    if filters_file_path:
        cmd.extend(["--filters_file", str(filters_file_path)])

    cmd.extend(extra_args)

    print(f"    CMD: {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(PROJ_ROOT), timeout=timeout)
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"    ERROR (rc={result.returncode}): {result.stderr[-300:]}")
            return False, elapsed, None
        if output_path.exists():
            with open(output_path) as f:
                return True, elapsed, json.load(f)
        return False, elapsed, None
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"    TIMEOUT after {elapsed:.0f}s")
        return False, elapsed, None


# ===========================================================================
# C++ result parsing helpers
# ===========================================================================
def _parse_sl_from_json(res, dataset, k, elapsed, params,
                        query_info, selected_buckets,
                        memory_bytes=0, rss_peak_query_mb=0):
    """Parse C++ JSON 'queries' array → SL stats (qps, recall, per_bucket)."""
    queries = res.get("queries", [])
    n_queries = len(queries)
    build_time_s = res.get("build_time_s", 0)

    if memory_bytes and memory_bytes < 1024 * 1024:
        memory_mb = float(memory_bytes) / (1024.0 * 1024.0)
    else:
        memory_mb = float(memory_bytes) / (1024.0 * 1024.0) if memory_bytes else 0

    if n_queries == 0:
        return {
            "params": params,
            "latency_ms": 0, "qps": 0, "avg_recall": 0, "min_recall": 0,
            "build_time_s": build_time_s, "memory_bytes": memory_bytes,
            "memory_mb": float(memory_mb),
            "rss_peak_query_mb": float(rss_peak_query_mb) if rss_peak_query_mb else 0,
            "per_bucket": {},
        }

    search_time_s = max(elapsed - build_time_s, 0.001)
    sl_gt = load_gt(dataset)
    latencies, recalls_list = [], []
    bucket_lats, bucket_recs = {}, {}

    for q, qr in enumerate(queries):
        r = compute_recall(qr["labels"], sl_gt[q], k)
        recalls_list.append(r)
        st = qr.get("search_time_us", 0)
        lat_ms = st / 1000.0 if st > 0 else search_time_s / n_queries * 1000
        latencies.append(lat_ms)

        if q < len(query_info):
            b = query_info[q]["bucket"]
            if "overflow" not in b.lower():
                bucket_lats.setdefault(b, []).append(lat_ms)
                bucket_recs.setdefault(b, []).append(r)

    avg_lat = float(np.mean(latencies)) if latencies else 0
    qps = float(1000.0 / avg_lat) if avg_lat > 0 else 0

    per_bucket = {}
    for b in sorted(bucket_lats):
        bl = bucket_lats[b]
        br = bucket_recs[b]
        per_bucket[b] = {
            "count": len(bl),
            "avg_latency_ms": float(np.mean(bl)),
            "avg_recall": float(np.mean(br)),
            "qps": float(1000.0 / np.mean(bl)) if np.mean(bl) > 0 else 0,
        }

    return {
        "params": params,
        "latency_ms": float(avg_lat),
        "qps": float(qps),
        "avg_recall": float(np.mean(recalls_list)) if recalls_list else 0,
        "min_recall": float(np.min(recalls_list)) if recalls_list else 0,
        "build_time_s": build_time_s,
        "memory_bytes": memory_bytes,
        "memory_mb": float(memory_mb),
        "rss_peak_query_mb": float(rss_peak_query_mb) if rss_peak_query_mb else 0,
        "per_bucket": per_bucket,
    }


def _parse_cp_from_json(res, cp_gt_map, k, rpn_to_polish=None):
    """Parse C++ JSON 'filters_results' → CP stats."""
    filters_results = res.get("filters_results", [])
    if not filters_results:
        return None

    cp_lats, cp_recs = [], []
    cp_per_filter = {}

    for fr in filters_results:
        formula = fr["filter"]
        gt_f = cp_gt_map.get(formula)
        if gt_f is None and rpn_to_polish:
            gt_f = cp_gt_map.get(rpn_to_polish.get(formula, formula))
        if gt_f is None:
            continue
        f_lats, f_recs = [], []
        for q_idx, qr in enumerate(fr.get("queries", [])):
            st = qr.get("search_time_us", 0)
            lat_ms = st / 1000.0 if st > 0 else 0
            f_lats.append(lat_ms)
            f_recs.append(compute_recall(qr["labels"], gt_f[q_idx], k))
        cp_lats.extend(f_lats)
        cp_recs.extend(f_recs)
        cp_per_filter[formula] = {
            "avg_latency_ms": float(np.mean(f_lats)) if f_lats else 0,
            "avg_recall": float(np.mean(f_recs)) if f_recs else 0,
            "qps": float(1000.0 / np.mean(f_lats)) if f_lats and np.mean(f_lats) > 0 else 0,
        }

    return {
        "n_filters": len(filters_results),
        "n_queries": len(filters_results[0]["queries"]) if filters_results else 0,
        "avg_latency_ms": float(np.mean(cp_lats)) if cp_lats else 0,
        "avg_recall": float(np.mean(cp_recs)) if cp_recs else 0,
        "qps": float(1000.0 / np.mean(cp_lats)) if cp_lats and np.mean(cp_lats) > 0 else 0,
        "per_filter": cp_per_filter,
    }


def _write_cp_filters_file(cp_filters, rpn=False):
    """Write CP filters to temp file. Returns file path. Caller must os.unlink."""
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                      delete=False, dir='/tmp')
    for formula in cp_filters:
        txt = polish_to_rpn(formula) if rpn else formula
        tmp.write(txt + "\n")
    tmp.close()
    return tmp.name


# ===========================================================================
# Curator C++ CLI sweep
# ===========================================================================
def run_curator_sweep(dataset, sweep_cfg, output_dir, run_sl=True, run_cp=False,
                      cp_selected_filters=None, cp_query_vecs_path=None,
                      cp_gt_map=None, query_info=None):
    dataset_key = dataset
    pq_M_list = sweep_cfg["build_params"]["pq_M"].get(dataset_key, [32])
    search_ef_list = sweep_cfg["search_params"]["search_ef"]
    fixed = sweep_cfg.get("fixed", {})

    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / dataset
    k = 10

    do_cp = run_cp and cp_selected_filters and cp_query_vecs_path
    cp_filters = list(cp_selected_filters) if do_cp else []
    rpn_to_polish = {}
    if do_cp:
        for formula in cp_filters:
            rpn_to_polish[polish_to_rpn(formula)] = formula

    if query_info is None:
        with open(gt_dir / "query_info.json") as f:
            query_info = json.load(f)
    selected_buckets = select_3_buckets(gt_dir / "query_info.json")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combos = list(product(pq_M_list, search_ef_list))
    sweep_results = []

    for i, (pq_m, search_ef) in enumerate(combos):
        tag = f"pq{pq_m}_ef{search_ef}"
        out_path = output_dir / f"curator_{dataset}_{tag}.json"
        print(f"  [{i+1}/{len(combos)}] pq_M={pq_m} search_ef={search_ef}...", end=" ", flush=True)

        base_cfg = dict(fixed)
        base_cfg["pq_M"] = pq_m
        base_cfg["search_ef"] = search_ef
        base_cfg["flash_path"] = str(output_dir / f"disk_data_{dataset}_{tag}")

        tmp_cfg = tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                              delete=False, dir='/tmp')
        json.dump(base_cfg, tmp_cfg)
        tmp_cfg.close()
        extra_args = ["--config", tmp_cfg.name]

        entry = {"params": {"pq_M": pq_m, "search_ef": search_ef}}

        # ── SL call ──
        if run_sl:
            ok, elapsed, res = run_cpp_bench("Curator", dataset, extra_args,
                                              output_dir / f"_sl_{tag}.json")
            if ok and res:
                sl = _parse_sl_from_json(res, dataset, k, elapsed,
                                         {"pq_M": pq_m, "search_ef": search_ef},
                                         query_info, selected_buckets,
                                         memory_bytes=res.get("memory_bytes", 0),
                                         rss_peak_query_mb=res.get("rss_peak_query_mb", 0))
                entry.update(sl)
            else:
                print("SL_FAILED", end=" ", flush=True)

        # ── CP call ──
        if do_cp:
            filters_file = _write_cp_filters_file(cp_filters, rpn=True)
            ok, elapsed, res = run_cpp_bench("Curator", dataset, extra_args,
                                              output_dir / f"_cp_{tag}.json",
                                              query_labels_opt=False,
                                              filters_file_path=filters_file,
                                              query_vecs_name="complex_predicate/query_vecs.npy")
            os.unlink(filters_file)
            if ok and res:
                entry["complex_predicate"] = _parse_cp_from_json(res, cp_gt_map, k, rpn_to_polish)
                # Also take memory/build_time from CP call if SL didn't run
                if not run_sl:
                    entry["build_time_s"] = res.get("build_time_s", 0)
                    entry["memory_bytes"] = res.get("memory_bytes", 0)
                    entry["memory_mb"] = float(res.get("memory_bytes", 0)) / (1024.0 * 1024.0) if res.get("memory_bytes") else 0
                    entry["rss_peak_query_mb"] = float(res.get("rss_peak_query_mb", 0))
            else:
                print("CP_FAILED", end=" ", flush=True)

        os.unlink(tmp_cfg.name)
        sweep_results.append(entry)

        if entry.get("complex_predicate"):
            cp = entry["complex_predicate"]
            print(f"SL QPS={entry.get('qps',0):.0f} R@10={entry.get('avg_recall',0):.4f}  "
                  f"CP QPS={cp['qps']:.0f} R@10={cp['avg_recall']:.4f}")
        else:
            print(f"QPS={entry.get('qps',0):.0f} R@10={entry.get('avg_recall',0):.4f}")

    output_path = output_dir / f"sweep_{dataset}.json"
    output_json = {
        "dataset": dataset, "method": "Curator",
        "n_combinations": len(sweep_results),
        "sweep_results": sweep_results,
    }
    if sweep_results:
        output_json["index_memory_mb"] = sweep_results[0].get("memory_mb", 0)
        output_json["rss_peak_query_mb"] = sweep_results[0].get("rss_peak_query_mb", 0)
        output_json["build_time_s"] = sweep_results[0].get("build_time_s", 0)

    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Results saved to {output_path}")
    return output_json


# ===========================================================================
# DiskIVF C++ CLI sweep
# ===========================================================================
def run_diskivf_sweep(dataset, sweep_cfg, output_dir, run_sl=True, run_cp=False,
                      cp_selected_filters=None, cp_query_vecs_path=None,
                      cp_gt_map=None, query_info=None):
    nprobe_list = sweep_cfg["nprobe"]
    nlist = sweep_cfg.get("nlist", 32)
    disk_dir_base = str(PROJ_ROOT / "4_Results/DiskIVF-PostFiltering/disk_data" / dataset)

    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / dataset
    k = 10

    do_cp = run_cp and cp_selected_filters and cp_query_vecs_path
    cp_filters = list(cp_selected_filters) if do_cp else []

    if query_info is None:
        with open(gt_dir / "query_info.json") as f:
            query_info = json.load(f)
    selected_buckets = select_3_buckets(gt_dir / "query_info.json")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = []

    for i, nprobe in enumerate(nprobe_list):
        disk_dir = f"{disk_dir_base}_nprobe{nprobe}"
        out_path = output_dir / f"diskivf_{dataset}_nprobe{nprobe}.json"
        print(f"  [{i+1}/{len(nprobe_list)}] nprobe={nprobe}...", end=" ", flush=True)

        extra_args = ["--nlist", str(nlist), "--nprobe", str(nprobe),
                      "--disk_dir", disk_dir]

        entry = {"params": {"nprobe": nprobe, "nlist": nlist}}

        # ── SL call ──
        if run_sl:
            ok, elapsed, res = run_cpp_bench("DiskIVF", dataset, extra_args,
                                              output_dir / f"_sl_diskivf_nprobe{nprobe}.json")
            if ok and res:
                sl = _parse_sl_from_json(res, dataset, k, elapsed,
                                         {"nprobe": nprobe, "nlist": nlist},
                                         query_info, selected_buckets,
                                         memory_bytes=res.get("memory_bytes", 0),
                                         rss_peak_query_mb=res.get("rss_peak_query_mb", 0))
                entry.update(sl)
            else:
                print("SL_FAILED", end=" ", flush=True)

        # ── CP call ──
        if do_cp:
            filters_file = _write_cp_filters_file(cp_filters, rpn=False)
            ok, elapsed, res = run_cpp_bench("DiskIVF", dataset, extra_args,
                                              output_dir / f"_cp_diskivf_nprobe{nprobe}.json",
                                              query_labels_opt=False,
                                              filters_file_path=filters_file,
                                              query_vecs_name="complex_predicate/query_vecs.npy")
            os.unlink(filters_file)
            if ok and res:
                entry["complex_predicate"] = _parse_cp_from_json(res, cp_gt_map, k)
                if not run_sl:
                    entry["build_time_s"] = res.get("build_time_s", 0)
                    entry["memory_bytes"] = res.get("memory_bytes", 0)
                    entry["memory_mb"] = float(res.get("memory_bytes", 0)) / (1024.0 * 1024.0) if res.get("memory_bytes") else 0
                    entry["rss_peak_query_mb"] = float(res.get("rss_peak_query_mb", 0))
            else:
                print("CP_FAILED", end=" ", flush=True)

        sweep_results.append(entry)

        if entry.get("complex_predicate"):
            cp = entry["complex_predicate"]
            print(f"SL QPS={entry.get('qps',0):.0f} R@10={entry.get('avg_recall',0):.4f}  "
                  f"CP QPS={cp['qps']:.0f} R@10={cp['avg_recall']:.4f}")
        else:
            print(f"QPS={entry.get('qps',0):.0f} R@10={entry.get('avg_recall',0):.4f}")

    output_path = output_dir / f"sweep_{dataset}.json"
    output_json = {
        "dataset": dataset, "method": "DiskIVF",
        "n_combinations": len(sweep_results),
        "sweep_results": sweep_results,
    }
    if sweep_results:
        output_json["index_memory_mb"] = sweep_results[0].get("memory_mb", 0)
        output_json["rss_peak_query_mb"] = sweep_results[0].get("rss_peak_query_mb", 0)
        output_json["build_time_s"] = sweep_results[0].get("build_time_s", 0)

    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Results saved to {output_path}")
    return output_json


# ===========================================================================
# SPANN C++ CLI sweep
# ===========================================================================
def run_spann_sweep(dataset, sweep_cfg, output_dir, run_sl=True, run_cp=False,
                    cp_selected_filters=None, cp_query_vecs_path=None,
                    cp_gt_map=None, query_info=None):
    max_check_list = sweep_cfg["max_check"]
    overfetch_list = sweep_cfg["overfetch_factor"]
    index_base = str(PROJ_ROOT / "4_Results/SPANN-PostFiltering/spann_index" / dataset)

    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / dataset
    k = 10

    do_cp = run_cp and cp_selected_filters and cp_query_vecs_path
    cp_filters = list(cp_selected_filters) if do_cp else []

    if query_info is None:
        with open(gt_dir / "query_info.json") as f:
            query_info = json.load(f)
    selected_buckets = select_3_buckets(gt_dir / "query_info.json")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combos = list(product(max_check_list, overfetch_list))
    sweep_results = []

    # Write SPANN fixed config (overfetch_adaptive=false so sweep params take effect)
    spann_fixed = sweep_cfg.get("fixed", {})
    spann_cfg_path = tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                                  delete=False, dir='/tmp')
    json.dump(spann_fixed, spann_cfg_path)
    spann_cfg_path.close()

    for i, (mc, of) in enumerate(combos):
        idx_dir = f"{index_base}_mc{mc}_of{of}"
        out_path = output_dir / f"spann_{dataset}_mc{mc}_of{of}.json"
        print(f"  [{i+1}/{len(combos)}] max_check={mc} overfetch={of}...", end=" ", flush=True)

        extra_args = ["--index_dir", idx_dir, "--config", spann_cfg_path.name,
                      "--max_check", str(mc), "--overfetch_factor", str(of)]
        timeout = 900

        entry = {"params": {"max_check": mc, "overfetch_factor": of}}

        # ── SL call ──
        if run_sl:
            ok, elapsed, res = run_cpp_bench("SPANN", dataset, extra_args,
                                              output_dir / f"_sl_spann_mc{mc}_of{of}.json",
                                              timeout=timeout)
            if ok and res:
                sl = _parse_sl_from_json(res, dataset, k, elapsed,
                                         {"max_check": mc, "overfetch_factor": of},
                                         query_info, selected_buckets,
                                         memory_bytes=res.get("memory_bytes", 0),
                                         rss_peak_query_mb=res.get("rss_peak_query_mb", 0))
                entry.update(sl)
            else:
                print("SL_FAILED", end=" ", flush=True)

        # ── CP call ──
        if do_cp:
            filters_file = _write_cp_filters_file(cp_filters, rpn=False)
            ok, elapsed, res = run_cpp_bench("SPANN", dataset, extra_args,
                                              output_dir / f"_cp_spann_mc{mc}_of{of}.json",
                                              query_labels_opt=False,
                                              filters_file_path=filters_file,
                                              query_vecs_name="complex_predicate/query_vecs.npy",
                                              timeout=timeout)
            os.unlink(filters_file)
            if ok and res:
                entry["complex_predicate"] = _parse_cp_from_json(res, cp_gt_map, k)
                if not run_sl:
                    entry["build_time_s"] = res.get("build_time_s", 0)
                    entry["memory_bytes"] = res.get("memory_bytes", 0)
                    entry["memory_mb"] = float(res.get("memory_bytes", 0)) / (1024.0 * 1024.0) if res.get("memory_bytes") else 0
                    entry["rss_peak_query_mb"] = float(res.get("rss_peak_query_mb", 0))
            else:
                print("CP_FAILED", end=" ", flush=True)

        sweep_results.append(entry)

        if entry.get("complex_predicate"):
            cp = entry["complex_predicate"]
            print(f"SL QPS={entry.get('qps',0):.0f} R@10={entry.get('avg_recall',0):.4f}  "
                  f"CP QPS={cp['qps']:.0f} R@10={cp['avg_recall']:.4f}")
        else:
            print(f"QPS={entry.get('qps',0):.0f} R@10={entry.get('avg_recall',0):.4f}")

    output_path = output_dir / f"sweep_{dataset}.json"
    output_json = {
        "dataset": dataset, "method": "SPANN",
        "n_combinations": len(sweep_results),
        "sweep_results": sweep_results,
    }
    if sweep_results:
        output_json["index_memory_mb"] = sweep_results[0].get("memory_mb", 0)
        output_json["rss_peak_query_mb"] = sweep_results[0].get("rss_peak_query_mb", 0)
        output_json["build_time_s"] = sweep_results[0].get("build_time_s", 0)

    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Results saved to {output_path}")
    return output_json


# ===========================================================================
# Pre-Filtering C++ CLI (single run)
# ===========================================================================
def run_prefiltering_sweep(dataset, sweep_cfg, output_dir, run_sl=True, run_cp=False,
                           cp_selected_filters=None, cp_query_vecs_path=None,
                           cp_gt_map=None, query_info=None):
    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / dataset
    k = 10

    do_cp = run_cp and cp_selected_filters and cp_query_vecs_path
    cp_filters = list(cp_selected_filters) if do_cp else []

    if query_info is None:
        with open(gt_dir / "query_info.json") as f:
            query_info = json.load(f)
    selected_buckets = select_3_buckets(gt_dir / "query_info.json")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"prefiltering_{dataset}.json"

    print(f"  Running Pre-Filtering...", end=" ", flush=True)

    entry = {"params": {}}

    # ── SL call ──
    if run_sl:
        ok, elapsed, res = run_cpp_bench("Pre-Filtering", dataset, [],
                                          output_dir / f"_sl_prefiltering_{dataset}.json")
        if ok and res:
            sl = _parse_sl_from_json(res, dataset, k, elapsed, {},
                                     query_info, selected_buckets,
                                     memory_bytes=res.get("memory_bytes", 0),
                                     rss_peak_query_mb=res.get("rss_peak_query_mb", 0))
            entry.update(sl)
        else:
            print("SL_FAILED", end=" ", flush=True)

    # ── CP call ──
    if do_cp:
        filters_file = _write_cp_filters_file(cp_filters, rpn=False)
        ok, elapsed, res = run_cpp_bench("Pre-Filtering", dataset, [],
                                          output_dir / f"_cp_prefiltering_{dataset}.json",
                                          query_labels_opt=False,
                                          filters_file_path=filters_file,
                                          query_vecs_name="complex_predicate/query_vecs.npy")
        os.unlink(filters_file)
        if ok and res:
            entry["complex_predicate"] = _parse_cp_from_json(res, cp_gt_map, k)
            if not run_sl:
                entry["build_time_s"] = res.get("build_time_s", 0)
                entry["memory_bytes"] = res.get("memory_bytes", 0)
                entry["memory_mb"] = float(res.get("memory_bytes", 0)) / (1024.0 * 1024.0) if res.get("memory_bytes") else 0
                entry["rss_peak_query_mb"] = float(res.get("rss_peak_query_mb", 0))
        else:
            print("CP_FAILED")

    sweep_results = [entry]

    if entry.get("complex_predicate"):
        cp = entry["complex_predicate"]
        print(f"SL QPS={entry.get('qps',0):.0f} R@10={entry.get('avg_recall',0):.4f}  "
              f"CP QPS={cp['qps']:.0f} R@10={cp['avg_recall']:.4f}")
    else:
        print(f"QPS={entry.get('qps',0):.0f} R@10={entry.get('avg_recall',0):.4f}")

    output_json = {
        "dataset": dataset, "method": "Pre-Filtering",
        "n_combinations": 1,
        "index_memory_mb": entry.get("memory_mb", 0),
        "rss_peak_query_mb": entry.get("rss_peak_query_mb", 0),
        "build_time_s": entry.get("build_time_s", 0),
        "sweep_results": sweep_results,
    }
    output_path = output_dir / f"sweep_{dataset}.json"
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Results saved to {output_path}")
    return output_json


# ===========================================================================
# Main dispatcher
# ===========================================================================
METHOD_RUNNERS = {
    "Curator": run_curator_sweep,
    "DiskIVF": run_diskivf_sweep,
    "SPANN": run_spann_sweep,
    "Pre-Filtering": run_prefiltering_sweep,
}

SWEEP_CONFIGS = {
    "Curator": PROJ_ROOT / "3_Config/Curator/sweep_100k.json",
    "DiskIVF": PROJ_ROOT / "3_Config/DiskIVF-PostFiltering/sweep_100k.json",
    "SPANN": PROJ_ROOT / "3_Config/SPANN-PostFiltering/sweep_100k.json",
    "Pre-Filtering": PROJ_ROOT / "3_Config/Pre-Filtering/sweep_100k.json",
}


def main():
    parser = argparse.ArgumentParser(description="100K benchmark sweep runner")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "Curator", "DiskIVF", "SPANN", "Pre-Filtering"])
    parser.add_argument("--sweep", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="4_Results")
    parser.add_argument("--run_sl", action="store_true", default=True,
                        help="Run single-label sweep")
    parser.add_argument("--run_cp", action="store_true", default=False,
                        help="Run complex-predicate sweep")
    args = parser.parse_args()

    methods = list(METHOD_RUNNERS.keys()) if args.method == "all" else [args.method]
    dataset = args.dataset

    print(f"\n{'='*60}")
    print(f"100K Sweep: {dataset}")
    print(f"  Methods: {methods}")
    print(f"  SL: {args.run_sl}, CP: {args.run_cp}")
    print(f"{'='*60}\n")

    # ── Pre-load CP shared data ──
    cp_selected = None
    cp_query_vecs_path = None
    cp_gt_map = None
    query_info = None

    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / dataset
    cp_dir = gt_dir / "complex_predicate"

    with open(gt_dir / "query_info.json") as f:
        query_info = json.load(f)

    if args.run_cp and cp_dir.exists():
        cp_selected = select_representative_filters(cp_dir / "filters.json")
        print(f"Selected {len(cp_selected)} representative CP filters:")
        for formula in cp_selected:
            print(f"    {formula} ({classify_filter(formula)})")
        print()

        cp_query_vecs_path = str(cp_dir / "query_vecs.npy")
        cp_gt_map = {}
        for formula in cp_selected:
            safe = formula.replace(" ", "_")
            cp_gt_map[formula] = np.load(cp_dir / f"gt_{safe}.npy")
        print(f"  CP queries + {len(cp_gt_map)} GTs loaded\n")

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  {method}")
        print(f"{'='*60}")

        sweep_path = args.sweep or str(SWEEP_CONFIGS[method])
        if not os.path.exists(sweep_path):
            print(f"  ERROR: sweep config not found: {sweep_path}")
            continue
        with open(sweep_path) as f:
            sweep_cfg = json.load(f)

        output_dir = Path(args.output_dir) / method
        runner = METHOD_RUNNERS[method]
        try:
            runner(dataset, sweep_cfg, str(output_dir),
                   run_sl=args.run_sl, run_cp=args.run_cp,
                   cp_selected_filters=cp_selected,
                   cp_query_vecs_path=cp_query_vecs_path,
                   cp_gt_map=cp_gt_map,
                   query_info=query_info)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All sweeps complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
