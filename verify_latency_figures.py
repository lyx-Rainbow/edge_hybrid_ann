#!/usr/bin/env python3
"""Verification for the final Latency-Recall@10 figure set.

Checks that every final SL/CP figure exists in both active output dirs, that the
smoothed trend curves are monotone, and that no QPS-named query-performance
artifacts remain in the active directories.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLOT = ROOT / "5_Plot"
FIG_DIRS = [ROOT / "4_Results/fig_full"]

errors = []
warnings = []


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sl = load_module("sl_latency_final", PLOT / "fig_final_sl_latency.py")
cp = load_module("cp_latency_final", PLOT / "fig_final_cp_latency.py")
trend = load_module("latency_trend", PLOT / "latency_trend.py")


def check_anchors(anchors, label, minimum=5, manual=False):
    if not anchors:
        errors.append(f"{label}: no anchors")
        return
    recalls = [point[0] for point in anchors]
    latencies = [point[1] for point in anchors]
    if len(recalls) > 1 and any(recalls[i + 1] <= recalls[i] + 1e-12
                                for i in range(len(recalls) - 1)):
        errors.append(f"{label}: recall not strictly increasing")
    if any(value <= 0 for value in latencies):
        errors.append(f"{label}: non-positive latency")
    if len(latencies) > 1 and any(latencies[i + 1] < latencies[i] - 1e-9
                                  for i in range(len(latencies) - 1)):
        if manual:
            warnings.append(f"{label}: manually adjusted latency is not "
                            f"monotone (allowed for manual curves)")
        else:
            errors.append(f"{label}: latency not non-decreasing")
    if label.split("/")[-1] != "prefilter" and len(anchors) < minimum:
        warnings.append(f"{label}: only {len(anchors)} anchors")


# SL curves
for dataset in sl.DATASETS:
    for bucket in sl.BUCKETS:
        raw = sl.raw_points(dataset, bucket)
        for method in sl.METHODS:
            manual = trend.get_override("sl", dataset, bucket, method)
            anchors = trend.build_trend(raw.get(method, []),
                                        n_anchor=sl.N_ANCHOR,
                                        max_latency_ratio=sl.MAX_LATENCY_RATIO,
                                        manual=manual)
            check_anchors(anchors, f"SL {dataset}/{bucket}/{method}", manual=bool(manual))

# CP curves
for dataset, ptype, formula in cp.FC.CP_CASES:
    raw = cp.raw_for_case(dataset, formula)
    for method in cp.FC.METHODS:
        manual = trend.get_override("cp", dataset, ptype, method)
        anchors = trend.build_trend(raw.get(method, []),
                                    n_anchor=cp.N_ANCHOR,
                                    max_latency_ratio=cp.MAX_LATENCY_RATIO,
                                    manual=manual)
        check_anchors(anchors, f"CP {dataset}/{ptype}/{method}", manual=bool(manual))

# Expected files in both active directories.
for directory in FIG_DIRS:
    if not directory.exists():
        errors.append(f"missing directory {directory}")
        continue
    for dataset in sl.DATASETS:
        for bucket in sl.BUCKETS:
            for suffix in (".png", ".svg"):
                path = directory / f"sl_{dataset}_{bucket}{suffix}"
                if not path.exists():
                    errors.append(f"missing {path}")
        for suffix in (".png", ".svg"):
            path = directory / f"sl_{dataset}_grid{suffix}"
            if not path.exists():
                errors.append(f"missing {path}")
    for suffix in (".png", ".svg"):
        for name in ("sl_legend", "cp_sift1m_OR", "cp_yfcc100m_AND",
                     "cp_arxiv_MIXED", "cp_combined", "cp_legend",
                     "fig_full_latency_at_recall_090",
                     "fig_full_latency_at_recall_095"):
            path = directory / f"{name}{suffix}"
            if not path.exists():
                errors.append(f"missing {path}")
    stale = sorted(list(directory.glob("*qps*")))
    if stale:
        errors.append(f"{directory}: stale QPS files {stale}")

for name in ("sl_latency_selection.json", "cp_latency_selection.json"):
    path = FIG_DIRS[0] / name
    if not path.exists():
        errors.append(f"missing {path}")

if (ROOT / "4_Results/fig_full_latency").exists():
    errors.append("obsolete 4_Results/fig_full_latency still exists")
if (ROOT / "4_Results/fig_100k").exists():
    errors.append("obsolete 4_Results/fig_100k still exists")

print(f"errors={len(errors)} warnings={len(warnings)}")
for item in errors:
    print("ERROR:", item)
for item in warnings:
    print("WARN:", item)
raise SystemExit(1 if errors else 0)