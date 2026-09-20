#!/usr/bin/env python3
"""Matched-Recall QPS-vs-selectivity figures.

For each dataset and method, build the measured QPS-Recall frontier per
selectivity bucket, interpolate QPS at a fixed Recall target, and plot QPS
against the bucket's median selectivity.  This is the correct horizontal
comparison for post-filter methods with selectivity-dependent candidate needs.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "5_Plot"))
import paper_style as PS  # noqa: E402

RESULTS_DIR = PROJ_ROOT / "4_Results"
OUTPUT_DIR = RESULTS_DIR / "fig_full_qps_legacy"
GT_DIR = PROJ_ROOT / "1_Data/ground_truth"
DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]
BUCKETS = ["1p", "25p", "50p", "75p", "99p"]
METHODS = ["curator", "diskivf", "spann", "prefilter"]
INDEX_DIRS = {"curator": "Curator", "diskivf": "DiskIVF",
              "spann": "SPANN", "prefilter": "Pre-Filtering"}
TARGETS = [0.90, 0.95]


def load_sweep(method, dataset):
    if method == "prefilter":
        inmem = RESULTS_DIR / "Pre-Filtering" / f"sweep_{dataset}_inmem.json"
        if inmem.exists():
            with open(inmem) as f:
                return json.load(f)
    path = RESULTS_DIR / INDEX_DIRS[method] / f"sweep_{dataset}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def bucket_points(method, dataset, bucket):
    data = load_sweep(method, dataset)
    if not data:
        return []
    out = []
    for result in data.get("sweep_results", []):
        pb = result.get("per_bucket", {}).get(bucket)
        if not pb:
            continue
        qps = float(pb.get("qps", 0.0) or 0.0)
        recall = float(pb.get("avg_recall", 0.0) or 0.0)
        if qps <= 0 or recall <= 0:
            continue
        # PreFilter is an exact baseline.  Tiny sub-1.0 values are distance
        # ties between duplicate vectors or argpartition tie-breaking.
        if method == "prefilter":
            recall = 1.0
        out.append((recall, qps))
    return out


def pareto(points):
    if not points:
        return []
    grouped = {}
    for recall, qps in points:
        key = round(recall, 9)
        grouped[key] = max(grouped.get(key, 0.0), qps)
    arr = sorted((r, q) for r, q in grouped.items())
    frontier = []
    best_q = -1.0
    for recall, qps in reversed(arr):
        if qps > best_q + 1e-12:
            frontier.append((recall, qps))
            best_q = qps
    frontier.reverse()
    return frontier


def qps_at_recall(frontier, target):
    if not frontier:
        return None
    if len(frontier) == 1:
        recall, qps = frontier[0]
        return qps if target <= recall + 1e-12 else None
    lo = frontier[0][0]
    hi = frontier[-1][0]
    if target <= lo + 1e-12:
        return max(q for _, q in frontier)
    if target > hi + 1e-12:
        return None
    recalls = [p[0] for p in frontier]
    log_q = [math.log(p[1]) for p in frontier]
    idx = int(np.searchsorted(recalls, target, side="left"))
    if idx <= 0:
        return frontier[0][1]
    if idx >= len(frontier):
        return frontier[-1][1]
    x0, x1 = recalls[idx - 1], recalls[idx]
    y0, y1 = log_q[idx - 1], log_q[idx]
    return math.exp(y0 + (y1 - y0) * (target - x0) / (x1 - x0))


def median_selectivity(dataset, bucket):
    path = GT_DIR / dataset / "query_info.json"
    with open(path) as f:
        info = json.load(f)
    vals = [q["selectivity"] for q in info
            if q.get("bucket") == bucket and q.get("selectivity") is not None]
    return float(np.median(vals)) if vals else None


def collect(dataset, target):
    rows = {}
    for method in METHODS:
        xs, ys = [], []
        for bucket in BUCKETS:
            sel = median_selectivity(dataset, bucket)
            if sel is None:
                continue
            qps = qps_at_recall(pareto(bucket_points(method, dataset, bucket)),
                                target)
            if qps is not None and qps > 0:
                xs.append(sel)
                ys.append(qps)
        rows[method] = (xs, ys)
    return rows


def plot_target(target):
    figure, axes = plt.subplots(1, len(DATASETS), figsize=(7.0 * len(DATASETS), 7.0))
    if len(DATASETS) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, DATASETS):
        rows = collect(dataset, target)
        for method in PS.PLOT_ORDER:
            xs, ys = rows.get(method, ([], []))
            if not xs:
                continue
            style = PS.METHOD_STYLE[method]
            axis.plot(xs, ys, color=style["color"], marker=style["marker"],
                      linestyle=style["linestyle"], lw=PS.LINE_WIDTH,
                      ms=PS.MARKER_SIZE, mec=style["color"],
                      mew=PS.MARKER_EDGE_WIDTH, mfc="none", label=method)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, linewidth=3, alpha=0.30)
        axis.set_xlabel("Median selectivity", fontsize=22)
        axis.set_ylabel(f"QPS @ Recall >= {target:.2f}", fontsize=22)
        axis.tick_params(axis="both", labelsize=16)
        title = dataset
        if dataset == "gist1m":
            title += " (kmeans labels: confounded)"
        axis.set_title(title, fontsize=24, fontweight="bold")
    figure.tight_layout()
    out = OUTPUT_DIR / f"fig_full_qps_at_recall_{int(target * 100):03d}.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(out.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print("Saved", out)


def main():
    PS.apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        plot_target(target)


if __name__ == "__main__":
    main()
