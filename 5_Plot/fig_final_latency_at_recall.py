#!/usr/bin/env python3
"""Matched-Recall latency-vs-selectivity figures with manual overrides.

The x-axis is median selectivity, the y-axis is latency at a fixed Recall@10
target.  Manual point selection and position adjustments work through
``5_Plot/manual_latency_overrides.json`` under the ``matched_recall`` section.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOT_DIR))

import fig_full_sl_by_percentile as SL  # noqa: E402
import latency_trend as LT  # noqa: E402
import paper_style as PS  # noqa: E402

OUTPUT_DIRS = [
    PROJ_ROOT / "4_Results/fig_full",
]
GT_DIR = PROJ_ROOT / "1_Data/ground_truth"
DATASETS = SL.DATASETS
BUCKETS = SL.BUCKETS
METHODS = SL.METHODS
TARGETS = [0.90, 0.95]


def median_selectivity(dataset: str, bucket: str):
    """Median selectivity of a bucket (same x definition as the old figure)."""
    try:
        with open(GT_DIR / dataset / "query_info.json", "r",
                  encoding="utf-8") as stream:
            info = json.load(stream)
    except Exception:
        return None
    values = [float(item["selectivity"]) for item in info
              if item.get("bucket") == bucket
              and item.get("selectivity") is not None]
    return float(np.median(values)) if values else None


def _curve_value(anchors, target: float, interpolation: str):
    """Read latency at target Recall from the same state as the SL figure."""
    if not anchors:
        return None
    if len(anchors) == 1:
        # Exact single-point baselines (PreFilter): Recall=1.0 satisfies any
        # target <= 1.0 at that measured latency.
        return (float(anchors[0][1])
                if target <= float(anchors[0][0]) + 1e-9 else None)
    recalls = [float(item[0]) for item in anchors]
    if target < recalls[0] - 1e-9 or target > recalls[-1] + 1e-9:
        return None
    dense_x, dense_y = LT.smooth_line(anchors, samples=512,
                                       interpolation=interpolation)
    if target < dense_x[0] - 1e-9 or target > dense_x[-1] + 1e-9:
        return None
    return float(np.interp(target, dense_x, dense_y))


def method_points(dataset: str, target: float, method: str):
    """Derive matched-recall points from the current manual SL curves."""
    points = []
    for bucket in BUCKETS:
        selectivity = median_selectivity(dataset, bucket)
        if selectivity is None:
            continue
        raw = SL.bucket_points(method, dataset, bucket)
        manual = LT.get_override("sl", dataset, bucket, method)
        anchors = LT.build_trend(raw, n_anchor=9, manual=manual)
        interpolation = (manual or {}).get("interpolation", "pchip")
        latency = _curve_value(anchors, target, interpolation)
        if latency is None or latency <= 0:
            continue
        points.append({
            "id": len(points),
            "bucket": bucket,
            "median_selectivity": float(selectivity),
            "latency_ms": float(latency),
        })
    return points


def collect(dataset: str, target: float):
    rows = {}
    for method in METHODS:
        points = method_points(dataset, target, method)
        rows[method] = ([p["median_selectivity"] for p in points],
                        [p["latency_ms"] for p in points])
    return rows


def _parse_adjustment(item):
    if isinstance(item, dict):
        index = item.get("i", item.get("index"))
        dx = float(item.get("dx", item.get("dr", 0.0)) or 0.0)
        dy = float(item.get("dlat_ms", item.get("dy", 0.0)) or 0.0)
        scale = float(item.get("scale", 1.0) or 1.0)
        return index, dx, dy, scale
    if isinstance(item, (list, tuple)) and len(item) >= 1:
        return (item[0],
                float(item[1]) if len(item) > 1 else 0.0,
                float(item[2]) if len(item) > 2 else 0.0,
                float(item[3]) if len(item) > 3 else 1.0)
    return None, 0.0, 0.0, 1.0


def apply_manual(points, override):
    """Apply manual selection/adjustments to one matched-recall series."""
    if not points:
        return []
    override = override or {}
    if override.get("anchors"):
        chosen = []
        for item in override.get("anchors", []):
            chosen.append({
                "median_selectivity": float(item[0]),
                "latency_ms": float(item[1]),
            })
    else:
        selected_ids = (override.get("selected_ids")
                        or override.get("selected") or [])
        if selected_ids:
            by_id = {point["id"]: point for point in points}
            chosen = []
            for raw_id in selected_ids:
                try:
                    point = by_id.get(int(raw_id))
                except Exception:
                    point = None
                if point is not None:
                    chosen.append(dict(point))
        else:
            chosen = [dict(point) for point in points]

    chosen.sort(key=lambda point: point["median_selectivity"])
    for item in override.get("adjustments", []) or []:
        index, dx, dy, scale = _parse_adjustment(item)
        if index is None:
            continue
        try:
            index = int(index)
        except Exception:
            continue
        if not (0 <= index < len(chosen)):
            continue
        chosen[index]["median_selectivity"] += dx
        chosen[index]["latency_ms"] = max(
            1e-12, chosen[index]["latency_ms"] + dy)
        if scale != 1.0:
            chosen[index]["latency_ms"] = max(
                1e-12, chosen[index]["latency_ms"] * scale)

    chosen = [point for point in chosen
              if point["median_selectivity"] > 0
              and point["latency_ms"] > 0]
    chosen.sort(key=lambda point: point["median_selectivity"])
    return [(point["median_selectivity"], point["latency_ms"])
            for point in chosen]


def save_figure(figure, name: str):
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        png = output_dir / f"{name}.png"
        figure.savefig(png, dpi=300, bbox_inches="tight")
        figure.savefig(png.with_suffix(".svg"), dpi=300,
                       bbox_inches="tight")


def plot_target(target: float):
    figure, axes = plt.subplots(1, len(DATASETS),
                                figsize=(7.0 * len(DATASETS), 7.0))
    if len(DATASETS) == 1:
        axes = [axes]
    target_key = f"{target:.2f}"
    for axis, dataset in zip(axes, DATASETS):
        for method in PS.PLOT_ORDER:
            points = method_points(dataset, target, method)
            override = LT.get_override("matched_recall", dataset, target_key,
                                       method)
            xy = apply_manual(points, override)
            if not xy:
                continue
            xs = [point[0] for point in xy]
            ys = [point[1] for point in xy]
            PS.draw_method(axis, method, xs, ys, single_point=False,
                           linestyle="-" if method == "prefilter" else None)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, linewidth=3, alpha=0.30)
        axis.set_xlabel("Median selectivity", fontsize=22)
        axis.set_ylabel(
            f"Latency @ Recall@10 ≥ {target:.2f} (ms)", fontsize=22)
        axis.tick_params(axis="both", labelsize=16)
        axis.set_title(dataset, fontsize=24, fontweight="bold")
    figure.tight_layout()
    name = f"fig_full_latency_at_recall_{int(round(target * 100)):03d}"
    save_figure(figure, name)
    plt.close(figure)
    print("Saved", name)


def main():
    PS.apply_style()
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        plot_target(target)


if __name__ == "__main__":
    main()