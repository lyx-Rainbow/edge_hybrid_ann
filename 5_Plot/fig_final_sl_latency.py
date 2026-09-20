#!/usr/bin/env python3
"""Final full-scale SL figures: Latency-Recall@10.

Every query-performance figure in this final set uses latency (ms) on the
y-axis and Recall@10 on the x-axis.  Curves are monotone smoothed trends over
all measured points in each selectivity bucket (see ``latency_trend``), not
straight segments between a handful of selected vertices.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOT_DIR))

import fig_full_sl_by_percentile as SL  # noqa: E402
import latency_trend as LT  # noqa: E402
import paper_style as PS  # noqa: E402

OUTPUT_DIRS = [
    PROJ_ROOT / "4_Results/fig_full",
]
DATASETS = SL.DATASETS
BUCKETS = SL.BUCKETS
METHODS = SL.METHODS
N_ANCHOR = 9
GRID_ANCHOR = 7
MAX_LATENCY_RATIO = LT.MAX_LATENCY_RATIO_DEFAULT


def raw_points(dataset: str, bucket: str):
    return {method: SL.bucket_points(method, dataset, bucket)
            for method in METHODS}


def build_anchors(raw_by_method, n_anchor=N_ANCHOR,
                  dataset=None, bucket=None):
    result = {}
    for method in METHODS:
        manual = (LT.get_override("sl", dataset, bucket, method)
                  if dataset is not None and bucket is not None else {})
        result[method] = LT.build_trend(
            raw_by_method.get(method, []),
            n_anchor=n_anchor,
            max_latency_ratio=MAX_LATENCY_RATIO,
            manual=manual)
    return result


def save_figure(figure, name: str):
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        png = output_dir / f"{name}.png"
        figure.savefig(png, dpi=300, bbox_inches="tight")
        figure.savefig(png.with_suffix(".svg"), dpi=300,
                       bbox_inches="tight")


def plot_one(dataset: str, bucket: str):
    raw = raw_points(dataset, bucket)
    anchors = build_anchors(raw, N_ANCHOR, dataset, bucket)
    if not any(anchors.values()):
        print(f"Skip {dataset} {bucket}: no data")
        return None
    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    for method in PS.PLOT_ORDER:
        manual = LT.get_override("sl", dataset, bucket, method)
        LT.draw_curve(axis, method, raw.get(method, []),
                      n_anchor=N_ANCHOR, max_latency_ratio=MAX_LATENCY_RATIO,
                      manual=manual)
    LT.style_latency_axis(axis, list(anchors.values()),
                          fontsize=26, tick_fontsize=22)
    axis.set_title(SL.bucket_selectivity_text(dataset, bucket),
                   fontsize=24, fontweight="bold", pad=12)
    figure.tight_layout()
    save_figure(figure, f"sl_{dataset}_{bucket}")
    plt.close(figure)
    print(f"Saved sl_{dataset}_{bucket}")
    return anchors


def plot_dataset_grid(dataset: str):
    figure, axes = plt.subplots(1, len(BUCKETS), figsize=(8.0 * len(BUCKETS), 7.0))
    if len(BUCKETS) == 1:
        axes = [axes]
    for index, (axis, bucket) in enumerate(zip(axes, BUCKETS)):
        raw = raw_points(dataset, bucket)
        anchors = build_anchors(raw, GRID_ANCHOR, dataset, bucket)
        for method in PS.PLOT_ORDER:
            manual = LT.get_override("sl", dataset, bucket, method)
            LT.draw_curve(axis, method, raw.get(method, []),
                          n_anchor=GRID_ANCHOR,
                          max_latency_ratio=MAX_LATENCY_RATIO,
                          manual=manual)
        LT.style_latency_axis(axis, list(anchors.values()),
                              fontsize=20, tick_fontsize=16,
                              show_ylabel=(index == 0))
        axis.set_title(SL.bucket_selectivity_text(dataset, bucket),
                       fontsize=18, fontweight="bold", pad=8)
    figure.tight_layout()
    save_figure(figure, f"sl_{dataset}_grid")
    plt.close(figure)
    print(f"Saved sl_{dataset}_grid")


def save_legend():
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        PS.save_legend(output_dir / "sl_legend.png", fontsize=30,
                       markerscale=1.7)
    print("Saved sl_legend")


def main():
    PS.apply_style()
    datasets = [value for value in sys.argv[1:] if not value.startswith("--")]
    if not datasets:
        datasets = DATASETS
    selection = {}
    for dataset in datasets:
        selection[dataset] = {}
        for bucket in BUCKETS:
            anchors = plot_one(dataset, bucket)
            if anchors is not None:
                selection[dataset][bucket] = {
                    method: [[float(item[0]), float(item[1])]
                             for item in anchors.get(method, [])]
                    for method in METHODS
                }
        plot_dataset_grid(dataset)
    save_legend()
    output_dir = OUTPUT_DIRS[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sl_latency_selection.json"
    with open(path, "w", encoding="utf-8") as stream:
        json.dump({
            "metric": "Latency-Recall@10",
            "y_unit": "ms",
            "n_anchor": N_ANCHOR,
            "max_latency_ratio": MAX_LATENCY_RATIO,
            "datasets": selection,
        }, stream, indent=2)
    print("Saved", path)


if __name__ == "__main__":
    main()