#!/usr/bin/env python3
"""Latency(ms) version of the full-scale SL figures.

The current QPS figures are left untouched; this script writes to
4_Results/fig_full_latency/.  Every plotted value is derived from the same
selected measured points via latency_ms = 1000 / qps.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOT_DIR))
import fig_full_sl_by_percentile as SL  # noqa: E402
import paper_style as PS  # noqa: E402

LATENCY_DIR = SL.PROJ_ROOT / "4_Results/fig_full_latency_legacy"
LATENCY_DIR.mkdir(parents=True, exist_ok=True)
YLABEL = "Latency (ms)"


def to_latency(curves):
    out = {}
    for method, curve in curves.items():
        points = []
        for recall, qps, params in curve:
            if qps and qps > 0:
                points.append((recall, 1000.0 / qps, params))
        out[method] = points
    return out


def all_latency(curves):
    values = []
    for curve in curves.values():
        values.extend(p[1] for p in curve)
    return values


def _plot_curves(axis, curves, xlabel, fontsize=26, tick_fontsize=22,
                 show_ylabel=True):
    latency = to_latency(curves)
    for method in PS.PLOT_ORDER:
        curve = latency.get(method) or []
        if not curve:
            continue
        PS.draw_method(axis, method, [p[0] for p in curve],
                       [p[1] for p in curve],
                       single_point=(method == "prefilter"))
    lo, hi = SL._xlim_for(curves)
    PS.style_axis(axis, (lo, hi), xlabel, ylabel=YLABEL, fontsize=fontsize,
                  tick_fontsize=tick_fontsize, show_ylabel=show_ylabel)
    PS.set_log_ylim(axis, all_latency(latency))


def plot_one(dataset, bucket):
    _, curves = SL.build_curves(dataset, bucket)
    if not any(curves.values()):
        print(f"Skip {dataset} {bucket}: no curve data")
        return
    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    _plot_curves(axis, curves, "Recall@10")
    axis.set_title(SL.bucket_selectivity_text(dataset, bucket),
                   fontsize=24, fontweight="bold", pad=12)
    figure.tight_layout()
    out = LATENCY_DIR / f"sl_{dataset}_{bucket}.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(str(out).replace(".png", ".svg"), dpi=300,
                   bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {out}")


def plot_dataset_grid(dataset):
    n = len(SL.BUCKETS)
    figure, axes = plt.subplots(1, n, figsize=(8.0 * n, 7.0))
    if n == 1:
        axes = [axes]
    for i, (axis, bucket) in enumerate(zip(axes, SL.BUCKETS)):
        _, curves = SL.build_curves(dataset, bucket)
        _plot_curves(axis, curves, "Recall@10", fontsize=20,
                     tick_fontsize=16, show_ylabel=(i == 0))
        axis.set_title(SL.bucket_selectivity_text(dataset, bucket),
                       fontsize=18, fontweight="bold", pad=8)
    figure.tight_layout()
    out = LATENCY_DIR / f"sl_{dataset}_grid.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(str(out).replace(".png", ".svg"), dpi=300,
                   bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {out}")


def save_legend():
    PS.save_legend(LATENCY_DIR / "sl_legend.png", fontsize=30)
    print(f"Saved {LATENCY_DIR / 'sl_legend.png'}")


def main():
    PS.apply_style()
    datasets = sys.argv[1:] if len(sys.argv) > 1 else SL.DATASETS
    for dataset in datasets:
        for bucket in SL.BUCKETS:
            plot_one(dataset, bucket)
        plot_dataset_grid(dataset)
    save_legend()


if __name__ == "__main__":
    main()