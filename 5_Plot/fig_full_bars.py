#!/usr/bin/env python3
"""Full-scale bar figures in the paper/reference style.

The three metrics are memory, build time and external volume.  Each metric can
be rendered either as one horizontal grid figure with one panel per dataset,
or as standalone per-dataset figures for free combination in the manuscript::

    python 5_Plot/fig_full_bars.py --metric memory --layout separate
    python 5_Plot/fig_full_bars.py --metric volume --layout grid
    python 5_Plot/fig_full_bars.py --metric all --layout both

Outputs (when layout=both):
    fig_full_memory_{dataset}.{png,svg}
    fig_full_build_time_{dataset}.{png,svg}
    fig_full_volume_{dataset}.{png,svg}
    fig_full_memory_grid / fig_full_build_time_grid / fig_full_volume_grid
    fig_full_memory / fig_full_build_time / fig_full_volume  (grid aliases)
    fig_full_bars_legend.{png,svg}

If ``4_Results/build_measure/index_metrics.json`` exists (produced by
``run_build_measure.py``) the figures use those unified measurements.  The old
sweep-based extraction is kept as a fallback for partially generated data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "5_Plot"))
import paper_style as PS  # noqa: E402

RESULTS_DIR = PROJ_ROOT / "4_Results"
BUILD_METRICS_PATH = RESULTS_DIR / "build_measure" / "index_metrics.json"
OUTPUT_DIR = RESULTS_DIR / "fig_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]
METHODS = ["curator", "diskivf", "spann", "prefilter"]
COLORS = [PS.METHOD_STYLE[k]["color"] for k in METHODS]
HATCHES = ["//", "\\\\", "xx", ".."]
LABELS = [PS.METHOD_LABELS[k] for k in METHODS]

BAR_WIDTH = 0.8
GROUP_GAP = 1.0
EDGE_LW = 2.0
HEADROOM = 1.15
BASE_LABEL_FONTSIZE = 26
BASE_TICK_FONTSIZE = 22
plt.rcParams["hatch.linewidth"] = 2.0


def load_build_metrics():
    if not BUILD_METRICS_PATH.exists():
        return {}
    with open(BUILD_METRICS_PATH, encoding="utf-8") as stream:
        return json.load(stream)


BUILD_METRICS = load_build_metrics()


def load_sweep(method_key, dataset):
    path = RESULTS_DIR / PS.INDEX_META[method_key][1] / f"sweep_{dataset}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def first_result(data):
    if not data:
        return {}
    return data.get("sweep_results", [{}])[0] if data.get("sweep_results") else {}


def get_value(method_key, dataset, field, fallback_in_result=None):
    data = load_sweep(method_key, dataset)
    if not data:
        return None
    if field in data and data[field]:
        return float(data[field])
    result = first_result(data)
    if field in result and result[field]:
        return float(result[field])
    if fallback_in_result and fallback_in_result in result:
        return float(result[fallback_in_result] or 0)
    return None


def master_value(section, dataset, method_key):
    value = BUILD_METRICS.get(section, {}).get(dataset, {}).get(method_key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def master_breakdown(dataset, method_key):
    value = (BUILD_METRICS.get("volume_breakdown_mb", {})
             .get(dataset, {}).get(method_key))
    if not value:
        return None
    return float(value.get("memory", 0.0)), float(value.get("external", 0.0))


def memory_value(dataset, method):
    value = master_value("query_memory_mb", dataset, method)
    if value is not None:
        return value
    return get_value(method, dataset, "rss_peak_query_mb",
                     fallback_in_result="rss_peak_query_mb")


def build_time_value(dataset, method):
    value = master_value("build_time_s", dataset, method)
    if value is not None:
        return value
    return get_value(method, dataset, "build_time_s",
                     fallback_in_result="build_time_s")


def volume_breakdown(dataset, method):
    """Return (memory_mb, external_mb) for the stacked volume bars."""
    memory = memory_value(dataset, method) or 0.0
    parts = master_breakdown(dataset, method)
    if parts is not None:
        return memory, parts[1]
    dsk = get_value(method, dataset, "disk_bytes",
                    fallback_in_result="disk_bytes")
    return memory, (dsk or 0.0) / (1024.0 ** 2)


def style_axis(axis, dataset, ylabel, show_ylabel=True):
    axis.grid(axis="y", color="white", linewidth=1.4, zorder=0)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks([])
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    xlabel = PS.text_content("bar_dataset_xlabel", "{dataset}")
    try:
        xlabel = xlabel.format(dataset=dataset)
    except Exception:
        pass
    PS.apply_axis_text(axis, xlabel, ylabel,
                       fontsize=BASE_LABEL_FONTSIZE,
                       tick_fontsize=BASE_TICK_FONTSIZE,
                       show_ylabel=show_ylabel)
    PS.apply_sci_offset_text(axis, base=BASE_TICK_FONTSIZE)


def set_headroom(axis, values):
    finite = [v for v in values if v is not None and not np.isnan(v)]
    upper = max(finite) * HEADROOM if finite and max(finite) > 0 else 1.0
    axis.set_ylim(0, upper)


def draw_grouped(axis, values, ylabel, dataset, show_ylabel=True):
    for x, value, color, hatch in zip(np.arange(len(METHODS)) * GROUP_GAP,
                                      values, COLORS, HATCHES):
        axis.bar(x, value, width=BAR_WIDTH, facecolor="white",
                 edgecolor=color, hatch=hatch, linewidth=EDGE_LW, zorder=3)
    style_axis(axis, dataset, ylabel, show_ylabel=show_ylabel)
    set_headroom(axis, values)


def draw_stacked(axis, memory, disk, ylabel, dataset, show_ylabel=True):
    totals = []
    for i, (mem, dsk, color, hatch) in enumerate(
            zip(memory, disk, COLORS, HATCHES)):
        x = i * GROUP_GAP
        axis.bar(x, mem, width=BAR_WIDTH, facecolor="white",
                 edgecolor=color, hatch=hatch, linewidth=EDGE_LW, zorder=3)
        axis.bar(x, dsk, width=BAR_WIDTH, bottom=mem, facecolor=color,
                 edgecolor=color, hatch=hatch, linewidth=EDGE_LW,
                 alpha=0.35, zorder=3)
        totals.append(mem + dsk)
    style_axis(axis, dataset, ylabel, show_ylabel=show_ylabel)
    set_headroom(axis, totals)


METRIC_LABEL_KEYS = {
    "memory": "bar_memory_ylabel",
    "build_time": "bar_build_time_ylabel",
    "volume": "bar_volume_ylabel",
}
METRIC_LABEL_DEFAULTS = {
    "memory": "Index Memory",
    "build_time": "Index Build Time (s)",
    "volume": "Index Volume",
}


def metric_ylabel(metric):
    return PS.text_content(METRIC_LABEL_KEYS[metric],
                           METRIC_LABEL_DEFAULTS[metric])


def draw_metric(axis, metric, dataset, show_ylabel=True):
    if metric == "memory":
        values = [memory_value(dataset, method) or 0.0
                  for method in METHODS]
        draw_grouped(axis, values, metric_ylabel(metric), dataset,
                     show_ylabel=show_ylabel)
    elif metric == "build_time":
        values = [build_time_value(dataset, method) or 0.0
                  for method in METHODS]
        draw_grouped(axis, values, metric_ylabel(metric), dataset,
                     show_ylabel=show_ylabel)
    elif metric == "volume":
        memory, disk = [], []
        for method in METHODS:
            mem, dsk = volume_breakdown(dataset, method)
            memory.append(mem)
            disk.append(dsk)
        draw_stacked(axis, memory, disk, metric_ylabel(metric), dataset,
                     show_ylabel=show_ylabel)
    else:
        raise ValueError(metric)


def save_figure(figure, output_name):
    figure.tight_layout()
    out = OUTPUT_DIR / f"{output_name}.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(str(out).replace(".png", ".svg"), dpi=300,
                   bbox_inches="tight")
    plt.close(figure)
    print("Saved", out)


def plot_grid(metric, datasets):
    figure, axes = plt.subplots(1, len(datasets),
                                figsize=(6.0 * len(datasets), 6.0))
    if len(datasets) == 1:
        axes = [axes]
    for index, (axis, dataset) in enumerate(zip(axes, datasets)):
        draw_metric(axis, metric, dataset, show_ylabel=(index == 0))
    # New explicit grid name + the legacy combined name for compatibility.
    save_figure(figure, f"fig_full_{metric}_grid")
    if list(datasets) == DATASETS:
        figure, axes = plt.subplots(1, len(DATASETS),
                                    figsize=(6.0 * len(DATASETS), 6.0))
        for index, (axis, dataset) in enumerate(zip(axes, DATASETS)):
            draw_metric(axis, metric, dataset, show_ylabel=(index == 0))
        save_figure(figure, f"fig_full_{metric}")


def plot_separate(metric, datasets):
    for dataset in datasets:
        figure, axis = plt.subplots(1, 1, figsize=(6.0, 6.0))
        draw_metric(axis, metric, dataset, show_ylabel=True)
        save_figure(figure, f"fig_full_{metric}_{dataset}")


def make_legend():
    handles = [Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=color,
                         hatch=hatch, linewidth=EDGE_LW)
               for color, hatch in zip(COLORS, HATCHES)]
    figure = plt.figure(figsize=(max(10, 8 * len(METHODS) * 0.75), 2.0))
    legend = figure.legend(handles, LABELS, loc="center", ncol=len(METHODS),
                           frameon=False, fontsize=26, handlelength=2.0,
                           columnspacing=0.8)
    for handle in legend.legend_handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(EDGE_LW)
    out = OUTPUT_DIR / "fig_full_bars_legend.png"
    figure.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1)
    figure.savefig(str(out).replace(".png", ".svg"), dpi=300,
                   bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)
    print("Saved", out)


def main():
    parser = argparse.ArgumentParser(
        description="Render split/grid paper bar figures.")
    parser.add_argument("--metric", default="all",
                        choices=["memory", "build_time", "volume", "all"])
    parser.add_argument("--layout", default="both",
                        choices=["separate", "grid", "both"])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--text-config", default=None,
                        help="optional path to figure_text_style.json")
    args = parser.parse_args()

    PS.apply_style(args.text_config)
    metrics = (["memory", "build_time", "volume"]
               if args.metric == "all" else [args.metric])
    for metric in metrics:
        if args.layout in ("grid", "both"):
            plot_grid(metric, args.datasets)
        if args.layout in ("separate", "both"):
            plot_separate(metric, args.datasets)
    make_legend()


if __name__ == "__main__":
    main()
