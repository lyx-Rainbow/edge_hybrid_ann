"""Shared paper-style plotting helpers, following 5_Plot/style_refer templates."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import INDEX_META  # noqa: E402

# 4-method palette/markers/line styles, in the spirit of the reference scripts.
METHOD_STYLE = {
    "curator":   dict(color="#6F2DBD", marker="P", linestyle="-"),
    "diskivf":   dict(color="#D81B60", marker="h", linestyle="-."),
    "spann":     dict(color="#009E9A", marker="X", linestyle="--"),
    "prefilter": dict(color="#C99700", marker="*", linestyle=":"),
}
METHOD_LABELS = {k: INDEX_META[k][0] for k in METHOD_STYLE}
# Plot the complete method last so it stays visible at crossings.
PLOT_ORDER = ("prefilter", "spann", "diskivf", "curator")
LEGEND_ORDER = ("curator", "diskivf", "spann", "prefilter")

LINE_WIDTH = 4.0
MARKER_SIZE = 13.0
PREFILTER_MARKER_SIZE = 30.0
PREFILTER_MARKER_EDGE = 3.0
MARKER_EDGE_WIDTH = 2.5
LEGEND_LINEWIDTH = 6.0
LEGEND_MARKER_EDGE = 4.0


def apply_style():
    plt.style.use("ggplot")


def set_log_ylim(ax, values, padding=1.6):
    qps = [float(v) for v in values if v and float(v) > 0]
    if qps:
        ax.set_ylim(bottom=max(min(qps) / padding, 1e-3),
                    top=max(qps) * padding)


def style_axis(ax, xlim, xlabel, ylabel=None, fontsize=26, tick_fontsize=22,
               show_ylabel=True):
    ax.set_xlim(*xlim)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel and show_ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    elif not show_ylabel:
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, linewidth=4, alpha=0.3)


def draw_method(ax, method_key, xs, ys, single_point=False, linestyle=None):
    style = METHOD_STYLE[method_key]
    zorder = 10 if method_key == "curator" else 3
    line_style = linestyle if linestyle is not None else style["linestyle"]
    if single_point:
        if method_key == "prefilter":
            marker_size, marker_edge = (
                PREFILTER_MARKER_SIZE, PREFILTER_MARKER_EDGE)
        else:
            marker_size, marker_edge = MARKER_SIZE, MARKER_EDGE_WIDTH
        ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                linestyle="none", ms=marker_size,
                mec=style["color"], mew=marker_edge, mfc="none",
                zorder=zorder)
        return
    ax.plot(xs, ys, color=style["color"], marker=style["marker"],
            linestyle=line_style, lw=LINE_WIDTH, ms=MARKER_SIZE,
            mec=style["color"], mew=MARKER_EDGE_WIDTH, mfc="none",
            zorder=zorder)


def method_handles(keys=None):
    handles = []
    for key in (keys or LEGEND_ORDER):
        style = METHOD_STYLE[key]
        handles.append(Line2D([0], [0], color=style["color"],
                              marker=style["marker"],
                              linestyle=style["linestyle"],
                              lw=LINE_WIDTH, ms=MARKER_SIZE,
                              mec=style["color"],
                              mew=MARKER_EDGE_WIDTH, mfc="none",
                              label=METHOD_LABELS[key]))
    return handles


def save_legend(path, keys=None, fontsize=30, ncol=None, markerscale=1.7):
    path = Path(path)
    keys = list(keys or LEGEND_ORDER)
    figure = plt.figure(figsize=(max(10, 4.4 * len(keys)), 2.0))
    legend = figure.legend(
        method_handles(keys), [METHOD_LABELS[k] for k in keys],
        loc="center", ncol=ncol or len(keys), frameon=False,
        fontsize=fontsize, markerscale=markerscale,
        handlelength=3.0, handletextpad=1.0, columnspacing=1.2,
    )
    for handle in legend.legend_handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(LEGEND_LINEWIDTH)
        if hasattr(handle, "set_markeredgewidth"):
            handle.set_markeredgewidth(LEGEND_MARKER_EDGE)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    if path.suffix.lower() == ".png":
        figure.savefig(path.with_suffix(".svg"), dpi=300, bbox_inches="tight",
                       pad_inches=0.25)
    plt.close(figure)