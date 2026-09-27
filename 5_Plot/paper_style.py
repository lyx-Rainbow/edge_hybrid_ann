"""Shared paper-style plotting helpers, following 5_Plot/style_refer templates.

Text-adjustment interface
-------------------------
All non-legend text sizes are controlled by ``figure_text_style.json`` in this
directory.  The effective font size is::

    base_size * global_scale * <kind>_scale

where ``<kind>`` is ``label``, ``tick`` or ``title``.  Font weights are taken
from the same file.  Legends are intentionally left at their own sizes (the
legend figures were already clear in the manuscript layout).

Set the environment variable ``FIGURE_TEXT_CONFIG`` to the path of another
JSON file to experiment with different text settings without editing the
repository file.  Calling ``reload_text_style()`` re-reads the file in the
current process.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

TEXT_STYLE_PATH = Path(__file__).resolve().parent / "figure_text_style.json"
_DEFAULT_TEXT_STYLE = {
    "global_scale": 1.0,
    "label_scale": 1.0,
    "tick_scale": 1.0,
    "title_scale": 1.0,
    "legend_scale": 1.0,
    "offset_scale": 1.0,
    "label_font_weight": "bold",
    "tick_font_weight": "bold",
    "title_font_weight": "bold",
    "offset_font_weight": "bold",
}

_DEFAULT_TEXT_CONTENT = {
    # Bar-chart axis text (kept short so titles fit inside the paper panels).
    "bar_memory_ylabel": "Index Memory",
    "bar_build_time_ylabel": "Index Build Time (s)",
    "bar_volume_ylabel": "Index Volume",
    "bar_dataset_xlabel": "{dataset}",
}
_TEXT_STYLE = None


def _style_path(path=None):
    if path:
        return Path(path)
    return Path(os.environ.get("FIGURE_TEXT_CONFIG", str(TEXT_STYLE_PATH)))


def _read_text_style(path=None):
    style = dict(_DEFAULT_TEXT_STYLE)
    style["content"] = dict(_DEFAULT_TEXT_CONTENT)
    style_path = _style_path(path)
    try:
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8-sig") as stream:
                user = json.load(stream)
            if isinstance(user, dict):
                user_content = user.get("content", {})
                style.update({key: value for key, value in user.items()
                              if key != "content"})
                if isinstance(user_content, dict):
                    style["content"].update(user_content)
    except Exception as exc:  # pragma: no cover - defensive only
        print(f"Warning: cannot read text style {style_path}: {exc}",
              file=sys.stderr)
    return style


def reload_text_style(path=None):
    """Reload the JSON text-style configuration and return it."""
    global _TEXT_STYLE
    _TEXT_STYLE = _read_text_style(path)
    return _TEXT_STYLE


def text_style():
    if _TEXT_STYLE is None:
        reload_text_style()
    style = dict(_TEXT_STYLE)
    style["content"] = dict(style.get("content", {}))
    return style


def text_content(key, default=None):
    """Return a user-adjustable text string from the same JSON file.

    The ``content`` object in ``figure_text_style.json`` stores all visible
    labels that are not purely numeric/scale settings.  This keeps style and
    wording in one place for later manual adjustment.
    """
    content = text_style().get("content", {})
    if key in content:
        return content[key]
    if default is not None:
        return default
    return _DEFAULT_TEXT_CONTENT.get(key, "")


def scaled_fontsize(base, kind="label"):
    """Return ``base`` scaled according to the shared text-style file."""
    style = text_style()
    global_scale = float(style.get("global_scale", 1.0))
    kind_scale = float(style.get(f"{kind}_scale", 1.0))
    return float(base) * global_scale * kind_scale


def font_weight(kind="label"):
    style = text_style()
    return str(style.get(f"{kind}_font_weight", "normal"))


def apply_axis_text(axis, xlabel, ylabel=None, fontsize=26,
                    tick_fontsize=22, show_ylabel=True):
    """Apply scaled and weighted axis labels/ticks to one axis."""
    axis.set_xlabel(xlabel, fontsize=scaled_fontsize(fontsize, "label"),
                    fontweight=font_weight("label"))
    if ylabel and show_ylabel:
        axis.set_ylabel(ylabel, fontsize=scaled_fontsize(fontsize, "label"),
                        fontweight=font_weight("label"))
    elif not show_ylabel:
        axis.set_ylabel("")
    axis.tick_params(axis="both",
                     labelsize=scaled_fontsize(tick_fontsize, "tick"))
    for tick_label in (list(axis.get_xticklabels())
                       + list(axis.get_yticklabels())):
        tick_label.set_fontweight(font_weight("tick"))


def apply_sci_offset_text(axis, base=22):
    """Apply the configured size/weight to a scientific-notation offset text.

    This is the small text such as "x10^3" shown above a ScalarFormatter
    axis.  It is controlled by ``offset_scale`` / ``offset_font_weight`` in
    figure_text_style.json.
    """
    offset = axis.yaxis.get_offset_text()
    if offset is None:
        return
    offset.set_fontsize(scaled_fontsize(base, "offset"))
    offset.set_fontweight(font_weight("offset"))


def set_title(axis, title, fontsize=24, pad=None):
    """Set a scaled and weighted axis title."""
    kwargs = {
        "fontsize": scaled_fontsize(fontsize, "title"),
        "fontweight": font_weight("title"),
    }
    if pad is not None:
        kwargs["pad"] = pad
    axis.set_title(title, **kwargs)


def apply_style(text_config=None):
    if text_config is not None:
        reload_text_style(text_config)
    else:
        reload_text_style()
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
    apply_axis_text(ax, xlabel, ylabel, fontsize=fontsize,
                    tick_fontsize=tick_fontsize, show_ylabel=show_ylabel)
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
