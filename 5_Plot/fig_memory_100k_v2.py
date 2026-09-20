#!/usr/bin/env python3
"""
fig_memory_100k_v2.py — Memory comparison, Curator with memory-optimized
configuration (n_clusters=16, bf_false_pos=0.05, pq_cache_max_blocks=32,
build-buffer release + heap trim + arena cap).

Curator values come from 4_Results/memory_tuned/curator_v2.json; the other
three methods use the same sweep data as fig_memory_100k.py. Output files are
suffixed _v2 so the original figure is preserved.

Usage:
    python 5_Plot/fig_memory_100k_v2.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "5_Plot"))
from utils import INDEX_META

RESULTS_DIR = PROJ_ROOT / "4_Results"
OUTPUT_DIR = PROJ_ROOT / "4_Results/_legacy_100k_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURATOR_V2_PATH = PROJ_ROOT / "4_Results/memory_tuned/curator_v2.json"

DATASETS_100K = ["sift1m_100k", "yfcc100m_100k", "arxiv_100k", "gist1m_100k", "wit_100k"]
DATASET_LABELS = {
    "sift1m_100k": "SIFT1M\n(d=128)",
    "yfcc100m_100k": "YFCC100M\n(d=192)",
    "arxiv_100k": "arXiv\n(d=384)",
    "gist1m_100k": "GIST1M\n(d=960)",
    "wit_100k": "WIT\n(d=384)",
}
METHOD_KEYS = ["curator", "diskivf", "spann", "prefilter"]
METHOD_LABELS = [INDEX_META[k][0] for k in METHOD_KEYS]
COLORS = [INDEX_META[k][2] for k in METHOD_KEYS]

HATCHES = ["//", "\\\\", "xx", ".."]
BAR_WIDTH = 0.7
GROUP_GAP = 1.0
EDGE_LW = 2.0
plt.rcParams["hatch.linewidth"] = 2.0


def load_curator_v2():
    if not CURATOR_V2_PATH.exists():
        return {}
    return json.load(open(CURATOR_V2_PATH))


def load_memory_mb(method_key, dataset, curator_v2):
    """Load memory from sweep JSON. Returns (rss_peak_query_mb, index_memory_mb)."""
    if method_key == "curator" and dataset in curator_v2:
        rec = curator_v2[dataset]
        return rec.get("rss_peak_query_mb"), rec.get("index_total_bytes")

    subdir = INDEX_META[method_key][1]
    path = RESULTS_DIR / subdir / f"sweep_{dataset}.json"
    if not path.exists():
        return None, None

    with open(path) as f:
        data = json.load(f)

    rss_peak = data.get("rss_peak_query_mb")
    if rss_peak is None:
        for r in data.get("sweep_results", []):
            rss_peak = r.get("rss_peak_query_mb")
            if rss_peak is not None:
                break

    idx_mem = data.get("index_memory_mb")
    return (float(rss_peak) if rss_peak else None,
            float(idx_mem) if idx_mem else None)


def plot_all():
    curator_v2 = load_curator_v2()
    n_ds = len(DATASETS_100K)
    n_methods = len(METHOD_KEYS)
    x_positions = np.arange(n_methods) * GROUP_GAP

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 6))
    if n_ds == 1:
        axes = [axes]

    bars_for_legend = []

    for ax, ds in zip(axes, DATASETS_100K):
        values = []
        for mk in METHOD_KEYS:
            rss_peak, idx_mem = load_memory_mb(mk, ds, curator_v2)
            val = rss_peak if rss_peak else (idx_mem if idx_mem else 0)
            values.append(val if val and val > 0 else 0)

        ax.grid(axis="y", color="white", linewidth=1.4, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for x, v, color, hatch in zip(x_positions, values, COLORS, HATCHES):
            bar = ax.bar(x, v, width=BAR_WIDTH, facecolor="white",
                         edgecolor=color, hatch=hatch, linewidth=EDGE_LW, zorder=3)
            if ax is axes[0]:
                bars_for_legend.append(bar[0])

        max_v = max(values) if values and max(values) > 0 else 1
        ax.set_ylim(0, max_v * 1.2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.ticklabel_format(style="plain", axis="y")
        ax.set_xticks([])
        ax.set_xlabel(DATASET_LABELS.get(ds, ds), fontsize=18)
        ax.set_ylabel("Query Peak RSS (MB)", fontsize=16)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_title(DATASET_LABELS.get(ds, ds).replace("\n", " "),
                     fontsize=16, fontweight="bold")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig_memory_100k_v2.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Separate legend figure
    fig_leg = plt.figure(figsize=(max(8, 2 * n_methods), 1.0))
    fig_leg.legend(bars_for_legend, METHOD_LABELS, loc="center",
                   ncol=n_methods, frameon=False, fontsize=20,
                   handlelength=2.0, columnspacing=0.8)
    plt.axis("off")
    leg_path = OUTPUT_DIR / "fig_memory_100k_v2_legend.png"
    fig_leg.savefig(leg_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Legend saved: {leg_path}")
    plt.close("all")


if __name__ == "__main__":
    plot_all()
