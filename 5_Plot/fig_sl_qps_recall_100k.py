#!/usr/bin/env python3
"""
fig_sl_qps_recall_100k.py — Single-label QPS vs Recall for 100K benchmark.

One subplot per dataset. Each method shown as a Pareto frontier curve
(log-scale y-axis for QPS). Uses INDEX_META from utils.py for colors/markers.

Usage:
    python 5_Plot/fig_sl_qps_recall_100k.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "5_Plot"))
from utils import INDEX_META  # {key: (display_name, subdir, color, marker)}

RESULTS_DIR = PROJ_ROOT / "4_Results"
OUTPUT_DIR = PROJ_ROOT / "4_Results/fig_100k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_100K = ["sift1m_100k", "yfcc100m_100k", "arxiv_100k", "gist1m_100k", "wit_100k"]
DATASET_LABELS = {
    "sift1m_100k": "SIFT1M (d=128)",
    "yfcc100m_100k": "YFCC100M (d=192)",
    "arxiv_100k": "arXiv (d=384)",
    "gist1m_100k": "GIST1M (d=960)",
    "wit_100k": "WIT (d=384)",
}
K = 10


def load_sweep(method_key, dataset):
    """Load sweep JSON for a method+dataset. Returns list of (recall, qps) or None."""
    subdir = INDEX_META[method_key][1]
    path = RESULTS_DIR / subdir / f"sweep_{dataset}.json"
    if not path.exists():
        # Try validation directory
        path = RESULTS_DIR / subdir / "sweep_val" / f"sweep_{dataset}.json"
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    points = []
    for r in data.get("sweep_results", []):
        # Keep every combo with a valid QPS so all operating points are visible.
        if r.get("qps", 0) > 0:
            points.append((r["avg_recall"], r["qps"]))
    if not points:
        return None
    points.sort()  # Sort by recall
    return [(p[0], p[1]) for p in points]


def compute_pareto_frontier(points):
    """Given sorted (recall, qps) points, return Pareto frontier."""
    if not points:
        return [], []
    frontier_x, frontier_y = [], []
    max_qps = 0
    for r, q in reversed(points):  # iterate from high recall to low
        if q > max_qps:
            max_qps = q
            frontier_x.append(r)
            frontier_y.append(q)
    return list(reversed(frontier_x)), list(reversed(frontier_y))


def adaptive_xlim_x_ticks(recalls):
    """Per-subplot x range: start just below the subplot's own min recall so
    the curves fill the plot (no large empty low-recall region)."""
    lo = min(recalls) if recalls else 0.5
    lo = np.floor(lo * 20) / 20  # snap to 0.05 grid
    lo = max(0.0, lo - 0.02)
    span = 1.0 - lo
    if span > 0.2:
        step = 0.1
    elif span > 0.05:
        step = 0.05
    else:
        step = 0.02
    ticks = np.arange(np.ceil(lo / step) * step, 1.0001, step)
    return lo, ticks


def plot_all():
    n_ds = len(DATASETS_100K)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Pre-load every method's points.
    points_map = {}
    for method_key in ["curator", "diskivf", "spann", "prefilter"]:
        for ds in DATASETS_100K:
            points_map[(method_key, ds)] = load_sweep(method_key, ds)

    all_methods_shown = set()

    for ax_idx, ds in enumerate(DATASETS_100K):
        ax = axes[ax_idx]
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=18, fontweight="bold")
        ax.set_xlabel(f"Recall@{K}", fontsize=14)
        ax.set_ylabel("QPS", fontsize=14)
        ax.set_yscale("log")

        ds_recalls = [p[0] for mk in ["curator", "diskivf", "spann", "prefilter"]
                      for p in (points_map.get((mk, ds)) or [])]
        x_lo, x_ticks = adaptive_xlim_x_ticks(ds_recalls)
        ax.set_xlim(x_lo, 1.005)
        ax.set_xticks(x_ticks)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=12)

        n_pts = {}
        for method_key in ["curator", "diskivf", "spann", "prefilter"]:
            points = points_map.get((method_key, ds))
            if not points:
                continue
            all_methods_shown.add(method_key)

            name, _, color, marker = INDEX_META[method_key]

            # Pareto frontier line with emphasized markers (only frontier points
            # are shown: the line stays monotonic, recall up -> QPS down).
            fx, fy = compute_pareto_frontier(points)
            if fx:
                ax.plot(fx, fy, color=color, marker=marker, label=name,
                        lw=3, ms=10, mec=color, mew=2, mfc="none", zorder=2)
            n_pts[method_key] = len(points)

        print(f"  {ds}: " + " ".join(f"{k}={v}pts" for k, v in n_pts.items()))

    # Hide unused subplot
    if n_ds < len(axes):
        axes[-1].set_visible(False)

    # Global legend
    handles, labels = [], []
    for method_key in ["curator", "diskivf", "spann", "prefilter"]:
        if method_key in all_methods_shown:
            name, _, color, marker = INDEX_META[method_key]
            from matplotlib.lines import Line2D
            h = Line2D([0], [0], color=color, marker=marker, lw=3, ms=12,
                       mec=color, mew=2, mfc="none", label=name)
            handles.append(h)
            labels.append(name)

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   fontsize=16, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = OUTPUT_DIR / "fig_sl_qps_recall_100k.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_all()
