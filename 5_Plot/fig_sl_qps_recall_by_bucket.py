#!/usr/bin/env python3
"""
fig_sl_qps_recall_by_bucket.py — SL QPS vs Recall faceted by selectivity bucket.

Layout: N dataset rows × 3 bucket columns (low / mid / high selectivity).
Each subplot shows Pareto frontier curves for all available methods.

Usage:
    python 5_Plot/fig_sl_qps_recall_by_bucket.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "5_Plot"))
from utils import INDEX_META

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
METHOD_ORDER = ["curator", "diskivf", "spann", "prefilter"]
N_COLS = 3
K = 10


def load_per_bucket_points(method_key, dataset):
    """Load per-bucket (recall, qps) from sweep JSON."""
    subdir = INDEX_META[method_key][1]
    path = RESULTS_DIR / subdir / f"sweep_{dataset}.json"
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    bucket_points = {}
    for r in data.get("sweep_results", []):
        pb = r.get("per_bucket", {})
        if not pb:
            continue
        for bname, bdata in pb.items():
            rec = bdata.get("avg_recall", 0)
            qps = bdata.get("qps", 0)
            if rec > 0 and qps > 0:
                bucket_points.setdefault(bname, []).append((rec, qps))

    return bucket_points


def compute_pareto_frontier(points):
    if not points:
        return [], []
    points = sorted(points)
    frontier_x, frontier_y = [], []
    max_qps = 0
    for r, q in reversed(points):
        if q > max_qps:
            max_qps = q
            frontier_x.append(r)
            frontier_y.append(q)
    return list(reversed(frontier_x)), list(reversed(frontier_y))


def short_bucket_name(name):
    try:
        parts = name.strip("[]()").split(",")
        lo = float(parts[0].strip()) * 100
        hi = float(parts[1].strip()) * 100
        return f"{lo:.0f}-{hi:.0f}%"
    except (ValueError, IndexError):
        return name[:16]


def plot_all():
    # Find which datasets actually have data
    active_datasets = []
    for ds in DATASETS_100K:
        for mk in METHOD_ORDER:
            bp = load_per_bucket_points(mk, ds)
            if bp:
                active_datasets.append(ds)
                break

    if not active_datasets:
        print("No sweep data found for any dataset. Run --run_sl first.")
        return

    print(f"Datasets with data: {active_datasets}")
    n_rows = len(active_datasets)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(6 * N_COLS, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    all_methods_shown = set()

    for row_idx, ds in enumerate(active_datasets):
        # Collect per-bucket data for all methods
        ds_buckets = {}
        for mk in METHOD_ORDER:
            bp = load_per_bucket_points(mk, ds)
            for bname, pts in bp.items():
                ds_buckets.setdefault(bname, {})[mk] = pts

        bucket_names = sorted(ds_buckets.keys())

        if not bucket_names:
            for col_idx in range(N_COLS):
                ax = axes[row_idx, col_idx]
                ax.set_title(f"{DATASET_LABELS.get(ds, ds)}\n(no per_bucket data)", fontsize=13)
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
            continue

        # Select 3 representative buckets
        if len(bucket_names) > N_COLS:
            N = len(bucket_names)
            bucket_names = [bucket_names[0], bucket_names[N // 2], bucket_names[-1]]

        print(f"  {ds}: {len(ds_buckets)} buckets total, showing {bucket_names}")

        for col_idx in range(N_COLS):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(bucket_names):
                ax.set_visible(False)
                continue

            bname = bucket_names[col_idx]
            bucket_data = ds_buckets.get(bname, {})

            ax.set_title(f"{DATASET_LABELS.get(ds, ds)}\nSelectivity: {short_bucket_name(bname)}",
                        fontsize=13, fontweight="bold")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, linewidth=1.0)
            ax.tick_params(labelsize=10)

            # Collect all points for y-axis scaling and adaptive x range
            all_pts = []
            for mk in METHOD_ORDER:
                pts = bucket_data.get(mk, [])
                if pts:
                    all_pts.extend(pts)

            if all_pts:
                all_qps = [q for _, q in all_pts]
                y_min = min(all_qps) * 0.5
                y_max = max(all_qps) * 2
                ax.set_ylim(y_min, y_max)

                lo = np.floor(min(r for r, _ in all_pts) * 20) / 20
                lo = max(0.0, lo - 0.02)
                span = 1.0 - lo
                step = 0.1 if span > 0.2 else (0.05 if span > 0.05 else 0.02)
                ax.set_xlim(lo, 1.005)
                ax.set_xticks(np.arange(np.ceil(lo / step) * step, 1.0001, step))

            if row_idx == n_rows - 1:
                ax.set_xlabel(f"Recall@{K}", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel("QPS", fontsize=12)

            for mk in METHOD_ORDER:
                pts = bucket_data.get(mk, [])
                if not pts:
                    continue
                all_methods_shown.add(mk)
                name, _, color, marker = INDEX_META[mk]
                fx, fy = compute_pareto_frontier(pts)
                if fx:
                    ax.plot(fx, fy, color=color, marker=marker, label=name,
                            lw=2.5, ms=8, mec=color, mew=1.5, mfc="none", zorder=2)

    # Global legend
    handles, labels = [], []
    for mk in METHOD_ORDER:
        if mk in all_methods_shown:
            name, _, color, marker = INDEX_META[mk]
            from matplotlib.lines import Line2D
            h = Line2D([0], [0], color=color, marker=marker, lw=3, ms=10,
                       mec=color, mew=2, mfc="none", label=name)
            handles.append(h)
            labels.append(name)

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = OUTPUT_DIR / "fig_sl_qps_recall_by_bucket.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_all()
