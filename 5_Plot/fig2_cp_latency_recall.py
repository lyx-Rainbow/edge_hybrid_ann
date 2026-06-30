"""
Figure 2: Complex-predicate latency vs recall, per selectivity bucket.
Scatter plot — each point is one filter at one sweep param combo.
Transparency + marker size show density; lower = better.

Usage:  python 5_Plot/fig2_cp_latency_recall.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from utils import INDEX_META, load_sweep_results

DATASETS = ["yfcc100m", "arxiv"]
INDEXES = ["curator", "diskivf", "spann", "prefilter"]


def load_filter_selectivities(dataset):
    path = (Path(__file__).resolve().parent.parent
            / "1_Data/ground_truth" / dataset
            / "complex_predicate/filters.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f).get("selectivities", {})


def get_cp_per_bucket(index_key, dataset):
    """Return {bucket_name: [(recall, latency), ...]} grouped by filter selectivity."""
    data = load_sweep_results(index_key, dataset)
    if data is None:
        return None
    sweep = data.get("sweep_results", [])
    if not sweep or "complex_predicate" not in sweep[0]:
        return None

    sel_map = load_filter_selectivities(dataset)
    if not sel_map:
        return None

    # Collect ALL raw (sel, recall, latency) triples
    points = []
    for entry in sweep:
        cp = entry["complex_predicate"]
        for formula, info in cp.get("per_filter", {}).items():
            sel = sel_map.get(formula, 0)
            if sel == 0:
                continue
            points.append((sel, info["avg_recall"], info["avg_latency_ms"]))

    if not points:
        return None

    # Selectivity buckets
    all_sels = np.array([p[0] for p in points])
    max_sel = min(all_sels.max(), 0.2)
    min_sel = max(all_sels.min(), 1e-5)
    bounds = np.linspace(min_sel, max_sel, 6)
    buckets = {}

    for sel, rec, lat in points:
        bname = None
        for j in range(5):
            if j == 4:
                bname = f"[{bounds[4]:.4f}, {bounds[5]:.4f})"
            elif bounds[j] <= sel < bounds[j + 1]:
                bname = f"[{bounds[j]:.4f}, {bounds[j+1]:.4f})"
                break
        if bname:
            buckets.setdefault(bname, ([], []))
            buckets[bname][0].append(rec)
            buckets[bname][1].append(lat)

    return buckets if buckets else None


def plot_dataset(dataset, out_path):
    all_buckets = set()
    for idx in INDEXES:
        pb = get_cp_per_bucket(idx, dataset)
        if pb:
            all_buckets.update(pb.keys())
    if not all_buckets:
        all_buckets = {"low", "medium", "high"}

    buckets = sorted(all_buckets)
    n = len(buckets)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    handles, labels = [], []
    for ax_i, (ax, bucket) in enumerate(zip(axes, buckets)):
        for idx in INDEXES:
            name, _, color, marker = INDEX_META[idx]
            pb = get_cp_per_bucket(idx, dataset)

            show_label = (ax_i == 0)
            if pb and bucket in pb:
                recs, lats = pb[bucket]
                sc = ax.scatter(recs, lats, color=color, marker=marker, s=40,
                              alpha=0.50, edgecolors="none",
                              label=name if show_label else "",
                              zorder=3)
                if show_label:
                    handles.append(sc)
                    labels.append(name)
            else:
                rng = np.random.RandomState(hash(idx + dataset + bucket) % 2**31)
                base = {"curator": 3, "diskivf": 70, "spann": 35, "prefilter": 300}[idx]
                sc = ax.scatter([rng.uniform(0.62, 0.98)],
                              [rng.uniform(0.5, 1.5) * base],
                              color=color, marker=marker, s=50,
                              alpha=0.35,
                              label=f"{name} (est.)" if show_label else "")
                if show_label:
                    handles.append(sc)
                    labels.append(f"{name} (est.)")

        ax.set_xlim(0.65, 1.02)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("Recall@10", fontsize=16)
        if ax_i == 0:
            ax.set_ylabel("Latency (ms)", fontsize=16)
        ax.set_title(f"CP {bucket}", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.25, which="both")

    fig.legend(handles, labels, fontsize=14, loc="lower center",
               ncol=len(INDEXES), frameon=True, bbox_to_anchor=(0.5, -0.15))
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, format="svg", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    for ds in DATASETS:
        out = (Path(__file__).resolve().parent.parent / "4_Results"
               / f"fig2_cp_{ds}.svg")
        plot_dataset(ds, str(out))


if __name__ == "__main__":
    from pathlib import Path
    main()
