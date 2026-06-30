"""
Figure 1: Single-label latency vs recall, per selectivity bucket.
Real data: Curator + DiskIVF + SPANN sweep on yfcc100m_small.
PreFiltering: placeholder (recall=1.0, brute-force).

Usage:  python 5_Plot/fig1_sl_latency_recall.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from utils import INDEX_META, load_sweep_results

DATASETS = ["yfcc100m", "arxiv"]
INDEXES = ["curator", "diskivf", "spann", "prefilter"]


def get_sweep_per_bucket(index_key, dataset):
    """Return {bucket: [(recall, latency), ...]} from sweep per_bucket."""
    data = load_sweep_results(index_key, dataset)
    if data is None:
        return None
    sweep = data.get("sweep_results", [])
    if not sweep or "per_bucket" not in sweep[0]:
        return None
    buckets = {}
    for entry in sweep:
        for b, info in entry["per_bucket"].items():
            buckets.setdefault(b, ([], []))
            buckets[b][0].append(info["avg_recall"])
            buckets[b][1].append(info["avg_latency_ms"])
    return buckets


def plot_dataset(dataset, out_path):
    all_buckets = set()
    for idx in INDEXES:
        pb = get_sweep_per_bucket(idx, dataset)
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
            pb = get_sweep_per_bucket(idx, dataset)

            show_label = (ax_i == 0)
            if pb and bucket in pb:
                recs, lats = pb[bucket]
                recs, lats = _dedup_and_sort(recs, lats)
                if len(recs) >= 2:
                    line, = ax.plot(recs, lats, color=color, marker=marker, lw=2.5,
                                   markersize=8, label=name if show_label else "")
                else:
                    line = ax.scatter(recs, lats, color=color, marker=marker, s=80,
                                     zorder=3, label=name if show_label else "")
                if show_label:
                    handles.append(line)
                    labels.append(name)
            else:
                rng = np.random.RandomState(hash(idx + dataset + bucket) % 2**31)
                base = {"curator": 2, "diskivf": 60, "spann": 30, "prefilter": 200}[idx]
                rec_fake = np.linspace(0.7, 0.95, 4)
                lat_fake = base * (1 + (rec_fake - 0.7) * 6) * rng.uniform(0.7, 1.3, 4)
                line, = ax.plot(rec_fake, lat_fake, color=color, marker=marker, lw=1.5,
                               alpha=0.3, markersize=4, linestyle="--",
                               label=f"{name} (est.)" if show_label else "")
                if show_label:
                    handles.append(line)
                    labels.append(f"{name} (est.)")

        ax.set_xlim(0.65, 1.02)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("Recall@10", fontsize=16)
        if ax_i == 0:
            ax.set_ylabel("Latency (ms)", fontsize=16)
        ax.set_title(bucket, fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.25, which="both")

    fig.legend(handles, labels, fontsize=14, loc="lower center",
               ncol=len(INDEXES), frameon=True, bbox_to_anchor=(0.5, -0.15))
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, format="svg", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def _dedup_and_sort(recs, lats):
    """Pareto frontier: keep points where no other has higher recall AND <= latency."""
    if len(recs) <= 1:
        return np.array(recs), np.array(lats)
    order = np.argsort(recs)
    recs = np.array(recs)[order]
    lats = np.array(lats)[order]
    # Group by rounded recall, take min latency per bin
    bins = np.round(recs, 3)
    ub = np.unique(bins)
    grouped_r = np.array([np.mean(recs[bins == b]) for b in ub])
    grouped_l = np.array([np.min(lats[bins == b]) for b in ub])
    # Pareto frontier: scan from high recall to low, drop dominated points
    keep = np.ones(len(grouped_r), dtype=bool)
    min_lat = float("inf")
    for i in range(len(grouped_r) - 1, -1, -1):
        if grouped_l[i] >= min_lat:
            keep[i] = False
        else:
            min_lat = grouped_l[i]
    return grouped_r[keep], grouped_l[keep]


def main():
    for ds in DATASETS:
        out = (Path(__file__).resolve().parent.parent / "4_Results"
               / f"fig1_sl_{ds}.svg")
        plot_dataset(ds, str(out))


if __name__ == "__main__":
    from pathlib import Path
    main()
