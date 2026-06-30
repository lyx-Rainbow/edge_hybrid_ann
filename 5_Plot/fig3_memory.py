"""
Figure 3: Memory overhead (peak RSS) per index, per dataset.

Real data from available single-run / sweep results.
Missing data filled with placeholders.

Usage:
    python 5_Plot/fig3_memory.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    INDEX_META, load_benchmark_results, load_sweep_results, get_index_memory_mb,
)

DATASETS = ["yfcc100m", "arxiv"]
INDEXES = ["curator", "diskivf", "spann", "prefilter"]

MEMORY_CORRECT_DIR = Path(__file__).resolve().parent.parent / "4_Results" / "memory_correct"


def get_memory_for(index_key: str, dataset: str) -> float | None:
    """Try corrected memory first, then sweep/benchmark results, then fall back."""
    # Priority 1: corrected memory from diagnostic
    corrected_path = MEMORY_CORRECT_DIR / f"{index_key}_{dataset}.json"
    if corrected_path.exists():
        with open(corrected_path) as f:
            data = json.load(f)
        return float(data.get("rss_peak_query_mb", 0))

    # Priority 2: benchmark results
    bench = load_benchmark_results(index_key, dataset)
    if bench:
        return get_index_memory_mb(bench)

    # Priority 3: sweep results
    sweep = load_sweep_results(index_key, dataset)
    if sweep:
        entries = sweep.get("sweep_results", [])
        if entries and "rss_peak_query_mb" in entries[0]:
            return float(entries[0]["rss_peak_query_mb"])
        return get_index_memory_mb(sweep)
    return None


def main():
    PLACEHOLDER_MB = {
        ("arxiv_small", "curator"): 250,
        ("arxiv_small", "diskivf"): 200,
        ("arxiv_small", "spann"): 180,
        ("arxiv_small", "prefilter"): 350,
        ("yfcc100m_small", "spann"): 160,
        ("yfcc100m_small", "prefilter"): 300,
    }

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12, 5))
    if len(DATASETS) == 1:
        axes = [axes]

    for ax, ds in zip(axes, DATASETS):
        names, values, colors, alphas = [], [], [], []
        for idx in INDEXES:
            val = get_memory_for(idx, ds)
            name, _, color, _ = INDEX_META[idx]
            if val is not None:
                names.append(f"{name}")
                values.append(val)
                alphas.append(1.0)
            else:
                names.append(f"{name}\n(est.)")
                values.append(PLACEHOLDER_MB.get((ds, idx), 200))
                alphas.append(0.4)
            colors.append(color)

        bars = ax.bar(range(len(names)), values, color=colors,
                      edgecolor="black", linewidth=0.5)
        for bar, val, a in zip(bars, values, alphas):
            bar.set_alpha(a)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.2f}", ha="center", fontsize=16)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=16)
        ax.set_ylabel("Peak RSS (MB)", fontsize=18)
        DISPLAY = {"yfcc100m": "YFCC-1M", "arxiv": "arXiv"}
        ax.set_title(DISPLAY.get(ds, ds), fontsize=20)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "4_Results" / "fig3_memory.svg"
    fig.savefig(out, format="svg", dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    from pathlib import Path
    main()
