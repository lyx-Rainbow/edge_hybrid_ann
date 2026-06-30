"""
Figure 4: Index build time per index, per dataset (table).
Reads from sweep / benchmark results. Missing data filled with placeholders.

Usage:
    python 5_Plot/fig4_build_time.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    INDEX_META, load_benchmark_results, load_sweep_results, get_build_time_s,
)

DATASETS = ["yfcc100m", "arxiv"]
INDEXES = ["curator", "diskivf", "spann", "prefilter"]


def get_build_time(index_key: str, dataset: str) -> float | None:
    bench = load_benchmark_results(index_key, dataset)
    if bench:
        return get_build_time_s(bench)
    sweep = load_sweep_results(index_key, dataset)
    if sweep:
        return get_build_time_s(sweep)
    return None


def main():
    PLACEHOLDER_S = {
        ("arxiv", "curator"): 180,
        ("arxiv", "spann"): 40,
        ("arxiv_small", "curator"): 4.0,
        ("arxiv_small", "diskivf"): 3.0,
        ("arxiv_small", "spann"): 5.0,
        ("arxiv_small", "prefilter"): 0.3,
        ("yfcc100m_small", "spann"): 2.5,
        ("yfcc100m_small", "prefilter"): 0.2,
    }

    # Gather data
    display_names = [INDEX_META[k][0] for k in INDEXES]
    col_labels = ["yfcc100m", "arxiv"]
    cell_text = []

    for idx in INDEXES:
        row = []
        for ds in DATASETS:
            val = get_build_time(idx, ds)
            if val is not None:
                row.append(f"{val:.1f}s")
            else:
                ph = PLACEHOLDER_S.get((ds, idx), 0)
                row.append(f"{ph:.1f}s (est.)")
        cell_text.append(row)

    # Build table figure
    fig, ax = plt.subplots(figsize=(6, 2.0))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=display_names,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#e0e0e0")
        cell.set_text_props(weight="bold")

    # Style row labels
    for i in range(len(display_names)):
        cell = table[i + 1, -1]
        cell.set_facecolor("#f5f5f5")
        cell.set_text_props(weight="bold")

    fig.tight_layout(pad=0.5)

    out = (Path(__file__).resolve().parent.parent
           / "4_Results" / "fig4_build_time.svg")
    fig.savefig(out, format="svg", dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    from pathlib import Path
    main()
