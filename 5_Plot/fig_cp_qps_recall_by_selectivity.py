#!/usr/bin/env python3
"""
fig_cp_qps_recall_by_selectivity.py — CP QPS/Recall by filter selectivity.

Each filter is shown as a marker, positioned by its selectivity (x) and
recall/QPS (y). One row per dataset, columns: recall and QPS.
Filter type (AND/AND_NOT/OR/NOT) distinguished by marker shape.

Usage:
    python 5_Plot/fig_cp_qps_recall_by_selectivity.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "5_Plot"))
from utils import INDEX_META

RESULTS_DIR = PROJ_ROOT / "4_Results"
OUTPUT_DIR = PROJ_ROOT / "4_Results/_legacy_100k_figures"
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

FILTER_TYPE_MARKERS = {"AND": "o", "AND_NOT": "s", "OR": "D", "NOT": "^"}
FILTER_TYPE_LABELS = {"AND": "AND", "AND_NOT": "AND_NOT", "OR": "OR", "NOT": "NOT"}


def classify_filter_formula(formula: str) -> str:
    t = formula.split()
    if t[0] == "NOT":
        return "NOT"
    if t[0] == "AND":
        return "AND_NOT" if "NOT" in t else "AND"
    if t[0] == "OR":
        return "OR"
    return "UNKNOWN"


def load_cp_per_filter(method_key, dataset):
    """Load per-filter CP data with selectivity info.
    Returns list of (selectivity, recall, qps, filter_type, formula).
    """
    subdir = INDEX_META[method_key][1]
    path = RESULTS_DIR / subdir / f"sweep_{dataset}.json"
    if not path.exists():
        return None, None

    with open(path) as f:
        data = json.load(f)

    # Load selectivities from filters.json
    filters_path = PROJ_ROOT / "1_Data/ground_truth" / dataset / "complex_predicate" / "filters.json"
    if not filters_path.exists():
        return None, None
    with open(filters_path) as f:
        filters_info = json.load(f)
    selectivities = filters_info.get("selectivities", {})

    # Collect per-filter results from the best sweep combo (highest CP recall)
    best_entry = None
    for r in data.get("sweep_results", []):
        cp = r.get("complex_predicate")
        if cp and cp.get("avg_recall", 0) > 0:
            if best_entry is None or cp["avg_recall"] > best_entry.get("complex_predicate", {}).get("avg_recall", 0):
                best_entry = r

    if best_entry is None:
        return None, None

    cp = best_entry["complex_predicate"]
    params = best_entry.get("params", {})

    per_filter = cp.get("per_filter", {})
    points = []
    for formula, fd in per_filter.items():
        sel = selectivities.get(formula, None)
        if sel is None:
            # Curator returns RPN keys ("0 24 AND"), filters.json has Polish ("AND 0 24")
            # Try reverse token order as fallback
            tokens = formula.split()
            reversed_formula = " ".join(reversed(tokens))
            sel = selectivities.get(reversed_formula, None)
        if sel is None:
            continue
        ftype = classify_filter_formula(formula)
        points.append((sel, fd["avg_recall"], fd["qps"], ftype, formula))

    return points, params


def plot_all():
    n_ds = len(DATASETS_100K)
    fig, axes = plt.subplots(n_ds, 2, figsize=(14, 5 * n_ds))

    if n_ds == 1:
        axes = axes.reshape(1, -1)

    all_types_shown = set()
    all_methods_shown = set()

    for row_idx, ds in enumerate(DATASETS_100K):
        for col_idx, metric in enumerate(["recall", "qps"]):
            ax = axes[row_idx, col_idx]

            for method_key in ["curator", "diskivf", "spann", "prefilter"]:
                points, params = load_cp_per_filter(method_key, ds)
                if not points:
                    continue
                all_methods_shown.add(method_key)
                name, _, color, _ = INDEX_META[method_key]

                for sel, rec, qps, ftype, formula in points:
                    all_types_shown.add(ftype)
                    marker = FILTER_TYPE_MARKERS.get(ftype, "o")
                    y_val = rec if metric == "recall" else qps

                    ax.scatter(sel, y_val,
                              color=color, marker=marker, s=100,
                              edgecolors="black", linewidth=0.5,
                              zorder=3, label=name if ftype == "AND" else "")

            ax.set_xscale("log")
            ax.set_xlabel("Filter Selectivity", fontsize=12)
            ylabel = "CP Recall@10" if metric == "recall" else "CP QPS"
            ax.set_ylabel(ylabel, fontsize=12)
            ytitle = "Recall" if metric == "recall" else "QPS"
            ax.set_title(f"{DATASET_LABELS.get(ds, ds)}\n{ytitle} by Selectivity",
                        fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linewidth=1.0)
            ax.tick_params(labelsize=10)

            if metric == "recall":
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            else:
                ax.set_yscale("log")

    # Method legend (top)
    method_handles = []
    for mk in ["curator", "diskivf", "spann", "prefilter"]:
        if mk in all_methods_shown:
            name, _, color, _ = INDEX_META[mk]
            from matplotlib.lines import Line2D
            h = Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markersize=12, markeredgecolor="black", markeredgewidth=0.5,
                       label=name, linestyle="None")
            method_handles.append(h)

    # Filter type legend (bottom)
    type_handles = []
    for ft in ["AND", "AND_NOT", "OR", "NOT"]:
        if ft in all_types_shown:
            from matplotlib.lines import Line2D
            h = Line2D([0], [0], marker=FILTER_TYPE_MARKERS[ft], color="w",
                       markerfacecolor="gray", markersize=10,
                       markeredgecolor="black", markeredgewidth=0.5,
                       label=FILTER_TYPE_LABELS[ft], linestyle="None")
            type_handles.append(h)

    if method_handles:
        leg1 = fig.legend(method_handles, [h.get_label() for h in method_handles],
                         loc="upper center", ncol=len(method_handles),
                         fontsize=14, frameon=False, bbox_to_anchor=(0.5, 1.01),
                         title="Methods")
    if type_handles:
        leg2 = fig.legend(type_handles, [h.get_label() for h in type_handles],
                         loc="upper center", ncol=len(type_handles),
                         fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.99),
                         title="Filter Types")
        # Remove title from second legend to keep it clean
        if method_handles:
            fig.add_artist(leg1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / "fig_cp_qps_recall_by_selectivity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_all()
