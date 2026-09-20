#!/usr/bin/env python3
"""Final full-scale CP figures: Latency-Recall@10.

CP case-study curves are drawn in the same smoothed Latency-Recall@10 style as
the final SL figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOT_DIR))

import fig_full_cp_by_predicate as FC  # noqa: E402
import latency_trend as LT  # noqa: E402
import paper_style as PS  # noqa: E402

OUTPUT_DIRS = [
    PROJ_ROOT / "4_Results/fig_full",
]
N_ANCHOR = 9
GRID_ANCHOR = 7
MAX_LATENCY_RATIO = LT.MAX_LATENCY_RATIO_DEFAULT


def raw_for_case(dataset: str, formula: str):
    return {method: FC.load_cp_points(method, dataset, formula)
            for method in FC.METHODS}


def anchors_for(raw_by_method, n_anchor: int = N_ANCHOR,
                dataset: str = None, ptype: str = None):
    result = {}
    for method in FC.METHODS:
        manual = (LT.get_override("cp", dataset, ptype, method)
                  if dataset is not None and ptype is not None else {})
        result[method] = LT.build_trend(
            raw_by_method.get(method, []),
            n_anchor=n_anchor,
            max_latency_ratio=MAX_LATENCY_RATIO,
            manual=manual)
    return result


def save_figure(figure, name: str):
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        png = output_dir / f"{name}.png"
        figure.savefig(png, dpi=300, bbox_inches="tight")
        figure.savefig(png.with_suffix(".svg"), dpi=300,
                       bbox_inches="tight")


def draw_panel(axis, dataset: str, ptype: str, formula: str,
               n_anchor: int = N_ANCHOR, fontsize: int = 20,
               tick_fontsize: int = 16, show_ylabel: bool = True):
    raw = raw_for_case(dataset, formula)
    if not any(raw.values()):
        axis.axis("off")
        return None
    anchors = anchors_for(raw, n_anchor=n_anchor, dataset=dataset, ptype=ptype)
    for method in PS.PLOT_ORDER:
        manual = LT.get_override("cp", dataset, ptype, method)
        LT.draw_curve(axis, method, raw.get(method, []),
                      n_anchor=n_anchor, max_latency_ratio=MAX_LATENCY_RATIO,
                      manual=manual)
    LT.style_latency_axis(
        axis, list(anchors.values()), fontsize=fontsize,
        tick_fontsize=tick_fontsize, show_ylabel=show_ylabel)
    selectivity = FC.get_case_selectivity(dataset, formula)
    axis.set_title(
        f"{dataset} {ptype} ({FC.selectivity_text(selectivity)})",
        fontsize=18 if fontsize < 22 else 24,
        fontweight="bold", pad=8 if fontsize < 22 else 12)
    return anchors


def plot_one(dataset: str, ptype: str, formula: str):
    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    anchors = draw_panel(axis, dataset, ptype, formula, n_anchor=N_ANCHOR,
                         fontsize=26, tick_fontsize=22)
    if anchors is None:
        plt.close(figure)
        print(f"Skip {dataset} {ptype}")
        return None
    figure.tight_layout()
    save_figure(figure, f"cp_{dataset}_{ptype}")
    plt.close(figure)
    print(f"Saved cp_{dataset}_{ptype}")
    return anchors


def plot_combined():
    n_cases = len(FC.CP_CASES)
    figure, axes = plt.subplots(1, n_cases, figsize=(8.0 * n_cases, 7.0))
    if n_cases == 1:
        axes = [axes]
    for index, (axis, (dataset, ptype, formula)) in enumerate(
            zip(axes, FC.CP_CASES)):
        draw_panel(axis, dataset, ptype, formula, n_anchor=GRID_ANCHOR,
                   fontsize=20, tick_fontsize=16, show_ylabel=(index == 0))
    figure.legend(
        PS.method_handles(), [PS.METHOD_LABELS[key] for key in PS.LEGEND_ORDER],
        loc="lower center", ncol=4, frameon=False, fontsize=22,
        markerscale=1.7, bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(figure, "cp_combined")
    plt.close(figure)
    print("Saved cp_combined")


def save_legend():
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        PS.save_legend(output_dir / "cp_legend.png", fontsize=30,
                       markerscale=1.7)
    print("Saved cp_legend")


def main():
    PS.apply_style()
    selection = {}
    for dataset, ptype, formula in FC.CP_CASES:
        anchors = plot_one(dataset, ptype, formula)
        if anchors is not None:
            selection[dataset] = {
                "predicate": ptype,
                "formula": formula,
                "metric": "Latency-Recall@10",
                "y_unit": "ms",
                "n_anchor": N_ANCHOR,
                "max_latency_ratio": MAX_LATENCY_RATIO,
                "curves": {
                    method: [[float(item[0]), float(item[1])]
                             for item in anchors.get(method, [])]
                    for method in FC.METHODS
                },
            }
    plot_combined()
    save_legend()
    output_dir = OUTPUT_DIRS[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cp_latency_selection.json"
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2)
    print("Saved", path)


if __name__ == "__main__":
    main()