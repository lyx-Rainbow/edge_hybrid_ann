#!/usr/bin/env python3
"""Latency(ms) version of the full-scale CP figures.

The current QPS figures are left untouched; this script writes to
4_Results/fig_full_latency/.  Latency is derived from the same selected
measured points via latency_ms = 1000 / qps.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOT_DIR))
import fig_full_cp_by_predicate as FC  # noqa: E402
import paper_style as PS  # noqa: E402

LATENCY_DIR = FC.RESULTS_DIR / "fig_full_latency_legacy"
LATENCY_DIR.mkdir(parents=True, exist_ok=True)
YLABEL = "Latency (ms)"


def to_latency(curves):
    out = {}
    for method, curve in curves.items():
        points = []
        for recall, qps, params in curve:
            if qps and qps > 0:
                points.append((recall, 1000.0 / qps, params))
        out[method] = points
    return out


def all_latency(curves):
    values = []
    for curve in curves.values():
        values.extend(p[1] for p in curve)
    return values


def load_curves(dataset, formula):
    raw = {method: FC.load_cp_points(method, dataset, formula)
           for method in FC.METHODS}
    if not any(raw.values()):
        return None
    return to_latency(FC.build_curves(raw, dataset=dataset))


def draw_panel(axis, dataset, ptype, formula, fontsize=20, tick_fontsize=16,
               show_ylabel=True):
    curves = load_curves(dataset, formula)
    if curves is None:
        axis.axis("off")
        return None
    for method in PS.PLOT_ORDER:
        curve = curves.get(method) or []
        if not curve:
            continue
        PS.draw_method(axis, method, [p[0] for p in curve],
                       [p[1] for p in curve],
                       single_point=(method == "prefilter"))
    lo, hi = FC._xlim_for(curves)
    sel = FC.get_case_selectivity(dataset, formula)
    PS.style_axis(
        axis, (lo, hi), "Recall@10",
        ylabel=YLABEL if show_ylabel else None,
        fontsize=fontsize, tick_fontsize=tick_fontsize,
        show_ylabel=show_ylabel)
    axis.set_title(f"{dataset} {ptype} ({FC.selectivity_text(sel)})",
                   fontsize=18 if fontsize < 22 else 24,
                   fontweight="bold", pad=8 if fontsize < 22 else 12)
    PS.set_log_ylim(axis, all_latency(curves))
    return curves


def plot_one(dataset, ptype, formula, selection_out):
    curves = load_curves(dataset, formula)
    if curves is None:
        print(f"Skip {dataset} {ptype}: no CP data")
        return False
    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    draw_panel(axis, dataset, ptype, formula, fontsize=26, tick_fontsize=22)
    figure.tight_layout()
    out = LATENCY_DIR / f"cp_{dataset}_{ptype}.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(str(out).replace(".png", ".svg"), dpi=300,
                   bbox_inches="tight")
    plt.close(figure)
    selection_out[dataset] = {
        "predicate": ptype,
        "formula": formula,
        "selectivity": FC.get_case_selectivity(dataset, formula),
        "curves": {method: [[float(p[0]), float(p[1]), p[2]]
                            for p in curves.get(method, [])]
                   for method in FC.METHODS},
    }
    print(f"Saved {out}")
    return True


def plot_combined():
    n = len(FC.CP_CASES)
    figure, axes = plt.subplots(1, n, figsize=(8.0 * n, 7.0))
    if n == 1:
        axes = [axes]
    for i, (axis, (dataset, ptype, formula)) in enumerate(
            zip(axes, FC.CP_CASES)):
        draw_panel(axis, dataset, ptype, formula, show_ylabel=(i == 0))
    figure.legend(PS.method_handles(),
                  [PS.METHOD_LABELS[k] for k in PS.LEGEND_ORDER],
                  loc="lower center", ncol=4, frameon=False, fontsize=22,
                  markerscale=2.0, bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    out = LATENCY_DIR / "cp_combined.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(str(out).replace(".png", ".svg"), dpi=300,
                   bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {out}")


def save_legend():
    PS.save_legend(LATENCY_DIR / "cp_legend.png", fontsize=30)
    print(f"Saved {LATENCY_DIR / 'cp_legend.png'}")


def main():
    PS.apply_style()
    selection_out = {}
    for dataset, ptype, formula in FC.CP_CASES:
        plot_one(dataset, ptype, formula, selection_out)
    plot_combined()
    save_legend()
    with open(LATENCY_DIR / "cp_selection_latency.json", "w") as stream:
        json.dump(selection_out, stream, indent=2)
    print("Saved", LATENCY_DIR / "cp_selection_latency.json")


if __name__ == "__main__":
    main()