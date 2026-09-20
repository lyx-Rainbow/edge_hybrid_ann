#!/usr/bin/env python3
"""Combined 1x3 CP QPS--Recall figure in the paper/reference style."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOT_DIR))
import fig_full_cp_by_predicate as FC  # noqa: E402
import paper_style as PS  # noqa: E402

OUTPUT_DIR = FC.OUTPUT_DIR


def draw_panel(axis, dataset, ptype, formula, show_ylabel=True):
    raw = {method: FC.load_cp_points(method, dataset, formula)
           for method in FC.METHODS}
    if not any(raw.values()):
        axis.axis("off")
        return
    curves = FC.build_curves(raw, dataset=dataset)
    for method in PS.PLOT_ORDER:
        curve = curves.get(method) or []
        if not curve:
            continue
        PS.draw_method(
            axis, method,
            [p[0] for p in curve], [p[1] for p in curve],
            single_point=(method == "prefilter"))
    sel = FC.get_case_selectivity(dataset, formula)
    lo, hi = FC._xlim_for(curves)
    PS.style_axis(
        axis, (lo, hi), "Recall@10",
        ylabel="QPS" if show_ylabel else None,
        fontsize=20, tick_fontsize=16, show_ylabel=show_ylabel)
    axis.set_title(f"{dataset} {ptype} ({FC.selectivity_text(sel)})",
                   fontsize=18, fontweight="bold", pad=8)
    PS.set_log_ylim(axis, FC._all_qps(curves))


def main():
    PS.apply_style()
    n = len(FC.CP_CASES)
    figure, axes = plt.subplots(1, n, figsize=(8.0 * n, 7.0))
    if n == 1:
        axes = [axes]
    for i, (axis, (dataset, ptype, formula)) in enumerate(
            zip(axes, FC.CP_CASES)):
        draw_panel(axis, dataset, ptype, formula, show_ylabel=(i == 0))
    figure.legend(PS.method_handles(), [PS.METHOD_LABELS[k]
                                        for k in PS.LEGEND_ORDER],
                  loc="lower center", ncol=4, frameon=False, fontsize=22,
                  markerscale=2.0, bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUTPUT_DIR / "cp_combined.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(str(out).replace(".png", ".svg"), dpi=300,
                   bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()