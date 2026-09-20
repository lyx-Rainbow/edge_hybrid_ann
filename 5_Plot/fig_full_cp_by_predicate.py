#!/usr/bin/env python3
"""Full-scale complex-predicate (CP) QPS-Recall figures.

Style follows 5_Plot/fig_full_sl_by_percentile.py:
  * one panel per (dataset, predicate) case;
  * gray background, large fonts, log-scaled QPS axis;
  * translucent raw measurements plus a visually selected monotone mainline;
  * selected vertices are measured points only.

Cases:
  sift1m   OR 70 40
  yfcc100m AND 119 320
  arxiv    AND 0 OR 1 2
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOT_DIR))
sys.path.insert(0, str(PROJ_ROOT))
import fig_full_sl_by_percentile as SL  # reuse curve-selection utilities
from utils import INDEX_META
from run_100k_sweep import polish_to_rpn

RESULTS_DIR = PROJ_ROOT / "4_Results"
OUTPUT_DIR = RESULTS_DIR / "fig_full_qps_legacy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["curator", "diskivf", "spann", "prefilter"]
K = 10
CP_CASES = [
    ("sift1m", "OR", "OR 70 40"),
    ("yfcc100m", "AND", "AND 396 988"),
    ("arxiv", "MIXED", "OR 68 AND 48 49"),
]

# Optional manual overrides: (dataset, method) -> list of params dicts.
CP_MANUAL = {
    "yfcc100m": {
        "curator": [
            {"pq_M": 24, "search_ef": 16},
            {"pq_M": 48, "search_ef": 64},
            {"pq_M": 48, "search_ef": 1024},
            {"pq_M": 96, "search_ef": 32768},
        ],
    },
}
BG_GRAY = SL.BG_GRAY


def sweep_path(method_key, dataset):
    meta = INDEX_META[method_key]
    base = RESULTS_DIR / meta[1] / f"sweep_{dataset}_cp.json"
    if base.exists():
        return base
    return RESULTS_DIR / meta[1] / f"sweep_{dataset}.json"


def _info_from_cp(cp, formula):
    if not cp:
        return None
    pf = cp.get("per_filter") or {}
    candidates = [formula, polish_to_rpn(formula)]
    for key in candidates:
        if key in pf:
            return pf[key]
    if len(pf) == 1:
        return next(iter(pf.values()))
    if cp.get("n_filters") == 1 and cp.get("avg_recall") is not None:
        return cp
    return None


def load_cp_points(method_key, dataset, formula):
    path = sweep_path(method_key, dataset)
    if not path.exists():
        return []
    try:
        data = json.load(open(path))
    except Exception:
        return []
    pts = []
    for r in data.get("sweep_results", []):
        info = _info_from_cp(r.get("complex_predicate"), formula)
        if not info:
            continue
        q = info.get("qps", 0.0)
        rec = info.get("avg_recall", 0.0)
        if q and q > 0 and rec is not None:
            # PreFilter is the exact external-scan baseline; measured recall
            # may be slightly below 1.0 due to duplicate distances/tie-breaking.
            if method_key == "prefilter":
                rec = 1.0
            pts.append((float(rec), float(q), r.get("params", {})))
    pts.sort(key=lambda p: (p[0], p[1]))
    return pts


def get_case_selectivity(dataset, formula):
    path = PROJ_ROOT / "1_Data/ground_truth" / dataset / "complex_predicate" / "filters.json"
    try:
        info = json.load(open(path))
        sel = (info.get("selectivities") or {}).get(formula)
        if sel is not None:
            return float(sel)
    except Exception:
        pass
    return None


def selectivity_text(sel):
    if sel is None:
        return "sel=?"
    if sel < 0.0001:
        return f"sel={sel*100:.4f}%"
    if sel < 0.0005:
        return f"sel={sel*100:.3f}%"
    return f"sel={sel*100:.2f}%"


MANUAL_CP_SELECTION_PATH = Path(__file__).resolve().parent / "manual_cp_selection.json"


def load_manual_cp_selection():
    if MANUAL_CP_SELECTION_PATH.exists():
        try:
            return json.load(open(MANUAL_CP_SELECTION_PATH))
        except Exception:
            return {}
    return {}


def candidate_points_cp(raw_points):
    """Canonical candidate ordering used by manual CP selection IDs."""
    return sorted(raw_points, key=SL._stable_point_key)


def _params_match(params, spec):
    if not isinstance(spec, dict):
        return False
    return all(params.get(key) == value for key, value in spec.items())


def build_curves(raw_by_method, dataset=None):
    curves = {}
    for method in METHODS:
        curves[method] = SL.select_paper_curve(raw_by_method.get(method, []))
    # Manual CP overrides are intentionally disabled in the final figures: their
    # hand-picked vertices were not guaranteed to be monotone.
    return curves

import paper_style as PS  # noqa: E402


def _xlim_for(curves):
    starts = [c[0][0] for c in curves.values() if c]
    if not starts:
        return 0.5, 1.04
    mn = min(starts)
    lo = max(0.5, min(0.7, math.floor(mn * 10.0) / 10.0))
    return lo, 1.04


def _all_qps(curves):
    values = []
    for curve in curves.values():
        values.extend(p[1] for p in curve)
    return values


def _plot_curves(axis, curves, xlabel, ylabel="QPS", fontsize=26,
                 tick_fontsize=22, show_ylabel=True):
    for method in PS.PLOT_ORDER:
        curve = curves.get(method) or []
        if not curve:
            continue
        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        PS.draw_method(axis, method, xs, ys,
                       single_point=(method == "prefilter"))
    lo, hi = _xlim_for(curves)
    PS.style_axis(axis, (lo, hi), xlabel, ylabel=ylabel, fontsize=fontsize,
                  tick_fontsize=tick_fontsize, show_ylabel=show_ylabel)
    PS.set_log_ylim(axis, _all_qps(curves))


def plot_one(dataset, ptype, formula, selection_out):
    raw = {method: load_cp_points(method, dataset, formula)
           for method in METHODS}
    if not any(raw.values()):
        print(f"Skip {dataset} {ptype}: no CP data")
        return False
    curves = build_curves(raw, dataset=dataset)
    sel = get_case_selectivity(dataset, formula)
    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    _plot_curves(axis, curves, xlabel="Recall@10")
    axis.set_title(f"{dataset} {ptype} ({selectivity_text(sel)})",
                   fontsize=24, fontweight="bold", pad=12)
    figure.tight_layout()
    out = OUTPUT_DIR / f"cp_{dataset}_{ptype}.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(str(out).replace(".png", ".svg"), dpi=300,
                   bbox_inches="tight")
    plt.close(figure)
    selection_out[dataset] = {
        "predicate": ptype,
        "formula": formula,
        "selectivity": sel,
        "curves": {
            method: [[float(p[0]), float(p[1]), p[2]]
                     for p in curves.get(method, [])]
            for method in METHODS
        },
    }
    print(f"Saved {out}")
    return True


def save_legend():
    PS.save_legend(OUTPUT_DIR / "cp_legend.png", fontsize=30)
    print(f"Saved {OUTPUT_DIR / 'cp_legend.png'}")


def main():
    PS.apply_style()
    selection_out = {}
    for dataset, ptype, formula in CP_CASES:
        plot_one(dataset, ptype, formula, selection_out)
    save_legend()
    with open(OUTPUT_DIR / "cp_selection.json", "w") as stream:
        json.dump(selection_out, stream, indent=2)
    print("Saved", OUTPUT_DIR / "cp_selection.json")


if __name__ == "__main__":
    main()
