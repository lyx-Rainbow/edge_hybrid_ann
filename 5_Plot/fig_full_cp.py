#!/usr/bin/env python3
"""
Full-scale complex-predicate QPS-Recall figures.

Expected data: 4_Results/<Method>/sweep_<dataset>_cp.json with entries that
contain 'complex_predicate' with 'avg_recall' and 'qps'.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "5_Plot"))
from utils import INDEX_META

RESULTS_DIR = PROJ_ROOT / "4_Results"
OUTPUT_DIR = PROJ_ROOT / "4_Results/fig_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# dataset -> (predicate display, filter formula used, CP data file suffix)
CP_CASES = [
    ("sift1m", "OR", "OR 70 40"),
    ("yfcc100m", "AND", "AND 119 320"),
    ("arxiv", "MIXED", "AND 0 OR 1 2"),
]
METHODS = ["curator", "diskivf", "spann", "prefilter"]
K = 10


def load_cp_data(method_key, dataset):
    path = RESULTS_DIR / INDEX_META[method_key][1] / f"sweep_{dataset}_cp.json"
    if not path.exists():
        path = RESULTS_DIR / INDEX_META[method_key][1] / f"sweep_{dataset}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    pts = []
    for r in data.get("sweep_results", []):
        cp = r.get("complex_predicate")
        if cp and cp.get("qps", 0) > 0 and cp.get("avg_recall", 0) > 0:
            pts.append((cp["avg_recall"], cp["qps"]))
    pts.sort()
    return pts


def pareto(points):
    if not points:
        return [], []
    fx, fy = [], []
    max_q = 0
    for r, q in reversed(points):
        if q > max_q:
            max_q = q
            fx.append(r)
            fy.append(q)
    return list(reversed(fx)), list(reversed(fy))


def monotone_binned(points, n_bins=20):
    if not points:
        return []
    pts = sorted(points)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    lo, hi = float(xs.min()), float(xs.max())
    if hi <= lo:
        return [(lo, float(np.percentile(ys, 25)))]
    edges = np.linspace(lo, hi, n_bins + 1)
    bx, by = [], []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (xs >= edges[i]) & (xs <= edges[i + 1] + 1e-12)
        else:
            mask = (xs >= edges[i]) & (xs < edges[i + 1])
        if mask.any():
            bx.append(float(xs[mask].mean()))
            by.append(float(np.percentile(ys[mask], 25)))
    bx = np.array(bx)
    by = np.array(by)
    by = np.minimum.accumulate(by)
    out = []
    for x, y in zip(bx, by):
        if not out or abs(out[-1][0] - x) > 1e-9 or abs(out[-1][1] - y) > 1e-9:
            out.append((x, y))
    return out


def get_selectivity(dataset, formula):
    try:
        info = json.load(open(PROJ_ROOT / "1_Data/ground_truth" / dataset / "complex_predicate" / "filters.json"))
        return info.get("selectivities", {}).get(formula)
    except Exception:
        return None


def main():
    for dataset, ptype, formula in CP_CASES:
        all_pts = []
        for mk in METHODS:
            all_pts.extend(load_cp_data(mk, dataset) or [])
        if not all_pts:
            print(f"Skip {dataset} {ptype}: no CP data", flush=True)
            continue
        sel = get_selectivity(dataset, formula)
        sel_text = f"sel≈{sel*100:.2f}%" if sel is not None else "sel=?"
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        all_r = []
        for mk in METHODS:
            pts = load_cp_data(mk, dataset)
            if pts:
                all_r.extend(p[0] for p in pts)
        lo = 0.0
        if all_r:
            lo = np.floor(min(all_r) * 20) / 20 - 0.02
            lo = max(0.0, lo)
        ax.set_xlim(lo, 1.005)
        ax.set_yscale("log")
        ax.set_xlabel(f"CP Recall@{K}", fontsize=12)
        ax.set_ylabel("CP QPS", fontsize=12)
        ax.set_title(f"{dataset}  {ptype}  {sel_text}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=1.0)
        for mk in METHODS:
            pts = load_cp_data(mk, dataset)
            if not pts:
                continue
            name, _, color, marker = INDEX_META[mk]
            mp = monotone_binned(pts)
            if len(mp) == 1:
                ax.plot([mp[0][0]], [mp[0][1]], color=color, marker=marker,
                        ms=10, lw=0, mec=color, mfc="none", label=name, zorder=3)
            elif mp:
                mx = [p[0] for p in mp]
                my = [p[1] for p in mp]
                ax.plot(mx, my, color=color, marker=marker, lw=2.2, ms=5,
                        mec=color, mfc="none", label=name, zorder=3)
        ax.legend(loc="best", fontsize=9, framealpha=0.8)
        plt.tight_layout()
        out = OUTPUT_DIR / f"cp_{dataset}_{ptype}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        fig.savefig(str(out).replace(".png", ".svg"), bbox_inches="tight")
        plt.close(fig)
        print("Saved", out)


if __name__ == "__main__":
    main()
