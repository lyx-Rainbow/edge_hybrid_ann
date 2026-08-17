#!/usr/bin/env python3
"""Frontier report tool: per method x dataset, print SL and CP Pareto
frontier point counts, left/right ends (recall, qps, params) and whether
the right end reaches recall >= 0.95.

Usage:
    python tests/report_frontier.py            # all methods x datasets
    python tests/report_frontier.py --method DiskIVF
"""
import argparse, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "5_Plot"))

METHODS = ["curator", "diskivf", "spann", "prefilter"]
DATASETS = ["sift1m_100k", "yfcc100m_100k", "arxiv_100k",
            "gist1m_100k", "wit_100k"]


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fsl = load_mod("fsl", ROOT / "5_Plot/fig_sl_qps_recall_100k.py")
fcp = load_mod("fcp", ROOT / "5_Plot/fig_cp_qps_recall_100k.py")


def frontier_with_params(points, params_map):
    """points: [(recall, qps)] sorted; params_map: (recall,qps)->params."""
    fx, fy, fparams = [], [], []
    max_qps = 0
    for r, q in reversed(points):
        if q > max_qps:
            max_qps = q
            fx.append(r)
            fy.append(q)
            fparams.append(params_map.get((r, q), {}))
    return list(reversed(fx)), list(reversed(fy)), list(reversed(fparams))


def report(method_key, dataset, kind):
    if kind == "SL":
        pts = fsl.load_sweep(method_key, dataset)
    else:
        pts = fcp.load_cp_points(method_key, dataset)
    if not pts:
        return f"no data"
    pts = sorted(pts)
    fx, fy, fp = frontier_with_params(pts, {})
    right = fx[-1] if fx else 0.0
    left = fx[0] if fx else 0.0
    flag = "OK" if right >= 0.95 else "LOW"
    return (f"{len(fx)}pts L=({left:.3f},{fy[0]:.0f}) "
            f"R=({right:.3f},{fy[-1]:.0f}) [{flag}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None)
    args = ap.parse_args()
    methods = [args.method] if args.method else METHODS

    for kind, fig in (("SL", fsl), ("CP", fcp)):
        print(f"\n===== {kind} frontier report =====")
        print(f"{'method':<10}{'dataset':<16}{'report'}")
        for mk in methods:
            for ds in DATASETS:
                print(f"{mk:<10}{ds:<16}{report(mk, ds, kind)}")


if __name__ == "__main__":
    main()
