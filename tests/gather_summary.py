#!/usr/bin/env python3
"""Gather all experiment result numbers for the summary document."""
import json, importlib.util, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "5_Plot"))
sys.path.insert(0, str(ROOT))

def lm(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

fsl = lm("fsl", ROOT / "5_Plot/fig_sl_qps_recall_100k.py")
fcp = lm("fcp", ROOT / "5_Plot/fig_cp_qps_recall_100k.py")

METHODS = [("Curator", "Curator"), ("DiskIVF", "DiskIVF"), ("SPANN", "SPANN"), ("Pre-Filtering", "Pre-Filtering")]
DSS = ["sift1m_100k", "yfcc100m_100k", "arxiv_100k", "gist1m_100k", "wit_100k"]
IMETA = {"Curator": "Curator", "DiskIVF": "DiskIVF", "SPANN": "SPANN", "Pre-Filtering": "Pre-Filtering"}
MK = {"Curator": "curator", "DiskIVF": "diskivf", "SPANN": "spann", "Pre-Filtering": "prefilter"}

def fpts(kind, mk, ds):
    fig = fsl if kind == "SL" else fcp
    pts = fig.load_sweep(mk, ds) if kind == "SL" else fig.load_cp_points(mk, ds)
    if not pts: return None
    pts = sorted(pts)
    fx, fy, mq = [], [], 0
    for r, q in reversed(pts):
        if q > mq: mq = q; fx.append(r); fy.append(q)
    fx, fy = list(reversed(fx)), list(reversed(fy))
    return len(fx), fx[-1], fy[-1]

print("#### SL ####")
for name, _ in METHODS:
    for ds in DSS:
        r = fpts("SL", MK[name], ds)
        if r: print(f"{name} {ds} pts={r[0]} R={r[1]:.3f} QPS={r[2]:.0f}")
print("#### CP ####")
for name, _ in METHODS:
    for ds in DSS:
        r = fpts("CP", MK[name], ds)
        if r: print(f"{name} {ds} pts={r[0]} R={r[1]:.3f} QPS={r[2]:.0f}")

print("#### SWEEP combos + build ####")
for name, _ in METHODS:
    for ds in DSS:
        d = json.load(open(ROOT / f"4_Results/{IMETA[name]}/sweep_{ds}.json"))
        res = d["sweep_results"]
        bt = [rr.get("build_time_s", 0) for rr in res if rr.get("build_time_s")]
        print(f"{name} {ds} n={len(res)} build_max={max(bt) if bt else 0:.0f}s")

print("#### Memory RSS ####")
for name, _ in METHODS:
    for ds in DSS:
        d = json.load(open(ROOT / f"4_Results/{IMETA[name]}/sweep_{ds}.json"))
        rss = d.get("rss_peak_query_mb")
        if rss is None:
            rss = next((r.get("rss_peak_query_mb") for r in d.get("sweep_results", []) if r.get("rss_peak_query_mb")), None)
        idx = d.get("index_memory_mb")
        print(f"{name} {ds} rss={rss} idxmem={idx}")

cur = json.load(open(ROOT / "4_Results/memory_tuned/curator_v2.json"))
for ds in DSS:
    print(f"CuratorV2 {ds} rss={cur[ds]['rss_peak_query_mb']:.1f} idx={cur[ds]['index_total_bytes']/1e6:.1f}")
