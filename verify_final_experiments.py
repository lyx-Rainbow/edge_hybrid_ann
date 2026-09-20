#!/usr/bin/env python3
"""Final version/storage audit for the full-scale experiments.

Checks:
  * every SL/CP sweep was generated from the unified ext4/index version
  * every non-PreFilter curve has >=5 visible measured vertices in [0.5, 1.0]
  * selected curves have monotonically non-increasing QPS
  * the ext4 migration manifest and index directories exist
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "5_Plot"))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

fig_sl = load("fig_sl", ROOT / "5_Plot/fig_full_sl_by_percentile.py")
fig_cp = load("fig_cp", ROOT / "5_Plot/fig_full_cp_by_predicate.py")

errors = []
warnings = []

# ---------------------------------------------------------------------------
# Storage manifest
# ---------------------------------------------------------------------------
manifest_path = ROOT / "4_Results/_validation_selectivity/index_migration_manifest.json"
if not manifest_path.exists():
    errors.append("missing index_migration_manifest.json")
else:
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    for ds, record in manifest.get("datasets", {}).items():
        for key in ("spann_dst", "diskivf_dst"):
            path = Path(record.get(key, ""))
            if not path.exists():
                errors.append(f"manifest {ds} missing {key}: {path}")

# ---------------------------------------------------------------------------
# SL
# ---------------------------------------------------------------------------
for ds in fig_sl.DATASETS:
    raw, curves = None, None
    for method in fig_sl.METHODS:
        if method == "prefilter":
            continue
        for bucket in fig_sl.BUCKETS:
            _, curves = fig_sl.build_curves(ds, bucket)
            curve = curves.get(method, [])
            if len(curve) < 5:
                errors.append(f"SL {ds}/{bucket}/{method}: only {len(curve)} visible points")
            for i in range(len(curve) - 1):
                r0, q0, _ = curve[i]
                r1, q1, _ = curve[i + 1]
                if r1 <= r0:
                    errors.append(f"SL {ds}/{bucket}/{method}: recall not increasing")
                if q1 > q0 * 1.02 + 1e-12:
                    errors.append(f"SL {ds}/{bucket}/{method}: QPS not monotone")
    # storage marke
    for method, sub in (("curator", "Curator"), ("diskivf", "DiskIVF"), ("spann", "SPANN")):
        path = ROOT / "4_Results" / sub / f"sweep_{ds}.json"
        if not path.exists():
            errors.append(f"missing SL sweep {path}")
            continue
        data = json.load(open(path, encoding="utf-8"))
        if data.get("index_storage") != "wsl_ext4":
            errors.append(f"SL {ds}/{method}: index_storage={data.get('index_storage')!r}")

# ---------------------------------------------------------------------------
# CP
# ---------------------------------------------------------------------------
for dataset, ptype, formula in fig_cp.CP_CASES:
    raw = {method: fig_cp.load_cp_points(method, dataset, formula)
           for method in fig_cp.METHODS}
    curves = fig_cp.build_curves(raw)
    for method in fig_cp.METHODS:
        if method == "prefilter":
            continue
        curve = curves.get(method, [])
        if len(curve) < 5:
            errors.append(f"CP {dataset}/{ptype}/{method}: only {len(curve)} visible points")
        for i in range(len(curve) - 1):
            r0, q0, _ = curve[i]
            r1, q1, _ = curve[i + 1]
            if r1 <= r0:
                errors.append(f"CP {dataset}/{ptype}/{method}: recall not increasing")
            if q1 > q0 * 1.02 + 1e-12:
                errors.append(f"CP {dataset}/{ptype}/{method}: QPS not monotone")
    path = ROOT / "4_Results/Curator" / f"sweep_{dataset}_cp.json"
    if not path.exists():
        errors.append(f"missing CP sweep {path}")
        continue
    data = json.load(open(path, encoding="utf-8"))
    if data.get("index_storage") != "wsl_ext4":
        errors.append(f"CP {dataset}: index_storage={data.get('index_storage')!r}")

print("errors", len(errors), "warnings", len(warnings))
for item in errors:
    print("ERROR:", item)
for item in warnings:
    print("WARN:", item)
raise SystemExit(1 if errors else 0)
