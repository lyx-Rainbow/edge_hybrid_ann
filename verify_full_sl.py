#!/usr/bin/env python3
"""Verify the final SL curves: >=5 visible measured vertices and monotone QPS."""
import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.argv = [arg for arg in sys.argv if not arg.startswith("--")]
spec = importlib.util.spec_from_file_location(
    "fig", str(ROOT / "5_Plot/fig_full_sl_by_percentile.py"))
fig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig)

warnings = 0
print("dataset bucket method raw visible recall_range max_recall")
for ds in fig.DATASETS:
    for bucket in fig.BUCKETS:
        raw, curves = fig.build_curves(ds, bucket)
        for method in fig.METHODS:
            curve = curves.get(method, [])
            allpts = raw.get(method, [])
            if method == "prefilter":
                print(ds, bucket, method, len(allpts), len(curve), "exact", "-")
                continue
            if not curve:
                print(ds, bucket, method, len(allpts), 0, "-", "-")
                warnings += 1
                continue
            rng = f"{curve[0][0]:.3f}-{curve[-1][0]:.3f}"
            max_r = max((p[0] for p in allpts), default=0.0)
            print(ds, bucket, method, len(allpts), len(curve), rng, f"{max_r:.4f}")
            if len(curve) < 5:
                print(f"  WARN visible_curve_points={len(curve)} (<5)")
                warnings += 1
            recalls = [p[0] for p in curve]
            for i in range(len(curve) - 1):
                r0, q0, _ = curve[i]
                r1, q1, _ = curve[i + 1]
                if r1 <= r0:
                    print(f"  WARN recall_not_increasing {r0:.3f}->{r1:.3f}")
                    warnings += 1
                if q1 > q0 * 1.02 + 1e-12:
                    print(f"  WARN qps_not_monotone {r0:.3f}->{r1:.3f}")
                    warnings += 1
            gaps = [recalls[i + 1] - recalls[i] for i in range(len(recalls) - 1)]
            if gaps and max(gaps) > 0.28:
                print(f"  WARN sparse_recall_gap={max(gaps):.3f}")
                warnings += 1
print("warnings", warnings)
raise SystemExit(1 if warnings else 0)
