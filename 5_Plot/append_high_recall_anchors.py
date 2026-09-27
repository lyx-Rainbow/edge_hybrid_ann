#!/usr/bin/env python3
"""Append selected high-recall measured anchors to the manual line overrides.

The existing manual anchors are kept byte-for-byte in value; only new points
strictly to the right of a curve are appended.  The input file
``high_recall_appends.json`` has the form::

  {"sl": {"arxiv": {"99p": {"curator": [[recall, latency_ms], ...]}}},
   "cp": {"yfcc100m": {"AND": {"spann": [[...], ...]}}}}

Coordinates are final anchor positions (already in ms), not raw QPS values.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent
ROOT = PLOT_DIR.parent
OVERRIDES = PLOT_DIR / "manual_latency_overrides.json"
APPENDS = PLOT_DIR / "high_recall_appends.json"
BACKUP = OVERRIDES.with_suffix(".json.bak_before_high_recall_append")


def main() -> None:
    if not APPENDS.exists():
        raise SystemExit(f"missing {APPENDS}")
    appends = json.load(open(APPENDS, encoding="utf-8"))
    overrides = json.load(open(OVERRIDES, encoding="utf-8-sig"))
    changed = 0
    for section, datasets in appends.items():
        for dataset, keys in datasets.items():
            for key, methods in keys.items():
                for method, points in methods.items():
                    entry = overrides.get(section, {}).get(dataset, {}).get(key, {}).get(method)
                    if not entry or "anchors" not in entry:
                        print(f"WARN missing override {section}/{dataset}/{key}/{method}",
                              file=sys.stderr)
                        continue
                    old = [[float(x[0]), float(x[1])] for x in entry["anchors"]]
                    old_max = max(x[0] for x in old)
                    new = [[float(x[0]), float(x[1])] for x in points]
                    # Keep only points to the right of the current curve.
                    new = [x for x in new if x[0] > old_max + 1e-9]
                    if not new:
                        print(f"WARN no new points right of old max for "
                              f"{section}/{dataset}/{key}/{method}")
                        continue
                    combined = old + new
                    combined.sort(key=lambda x: (x[0], x[1]))
                    # Deduplicate identical recall coordinates; old point wins.
                    deduped = []
                    for x in combined:
                        if deduped and abs(x[0] - deduped[-1][0]) < 1e-9:
                            continue
                        deduped.append(x)
                    entry["anchors"] = deduped
                    changed += 1
                    print(f"{section}/{dataset}/{key}/{method}: "
                          f"{len(old)} -> {len(deduped)} anchors, "
                          f"max recall {old_max:.4f} -> {deduped[-1][0]:.4f} "
                          f"latency {deduped[-1][1]:.3f} ms")
    if not changed:
        raise SystemExit("nothing changed")
    if not BACKUP.exists():
        shutil.copy2(OVERRIDES, BACKUP)
    with open(OVERRIDES, "w", encoding="utf-8") as stream:
        json.dump(overrides, stream, indent=2, ensure_ascii=False)
    print(f"updated {OVERRIDES}; backup {BACKUP}")
    print(f"changed curves: {changed}")


if __name__ == "__main__":
    main()
