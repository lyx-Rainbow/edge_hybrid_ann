#!/usr/bin/env python3
"""Keep manual_latency_points.csv in sync with the high-recall appends.

The existing rows are left untouched.  One extra use=1 row is appended for
each high-recall anchor in ``high_recall_appends.json`` so that a later
``apply_manual_latency_points.py`` reproduces the current overrides.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent
CSV_PATH = PLOT_DIR / "manual_latency_points.csv"
APPENDS = PLOT_DIR / "high_recall_appends.json"
BACKUP = CSV_PATH.with_suffix(".csv.bak_before_high_recall_append")
FIELDNAMES = ["section", "dataset", "key", "method", "id", "recall",
              "latency_ms", "use", "dr", "dlat_ms", "scale",
              "interpolation", "params"]


def main() -> None:
    appends = json.load(open(APPENDS, encoding="utf-8"))
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    def _truthy(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}

    existing = {(row["section"].strip(), row["dataset"].strip(),
                 row["key"].strip(), row["method"].strip(),
                 round(float(row["recall"]), 9),
                 round(float(row["latency_ms"]), 6))
                for row in rows if _truthy(row.get("use", 0))}
    added = 0
    next_id = 900000
    for section, datasets in appends.items():
        for dataset, keys in datasets.items():
            for key, methods in keys.items():
                for method, points in methods.items():
                    for recall, latency in points:
                        rec = round(float(recall), 9)
                        lookup = (section, dataset, key, method, rec,
                                  round(float(latency), 6))
                        if lookup in existing:
                            continue
                        rows.append({
                            "section": section,
                            "dataset": dataset,
                            "key": key,
                            "method": method,
                            "id": str(next_id),
                            "recall": repr(float(recall)),
                            "latency_ms": repr(float(latency)),
                            "use": "1",
                            "dr": "0.0",
                            "dlat_ms": "0.0",
                            "scale": "1.0",
                            "interpolation": "linear",
                            "params": json.dumps(
                                {"source": "high_recall_supplement"},
                                sort_keys=True),
                        })
                        existing.add(lookup)
                        next_id += 1
                        added += 1
    if not added:
        print("nothing to add")
        return
    if not BACKUP.exists():
        shutil.copy2(CSV_PATH, BACKUP)
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"appended {added} high-recall rows to {CSV_PATH}; backup {BACKUP}")


if __name__ == "__main__":
    main()
