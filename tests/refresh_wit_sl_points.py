#!/usr/bin/env python3
"""Refresh wit SL manual-table x-coordinates after the tie-aware recall fix.

Only the base ``recall`` of selected (``use=1``) wit SL rows is updated by
matching each row's ``params`` to the newly computed candidate pool.  The
latency and all manual adjustments (``dr``, ``dlat_ms``, ``scale``,
``interpolation``, ``use``) are preserved.  ``matched_recall`` rows for wit
are removed so that the matched-recall figures derive from the updated SL
curves automatically.

Run:
    python tests/refresh_wit_sl_points.py
Then:
    python 5_Plot/apply_manual_latency_points.py
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = ROOT / "5_Plot"
CSV_PATH = PLOT_DIR / "manual_latency_points.csv"
BACKUP_PATH = CSV_PATH.with_suffix(".csv.bak_before_tieaware")

sys.path.insert(0, str(PLOT_DIR))
import fig_full_sl_by_percentile as SL  # noqa: E402


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def main() -> None:
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)

    cache = {}

    def points_for(bucket: str, method: str):
        key = (bucket, method)
        if key not in cache:
            cache[key] = SL.bucket_points(method, "wit", bucket)
        return cache[key]

    updated = unmatched = removed = 0
    output_rows = []
    for row in rows:
        if row.get("section") == "matched_recall" and row.get("dataset") == "wit":
            removed += 1
            continue
        if (row.get("section") == "sl" and row.get("dataset") == "wit"
                and truthy(row.get("use", 0))):
            try:
                old_params = json.loads(row.get("params") or "{}")
            except Exception:
                old_params = {}
            match = None
            for point in points_for(row.get("key", ""), row.get("method", "")):
                params = point[2] if len(point) > 2 else {}
                if params == old_params:
                    match = point
                    break
            if match is None:
                unmatched += 1
            else:
                row["recall"] = format(float(match[0]), ".15g")
                updated += 1
        output_rows.append(row)

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"updated selected wit SL rows: {updated}")
    print(f"unmatched selected wit SL rows: {unmatched}")
    print(f"removed wit matched_recall rows: {removed}")
    print(f"backup: {BACKUP_PATH}")


if __name__ == "__main__":
    main()
