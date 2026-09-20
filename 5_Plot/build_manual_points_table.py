#!/usr/bin/env python3
"""Build the human-editable manual latency points spreadsheet.

Output: 5_Plot/manual_latency_points.csv

The script computes the candidate point pool and the current automatic anchors
directly from the experiment data.  If the CSV already exists, user-editable
fields (use / dr / dlat_ms / scale / interpolation and coordinates) are
preserved for rows with the same curve + candidate id.  Pass ``--reset`` to
regenerate all rows from fresh defaults.

Columns:
  section,dataset,key,method,id,recall,latency_ms,use,dr,dlat_ms,scale,
  interpolation,params
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOT_DIR))

import fig_full_cp_by_predicate as FC  # noqa: E402
import fig_final_latency_at_recall as FR  # noqa: E402
import fig_full_sl_by_percentile as SL  # noqa: E402
import latency_trend as LT  # noqa: E402

N_ANCHOR = 9
OUTPUT_PATH = PLOT_DIR / "manual_latency_points.csv"
FIELDNAMES = ["section", "dataset", "key", "method", "id", "recall",
              "latency_ms", "use", "dr", "dlat_ms", "scale",
              "interpolation", "params"]
PRESERVE_FIELDS = ["recall", "latency_ms", "use", "dr", "dlat_ms", "scale",
                   "interpolation"]


def automatic_ids(records, anchors):
    ids = []
    for anchor in anchors:
        recall, latency = float(anchor[0]), float(anchor[1])
        best = None
        best_score = float("inf")
        for record in records:
            score = (abs(record["recall"] - recall)
                     + abs(record["latency"] - latency) * 1e-6)
            if score < best_score:
                best_score = score
                best = record
        if best is not None:
            ids.append(best["id"])
    return ids


def iter_curves():
    for dataset in SL.DATASETS:
        for bucket in SL.BUCKETS:
            for method in SL.METHODS:
                points = SL.bucket_points(method, dataset, bucket)
                records = LT.candidate_records(points)
                if not records:
                    continue
                anchors = LT.build_trend(points, n_anchor=N_ANCHOR)
                yield ("sl", dataset, bucket, method, records, anchors,
                       automatic_ids(records, anchors))

    for dataset, ptype, formula in FC.CP_CASES:
        for method in FC.METHODS:
            points = FC.load_cp_points(method, dataset, formula)
            records = LT.candidate_records(points)
            if not records:
                continue
            anchors = LT.build_trend(points, n_anchor=N_ANCHOR)
            yield ("cp", dataset, ptype, method, records, anchors,
                   automatic_ids(records, anchors))

    for target in FR.TARGETS:
        key = f"{target:.2f}"
        for dataset in FR.DATASETS:
            for method in FR.METHODS:
                points = FR.method_points(dataset, target, method)
                if not points:
                    continue
                records = [{
                    "id": point["id"],
                    "recall": point["median_selectivity"],
                    "latency": point["latency_ms"],
                    "qps": 0.0,
                    "params": {"bucket": point["bucket"]},
                } for point in points]
                anchors = [[point["median_selectivity"],
                            point["latency_ms"]] for point in points]
                yield ("matched_recall", dataset, key, method, records,
                       anchors, [point["id"] for point in points])


def load_previous_rows():
    if not OUTPUT_PATH.exists():
        return {}
    previous = {}
    with open(OUTPUT_PATH, "r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row.get("section", "").strip(),
                   row.get("dataset", "").strip(),
                   row.get("key", "").strip(),
                   row.get("method", "").strip(),
                   row.get("id", "").strip())
            previous[key] = {field: row.get(field, "")
                             for field in PRESERVE_FIELDS}
    return previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="ignore the existing CSV and regenerate defaults")
    args = parser.parse_args()

    previous = {} if args.reset else load_previous_rows()
    preserved = 0
    rows = []

    for section, dataset, key, method, records, anchors, auto_ids in iter_curves():
        automatic = set(auto_ids)
        auto_coords = {}
        for index, auto_id in enumerate(auto_ids):
            if index < len(anchors):
                auto_coords[auto_id] = anchors[index]
        for candidate in records:
            recall = candidate["recall"]
            latency = candidate.get("latency", candidate.get("latency_ms"))
            if candidate["id"] in auto_coords:
                anchor = auto_coords[candidate["id"]]
                recall, latency = float(anchor[0]), float(anchor[1])
            row = {
                "section": section,
                "dataset": dataset,
                "key": key,
                "method": method,
                "id": candidate["id"],
                "recall": recall,
                "latency_ms": latency,
                "use": 1 if candidate["id"] in automatic else 0,
                "dr": 0.0,
                "dlat_ms": 0.0,
                "scale": 1.0,
                "interpolation": "pchip",
                "params": json.dumps(candidate.get("params", {}),
                                     sort_keys=True, ensure_ascii=False,
                                     default=str),
            }
            old_key = (str(section), str(dataset), str(key), str(method),
                       str(candidate["id"]))
            old = previous.get(old_key)
            if old:
                for field in PRESERVE_FIELDS:
                    value = str(old.get(field, "")).strip()
                    if value != "":
                        row[field] = value
                preserved += 1
            rows.append(row)

    rows.sort(key=lambda row: (row["section"], row["dataset"], row["key"],
                               row["method"], row["recall"]))
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"rows: {len(rows)}  preserved existing rows: {preserved}")
    if args.reset:
        print("Reset mode: all rows are fresh defaults")
    print("Set use=1/0, then run apply_manual_latency_points.py")


if __name__ == "__main__":
    main()