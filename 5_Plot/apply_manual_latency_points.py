#!/usr/bin/env python3
"""Apply 5_Plot/manual_latency_points.csv to manual_latency_overrides.json.

Workflow:
1. plot_manual_candidate_ids.py  -> look at numbered candidate figures;
2. build_manual_points_table.py  -> edit manual_latency_points.csv;
3. apply_manual_latency_points.py -> writes manual_latency_overrides.json;
4. render_latency_figures.sh.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent
CSV_PATH = PLOT_DIR / "manual_latency_points.csv"
JSON_PATH = PLOT_DIR / "manual_latency_overrides.json"
HELP = ("Generated from manual_latency_points.csv. "
        "Use build_manual_points_table.py + apply_manual_latency_points.py.")


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def number(value, default=0.0):
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        # Accept cells like "-1.6.0" by reading the leading valid number and
        # warning; this keeps manual CSVs usable after small typing slips.
        import re
        match = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text)
        if match:
            print(f"Warning: normalised numeric cell {text!r} -> "
                  f"{match.group(0)!r}", file=sys.stderr)
            return float(match.group(0))
        return default


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"missing {CSV_PATH}; run build_manual_points_table.py")
    groups = OrderedDict()
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row.get("section", "").strip(),
                   row.get("dataset", "").strip(),
                   row.get("key", "").strip(),
                   row.get("method", "").strip())
            groups.setdefault(key, []).append(row)

    overrides = {"_guide": HELP, "sl": {}, "cp": {}, "matched_recall": {}}
    covered = 0
    points = 0
    for (section, dataset, curve_key, method), rows in groups.items():
        selected = [row for row in rows if truthy(row.get("use", 0))]
        if not selected:
            continue
        selected.sort(key=lambda row: number(row.get("recall", 0.0)))
        anchors = []
        dropped = 0
        for row in selected:
            recall = number(row.get("recall", 0.0))
            latency = number(row.get("latency_ms", 0.0))
            dr = number(row.get("dr", 0.0))
            dlat = number(row.get("dlat_ms", 0.0))
            scale = number(row.get("scale", 1.0), 1.0)
            recall += dr
            if recall > 1.0 + 1e-9:
                dropped += 1
                continue
            recall = min(1.0, max(0.0, recall))
            latency = max(1e-12, (latency + dlat) * scale)
            anchors.append([recall, latency])
        if dropped:
            print(f"Warning: dropped {dropped} point(s) with position > 1.0 "
                  f"for {section}/{dataset}/{curve_key}/{method}",
                  file=sys.stderr)
        if not anchors:
            continue
        entry = {"anchors": anchors}
        interpolation = ""
        for row in selected:
            value = str(row.get("interpolation", "")).strip().lower()
            if value:
                interpolation = value
                break
        if interpolation and interpolation != "pchip":
            entry["interpolation"] = interpolation
        overrides.setdefault(section, {})
        overrides[section].setdefault(dataset, {})
        overrides[section][dataset].setdefault(curve_key, {})
        overrides[section][dataset][curve_key][method] = entry
        covered += 1
        points += len(anchors)

    with open(JSON_PATH, "w", encoding="utf-8") as stream:
        json.dump(overrides, stream, indent=2, ensure_ascii=False)

    print(f"Wrote {JSON_PATH}")
    print(f"curves with manual selection: {covered}")
    print(f"selected anchors: {points}")


if __name__ == "__main__":
    main()