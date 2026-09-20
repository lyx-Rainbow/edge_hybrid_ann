#!/usr/bin/env python3
"""Render numbered candidate-ID figures for human manual selection.

Output directory:
    4_Results/fig_full/manual_candidates/

Each figure shows:
- all measured candidate points, labelled with their candidate id;
- the current automatic anchor curve in the method colour;
- the automatic anchor ids in bold.

Use the visible ids in 5_Plot/manual_latency_points.csv
(``use`` column) and then run apply_manual_latency_points.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = PLOT_DIR.parent
sys.path.insert(0, str(PLOT_DIR))

import fig_full_cp_by_predicate as FC  # noqa: E402
import fig_final_latency_at_recall as FR  # noqa: E402
import fig_full_sl_by_percentile as SL  # noqa: E402
import latency_trend as LT  # noqa: E402
import paper_style as PS  # noqa: E402

OUT_DIR = PROJ_ROOT / "4_Results/fig_full/manual_candidates"
N_ANCHOR = 9
OFFSETS = [(6, 4), (-16, 4), (6, -11), (-16, -11),
           (10, 9), (-20, -9), (10, -14), (-20, 14)]


def _annotate(axis, xs, ys, labels, color, fontsize=7, bold=False):
    for index, (x, y, label) in enumerate(zip(xs, ys, labels)):
        dx, dy = OFFSETS[index % len(OFFSETS)]
        axis.annotate(str(label), (x, y), textcoords="offset points",
                      xytext=(dx, dy), fontsize=fontsize, color=color,
                      fontweight=("bold" if bold else "normal"),
                      zorder=8)


def _finish(axis, title, xlabel, ylabel):
    axis.set_yscale("log")
    axis.grid(True, linewidth=2.0, alpha=0.28)
    axis.set_xlabel(xlabel, fontsize=16)
    axis.set_ylabel(ylabel, fontsize=16)
    axis.tick_params(axis="both", labelsize=12)
    axis.set_title(title, fontsize=14, fontweight="bold", pad=10)


def _save(figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_records(title, records, automatic_anchors, filename_stem,
                 method_key, xlabel="Recall@10", ylabel="Latency (ms)"):
    if not records:
        return
    xs = [record["recall"] for record in records]
    ys = [record["latency"] for record in records]
    ids = [record["id"] for record in records]
    style = PS.METHOD_STYLE[method_key]

    figure, axis = plt.subplots(1, 1, figsize=(14, 9))
    axis.scatter(xs, ys, s=36, facecolors="white", edgecolors="#777777",
                 linewidths=1.2, zorder=3)
    _annotate(axis, xs, ys, ids, color="#444444", fontsize=7)

    auto_x = [float(item[0]) for item in automatic_anchors]
    auto_y = [float(item[1]) for item in automatic_anchors]
    if auto_x:
        line_x, line_y = LT.smooth_line(automatic_anchors)
        axis.plot(line_x, line_y, color=style["color"], lw=2.8,
                  alpha=0.85, zorder=4)
        axis.plot(auto_x, auto_y, marker=style["marker"], linestyle="none",
                  ms=13, mec=style["color"], mew=2.0, mfc="none", zorder=6)
    auto_ids = []
    for anchor in automatic_anchors:
        best = min(records,
                   key=lambda record: (abs(record["recall"] - float(anchor[0]))
                                       + 1e-6 * abs(record["latency"]
                                                   - float(anchor[1]))))
        auto_ids.append(best["id"])
    if auto_ids:
        auto_lookup = {record["id"]: record for record in records}
        _annotate(axis,
                  [auto_lookup[i]["recall"] for i in auto_ids],
                  [auto_lookup[i]["latency"] for i in auto_ids],
                  auto_ids, color=style["color"], fontsize=8, bold=True)

    _finish(axis, title, xlabel, ylabel)
    _save(figure, OUT_DIR / filename_stem)


def plot_matched(title, points, filename_stem, method_key, current=None):
    if not points:
        return
    xs = [point["median_selectivity"] for point in points]
    ys = [point["latency_ms"] for point in points]
    ids = [point["id"] for point in points]
    style = PS.METHOD_STYLE[method_key]
    figure, axis = plt.subplots(1, 1, figsize=(14, 9))
    axis.scatter(xs, ys, s=36, facecolors="white", edgecolors="#777777",
                 linewidths=1.2, zorder=3)
    _annotate(axis, xs, ys, ids, color="#444444", fontsize=8)
    if current is None:
        current = list(zip(xs, ys))
    cur_x = [float(item[0]) for item in current]
    cur_y = [float(item[1]) for item in current]
    if len(cur_x) > 1:
        axis.plot(cur_x, cur_y, color=style["color"], lw=2.8, alpha=0.85,
                  zorder=4)
    axis.plot(cur_x, cur_y, marker=style["marker"], linestyle="none", ms=13,
              mec=style["color"], mew=2.0, mfc="none", zorder=6)
    cur_ids = []
    for x, y in zip(cur_x, cur_y):
        best = min(points,
                   key=lambda point: (abs(point["median_selectivity"] - x)
                                      + 1e-6 * abs(point["latency_ms"] - y)))
        cur_ids.append(best["id"])
    _annotate(axis, cur_x, cur_y, cur_ids, color=style["color"], fontsize=9,
              bold=True)
    axis.set_xscale("log")
    _finish(axis, title, "Median selectivity", "Latency @ Recall@10 (ms)")
    _save(figure, OUT_DIR / filename_stem)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for dataset in SL.DATASETS:
        for bucket in SL.BUCKETS:
            for method in SL.METHODS:
                points = SL.bucket_points(method, dataset, bucket)
                records = LT.candidate_records(points)
                if len(records) < 2:
                    continue
                manual = LT.get_override("sl", dataset, bucket, method)
                anchors = LT.build_trend(points, n_anchor=N_ANCHOR,
                                         manual=manual)
                plot_records(
                    f"SL  {dataset} / {bucket} / {method}",
                    records, anchors,
                    f"{dataset}_{bucket}_{method}",
                    method)
                count += 1

    for dataset, ptype, formula in FC.CP_CASES:
        for method in FC.METHODS:
            points = FC.load_cp_points(method, dataset, formula)
            records = LT.candidate_records(points)
            if len(records) < 2:
                continue
            manual = LT.get_override("cp", dataset, ptype, method)
            anchors = LT.build_trend(points, n_anchor=N_ANCHOR,
                                     manual=manual)
            plot_records(
                f"CP  {dataset} / {ptype} / {method}",
                records, anchors,
                f"cp_{dataset}_{ptype}_{method}",
                method)
            count += 1

    for target in FR.TARGETS:
        key = f"{int(round(target * 100)):03d}"
        for dataset in FR.DATASETS:
            for method in FR.METHODS:
                points = FR.method_points(dataset, target, method)
                if len(points) < 2:
                    continue
                override = LT.get_override("matched_recall", dataset, key,
                                           method)
                current = FR.apply_manual(points, override)
                plot_matched(
                    f"Latency@Recall>={target:.2f}  {dataset} / {method}",
                    points,
                    f"recall{key}_{dataset}_{method}",
                    method,
                    current=current)
                count += 1

    print(f"Wrote {count} candidate-ID figures to {OUT_DIR}")


if __name__ == "__main__":
    main()