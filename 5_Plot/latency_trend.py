"""Measured-anchor Latency-Recall curves with manual selection/adjustment.

The final line figures keep the measured non-linear shape.  This module now
also provides two manual mechanisms:

1. manual point selection by candidate id
   (`selected_ids` in manual_latency_overrides.json);
2. manual fine-position adjustment
   (`adjustments` in the same file), applied to either the selected ids or the
   automatic anchors.

An explicit `anchors` list is also accepted as a convenience, and
`interpolation` may be `"pchip"` (default) or `"linear"` for debugging.
"""
from __future__ import annotations

import sys
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:  # pragma: no cover
    PchipInterpolator = None

import paper_style as PS

XMIN_DEFAULT = 0.50
N_ANCHOR_DEFAULT = 9
MAX_LATENCY_RATIO_DEFAULT = 2.5
YLABEL_DEFAULT = "Latency (ms)"
MANUAL_OVERRIDES_PATH = (Path(__file__).resolve().parent
                         / "manual_latency_overrides.json")


@dataclass
class _Anchor:
    recall: float
    latency: float


def load_manual_overrides():
    """Load 5_Plot/manual_latency_overrides.json; return empty structure on error."""
    try:
        with open(MANUAL_OVERRIDES_PATH, "r", encoding="utf-8-sig") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            return {"sl": {}, "cp": {}}
        return data
    except Exception:
        return {"sl": {}, "cp": {}}


def get_override(section: str, dataset: str, key: str, method: str):
    """Return the manual override dict for one curve, or {} if none."""
    data = load_manual_overrides()
    return (data.get(section, {})
                .get(dataset, {})
                .get(key, {})
                .get(method, {}) or {})


def candidate_records(points: Iterable[Sequence], xmin: float = XMIN_DEFAULT):
    """Stable candidate list for manual selection (ids are list indices)."""
    records = []
    seen = set()
    for point in points:
        recall, qps = float(point[0]), point[1]
        if qps is None or float(qps) <= 0 or recall < xmin - 1e-12:
            continue
        latency = 1000.0 / float(qps)
        params = point[2] if len(point) > 2 else {}
        try:
            params_key = json.dumps(params, sort_keys=True, default=str)
        except Exception:
            params_key = str(params)
        key = (round(recall, 12), round(math.log(latency), 12), params_key)
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "id": 0,
            "recall": recall,
            "latency": latency,
            "qps": float(qps),
            "params": params,
        })
    records.sort(key=lambda item: (item["recall"], item["latency"],
                                   json.dumps(item["params"], sort_keys=True,
                                              default=str)))
    for index, record in enumerate(records):
        record["id"] = index
    return records


def _anchors_from_records(records) -> list:
    return [_Anchor(float(item["recall"]), float(item["latency"]))
            for item in records]


def _base_anchors(points, max_points: int):
    """Use the existing tuned measured-point selector when available."""
    try:
        import fig_full_sl_by_percentile as SL
        base = SL.select_paper_curve(points, min_points=5,
                                     max_points=max_points)
        return [_Anchor(float(p[0]), 1000.0 / float(p[1])) for p in base
                if p[1] > 0]
    except Exception:
        records = candidate_records(points)
        if not records:
            return []
        if len(records) <= max_points:
            return _anchors_from_records(records)
        indices = np.linspace(0, len(records) - 1, max_points).astype(int)
        return [_Anchor(records[i]["recall"], records[i]["latency"])
                for i in indices]


def _merge_close(anchors, min_dx: float):
    """Merge near-duplicate anchors, never dropping the range endpoints."""
    index = 1
    while index < len(anchors):
        if anchors[index].recall - anchors[index - 1].recall < min_dx:
            if index - 1 == 0:
                del anchors[index]
            elif index == len(anchors) - 1:
                del anchors[index - 1]
            else:
                left = anchors[index - 1]
                right = anchors[index]
                anchors[index - 1] = _Anchor(
                    (left.recall + right.recall) * 0.5,
                    math.sqrt(left.latency * right.latency))
                del anchors[index]
            return True
        index += 1
    return False


def _repair_auto_anchors(points, max_anchor: int, max_latency_ratio: float,
                         min_dx: float = 0.008):
    """Automatic local repair on real measured anchors."""
    records = candidate_records(points)
    measured = _anchors_from_records(records)
    if not measured:
        return []
    if len(measured) == 1:
        return [(measured[0].recall, measured[0].latency, None)]

    r_min = min(item.recall for item in measured)
    r_max = max(item.recall for item in measured)
    starts = [item for item in measured if abs(item.recall - r_min) < 1e-12]
    ends = [item for item in measured if abs(item.recall - r_max) < 1e-12]
    start_anchor = min(starts, key=lambda item: item.latency)
    end_anchor = min(ends, key=lambda item: item.latency)

    anchors = _base_anchors(points, max_points=max(5, max_anchor - 1))
    anchors = [item for item in anchors if item.latency > 0]
    if not anchors:
        anchors = [start_anchor, end_anchor]
    anchors.sort(key=lambda item: item.recall)
    anchors[0] = _Anchor(start_anchor.recall, start_anchor.latency)
    anchors[-1] = _Anchor(end_anchor.recall, end_anchor.latency)

    log_cap = math.log(float(max_latency_ratio))
    for _ in range(80):
        changed = False
        if _merge_close(anchors, min_dx):
            changed = True
        if changed:
            continue
        for index in range(len(anchors) - 1):
            left = anchors[index]
            right = anchors[index + 1]
            if math.log(right.latency / left.latency) <= log_cap + 1e-9:
                continue
            if len(anchors) >= max_anchor:
                break
            best = None
            best_score = float("inf")
            for candidate in measured:
                if not (left.recall + min_dx < candidate.recall
                        < right.recall - min_dx):
                    continue
                if not (left.latency * 1.0001 < candidate.latency
                        < right.latency * 0.9999):
                    continue
                score = max(math.log(candidate.latency / left.latency),
                            math.log(right.latency / candidate.latency))
                if score < best_score:
                    best_score = score
                    best = candidate
            if best is not None:
                anchors.insert(index + 1, _Anchor(best.recall, best.latency))
                changed = True
                break
        if changed:
            continue
        break

    for index in range(len(anchors) - 1):
        upper = anchors[index].latency * max_latency_ratio
        if anchors[index + 1].latency > upper:
            anchors[index + 1].latency = upper
        elif anchors[index + 1].latency < anchors[index].latency:
            anchors[index + 1].latency = anchors[index].latency

    return [(item.recall, item.latency, None) for item in anchors]


def _parse_adjustment(item):
    """Return (index, dr, dlat_ms, scale) from a list or dict item."""
    if isinstance(item, dict):
        index = item.get("i", item.get("anchor", item.get("index")))
        dr = float(item.get("dr", item.get("dx", 0.0)) or 0.0)
        dlat = float(item.get("dlat_ms", item.get("dlat", item.get("dy", 0.0))) or 0.0)
        scale = float(item.get("scale", item.get("latency_scale", 1.0)) or 1.0)
        return index, dr, dlat, scale
    if isinstance(item, (list, tuple)) and len(item) >= 1:
        index = item[0]
        dr = float(item[1]) if len(item) > 1 else 0.0
        dlat = float(item[2]) if len(item) > 2 else 0.0
        scale = float(item[3]) if len(item) > 3 else 1.0
        return index, dr, dlat, scale
    return None, 0.0, 0.0, 1.0


def _apply_adjustments(anchors, adjustments):
    for item in adjustments or []:
        index, dr, dlat, scale = _parse_adjustment(item)
        if index is None:
            continue
        try:
            index = int(index)
        except Exception:
            continue
        if not (0 <= index < len(anchors)):
            continue
        anchors[index].recall += dr
        anchors[index].latency = max(1e-9, anchors[index].latency + dlat)
        if scale != 1.0:
            anchors[index].latency = max(1e-9,
                                         anchors[index].latency * scale)
    anchors.sort(key=lambda item: item.recall)
    return anchors


def _manual_anchors(records, manual):
    """Build anchors from explicit manual settings, or return None."""
    if not manual:
        return None
    if manual.get("anchors"):
        anchors = []
        for item in manual.get("anchors", []):
            if isinstance(item, dict):
                anchors.append(_Anchor(float(item["recall"]),
                                       float(item.get("latency_ms",
                                                      item.get("latency")))))
            elif len(item) >= 2:
                anchors.append(_Anchor(float(item[0]), float(item[1])))
        anchors.sort(key=lambda item: item.recall)
        return _apply_adjustments(anchors, manual.get("adjustments"))

    selected_ids = manual.get("selected_ids") or manual.get("selected") or []
    if selected_ids:
        by_id = {record["id"]: record for record in records}
        anchors = []
        invalid = []
        for raw_id in selected_ids:
            try:
                record = by_id.get(int(raw_id))
            except Exception:
                record = None
            if record is not None:
                anchors.append(_Anchor(record["recall"], record["latency"]))
            else:
                invalid.append(raw_id)
        if invalid:
            print(f"Warning: manual selected_ids not found: {invalid}",
                  file=sys.stderr)
        if anchors:
            anchors.sort(key=lambda item: item.recall)
            return _apply_adjustments(anchors, manual.get("adjustments"))
    return None


def build_trend(points: Iterable[Sequence], xmin: float = XMIN_DEFAULT,
                n_anchor: int = N_ANCHOR_DEFAULT,
                max_latency_ratio: float = MAX_LATENCY_RATIO_DEFAULT,
                manual=None, **ignored):
    """Return final anchors, applying manual overrides when supplied."""
    records = candidate_records(points, xmin)
    if not records:
        return []
    manual = manual or {}
    anchors = _manual_anchors(records, manual)
    if anchors is None:
        repaired = _repair_auto_anchors(
            points, max_anchor=max(5, int(n_anchor)),
            max_latency_ratio=max_latency_ratio)
        anchors = [_Anchor(item[0], item[1]) for item in repaired]
        anchors = _apply_adjustments(anchors, manual.get("adjustments"))
    if not anchors:
        return []
    # Sanitise manual coordinates: Recall@10 is defined in [0, 1].
    # A point pushed to >1 by a manual dr is dropped entirely, otherwise the
    # plotted line would continue to the right of Recall=1.
    deduped = []
    dropped = 0
    for anchor in anchors:
        if not math.isfinite(anchor.recall) or not math.isfinite(anchor.latency):
            dropped += 1
            continue
        if anchor.recall > 1.0 + 1e-9:
            dropped += 1
            continue
        anchor.recall = min(1.0, max(0.0, anchor.recall))
        anchor.latency = max(1e-9, anchor.latency)
        if deduped and abs(anchor.recall - deduped[-1].recall) < 1e-9:
            deduped[-1].latency = math.sqrt(deduped[-1].latency
                                            * anchor.latency)
        else:
            deduped.append(anchor)
    if dropped:
        print(f"Warning: dropped {dropped} manual anchor(s) with "
              f"Recall outside [0, 1]", file=sys.stderr)
    deduped.sort(key=lambda item: item.recall)
    return [(item.recall, item.latency, None) for item in deduped]


def smooth_line(anchors: Sequence[Sequence], samples: int = 260,
                interpolation: str = "pchip"):
    """Return dense (recall, latency) samples through the final anchors."""
    if not anchors:
        return np.array([]), np.array([])
    if len(anchors) == 1:
        return (np.array([float(anchors[0][0])]),
                np.array([float(anchors[0][1])]))
    x = np.array([float(item[0]) for item in anchors], dtype=float)
    log_y = np.log(np.array([float(item[1]) for item in anchors], dtype=float))
    dense_x = np.linspace(float(x[0]), float(x[-1]), int(samples))
    if interpolation == "linear" or PchipInterpolator is None:
        dense_log_y = np.interp(dense_x, x, log_y)
    else:
        dense_log_y = PchipInterpolator(x, log_y)(dense_x)
    return dense_x, np.exp(dense_log_y)


def draw_curve(axis, method_key: str, points: Iterable[Sequence],
               n_anchor: int = N_ANCHOR_DEFAULT,
               max_latency_ratio: float = MAX_LATENCY_RATIO_DEFAULT,
               manual=None, **ignored):
    anchors = build_trend(points, n_anchor=n_anchor,
                          max_latency_ratio=max_latency_ratio,
                          manual=manual, **ignored)
    if not anchors:
        return []
    style = PS.METHOD_STYLE[method_key]
    zorder = 10 if method_key == "curator" else 3
    if method_key == "prefilter" or len(anchors) == 1:
        PS.draw_method(axis, method_key,
                       [anchors[0][0]], [anchors[0][1]], single_point=True)
        return anchors

    interpolation = (manual or {}).get("interpolation", "pchip")
    line_x, line_y = smooth_line(anchors, interpolation=interpolation)
    axis.plot(line_x, line_y, color=style["color"],
              linestyle=style["linestyle"], lw=PS.LINE_WIDTH, zorder=zorder)
    axis.plot([item[0] for item in anchors], [item[1] for item in anchors],
              color=style["color"], marker=style["marker"],
              linestyle="none", ms=PS.MARKER_SIZE,
              mec=style["color"], mew=PS.MARKER_EDGE_WIDTH, mfc="none",
              zorder=zorder + 1)
    return anchors


def x_limits_for(anchor_lists, xmin: float = XMIN_DEFAULT):
    starts = [items[0][0] for items in anchor_lists if items]
    if not starts:
        return xmin, 1.04
    low = max(xmin, min(0.7, math.floor(min(starts) * 10.0) / 10.0))
    return low, 1.04


def y_values(anchor_lists):
    values = []
    for anchors in anchor_lists:
        values.extend(float(item[1]) for item in anchors)
    return values


def style_latency_axis(axis, anchor_lists, xlabel="Recall@10",
                       ylabel=YLABEL_DEFAULT, fontsize=26, tick_fontsize=22,
                       show_ylabel=True, xmin=XMIN_DEFAULT):
    low, high = x_limits_for(anchor_lists, xmin=xmin)
    PS.style_axis(axis, (low, high), xlabel, ylabel=ylabel,
                  fontsize=fontsize, tick_fontsize=tick_fontsize,
                  show_ylabel=show_ylabel)
    values = y_values(anchor_lists)
    if values:
        PS.set_log_ylim(axis, values)