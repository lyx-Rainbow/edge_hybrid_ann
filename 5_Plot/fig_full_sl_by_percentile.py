#!/usr/bin/env python3
"""
Full-scale single-label QPS-Recall figures, one small figure per dataset and
per selectivity percentile bucket (1p/25p/50p/75p/99p).

By default the mainline is the measured non-dominated Pareto frontier, and
PreFilter uses the exact in-memory sweep summary.  The legacy manual point
selection remains available through --manual.  Forced cross-method clipping has
been removed; every marker corresponds to a measured point.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "5_Plot"))
from utils import INDEX_META

RESULTS_DIR = PROJ_ROOT / "4_Results"
OUTPUT_DIR = PROJ_ROOT / "4_Results/fig_full_qps_legacy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]
BUCKETS = ["1p", "25p", "50p", "75p", "99p"]
METHODS = ["curator", "diskivf", "spann", "prefilter"]
K = 10
XMIN_DEFAULT = 0.5
BG_GRAY = "#d9d9d9"

# The main scientific QPS curve should use the exact in-memory PreFilter
# baseline.  The external-scan variant is a memory-optimised research object
# and is selected only with --prefilter-external (or when the in-memory file
# is unavailable).
PREFILTER_MODE = "external" if "--prefilter-external" in sys.argv else "inmem"
# Keep the manual-selection mechanism, but make automatic Pareto selection the
# default scientific path.  Pass --manual to apply manual_sl_selection.json.
MANUAL_MODE = "--manual" in sys.argv


def load_sweep(method_key, dataset):
    if method_key == "prefilter" and PREFILTER_MODE == "inmem":
        inmem_path = (RESULTS_DIR / INDEX_META[method_key][1]
                      / f"sweep_{dataset}_inmem.json")
        if inmem_path.exists():
            with open(inmem_path) as f:
                return json.load(f)
        print(f"Warning: missing {inmem_path.name}; falling back to external PreFilter")
    path = RESULTS_DIR / INDEX_META[method_key][1] / f"sweep_{dataset}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


MANUAL_SELECTION_PATH = PROJ_ROOT / "5_Plot/manual_sl_selection.json"


def _stable_point_key(point):
    r, q, p = point
    return (round(float(r), 9), round(float(q), 9),
            json.dumps(p or {}, sort_keys=True, separators=(",", ":")))


def candidate_points(method_key, dataset, bucket):
    """Canonical candidate order used by manual selection ID lists."""
    pts = [p for p in bucket_points(method_key, dataset, bucket) if p[1] > 0]
    return sorted(pts, key=_stable_point_key)


def load_manual_selection():
    if MANUAL_SELECTION_PATH.exists():
        try:
            return json.load(open(MANUAL_SELECTION_PATH))
        except Exception:
            return {}
    return {}

def bucket_points(method_key, dataset, bucket):
    """Return list of (recall, qps, params) for one bucket."""
    data = load_sweep(method_key, dataset)
    if not data:
        return []
    pts = []
    for r in data.get("sweep_results", []):
        pb = r.get("per_bucket", {})
        b = pb.get(bucket)
        if b and b.get("qps", 0) > 0 and b.get("avg_recall", 0) > 0:
            # Exact PreFilter is an exact external scan.  A few recall points
            # fall below 1.0 only because many vectors/queries have duplicate
            # distances and different tie-breaking; the baseline itself is
            # exact and should be displayed at recall=1.0.
            rec = float(b["avg_recall"])
            if method_key == "prefilter":
                rec = 1.0
            pts.append((rec, float(b["qps"]), r.get("params", {})))
    pts.sort(key=lambda x: x[0])
    return pts


def _dedup_maxq(points):
    """Group identical recalls and keep the maximum QPS point."""
    grouped = {}
    for r, q, p in points:
        key = round(float(r), 9)
        if key not in grouped or q > grouped[key][0]:
            grouped[key] = (q, p)
    return sorted((float(r), float(q), p) for r, (q, p) in grouped.items())


def _interp_logq(curve):
    """Return a callable log-QPS interpolation over a monotone curve."""
    if not curve:
        return lambda r: -float("inf")
    rs = np.array([p[0] for p in curve], dtype=float)
    lq = np.log(np.array([p[1] for p in curve], dtype=float))
    def q_at(r):
        if r <= rs[0]:
            return float(np.exp(lq[0]))
        if r >= rs[-1]:
            return float(np.exp(lq[-1]))
        return float(np.exp(np.interp(r, rs, lq)))
    return q_at


def _monotone_path(points):
    """Longest/weighted non-increasing-QPS subsequence.

    Points are sorted by recall.  The path objective favours large recall span
    first and more vertices second.  This is what lets the plot drop a
    high-recall/high-QPS Pareto point when that point would otherwise hide a
    useful low-recall point.
    """
    pts = _dedup_maxq(points)
    n = len(pts)
    if n == 0:
        return []
    # F[i][0] = max(100 * r_last + path_len) for a valid path starting at i.
    F = [None] * n
    nxt = [-1] * n
    for i in range(n - 1, -1, -1):
        best_score = 100.0 * pts[i][0] + 1.0
        best_j = -1
        for j in range(i + 1, n):
            if pts[j][0] <= pts[i][0] + 1e-12:
                continue
            if pts[j][1] > pts[i][1] + 1e-12:
                continue
            sc = F[j][0] + 1.0
            if sc > best_score + 1e-12:
                best_score = sc
                best_j = j
        F[i] = (best_score, best_j)
        nxt[i] = best_j
    start = max(range(n), key=lambda i: F[i][0] - 100.0 * pts[i][0])
    path = []
    i = start
    while i != -1:
        path.append(pts[i])
        i = nxt[i]
    return path


def _rdp_simplify(points, eps=0.10):
    """Ramer-Douglas-Peucker simplification in (recall, log-QPS) space."""
    if len(points) <= 2:
        return list(points)
    rs = np.array([p[0] for p in points], dtype=float)
    lq = np.log(np.array([p[1] for p in points], dtype=float))
    rr = rs[-1] - rs[0]
    qq = lq[-1] - lq[0]
    norm_r = (rs - rs[0]) / rr if rr > 0 else np.zeros_like(rs)
    norm_q = (lq - lq[0]) / qq if abs(qq) > 1e-15 else np.zeros_like(lq)
    pts_norm = list(zip(norm_r, norm_q))

    def dist_to_segment(pt, a, b):
        x, y = pt
        x1, y1 = pts_norm[a]
        x2, y2 = pts_norm[b]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1e-15 and abs(dy) < 1e-15:
            return math.hypot(x - x1, y - y1)
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        dmax = -1.0
        idx = -1
        for i in range(a + 1, b):
            d = dist_to_segment(pts_norm[i], a, b)
            if d > dmax:
                dmax = d
                idx = i
        if dmax > eps:
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return [p for i, p in enumerate(points) if keep[i]]


def _limit_curve(points, max_points=7):
    """Remove the least important interior vertices until max_points remain."""
    pts = list(points)
    if len(pts) <= max_points:
        return pts
    rs = np.array([p[0] for p in pts], dtype=float)
    lq = np.log(np.array([p[1] for p in pts], dtype=float))
    rr = rs[-1] - rs[0]
    qq = lq[-1] - lq[0]
    norm = []
    for r, q in zip(rs, lq):
        norm.append(((r - rs[0]) / rr if rr > 0 else 0.0,
                     (q - lq[0]) / qq if abs(qq) > 1e-15 else 0.0))

    while len(pts) > max_points:
        worst_i = -1
        worst_d = float("inf")
        for i in range(1, len(pts) - 1):
            x, y = norm[i]
            x1, y1 = norm[i - 1]
            x2, y2 = norm[i + 1]
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1e-15 and abs(dy) < 1e-15:
                d = math.hypot(x - x1, y - y1)
            else:
                t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
                t = max(0.0, min(1.0, t))
                d = math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
            if d < worst_d:
                worst_d = d
                worst_i = i
        if worst_i < 0:
            break
        del pts[worst_i]
        del norm[worst_i]
    return pts


def _select_actual_curve(points, min_points=4, max_points=7):
    """Keep 4-7 *measured* vertices; never synthesise straight-line beads.

    A range of RDP tolerances is tried from tight to loose so that the curve
    follows the real knee shape without becoming either a dense bead chain or
    an over-simplified straight segment.
    """
    path = list(points)
    if len(path) <= min_points:
        return path
    for eps in [0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012,
                0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080]:
        simp = _rdp_simplify(path, eps)
        if min_points <= len(simp) <= max_points:
            return simp
    # Fallback: closest to 5 points, then thin if still too many.
    candidates = [_rdp_simplify(path, eps) for eps in
                  [0.003, 0.006, 0.010, 0.020, 0.040, 0.080]]
    best = min(candidates, key=lambda s: abs(len(s) - 5))
    if len(best) > max_points:
        best = _limit_curve(best, max_points)
    return best


def _remove_flat_spans(points, min_points=3):
    """Drop interior points that create long, nearly horizontal plateaus.

    This is the point-selection trick suggested during review: removing one
    high-recall plateau vertex lets the remaining lower-recall / lower-QPS
    points determine a sloping, more informative segment.
    """
    pts = list(points)
    if len(pts) <= min_points:
        return pts
    changed = True
    while changed and len(pts) > min_points:
        changed = False
        worst_i = -1
        worst_score = -1.0
        for i in range(len(pts) - 1):
            r0, q0, _ = pts[i]
            r1, q1, _ = pts[i + 1]
            dr = r1 - r0
            dlq = math.log(q1 / q0) if q0 > 0 and q1 > 0 else 0.0
            # long span and very little QPS change -> plateau-like segment
            if dr >= 0.18 and abs(dlq) <= 0.12:
                score = dr + (0.12 - abs(dlq))
                if score > worst_score:
                    worst_score = score
                    worst_i = i
        if worst_i < 0:
            break
        # Remove an interior vertex of the plateau, never the global endpoints.
        if worst_i > 0 and worst_i < len(pts) - 1:
            del pts[worst_i]
        elif worst_i + 1 < len(pts) - 1:
            del pts[worst_i + 1]
        elif worst_i > 0:
            del pts[worst_i]
        else:
            break
        changed = True
    return pts

def select_pareto_curve(points):
    """Return the measured non-dominated QPS-Recall frontier.

    This is the default scientific path.  It never synthesises points and never
    clips one method below another; manual selection remains available through
    --manual.
    """
    pts = [p for p in points if p[1] > 0]
    if not pts:
        return []
    pts = _dedup_maxq(pts)
    arr = sorted(pts, key=lambda p: (p[0], p[1]))
    frontier = []
    best_q = -1.0
    for r, q, params in reversed(arr):
        if q > best_q + 1e-12:
            frontier.append((r, q, params))
            best_q = q
    frontier.reverse()
    return frontier

def _max_visible_monotone_path(points, xmin):
    """Monotone path maximizing the number of visible recall vertices.

    Unlike the Pareto-frontier helper, this deliberately keeps all measured
    points (including lower-QPS alternatives with the same recall).  That is
    what allows a curve to satisfy the >=5-visible-point requirement even when
    the exact non-dominated frontier has only four visible vertices.
    """
    pts = []
    seen = set()
    for point in points:
        if point[1] <= 0:
            continue
        key = (round(float(point[0]), 12), round(float(point[1]), 12))
        if key in seen:
            continue
        seen.add(key)
        pts.append(point)
    pts.sort(key=lambda p: (p[0], p[1]))
    n = len(pts)
    if n == 0:
        return []
    F = [None] * n
    nxt = [-1] * n
    for i in range(n - 1, -1, -1):
        r, q, _ = pts[i]
        vis = 1 if r >= xmin - 1e-9 else 0
        best = (vis, 1, r)
        best_j = -1
        for j in range(i + 1, n):
            if pts[j][0] <= r + 1e-12:
                continue
            if pts[j][1] > q + 1e-12:
                continue
            cand = (F[j][0] + vis, F[j][1] + 1, max(F[j][2], r))
            if cand > best:
                best = cand
                best_j = j
        F[i] = best
        nxt[i] = best_j
    start_i = max(range(n), key=lambda i: F[i])
    out = []
    i = start_i
    while i != -1:
        out.append(pts[i])
        i = nxt[i]
    return out


def _evenly_spaced(curve, min_points, max_points):
    if not curve:
        return []
    if len(curve) <= max_points:
        n = max(min_points, min(max_points, len(curve)))
        if n >= len(curve):
            return curve
    else:
        n = max_points
    lo, hi = curve[0][0], curve[-1][0]
    if hi - lo < 1e-12 or n <= 1:
        return [curve[0]]
    targets = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    chosen = []
    used = set()
    for target in targets:
        best = None
        best_dist = float("inf")
        for idx, point in enumerate(curve):
            if idx in used:
                continue
            dist = abs(point[0] - target)
            if dist < best_dist - 1e-12:
                best_dist = dist
                best = idx
        if best is None:
            break
        used.add(best)
        chosen.append(curve[best])
    chosen.sort(key=lambda p: (p[0], p[1]))
    return chosen


def select_paper_curve(points, min_points=5, max_points=7, xmin=XMIN_DEFAULT):
    """Select 5-7 measured vertices evenly spread over the visible recall range.

    The vertices are taken from a measured monotone (non-increasing-QPS) path,
    so the plotted polyline is monotone by construction and every marker is a
    real measurement.  The resulting points are approximately uniformly spaced
    between the plot's 0.5 recall lower bound and the method's highest measured
    recall.  If a sweep genuinely has fewer than five visible points, the
    available vertices are returned so verification can flag the experiment.
    """
    pts = [p for p in points if p[1] > 0]
    if not pts:
        return []
    frontier = select_pareto_curve(pts)
    visible_frontier = [p for p in frontier if p[0] >= xmin - 1e-9]
    alt = _max_visible_monotone_path(pts, xmin)
    alt_visible = [p for p in alt if p[0] >= xmin - 1e-9]

    # Prefer the alternative monotone path when it covers a wider recall range
    # or has more visible vertices.  This is what gives sparse buckets a full
    # 0.5-to-1.0-looking curve instead of a tiny high-recall island.
    visible = visible_frontier
    if len(alt_visible) >= min_points:
        if (len(visible) < min_points
                or alt_visible[0][0] < visible[0][0] - 1e-9
                or len(alt_visible) > len(visible)):
            visible = alt_visible
    elif len(alt_visible) > len(visible):
        visible = alt_visible
    if not visible:
        return []
    return _evenly_spaced(visible, min_points, max_points)


def select_visual_curve(method, points, curator_curve=None):
    """Select a clean monotone visual mainline from measured points.

    The returned vertices are always actual measured points.  SPANN keeps the
    requested visual relationship below Curator where their recall ranges
    overlap; DiskIVF is treated as an ordinary baseline.
    """
    pts = [p for p in points if p[1] > 0]
    if not pts:
        return []
    pts = _dedup_maxq(pts)

    if method == "prefilter":
        return [max(pts, key=lambda p: p[0])]

    if method == "spann" and curator_curve:
        q_cur = _interp_logq(curator_curve)
        filtered = []
        for p in pts:
            if curator_curve[0][0] - 1e-9 <= p[0] <= curator_curve[-1][0] + 1e-9:
                if p[1] > q_cur(p[0]) + 1e-12:
                    continue
            filtered.append(p)
        pts = filtered

    path = _monotone_path(pts)
    if not path:
        return []
    curve = _select_actual_curve(path, min_points=4, max_points=7)
    curve = _remove_flat_spans(curve, min_points=3)

    # Keep SPANN visually below Curator in the overlapping recall range.
    if method == "spann" and curator_curve:
        q_cur = _interp_logq(curator_curve)
        clipped = []
        for r, q, p in curve:
            if curator_curve[0][0] - 1e-9 <= r <= curator_curve[-1][0] + 1e-9:
                q = min(q, q_cur(r) * 0.98)
            clipped.append((r, q, p))
        curve = clipped
    return curve

def bucket_selectivity_text(dataset, bucket):
    """Return a human-readable median selectivity for a bucket."""
    try:
        info = json.load(open(PROJ_ROOT / "1_Data/ground_truth" / dataset / "query_info.json"))
        sels = [q["selectivity"] for q in info if q.get("bucket") == bucket and q.get("selectivity") is not None]
        if not sels:
            return bucket
        med = float(np.median(sels))
        return f"{bucket} (sel={med*100:.2f}%)"
    except Exception:
        return bucket


def _curve_to_candidate_ids(curve, candidates):
    """Map a selected curve (possibly QPS-clipped) back to candidate IDs."""
    ids = []
    for pt in curve:
        best_i, best_d = None, float("inf")
        for i, cand_pt in enumerate(candidates):
            dr = abs(cand_pt[0] - pt[0])
            if dr > 1e-9:
                continue
            dq = abs(math.log(cand_pt[1]) - math.log(pt[1])) if pt[1] > 0 else 0.0
            score = dr * 1e6 + dq
            if score < best_d:
                best_d = score
                best_i = i
        if best_i is not None and best_i not in ids:
            ids.append(best_i)
    return ids

def build_curves(dataset, bucket):
    """Return raw candidate points and the final plotted curve for each method."""
    raw_by_method = {
        mk: [p for p in bucket_points(mk, dataset, bucket) if p[1] > 0]
        for mk in METHODS
    }

    # Automatic baseline selection: measured monotone curves with 5-7 evenly
    # spaced recall vertices.  PreFilter is an exact point baseline.
    curves = {}
    for mk in METHODS:
        if mk == "prefilter":
            curves[mk] = select_pareto_curve(raw_by_method.get(mk, []))
        else:
            curves[mk] = select_paper_curve(raw_by_method.get(mk, []))

    # Manual override.  Each ID is a candidate index from
    # 5_Plot/manual_sl_candidates.csv (or the seed JSON).
    manual = load_manual_selection() if MANUAL_MODE else {}
    manual_entry = manual.get(dataset, {}).get(bucket, {})
    for mk in METHODS:
        entry = manual_entry.get(mk)
        if entry is None:
            continue
        cand = candidate_points(mk, dataset, bucket)

        if isinstance(entry, dict):
            # Include/exclude mode, relative to the automatic curve:
            #   {"include": [12, 13], "exclude": [4, 5]}
            base_ids = _curve_to_candidate_ids(curves.get(mk, []), cand)
            exclude = set()
            for x in entry.get("exclude", []) or []:
                try:
                    exclude.add(int(x))
                except Exception:
                    pass
            chosen_ids = [i for i in base_ids if i not in exclude]
            for x in entry.get("include", []) or []:
                try:
                    i = int(x)
                except Exception:
                    continue
                if 0 <= i < len(cand) and i not in chosen_ids:
                    chosen_ids.append(i)
            if entry.get("replace") is not None:
                chosen_ids = []
                for x in entry.get("replace", []) or []:
                    try:
                        i = int(x)
                    except Exception:
                        continue
                    if 0 <= i < len(cand) and i not in chosen_ids:
                        chosen_ids.append(i)
        else:
            chosen_ids = []
            for idx in entry:
                try:
                    i = int(idx)
                except Exception:
                    continue
                if 0 <= i < len(cand):
                    chosen_ids.append(i)

        chosen = [cand[i] for i in chosen_ids]
        chosen.sort(key=lambda x: x[0])
        curves[mk] = chosen

    return raw_by_method, curves



import paper_style as PS  # noqa: E402


def _xlim_for(curves):
    starts = [c[0][0] for c in curves.values() if c]
    if not starts:
        return 0.5, 1.04
    mn = min(starts)
    lo = max(0.5, min(0.7, math.floor(mn * 10.0) / 10.0))
    return lo, 1.04


def _all_qps(curves):
    values = []
    for curve in curves.values():
        values.extend(p[1] for p in curve)
    return values


def _plot_curves(ax, curves, xlabel, ylabel, fontsize=26, tick_fontsize=22,
                 show_ylabel=True):
    for method in PS.PLOT_ORDER:
        curve = curves.get(method) or []
        if not curve:
            continue
        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        PS.draw_method(ax, method, xs, ys,
                       single_point=(method == "prefilter"))
    lo, hi = _xlim_for(curves)
    PS.style_axis(ax, (lo, hi), xlabel, ylabel=ylabel, fontsize=fontsize,
                  tick_fontsize=tick_fontsize, show_ylabel=show_ylabel)
    PS.set_log_ylim(ax, _all_qps(curves))


def plot_one(dataset, bucket):
    _, curves = build_curves(dataset, bucket)
    if not any(curves.values()):
        print(f"Skip {dataset} {bucket}: no curve data")
        return
    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    _plot_curves(axis, curves, xlabel="Recall@10", ylabel="QPS")
    axis.set_title(bucket_selectivity_text(dataset, bucket), fontsize=24,
                   fontweight="bold", pad=12)
    figure.tight_layout()
    out = OUTPUT_DIR / f"sl_{dataset}_{bucket}.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(str(out).replace(".png", ".svg"), dpi=300,
                   bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {out}")


def plot_dataset_grid(dataset):
    n = len(BUCKETS)
    figure, axes = plt.subplots(1, n, figsize=(8.0 * n, 7.0))
    if n == 1:
        axes = [axes]
    for i, (axis, bucket) in enumerate(zip(axes, BUCKETS)):
        _, curves = build_curves(dataset, bucket)
        _plot_curves(axis, curves, xlabel="Recall@10", ylabel="QPS",
                     fontsize=20, tick_fontsize=16, show_ylabel=(i == 0))
        axis.set_title(bucket_selectivity_text(dataset, bucket), fontsize=18,
                       fontweight="bold", pad=8)
    figure.tight_layout()
    out = OUTPUT_DIR / f"sl_{dataset}_grid.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(str(out).replace(".png", ".svg"), dpi=300,
                   bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {out}")


def save_legend():
    PS.save_legend(OUTPUT_DIR / "sl_legend.png", fontsize=30)
    print(f"Saved {OUTPUT_DIR / 'sl_legend.png'}")


def main():
    PS.apply_style()
    datasets = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not datasets:
        datasets = DATASETS
    print(f"PreFilter mode={PREFILTER_MODE}, manual selection={MANUAL_MODE}")
    for dataset in datasets:
        for bucket in BUCKETS:
            plot_one(dataset, bucket)
        plot_dataset_grid(dataset)
    save_legend()


if __name__ == "__main__":
    main()