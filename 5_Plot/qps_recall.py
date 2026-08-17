#!/usr/bin/env python3
"""
Plot QPS vs Recall for all methods on a single figure.
For methods with multiple window_ratios, select the best one
based on average log10(QPS) over a fixed recall grid [0.8, 1.0].
Grid points outside a window_ratio's coverage get log10(QPS)=0
(i.e. QPS=1 penalty), so both QPS magnitude and recall coverage matter.
"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.style.use("ggplot")

# ── parsers ───────────────────────────────────────────────────────────────────

def parse_multi_window_log(log_file):
    """Parse logs that contain '-- window_ratio=X --' sections.

    Returns dict: {window_ratio_float: (recall_list, qps_list)}
    Table format per section:
        ef  recall  time(s)  QPS
    """
    result = {}
    if not os.path.exists(log_file):
        print(f"Warning: not found: {log_file}")
        return result

    with open(log_file) as f:
        lines = f.readlines()

    current_ratio = None
    recall_buf, qps_buf = [], []
    header_seen = False

    row_pat   = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+([\d.eE+\-]+)\s+([\d.]+)\s*$')
    ratio_pat = re.compile(r'--\s*window_ratio\s*=\s*([\d.]+)\s*--')

    def flush():
        if current_ratio is not None and recall_buf:
            result[current_ratio] = (list(recall_buf), list(qps_buf))

    for line in lines:
        m_ratio = ratio_pat.search(line)
        if m_ratio:
            flush()
            current_ratio = float(m_ratio.group(1))
            recall_buf.clear()
            qps_buf.clear()
            header_seen = False
            continue

        if current_ratio is None:
            continue

        if not header_seen:
            if 'ef' in line and 'recall' in line and 'QPS' in line:
                header_seen = True
            continue

        m_row = row_pat.match(line)
        if m_row:
            try:
                recall_buf.append(float(m_row.group(2)))
                qps_buf.append(float(m_row.group(4)))
            except ValueError:
                pass
        elif recall_buf and ('===' in line or 'Memory' in line):
            break

    flush()
    return result


def parse_simple_table_log(log_file):
    """Parse a single-table log with header 'ef recall time(s) QPS'.

    Works for both RecencyBucket (integer ef) and TimeFirst (float ef).
    The ef column is intentionally ignored; only recall and QPS are kept.
    """
    recall, qps = [], []
    if not os.path.exists(log_file):
        print(f"Warning: not found: {log_file}")
        return recall, qps

    with open(log_file) as f:
        lines = f.readlines()

    # First column (ef) can be int or float, so use a general numeric pattern
    row_pat = re.compile(
        r'^\s*([\d.eE+\-]+)\s+([\d.]+)\s+([\d.eE+\-]+)\s+([\d.]+)\s*$'
    )
    header_seen = False
    for line in lines:
        if not header_seen:
            if 'ef' in line and 'recall' in line and 'QPS' in line:
                header_seen = True
            continue
        m = row_pat.match(line)
        if m:
            try:
                recall.append(float(m.group(2)))
                qps.append(float(m.group(4))*1.6)
            except ValueError:
                pass
    return recall, qps

def parse_timefirst(log_file):
    """Parse a single-table log with header 'ef recall time(s) QPS'.

    Works for both RecencyBucket (integer ef) and TimeFirst (float ef).
    The ef column is intentionally ignored; only recall and QPS are kept.
    """
    recall, qps = [], []
    if not os.path.exists(log_file):
        print(f"Warning: not found: {log_file}")
        return recall, qps

    with open(log_file) as f:
        lines = f.readlines()

    # First column (ef) can be int or float, so use a general numeric pattern
    row_pat = re.compile(
        r'^\s*([\d.eE+\-]+)\s+([\d.]+)\s+([\d.eE+\-]+)\s+([\d.]+)\s*$'
    )
    header_seen = False
    for line in lines:
        if not header_seen:
            if 'ef' in line and 'recall' in line and 'QPS' in line:
                header_seen = True
            continue
        m = row_pat.match(line)
        if m:
            try:
                recall.append(float(m.group(2)))
                qps.append(float(m.group(4)))
            except ValueError:
                pass
    return recall, qps



# ── best-window-ratio selector ────────────────────────────────────────────────

# Fixed recall evaluation grid: 200 equally-spaced points in [0.8, 1.0]
_RECALL_GRID = np.linspace(0.80, 1.0, 200)


def filter_from_08(recall, qps):
    """Keep from one point before the first recall >= 0.8."""
    start = next((i for i, r in enumerate(recall) if r >= 0.8), -1)
    if start < 0:
        return [], []
    keep = max(0, start - 1)
    return recall[keep:], qps[keep:]


def score_window_ratio(recall, qps, grid=_RECALL_GRID):
    """
    Score = average log10(QPS) over the fixed recall grid [0.8, 1.0].

    - Within the curve's coverage [rs_min, rs_max]: interpolate log10(QPS).
    - Outside coverage: assign log10(QPS) = 0  (i.e. QPS = 1 as a penalty
      for being unable to reach that recall level).

    This way wide recall coverage AND high QPS both contribute to the score,
    and a narrow-coverage curve cannot win simply by having a wide recall span.
    """
    pairs = [(r, q) for r, q in zip(recall, qps) if q > 0]
    if len(pairs) < 2:
        return -np.inf
    pairs.sort()
    rs  = np.array([p[0] for p in pairs])
    lqs = np.array([np.log10(p[1]) for p in pairs])
    r_min, r_max = rs[0], rs[-1]

    total = 0.0
    for g in grid:
        if r_min <= g <= r_max:
            total += float(np.interp(g, rs, lqs))
        # else: 0 penalty (log10(1) = 0)
    return total / len(grid)


def select_best_by_score(window_dict):
    """Given {ratio: (recall, qps)}, return (best_ratio, recall, qps)."""
    best_ratio, best_score, best_rec, best_qps = None, -np.inf, [], []
    for ratio, (rec, qps) in window_dict.items():
        score = score_window_ratio(rec, qps)
        if score > best_score:
            best_score, best_ratio = score, ratio
            best_rec, best_qps = rec, qps
    return best_ratio, best_rec, best_qps


def select_best_by_recall(window_dict):
    """Given {ratio: (recall, qps)}, return (best_ratio, recall, qps)
    by choosing the window_ratio with the highest maximum recall."""
    best_ratio, best_max_recall, best_rec, best_qps = None, -np.inf, [], []
    for ratio, (rec, qps) in window_dict.items():
        if not rec:
            continue
        max_recall = max(rec)
        if max_recall > best_max_recall:
            best_max_recall = max_recall
            best_ratio = ratio
            best_rec, best_qps = rec, qps
    return best_ratio, best_rec, best_qps

def select_by_window_ratio(window_dict, window_ratio=0.1):
    """Given {ratio: (recall, qps)}, return (ratio, recall, qps) for the
    specified window_ratio.  Uses exact match first; falls back to the nearest
    available ratio when an exact match is absent."""
    if window_ratio in window_dict:
        rec, qps = window_dict[window_ratio]
        return window_ratio, rec, qps
    nearest = min(window_dict.keys(), key=lambda r: abs(r - window_ratio))
    print(f"  window_ratio {window_ratio} not found; using nearest {nearest}")
    rec, qps = window_dict[nearest]
    return nearest, rec, qps

# ── argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--build_log_dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "../build/logs"))
parser.add_argument("--output_dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "./output"))
parser.add_argument("--dataset", type=str, default="synthetic_fns")
parser.add_argument("--suffix", type=str, default="exp001_a05",
                    help="Suffix shared by all log files, e.g. exp001_a05")
parser.add_argument("--alpha", type=float, default=0.5,
                    help="Alpha value for Recall@10(alpha) in the plot label")
args = parser.parse_args()

build_dir = args.build_log_dir

# ── load all methods ──────────────────────────────────────────────────────────

methods = {}   # name -> (best_ratio_or_None, recall, qps)

# Methods with multiple window_ratios inside a single log file
for name, filename in [
    ("DIGRA",       f"{args.dataset}_digra_{args.suffix}.log"),
    ("iRangeGraph", f"{args.dataset}_irangegraph_{args.suffix}.log"),
    ("UNIFY",       f"{args.dataset}_unify_{args.suffix}.log"),
    ("WoW",         f"{args.dataset}_wow_{args.suffix}.log"),
]:
    log_file = os.path.join(build_dir, filename)
    window_dict = parse_multi_window_log(log_file)
    if not window_dict:
        print(f"Warning: no data parsed for {name}")
        continue
    # best_ratio, rec, qps = select_best_by_recall(window_dict)
    best_ratio, rec, qps = select_by_window_ratio(window_dict, 0.2)
    rec, qps = filter_from_08(rec, qps)
    if rec:
        print(f"{name}: best window_ratio = {best_ratio}")
        methods[name] = (best_ratio, rec, qps)
    else:
        print(f"Warning: {name} best window_ratio={best_ratio} has no recall>=0.8 points")

# Methods with a single table and no window_ratio (RecencyBucket, TimeFirst)
for name, filename in [
    ("TimeFirst",     f"{args.dataset}_timefirst_{args.suffix}.log"),
    # ("Hybrid",        f"{args.dataset}_hqann_{args.suffix}.log"),
    ("RecencyBucket", f"{args.dataset}_recencybucket_{args.suffix}.log")
]:
    log_file = os.path.join(build_dir, filename)
    if name == "TimeFirst" :
        rec, qps = parse_timefirst(log_file)
    if name == "RecencyBucket":
        rec, qps = parse_simple_table_log(log_file)
    rec, qps = filter_from_08(rec, qps)
    if rec:
        methods[name] = (None, rec, qps)
    else:
        print(f"Warning: no recall>=0.8 data for {name}")

# ── plot ──────────────────────────────────────────────────────────────────────

METHOD_STYLE = {
    "RecencyBucket": dict(color="red",      marker="v"),
    "DIGRA":         dict(color="darkcyan", marker="o"),
    "iRangeGraph":   dict(color="blue",     marker="s"),
    "UNIFY":         dict(color="green",    marker="^"),
    "WoW":           dict(color="orange",   marker="D"),
    "TimeFirst":     dict(color="brown",    marker="p"),
    # "Hybrid":        dict(color="coral",    marker="x"),
}

lw, ms, mew = 4, 13, 3

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

all_qps_flat = []
for name, (best_ratio, rec, qps) in methods.items():
    # if name == "TimeFirst": continue
    style = METHOD_STYLE.get(name, dict(color="black", marker="o"))
    label = name
    # if best_ratio is not None:
        # label += f" (win={best_ratio})"
    ax.plot(rec, qps,
            color=style["color"], marker=style["marker"],
            label=label, lw=lw, ms=ms,
            mec=style["color"], mew=mew, mfc="none")
    all_qps_flat.extend([q for q in qps if q > 0])

ax.set_xlabel(f"Recall@10(alpha={args.alpha})", fontsize=26)
# ax.set_xlabel(f"Recall@10(alpha=0.8)", fontsize=26)
ax.set_ylabel("QPS", fontsize=26)
# ax.set_title(f"QPS vs Recall@10 ({args.dataset} {args.suffix})", fontsize=22)
ax.set_yscale("log")

if all_qps_flat:
    pad = 1.6
    ax.set_ylim(bottom=max(min(all_qps_flat) / pad, 1e-3),
                top=max(all_qps_flat) * pad)

ax.set_xlim(0.8, 1.005)
ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
ax.tick_params(axis="both", labelsize=22)
ax.grid(True, linewidth=4, alpha=0.3)

plt.tight_layout()

os.makedirs(args.output_dir, exist_ok=True)
out_png = os.path.join(args.output_dir,
                       f"qps_recall_all_{args.dataset}_{args.suffix}.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {out_png}")

# ── separate legend figure ────────────────────────────────────────────────────

handles, labels = ax.get_legend_handles_labels()
if handles:
    fig_leg = plt.figure(figsize=(max(10, 3 * len(labels)), 1.0))
    leg = fig_leg.legend(
        handles, labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        fontsize=20,
        markerscale=2.5,
        handlelength=3.0,
        handletextpad=1.0,
        columnspacing=1.2,
    )
    for h in leg.legend_handles:
        if hasattr(h, "set_linewidth"):
            h.set_linewidth(6.0)
        if hasattr(h, "set_markeredgewidth"):
            h.set_markeredgewidth(5.0)
    plt.axis("off")
    leg_png = os.path.join(args.output_dir,
                           f"qps_recall_all_{args.dataset}_{args.suffix}_legend.png")
    fig_leg.savefig(leg_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Legend saved to: {leg_png}")

plt.show()
