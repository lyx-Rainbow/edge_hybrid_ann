#!/usr/bin/env python3
"""Plot QPS--Recall curves for the GIST1M heterogeneous-decay experiment."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


plt.style.use("ggplot")

METHOD_LABELS = {
    "single": "Single",
    "multi_same": "Multi-Same",
    "hetero_rate": "Hetero-Rate",
    "hetero_function": "Hetero-Function",
}

# Use a palette, marker set, and line styles distinct from the baseline figures.
METHOD_STYLES = {
    "single": dict(color="#6F2DBD", marker="P", linestyle="-"),
    "multi_same": dict(color="#009E9A", marker="X", linestyle="--"),
    "hetero_rate": dict(color="#D81B60", marker="h", linestyle="-."),
    "hetero_function": dict(color="#C99700", marker="*", linestyle=":"),
}

# Draw the homogeneous reference last so it remains visible at intersections.
PLOT_ORDER = ("hetero_function", "hetero_rate", "multi_same", "single")
LEGEND_ORDER = ("single", "multi_same", "hetero_rate", "hetero_function")

LINE_WIDTH = 4
MARKER_SIZE = 13
MARKER_EDGE_WIDTH = 2.5

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TABLE_ROW = re.compile(
    rf"^\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})(?:\s+.*)?$"
)
LOG_NAME = re.compile(
    r"^gist1m_(single|multi_same|hetero_rate|hetero_function)_a(\d+)\.log$",
    re.IGNORECASE,
)


def parse_qps_recall(log_file: Path) -> tuple[list[float], list[float]]:
    """Read recall and QPS only from the final ef/recall/time/QPS table."""
    recall: list[float] = []
    qps: list[float] = []
    header_seen = False

    with log_file.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not header_seen:
                if "ef" in line and "recall" in line and "QPS" in line:
                    header_seen = True
                continue

            match = TABLE_ROW.match(line)
            if match:
                recall.append(float(match.group(2)))
                qps.append(float(match.group(4)))

    if not header_seen:
        print(f"Warning: result-table header not found in {log_file.name}")
    elif not recall:
        print(f"Warning: no result rows found in {log_file.name}")
    return recall, qps


def filter_from_recall(
    recall: list[float],
    qps: list[float],
    minimum: float,
) -> tuple[list[float], list[float]]:
    """Keep one point before the first point reaching the requested recall."""
    first = next((i for i, value in enumerate(recall) if value >= minimum), -1)
    if first < 0:
        return [], []
    start = max(0, first - 1)
    pairs = sorted(
        (r, q) for r, q in zip(recall[start:], qps[start:]) if q > 0
    )
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def discover_logs(log_dir: Path) -> dict[str, dict[str, Path]]:
    """Return alpha code -> method -> log file."""
    groups: dict[str, dict[str, Path]] = {}
    for log_file in sorted(log_dir.glob("gist1m_*_a*.log")):
        match = LOG_NAME.match(log_file.name)
        if match is None:
            print(f"Warning: ignoring unrecognized log name: {log_file.name}")
            continue
        method = match.group(1).lower()
        alpha_code = match.group(2)
        groups.setdefault(alpha_code, {})[method] = log_file
    return groups


def alpha_code_to_value(code: str) -> float:
    if code.startswith("0") and len(code) > 1:
        return float(f"0.{code[1:]}")
    return float(code)


def alpha_value_to_code(value: str) -> str:
    return format(float(value), "g").replace(".", "")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot heterogeneous-decay GIST1M QPS--Recall curves."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=script_dir.parent / "logs" / "heterogeneous_gist1m",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "output"
    )
    parser.add_argument(
        "--alphas",
        nargs="*",
        help="optional alpha filter, for example: --alphas 0.2 0.4",
    )
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def plot_alpha(
    alpha_code: str,
    files: dict[str, Path],
    output_dir: Path,
    minimum_recall: float,
    show: bool,
) -> tuple[list[object], list[str]] | None:
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for method in LEGEND_ORDER:
        log_file = files.get(method)
        if log_file is None:
            print(f"Warning: missing {method} log for alpha code a{alpha_code}")
            continue
        recall, qps = parse_qps_recall(log_file)
        recall, qps = filter_from_recall(recall, qps, minimum_recall)
        if recall:
            curves[method] = (recall, qps)
        else:
            print(
                f"Warning: {log_file.name} has no recall >= {minimum_recall:g}"
            )

    if not curves:
        print(f"Warning: no plottable curves for alpha code a{alpha_code}")
        return None

    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    all_qps: list[float] = []

    for method in PLOT_ORDER:
        if method not in curves:
            continue
        recall, qps = curves[method]
        style = METHOD_STYLES[method]
        axis.plot(
            recall,
            qps,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=METHOD_LABELS[method],
            lw=LINE_WIDTH,
            ms=MARKER_SIZE,
            mec=style["color"],
            mew=MARKER_EDGE_WIDTH,
            mfc="none",
            zorder=10 if method == "single" else 3,
        )
        all_qps.extend(qps)

    alpha = alpha_code_to_value(alpha_code)
    axis.set_xlabel(f"Recall@10(alpha={alpha:g})", fontsize=26)
    axis.set_ylabel("QPS", fontsize=26)
    axis.set_yscale("log")
    axis.set_xlim(minimum_recall, 1.005)
    if abs(minimum_recall - 0.8) < 1e-12:
        axis.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])

    if all_qps:
        padding = 1.6
        axis.set_ylim(
            bottom=max(min(all_qps) / padding, 1e-3),
            top=max(all_qps) * padding,
        )

    axis.tick_params(axis="both", labelsize=22)
    axis.grid(True, linewidth=4, alpha=0.3)
    figure.tight_layout()

    output_file = output_dir / (
        f"qps_recall_heterogeneous_gist1m_a{alpha_code}.png"
    )
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_file}")

    handles, labels = axis.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    ordered_labels = [
        METHOD_LABELS[method]
        for method in LEGEND_ORDER
        if METHOD_LABELS[method] in handle_by_label
    ]
    ordered_handles = [handle_by_label[label] for label in ordered_labels]
    if not show:
        plt.close(figure)
    return ordered_handles, ordered_labels


def save_legend(
    handles: list[object],
    labels: list[str],
    output_file: Path,
    show: bool,
) -> None:
    figure = plt.figure(figsize=(max(12, 3.8 * len(labels)), 2.0))
    legend = figure.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        fontsize=28,
        markerscale=2.3,
        handlelength=3.0,
        handletextpad=0.9,
        columnspacing=1.2,
    )
    for handle in legend.legend_handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(6.0)
        if hasattr(handle, "set_markeredgewidth"):
            handle.set_markeredgewidth(4.0)

    figure.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.25)
    print(f"Legend saved to: {output_file}")
    if not show:
        plt.close(figure)


def main() -> int:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not log_dir.is_dir():
        raise FileNotFoundError(f"log directory not found: {log_dir}")
    if not 0.0 <= args.min_recall < 1.0:
        raise ValueError("--min-recall must be in [0, 1)")

    groups = discover_logs(log_dir)
    if args.alphas:
        requested = {alpha_value_to_code(value) for value in args.alphas}
        groups = {
            code: files for code, files in groups.items() if code in requested
        }
    if not groups:
        raise FileNotFoundError(f"no heterogeneous GIST1M logs found in {log_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    legend_data: tuple[list[object], list[str]] | None = None
    figure_count = 0
    for alpha_code in sorted(groups, key=alpha_code_to_value):
        result = plot_alpha(
            alpha_code, groups[alpha_code], output_dir, args.min_recall, args.show
        )
        if result is not None:
            figure_count += 1
            if legend_data is None:
                legend_data = result

    if legend_data is not None:
        save_legend(
            *legend_data,
            output_dir / "qps_recall_heterogeneous_gist1m_legend.png",
            args.show,
        )

    if args.show:
        plt.show()
    print(f"Generated {figure_count} heterogeneous QPS--Recall figure(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        raise SystemExit(1)
