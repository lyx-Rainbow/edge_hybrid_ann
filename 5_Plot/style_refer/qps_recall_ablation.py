#!/usr/bin/env python3
'''Plot QPS-versus-recall curves from recency-bucket ablation logs.'''

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


plt.style.use('ggplot')

VARIANT_LABELS = {
    'recencybucket': 'RecencyBucket',
    'fixedbucket': 'FixedBucket',
    'nocross': 'w/o Cross-Bucket Navigation',
    'noearlystop': 'w/o Early Termination',
    'nomerge': 'w/o Bucket Merging',
}

VARIANT_ALIASES = {
    'recencybucket': 'recencybucket',
    'recency_bucket': 'recencybucket',
    'fixedbucket': 'fixedbucket',
    'fixed_bucket': 'fixedbucket',
    'nocross': 'nocross',
    'nocorss': 'nocross',  # Keep compatibility with the current file typo.
    'no_cross': 'nocross',
    'nocross_nav': 'nocross',
    'no_cross_nav': 'nocross',
    'noearlystop': 'noearlystop',
    'no_early_stop': 'noearlystop',
    'nomerge': 'nomerge',
    'no_merge': 'nomerge',
}

METHOD_STYLE = {
    'recencybucket': dict(color='red', marker='v'),
    'fixedbucket': dict(color='darkviolet', marker='o'),
    'nocross': dict(color='blue', marker='s'),
    'noearlystop': dict(color='green', marker='^'),
    'nomerge': dict(color='orange', marker='D'),
}

# Draw ablations first and the complete method last so it stays visible.
PLOT_ORDER = (
    'fixedbucket', 'nocross', 'noearlystop', 'nomerge', 'recencybucket'
)
LEGEND_ORDER = (
    'recencybucket', 'fixedbucket', 'nocross', 'noearlystop', 'nomerge'
)

LINE_WIDTH = 4
MARKER_SIZE = 13
MARKER_EDGE_WIDTH = 3

NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
TABLE_ROW = re.compile(
    rf'^\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})(?:\s+.*)?$'
)


def parse_qps_recall(log_file: Path) -> tuple[list[float], list[float]]:
    '''Read the first four numeric columns: ef, recall, time, and QPS.'''
    recall: list[float] = []
    qps: list[float] = []
    header_seen = False

    with log_file.open('r', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if not header_seen:
                if 'ef' in line and 'recall' in line and 'QPS' in line:
                    header_seen = True
                continue

            match = TABLE_ROW.match(line)
            if match:
                recall.append(float(match.group(2)))
                qps.append(float(match.group(4)))

    if not header_seen:
        print(f'Warning: table header not found in {log_file.name}')
    elif not recall:
        print(f'Warning: no table rows parsed from {log_file.name}')
    return recall, qps


def filter_from_recall(
    recall: list[float],
    qps: list[float],
    minimum: float = 0.8,
) -> tuple[list[float], list[float]]:
    '''Keep one point before the first point at or above minimum recall.'''
    first = next((i for i, value in enumerate(recall) if value >= minimum), -1)
    if first < 0:
        return [], []
    start = max(0, first - 1)
    pairs = sorted(zip(recall[start:], qps[start:]))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def canonical_variant(raw_name: str) -> str | None:
    compact = raw_name.lower().replace('-', '_')
    return VARIANT_ALIASES.get(compact)


def discover_logs(
    log_dir: Path,
    dataset: str,
    decay: str,
) -> dict[str, dict[str, Path]]:
    '''Return alpha_code -> variant -> log path.'''
    result: dict[str, dict[str, Path]] = {}
    name_pattern = re.compile(
        rf'^{re.escape(dataset)}_(.+)_{re.escape(decay)}_a(\d+)\.log$',
        re.IGNORECASE,
    )

    for log_file in sorted(log_dir.glob(f'{dataset}_*_{decay}_a*.log')):
        match = name_pattern.match(log_file.name)
        if not match:
            continue
        variant = canonical_variant(match.group(1))
        if variant is None:
            print(f'Warning: unknown ablation variant in {log_file.name}')
            continue
        alpha_code = match.group(2)
        result.setdefault(alpha_code, {})[variant] = log_file
    return result


def alpha_code_to_value(code: str) -> float:
    if code.startswith('0') and len(code) > 1:
        return float(f'0.{code[1:]}')
    return float(code)


def alpha_value_to_code(value: str) -> str:
    return format(float(value), 'g').replace('.', '')


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description='Plot QPS-recall curves for ablation experiment logs.'
    )
    parser.add_argument('--log-dir', type=Path, default=script_dir.parent / 'logs')
    parser.add_argument('--output-dir', type=Path, default=script_dir / 'output')
    parser.add_argument('--dataset', default='gist1m')
    parser.add_argument('--decay', default='exp001')
    parser.add_argument(
        '--alphas', nargs='*',
        help='optional alpha filter, for example: --alphas 0.2 0.4',
    )
    parser.add_argument('--min-recall', type=float, default=0.8)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def plot_alpha(
    alpha_code: str,
    files: dict[str, Path],
    args: argparse.Namespace,
):
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for variant in VARIANT_LABELS:
        log_file = files.get(variant)
        if log_file is None:
            continue
        recall, qps = parse_qps_recall(log_file)
        recall, qps = filter_from_recall(recall, qps, args.min_recall)
        if recall:
            curves[variant] = (recall, qps)
        else:
            print(
                f'Warning: {log_file.name} has no recall >= {args.min_recall}'
            )

    if not curves:
        print(f'Warning: no plottable curves for alpha code a{alpha_code}')
        return None

    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    all_qps: list[float] = []

    for variant in PLOT_ORDER:
        if variant not in curves:
            continue
        recall, qps = curves[variant]
        style = METHOD_STYLE[variant]
        axis.plot(
            recall,
            qps,
            color=style['color'],
            marker=style['marker'],
            label=VARIANT_LABELS[variant],
            lw=LINE_WIDTH,
            ms=MARKER_SIZE,
            mec=style['color'],
            mew=MARKER_EDGE_WIDTH,
            mfc='none',
            zorder=10 if variant == 'recencybucket' else 3,
        )
        all_qps.extend(value for value in qps if value > 0)

    alpha = alpha_code_to_value(alpha_code)
    axis.set_xlabel(f'Recall@10(alpha={alpha:g})', fontsize=26)
    axis.set_ylabel('QPS', fontsize=26)
    axis.set_yscale('log')

    if all_qps:
        padding = 1.6
        axis.set_ylim(
            bottom=max(min(all_qps) / padding, 1e-3),
            top=max(all_qps) * padding,
        )

    axis.set_xlim(args.min_recall, 1.005)
    if abs(args.min_recall - 0.8) < 1e-12:
        axis.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    axis.tick_params(axis='both', labelsize=22)
    axis.grid(True, linewidth=4, alpha=0.3)
    figure.tight_layout()

    output_file = args.output_dir / (
        f'qps_recall_ablation_{args.dataset}_{args.decay}_a{alpha_code}.png'
    )
    figure.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Figure saved to: {output_file}')

    handles, labels = axis.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    labels = [
        VARIANT_LABELS[variant]
        for variant in LEGEND_ORDER
        if VARIANT_LABELS[variant] in handle_by_label
    ]
    handles = [handle_by_label[label] for label in labels]
    if not args.show:
        plt.close(figure)
    return handles, labels


def save_legend(
    handles,
    labels: list[str],
    output_file: Path,
    show: bool,
) -> None:
    figure = plt.figure(figsize=(max(10, 4.4 * len(labels)), 2.0))
    legend = figure.legend(
        handles,
        labels,
        loc='center',
        ncol=len(labels),
        frameon=False,
        fontsize=30,
        markerscale=2.5,
        handlelength=3.0,
        handletextpad=1.0,
        columnspacing=1.2,
    )
    for handle in legend.legend_handles:
        if hasattr(handle, 'set_linewidth'):
            handle.set_linewidth(6.0)
        if hasattr(handle, 'set_markeredgewidth'):
            handle.set_markeredgewidth(5.0)

    figure.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.25)
    print(f'Legend saved to: {output_file}')
    if not show:
        plt.close(figure)


def main() -> int:
    args = parse_args()
    args.log_dir = args.log_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.log_dir.is_dir():
        raise FileNotFoundError(f'log directory not found: {args.log_dir}')
    if not 0.0 <= args.min_recall < 1.0:
        raise ValueError('--min-recall must be in [0, 1)')

    groups = discover_logs(args.log_dir, args.dataset, args.decay)
    if args.alphas:
        requested = {alpha_value_to_code(value) for value in args.alphas}
        groups = {code: files for code, files in groups.items() if code in requested}
    if not groups:
        raise FileNotFoundError(
            f'no matching logs in {args.log_dir} for '
            f'{args.dataset}_*_{args.decay}_a*.log'
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    legend_data = None
    figure_count = 0
    for alpha_code in sorted(groups, key=alpha_code_to_value):
        result = plot_alpha(alpha_code, groups[alpha_code], args)
        if result is not None:
            figure_count += 1
            if legend_data is None:
                legend_data = result

    if legend_data is not None:
        legend_file = args.output_dir / (
            f'qps_recall_ablation_{args.dataset}_{args.decay}_legend.png'
        )
        save_legend(*legend_data, legend_file, args.show)

    if args.show:
        plt.show()
    print(f'Generated {figure_count} ablation QPS-recall figure(s).')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f'error: {error}')
        raise SystemExit(1)
