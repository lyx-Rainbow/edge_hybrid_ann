#!/usr/bin/env python3
"""Write a small Markdown record for the index-build bar figures."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "4_Results" / "build_measure" / "index_metrics.json"
OUT = ROOT / "4_Results" / "build_measure" / "BUILD_EXPERIMENT_RECORD.md"
DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]
METHODS = [("curator", "Proposed"), ("diskivf", "DiskIVF"),
           ("spann", "SPANN"), ("prefilter", "PreFilter")]


def table(section, title, fmt=".2f"):
    lines = [f"## {title}", "", "| Dataset | " +
             " | ".join(label for _, label in METHODS) + " |",
             "|---|" + "|".join("---:" for _ in METHODS) + "|"]
    data = metrics.get(section, {})
    for dataset in DATASETS:
        row = [dataset]
        for key, _ in METHODS:
            value = data.get(dataset, {}).get(key)
            row.append("--" if value is None else format(float(value), fmt))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


if not METRICS.exists():
    raise SystemExit(f"missing {METRICS}")
metrics = json.load(open(METRICS, encoding="utf-8"))
lines = [
    "# Index Build Experiment Record",
    "",
    f"> Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    "> Methods: Curator (Proposed), DiskIVF-PostFiltering, "
    "SPANN-PostFiltering, PreFiltering",
    "> Datasets: sift1m, gist1m, arxiv, yfcc100m, wit",
    "",
    "## Measurement Definitions",
    "",
]
for key in ("query_memory_mb", "build_time_s", "index_volume_mb",
            "volume_breakdown_mb"):
    value = metrics.get("definitions", {}).get(key)
    if value:
        lines.append(f"- **{key}**: {value}")
lines.extend(["",
             "## Figures",
             "",
             "- `4_Results/fig_full/fig_full_memory.png` / `.svg`",
             "- `4_Results/fig_full/fig_full_build_time.png` / `.svg`",
             "- `4_Results/fig_full/fig_full_volume.png` / `.svg`",
             "- `4_Results/fig_full/fig_full_bars_legend.png` / `.svg`",
             ""])
lines.extend(table("query_memory_mb", "Query-time Peak RSS (MB)"))
lines.extend(table("build_time_s", "Index Build Time (s)", ".1f"))
lines.extend(table("index_volume_mb", "Memory + External (MB)"))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUT)