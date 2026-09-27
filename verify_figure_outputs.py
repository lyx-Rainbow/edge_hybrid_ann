#!/usr/bin/env python3
"""Check that the split/grid bar figures and the text-style config exist.

Run after ``5_Plot/fig_full_bars.py`` and the latency figure scripts.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "4_Results/fig_full"
STYLE_PATH = ROOT / "5_Plot/figure_text_style.json"
DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]
METRICS = ["memory", "build_time", "volume"]

errors = []

if not STYLE_PATH.exists():
    errors.append(f"missing text style config: {STYLE_PATH}")
else:
    try:
        style = json.load(open(STYLE_PATH, encoding="utf-8-sig"))
    except Exception as exc:
        errors.append(f"cannot read {STYLE_PATH}: {exc}")
    else:
        for key in ("global_scale", "label_scale", "tick_scale",
                    "title_scale", "label_font_weight", "tick_font_weight",
                    "title_font_weight"):
            if key not in style:
                errors.append(f"text style missing key: {key}")
        content = style.get("content", {})
        for key in ("bar_memory_ylabel", "bar_build_time_ylabel",
                    "bar_volume_ylabel", "bar_dataset_xlabel"):
            if key not in content:
                errors.append(f"text content missing key: {key}")

for metric in METRICS:
    for stem in [f"fig_full_{metric}", f"fig_full_{metric}_grid"]:
        for suffix in (".png", ".svg"):
            path = FIG_DIR / f"{stem}{suffix}"
            if not path.exists():
                errors.append(f"missing {path}")
    for dataset in DATASETS:
        for suffix in (".png", ".svg"):
            path = FIG_DIR / f"fig_full_{metric}_{dataset}{suffix}"
            if not path.exists():
                errors.append(f"missing {path}")

for suffix in (".png", ".svg"):
    path = FIG_DIR / f"fig_full_bars_legend{suffix}"
    if not path.exists():
        errors.append(f"missing {path}")

print(f"figure-output errors={len(errors)}")
for item in errors:
    print("ERROR:", item)
raise SystemExit(1 if errors else 0)
