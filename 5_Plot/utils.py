"""Shared helpers for plotting scripts."""

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "4_Results"

# Index name -> (display name, results subdir, color, marker)
INDEX_META = {
    "curator":   ("Proposed",  "Curator",        "#1f77b4", "o"),
    "diskivf":   ("DiskIVF",   "DiskIVF",        "#ff7f0e", "s"),
    "spann":     ("SPANN",     "SPANN",          "#2ca02c", "D"),
    "prefilter": ("PreFilter", "Pre-Filtering",  "#d62728", "^"),
}


def load_sweep_results(index_key: str, dataset: str):
    """Load sweep result JSON. Returns dict or None."""
    path = RESULTS_DIR / INDEX_META[index_key][1] / f"sweep_{dataset}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_benchmark_results(index_key: str, dataset: str):
    """Load single-run benchmark result JSON. Returns dict or None."""
    meta = INDEX_META[index_key]
    # Try index-specific naming first
    for name in [meta[0].lower(), index_key]:
        path = RESULTS_DIR / meta[1] / f"{name}_{dataset}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def sweep_to_xy(data: dict) -> tuple[list, list]:
    """Extract (recall, latency) pairs from sweep_results."""
    points = data.get("sweep_results", [])
    x = [p["recall"] for p in points]
    y = [p["latency_ms"] for p in points]
    return x, y


def get_index_memory_mb(data: dict) -> float:
    """Extract memory info: prefer query-peak RSS, fall back to old fields."""
    # New field: query-time peak RSS (Curator/DiskIVF/SPANN)
    mem = data.get("memory", {})
    for key in ("rss_peak_query_mb", "peak_rss_during_build_mb", "peak_rss_build_mb"):
        if key in mem and mem[key]:
            return float(mem[key])
    # Direct field (PreFiltering or old data)
    for key in ("rss_peak_query_mb", "peak_rss_build_mb"):
        if key in data and data[key]:
            return float(data[key])
    # Sweep results top-level (old format fallback)
    if "rss_after_build_mb" in data:
        return float(data["rss_after_build_mb"])
    return float(data.get("index_memory_mb", 0))


def get_build_time_s(data: dict) -> float:
    """Extract build time in seconds."""
    return float(data.get("build_time_s", 0))
