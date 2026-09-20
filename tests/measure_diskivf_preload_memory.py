#!/usr/bin/env python3
"""Measure DiskIVF query-time memory with all nprobe clusters preloaded.

This is an overhead-only experiment: search results are unchanged, but the
clusters selected for one query are loaded together instead of streamed one at
a time.  Peak child RSS is measured externally with /usr/bin/time -v.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]
NPROBE = 128
BIN = ROOT / "DiskIVF-PostFiltering/build/diskivf"
OUT_DIR = ROOT / "4_Results/build_measure/raw"


def measure(dataset: str):
    index_dir = Path("/home/lyx") / f"diskivf_{dataset}_build_sqrt"
    if not index_dir.exists():
        raise FileNotFoundError(index_dir)
    query_dir = ROOT / "4_Results/build_measure/query_subsets" / dataset
    if not query_dir.exists():
        raise FileNotFoundError(query_dir)
    out_path = Path("/tmp") / f"diskivf_preload_{dataset}_nprobe{NPROBE}.json"
    cmd = [
        "/usr/bin/time", "-v", str(BIN), "search",
        "--disk_dir", str(index_dir),
        "--queries", str(query_dir / "query_vecs.npy"),
        "--query_labels", str(query_dir / "query_labels.npy"),
        "--nprobe", str(NPROBE),
        "--preload-clusters",
        "--k", "10",
        "--output", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr)
    if not match:
        raise RuntimeError("cannot parse /usr/bin/time -v output")
    rss_mb = int(match.group(1)) / 1024.0
    record = {
        "method": "DiskIVF",
        "method_key": "diskivf",
        "dataset": dataset,
        "nprobe": NPROBE,
        "query_count": 1,
        "preload_clusters": True,
        "rss_peak_query_mb": rss_mb,
        "index_dir": str(index_dir),
        "measurement": (
            "one-query overhead experiment; all nprobe clusters are loaded "
            "into memory before scanning, peak child RSS measured with "
            "/usr/bin/time -v"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"diskivf_query_preload_{dataset}_nprobe{NPROBE}_q1.json"
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"{dataset}: rss={rss_mb:.1f} MiB -> {output.name}")


def main():
    for dataset in DATASETS:
        measure(dataset)


if __name__ == "__main__":
    main()