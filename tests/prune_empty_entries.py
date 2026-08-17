#!/usr/bin/env python3
"""Prune empty (params-only) entries from sweep JSONs.
An entry is empty when it has neither SL stats (qps) nor complex_predicate.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "4_Results/DiskIVF/sweep_gist1m_100k.json",
]

for path in FILES:
    d = json.load(open(path))
    before = len(d["sweep_results"])
    kept = [r for r in d["sweep_results"]
            if r.get("qps") is not None or r.get("complex_predicate")]
    removed = before - len(kept)
    if removed:
        d["sweep_results"] = kept
        d["n_combinations"] = len(kept)
        json.dump(d, open(path, "w"), indent=2)
        print(f"{path.name}: removed {removed} empty entry/ies, "
              f"{before} -> {len(kept)}")
    else:
        print(f"{path.name}: nothing to prune")
