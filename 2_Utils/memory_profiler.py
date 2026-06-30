"""
MemoryProfiler — component-level memory breakdown for Curator index.

Usage:
    from memory_profiler import MemoryProfiler

    profiler = MemoryProfiler(index)
    profiler.print_breakdown()          # human-readable table
    snapshots = profiler.build_phase_snapshots(...)  # track during build
    profiler.compare(snap1, snap2)      # diff before/after
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow import from memory_utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_utils import getCurrentRSS  # noqa: E402


class MemoryProfiler:
    """Component-level memory analysis for Curator index."""

    def __init__(self, index):
        """*index* must be a Curator instance (or compatible)."""
        self.index = index

    # ------------------------------------------------------------------
    # Breakdown
    # ------------------------------------------------------------------

    def get_breakdown(self) -> dict[str, Any]:
        """Return structured breakdown dict from C++ get_memory_breakdown()."""
        raw = self.index.get_memory_breakdown()
        return raw  # already a nested dict from curator.py

    def get_snapshot(self) -> dict[str, Any]:
        """Full memory snapshot: RSS + index-reported + component breakdown."""
        rss = getCurrentRSS()
        idx_total = self.index.get_index_memory_bytes()
        breakdown = self.get_breakdown()
        return {
            "rss_bytes": rss,
            "rss_mb": rss / (1024 * 1024),
            "index_total_bytes": idx_total,
            "index_total_mb": idx_total / (1024 * 1024),
            "breakdown": breakdown,
            "overhead_bytes": rss - idx_total,
            "overhead_mb": (rss - idx_total) / (1024 * 1024),
        }

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_breakdown(self) -> None:
        """Print a human-readable component memory table."""
        snap = self.get_snapshot()
        bd = snap["breakdown"]

        def mb(b: int) -> float:
            return b / (1024 * 1024)

        lines = []
        total = bd["total_bytes"]
        pct = lambda b: (b / total * 100) if total > 0 else 0.0

        lines.append(f"{'Component':<38s} {'Size (MB)':>10s} {'%':>7s}")
        lines.append(f"{'─'*38} {'─'*10} {'─'*7}")

        tree = bd["tree"]
        lines.append(f"{'Tree Structure':<38s} {'':>10s} {'':>7s}")
        lines.append(
            f"  ├─ Node attributes{'':<21s} {mb(tree['node_attrs_bytes']):10.2f} {pct(tree['node_attrs_bytes']):6.1f}%"
        )
        lines.append(
            f"  ├─ Centroids{'':<27s} {mb(tree['centroids_bytes']):10.2f} {pct(tree['centroids_bytes']):6.1f}%"
        )
        lines.append(
            f"  └─ Num nodes{'':<26s} {tree['num_nodes']:>10d} {'':>7s}"
        )

        lines.append(
            f"{'Bloom Filters':<38s} {mb(bd['bloom_filters_bytes']):10.2f} {pct(bd['bloom_filters_bytes']):6.1f}%"
        )

        sl = bd["shortlists"]
        lines.append(f"{'Short Lists':<38s} {'':>10s} {'':>7s}")
        lines.append(
            f"  ├─ Hash-table overhead{'':<16s} {mb(sl['overhead_bytes']):10.2f} {pct(sl['overhead_bytes']):6.1f}%"
        )
        lines.append(
            f"  └─ Payload (vid data){'':<16s} {mb(sl['payload_bytes']):10.2f} {pct(sl['payload_bytes']):6.1f}%"
        )

        lines.append(
            f"{'Vector Indices (leaf)':<38s} {mb(bd['vector_indices_bytes']):10.2f} {pct(bd['vector_indices_bytes']):6.1f}%"
        )

        alloc = bd["id_allocators"]
        lines.append(f"{'ID Allocators':<38s} {'':>10s} {'':>7s}")
        lines.append(
            f"  ├─ Vector allocator{'':<19s} {mb(alloc['vector_allocator_bytes']):10.2f} {pct(alloc['vector_allocator_bytes']):6.1f}%"
        )
        lines.append(
            f"  └─ Tenant allocator{'':<19s} {mb(alloc['tenant_allocator_bytes']):10.2f} {pct(alloc['tenant_allocator_bytes']):6.1f}%"
        )

        pq = bd["pq"]
        lines.append(f"{'PQ':<38s} {'':>10s} {'':>7s}")
        lines.append(
            f"  ├─ Codebook{'':<27s} {mb(pq['codebook_bytes']):10.2f} {pct(pq['codebook_bytes']):6.1f}%"
        )
        lines.append(
            f"  └─ Codes (vid_to_pq_code){'':<13s} {mb(pq['codes_bytes']):10.2f} {pct(pq['codes_bytes']):6.1f}%"
        )

        lines.append(
            f"{'Flash Index (offset maps)':<38s} {mb(bd['flash_index_bytes']):10.2f} {pct(bd['flash_index_bytes']):6.1f}%"
        )
        lines.append(
            f"{'Raw Vectors Buffer':<38s} {mb(bd['raw_vectors_buffer_bytes']):10.2f} {pct(bd['raw_vectors_buffer_bytes']):6.1f}%"
        )

        tc = bd["temp_cache"]
        lines.append(f"{'Temp Index Cache':<38s} {'':>10s} {'':>7s}")
        lines.append(
            f"  ├─ Index structures{'':<19s} {mb(tc['index_bytes']):10.2f} {pct(tc['index_bytes']):6.1f}%"
        )
        lines.append(
            f"  └─ Qualified vecs{'':<21s} {mb(tc['qualified_vecs_bytes']):10.2f} {pct(tc['qualified_vecs_bytes']):6.1f}%"
        )

        lines.append(f"{'─'*38} {'─'*10} {'─'*7}")
        lines.append(
            f"{'Total (index-reported)':<38s} {snap['index_total_mb']:10.2f} {'100.0':>7s}%"
        )
        lines.append(
            f"{'RSS (OS-reported)':<38s} {snap['rss_mb']:10.2f} {'':>7s}"
        )
        lines.append(
            f"{'Python/OS overhead':<38s} {snap['overhead_mb']:10.2f} {'':>7s}"
        )

        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Build-phase snapshots
    # ------------------------------------------------------------------

    def snapshot_at(self, label: str) -> dict[str, Any]:
        """Take a labelled memory snapshot."""
        snap = self.get_snapshot()
        snap["label"] = label
        return snap

    def compare_snapshots(
        self, before: dict[str, Any], after: dict[str, Any]
    ) -> dict[str, Any]:
        """Return per-component delta between two snapshots."""
        b_bd = before["breakdown"]
        a_bd = after["breakdown"]

        def delta(key_path: list[str]) -> float:
            v_before = _get_nested(b_bd, key_path)
            v_after = _get_nested(a_bd, key_path)
            if isinstance(v_before, (int, float)) and isinstance(v_after, (int, float)):
                return (v_after - v_before) / (1024 * 1024)
            return 0.0

        components = [
            (["tree", "node_attrs_bytes"], "Tree: node attrs"),
            (["tree", "centroids_bytes"], "Tree: centroids"),
            (["bloom_filters_bytes"], "Bloom filters"),
            (["shortlists", "overhead_bytes"], "Shortlists: overhead"),
            (["shortlists", "payload_bytes"], "Shortlists: payload"),
            (["vector_indices_bytes"], "Vector indices"),
            (["id_allocators", "vector_allocator_bytes"], "Vector allocator"),
            (["id_allocators", "tenant_allocator_bytes"], "Tenant allocator"),
            (["pq", "codebook_bytes"], "PQ codebook"),
            (["pq", "codes_bytes"], "PQ codes"),
            (["flash_index_bytes"], "Flash index"),
            (["raw_vectors_buffer_bytes"], "Raw vectors buffer"),
            (["temp_cache", "index_bytes"], "Temp cache: index"),
            (["temp_cache", "qualified_vecs_bytes"], "Temp cache: vecs"),
            (["total_bytes"], "Total (index)"),
        ]

        diffs = {}
        for path, name in components:
            diffs[name] = delta(path)

        diffs["RSS"] = (after["rss_bytes"] - before["rss_bytes"]) / (1024 * 1024)
        return diffs

    def print_compare(
        self, before: dict[str, Any], after: dict[str, Any]
    ) -> None:
        """Print a before/after comparison table."""
        diffs = self.compare_snapshots(before, after)
        print(f"\n{'Component':<30s} {'Δ (MB)':>10s}")
        print(f"{'─'*30} {'─'*10}")
        for name, d in diffs.items():
            print(f"  {name:<28s} {d:10.2f}")
        print(f"{'─'*30} {'─'*10}")


def _get_nested(d: dict, keys: list[str]) -> Any:
    for k in keys:
        d = d[k]
    return d
