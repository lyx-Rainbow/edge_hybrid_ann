"""
QueryProfiler — time-profile Curator queries at step-level granularity.

Usage:
    from query_profiler import QueryProfiler

    profiler = QueryProfiler(index)
    profile = profiler.profile_single(x, k=10, tenant_id=3)
    df = profiler.profile_batch(queries, k=10, tenant_ids=[...])
    grouped = profiler.profile_by_selectivity(queries, query_info, k=10)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow import from Curator/python
_sys_added = False


def _ensure_curator_import():
    global _sys_added
    if not _sys_added:
        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent / "Curator" / "python")
        )
        _sys_added = True


class QueryProfiler:
    """Per-step query-time profiler for Curator index."""

    def __init__(self, index):
        """*index* must be a Curator instance."""
        self.index = index
        _ensure_curator_import()

    # ------------------------------------------------------------------
    # Single query
    # ------------------------------------------------------------------

    def profile_single(
        self, x: np.ndarray, k: int, tenant_id: int | None = None
    ) -> dict[str, Any]:
        """Profile a single query.  Returns (result_ids, profile_dict)."""
        result, profile = self.index.query_with_profile(x, k, tenant_id)
        return {"result_ids": result, **profile}

    # ------------------------------------------------------------------
    # Batch profile
    # ------------------------------------------------------------------

    def profile_batch(
        self,
        queries: np.ndarray,
        k: int,
        tenant_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Profile a batch of queries.  Returns list of per-query dicts."""
        results = []
        n = len(queries)
        for i in range(n):
            tid = None if tenant_ids is None else tenant_ids[i]
            r = self.profile_single(queries[i], k, tid)
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # By selectivity
    # ------------------------------------------------------------------

    def profile_by_selectivity(
        self,
        queries: np.ndarray,
        query_info: list[dict],
        k: int,
        tenant_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Group queries by selectivity bucket and collect per-step stats.

        query_info must be a list of dicts with at least 'bucket' and
        'selectivity' keys (matching the format from gt_computing.py).
        """
        buckets: dict[str, list[dict]] = {}

        n = len(queries)
        for i in range(n):
            tid = None if tenant_ids is None else tenant_ids[i]
            r = self.profile_single(queries[i], k, tid)
            bucket = query_info[i].get("bucket", "unknown")
            buckets.setdefault(bucket, []).append(r)

        summary = {}
        for bname, profiles in buckets.items():
            metric_names = [
                "total_search_time_ms",
                "beam_search_time_ms",
                "frontier_search_time_ms",
                "pq_table_build_time_ms",
                "pq_distance_compute_time_ms",
                "exact_distance_compute_time_ms",
                "candidate_merge_time_ms",
                "rerank_time_ms",
            ]
            agg = {"count": len(profiles)}
            for mn in metric_names:
                vals = [p.get(mn, 0.0) or 0.0 for p in profiles]
                if vals:
                    agg[f"{mn}_mean"] = float(np.mean(vals))
                    agg[f"{mn}_p50"] = float(np.percentile(vals, 50))
                    agg[f"{mn}_p95"] = float(np.percentile(vals, 95))
            summary[bname] = agg

        return summary

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_profile(profile: dict[str, Any]) -> None:
        """Pretty-print a single-query profile."""
        def ms(key: str) -> str:
            v = profile.get(key, 0.0) or 0.0
            return f"{v:8.2f} ms"

        total = profile.get("total_search_time_ms", 1.0) or 1.0

        def pct(key: str) -> str:
            v = profile.get(key, 0.0) or 0.0
            return f"{v / total * 100:5.1f}%"

        print(f"\nQuery Profile (total: {ms('total_search_time_ms')})")
        print(f"  {'Phase':<32s} {'Time':>10s} {'%':>7s}")
        print(f"  {'─'*32} {'─'*10} {'─'*7}")
        print(f"  {'Beam Search':<32s} {ms('beam_search_time_ms')} {pct('beam_search_time_ms')}")
        print(f"  {'PQ Table Build':<32s} {ms('pq_table_build_time_ms')} {pct('pq_table_build_time_ms')}")
        print(f"  {'Frontier Search':<32s} {ms('frontier_search_time_ms')} {pct('frontier_search_time_ms')}")
        print(f"    nodes popped:     {profile.get('frontier_nodes_popped', 0)}")
        print(f"    shortlists scanned: {profile.get('frontier_shortlists_scanned', 0)}")
        print(f"    children expanded:  {profile.get('frontier_children_expanded', 0)}")
        print(f"  {'PQ Distance Compute':<32s} {ms('pq_distance_compute_time_ms')} {pct('pq_distance_compute_time_ms')}")
        print(f"  {'Exact Distance Compute':<32s} {ms('exact_distance_compute_time_ms')} {pct('exact_distance_compute_time_ms')}")
        print(f"  {'Candidate Merge':<32s} {ms('candidate_merge_time_ms')} {pct('candidate_merge_time_ms')}")
        print(f"  {'ADC Rerank':<32s} {ms('rerank_time_ms')} {pct('rerank_time_ms')}")
        print(f"    rerank count:     {profile.get('rerank_count', 0)}")
        print(f"  {'─'*32} {'─'*10} {'─'*7}")

    @staticmethod
    def print_selectivity_summary(summary: dict[str, Any]) -> None:
        """Pretty-print a selectivity-bucket summary."""
        print(f"\n{'Bucket':<32s} {'Count':>6s} {'Total':>8s} {'Beam':>8s} {'Frontier':>9s} {'PQDist':>8s} {'Rerank':>8s}")
        print(f"{'─'*32} {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*8} {'─'*8}")
        for bname in sorted(summary.keys()):
            b = summary[bname]
            print(
                f"  {bname:<30s} {b['count']:6d} "
                f"{b.get('total_search_time_ms_mean', 0):8.2f} "
                f"{b.get('beam_search_time_ms_mean', 0):8.2f} "
                f"{b.get('frontier_search_time_ms_mean', 0):9.2f} "
                f"{b.get('pq_distance_compute_time_ms_mean', 0):8.2f} "
                f"{b.get('rerank_time_ms_mean', 0):8.2f}"
            )
