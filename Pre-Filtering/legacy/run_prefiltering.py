"""
Pre-Filtering index: brute-force filtered search, all data in memory.

Build:  load all vectors + metadata into RAM (no index structure).
Query:  filter candidates by label/predicate, then brute-force L2.

Usage:
    python Pre-Filtering/run_prefiltering.py --dataset yfcc100m
    python Pre-Filtering/run_prefiltering.py --config config/Pre-Filtering/prefiltering_yfcc100m.json
"""

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_Utils"))
from memory_utils import getPeakRSS, getCurrentRSS  # noqa: E402
from predicate import (  # noqa: E402
    build_inverted_index,
    compute_filter_selectivity,
    compute_qualified_indices,
)


# ---------------------------------------------------------------------------
# Index class
# ---------------------------------------------------------------------------

class PreFilteringIndex:
    """Brute-force filtered search. All vectors + metadata in memory."""

    def __init__(self):
        self.train_vecs = None       # [n, d] float32
        self.train_mds = None        # list[list[int]]
        self.label_to_indices = None # dict[int, np.ndarray]
        self.n_labels = 0
        self.d = 0
        self.n = 0
        self._built = False

    def build(self, train_vecs: np.ndarray, train_mds: list):
        """One-shot build: store data and build inverted index."""
        self.d = train_vecs.shape[1]
        self.n = len(train_vecs)
        self.train_vecs = np.ascontiguousarray(train_vecs, dtype=np.float32)
        self.train_mds = train_mds

        all_labels = set().union(*train_mds)
        self.n_labels = max(all_labels) + 1 if all_labels else 0
        self.label_to_indices = build_inverted_index(train_mds, self.n_labels)
        self._built = True

    def query(self, x: np.ndarray, k: int, tenant_id: int) -> list[int]:
        """Single-label filtered top-k."""
        candidates = self.label_to_indices.get(int(tenant_id),
                                               np.array([], dtype=np.int32))
        return self._search_candidates(x, k, candidates)

    def query_with_complex_predicate(
        self, x: np.ndarray, k: int, predicate: str
    ) -> list[int]:
        """Complex-predicate filtered top-k."""
        candidates = compute_qualified_indices(predicate, self.train_mds)
        return self._search_candidates(x, k, candidates)

    def _search_candidates(
        self, x: np.ndarray, k: int, candidates: np.ndarray
    ) -> list[int]:
        """Brute-force L2 among candidates, return top-k indices."""
        if len(candidates) == 0:
            return [-1] * k
        cand_vecs = self.train_vecs[candidates]
        dists = np.sum((cand_vecs - x.astype(np.float32)) ** 2, axis=1)
        k_act = min(k, len(candidates))
        top_k = np.argpartition(dists, k_act - 1)[:k_act]
        top_k = top_k[np.argsort(dists[top_k])]
        result = candidates[top_k].tolist()
        if len(result) < k:
            result.extend([-1] * (k - len(result)))
        return result

    def get_index_memory_bytes(self) -> int:
        mem = 0
        if self.train_vecs is not None:
            mem += self.train_vecs.nbytes
        if self.label_to_indices:
            for v in self.label_to_indices.values():
                mem += v.nbytes
        return mem


# ---------------------------------------------------------------------------
# Runner (follows Curator/run_curator.py pattern)
# ---------------------------------------------------------------------------

BUILTIN_DEFAULTS = {
    "arxiv":    {"d": 384, "n_labels": 100,  "k": 10, "n_queries": 1000, "num_warmup": 20},
    "yfcc100m": {"d": 192, "n_labels": 1000, "k": 10, "n_queries": 1000, "num_warmup": 20},
    "arxiv_small": {"d": 384, "n_labels": 100, "k": 10, "n_queries": 500, "num_warmup": 10},
    "yfcc100m_small": {"d": 192, "n_labels": 200, "k": 10, "n_queries": 500, "num_warmup": 10},
}


def load_config(config_path, dataset):
    if config_path:
        with open(config_path) as f:
            return json.load(f)
    return BUILTIN_DEFAULTS[dataset]


def load_data(dataset, data_dir):
    gt_dir = Path(data_dir) / "ground_truth" / dataset
    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        train_mds = pickle.load(f)
    query_vecs = np.load(gt_dir / "query_vecs.npy").astype(np.float32)
    query_labels = np.load(gt_dir / "query_labels.npy").astype(np.int32)
    with open(gt_dir / "query_info.json") as f:
        query_info = json.load(f)
    gt = np.load(gt_dir / "ground_truth.npy")
    return train_vecs, train_mds, query_vecs, query_labels, query_info, gt


def load_cp_data(dataset, data_dir):
    cp_dir = Path(data_dir) / "ground_truth" / dataset / "complex_predicate"
    query_vecs = np.load(cp_dir / "query_vecs.npy").astype(np.float32)
    with open(cp_dir / "filters.json") as f:
        filters_info = json.load(f)
    gt_dict = {}
    for formula in filters_info["filters"]:
        safe = formula.replace(" ", "_")
        gt_dict[formula] = np.load(cp_dir / f"gt_{safe}.npy")
    return query_vecs, filters_info, gt_dict


def compute_recall(pred_ids, gt_ids, k):
    valid = set(int(i) for i in gt_ids[:k] if i >= 0)
    if not valid:
        return 1.0
    return len(set(pred_ids[:k]) & valid) / len(valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["arxiv", "arxiv_small", "yfcc100m", "yfcc100m_small"])
    parser.add_argument("--data_dir", type=str, default="1_Data")
    parser.add_argument("--output_dir", type=str, default="4_Results/Pre-Filtering")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config, None)
        dataset = cfg.get("dataset", args.dataset)
    elif args.dataset:
        cfg = load_config(None, args.dataset)
        dataset = args.dataset
    else:
        parser.error("Need --config or --dataset")

    k = cfg["k"]
    num_warmup = cfg.get("num_warmup", 20)

    # ---- Load ----
    print(f"\n{'='*60}\nPre-Filtering Benchmark: {dataset}\n{'='*60}")
    print("Loading data...")
    t0 = time.perf_counter()
    train_vecs, train_mds, query_vecs, query_labels, query_info, sl_gt = \
        load_data(dataset, args.data_dir)
    rss_after_load = getCurrentRSS() / (1024 * 1024)
    print(f"  Train: {train_vecs.shape[0]:,} x {train_vecs.shape[1]}, "
          f"labels={cfg['n_labels']}")
    print(f"  Queries: {len(query_vecs)} (single-label)")
    print(f"  RSS after load: {rss_after_load:.1f} MB")

    # ---- Build ----
    print(f"\n{'='*60}\nBuilding index\n{'='*60}")
    index = PreFilteringIndex()
    t0 = time.perf_counter()
    index.build(train_vecs, train_mds)
    t_build = time.perf_counter() - t0
    rss_after_build = getCurrentRSS() / (1024 * 1024)
    peak_build = getPeakRSS() / (1024 * 1024)
    idx_mem = index.get_index_memory_bytes() / (1024 * 1024)
    print(f"  Build: {t_build:.1f}s, memory: {idx_mem:.1f} MB, "
          f"RSS: {rss_after_build:.1f} MB, peak: {peak_build:.1f} MB")

    # ---- Single-label queries ----
    print(f"\n{'='*60}\nSingle-label queries (k={k})\n{'='*60}")
    for i in range(min(num_warmup, len(query_vecs))):
        index.query(query_vecs[i], k, int(query_labels[i]))

    latencies, recalls = [], []
    bucket_lats, bucket_recs = {}, {}
    empty = 0
    rss_before_q = getCurrentRSS() / (1024 * 1024)

    for i in range(len(query_vecs)):
        t0 = time.perf_counter()
        res = index.query(query_vecs[i], k, int(query_labels[i]))
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        r = compute_recall(res, sl_gt[i], k)
        recalls.append(r)
        if not res or all(x == -1 for x in res):
            empty += 1
        b = query_info[i]["bucket"]
        bucket_lats.setdefault(b, []).append(ms)
        bucket_recs.setdefault(b, []).append(r)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(query_vecs)} done")

    rss_after_q = getCurrentRSS() / (1024 * 1024)
    peak_q = getPeakRSS() / (1024 * 1024)

    sl_results = {
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "qps": float(1000.0 / np.mean(latencies)),
        "avg_recall": float(np.mean(recalls)),
        "min_recall": float(np.min(recalls)),
        "empty": empty,
        "rss_before_mb": float(rss_before_q),
        "rss_after_mb": float(rss_after_q),
        "peak_rss_mb": float(peak_q),
        "per_bucket": {},
    }
    for b in sorted(bucket_lats):
        lats = bucket_lats[b]
        recs = bucket_recs[b]
        sl_results["per_bucket"][b] = {
            "count": len(lats),
            "avg_latency_ms": float(np.mean(lats)),
            "qps": float(1000 / np.mean(lats)) if np.mean(lats) > 0 else 0,
            "avg_recall": float(np.mean(recs)),
        }

    print(f"\n  Avg recall@{k}: {sl_results['avg_recall']:.4f}, "
          f"QPS: {sl_results['qps']:.1f}, "
          f"latency: {sl_results['avg_latency_ms']:.2f} ms")

    # ---- Complex-predicate queries ----
    print(f"\n{'='*60}\nComplex-predicate queries (k={k})\n{'='*60}")
    cp_query_vecs, filters_info, cp_gt_dict = load_cp_data(dataset, args.data_dir)
    cp_filters = filters_info["filters"]
    cp_selectivities = filters_info.get("selectivities", {})
    n_cp = len(cp_query_vecs)
    print(f"  {len(cp_filters)} filters x {n_cp} queries")

    cp_results = {}
    total_cp_time = 0.0
    cp_all_recalls = []
    cp_all_lats = []

    for fi, formula in enumerate(cp_filters):
        print(f"  [{fi + 1}/{len(cp_filters)}] {formula}", flush=True)
        gt = cp_gt_dict[formula]
        sel = cp_selectivities.get(formula, compute_filter_selectivity(formula, train_mds))
        lats, recs = [], []
        t0 = time.perf_counter()
        for qi in range(n_cp):
            qt0 = time.perf_counter()
            res = index.query_with_complex_predicate(cp_query_vecs[qi], k, formula)
            ms = (time.perf_counter() - qt0) * 1000
            lats.append(ms)
            recs.append(compute_recall(res, gt[qi], k))
        elapsed = time.perf_counter() - t0
        total_cp_time += elapsed
        cp_results[formula] = {
            "selectivity": float(sel),
            "avg_latency_ms": float(np.mean(lats)),
            "qps": float(1000 / np.mean(lats)) if np.mean(lats) > 0 else 0,
            "avg_recall": float(np.mean(recs)),
            "time_s": float(elapsed),
        }
        cp_all_recalls.extend(recs)
        cp_all_lats.extend(lats)
        print(f"    sel={sel:.4f}, recall={cp_results[formula]['avg_recall']:.4f}, "
              f"lat={np.mean(lats):.2f}ms, time={elapsed:.1f}s")

    # ---- Save ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "dataset": dataset,
        "index": "PreFiltering",
        "k": k,
        "train_size": int(len(train_vecs)),
        "n_sl_queries": len(query_vecs),
        "n_cp_filters": len(cp_filters),
        "n_cp_queries": n_cp,
        "build_time_s": float(t_build),
        "index_memory_mb": float(idx_mem),
        "rss_after_load_mb": float(rss_after_load),
        "rss_after_build_mb": float(rss_after_build),
        "peak_rss_build_mb": float(peak_build),
        "single_label": sl_results,
        "complex_predicate": {
            "total_time_s": float(total_cp_time),
            "avg_recall": float(np.mean(cp_all_recalls)) if cp_all_recalls else 0,
            "avg_latency_ms": float(np.mean(cp_all_lats)) if cp_all_lats else 0,
            "per_filter": cp_results,
        },
    }

    out_path = output_dir / f"prefiltering_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results: {out_path.resolve()}")
    print(f"  SL recall@{k}: {sl_results['avg_recall']:.4f}, QPS: {sl_results['qps']:.1f}")
    if cp_all_recalls:
        print(f"  CP recall@{k}: {np.mean(cp_all_recalls):.4f}, "
              f"total time: {total_cp_time:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
