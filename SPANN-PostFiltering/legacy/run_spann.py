"""
SPANN-PostFiltering index: fine-grained k-means partitions on disk, post-filtered search.

Same architecture as DiskIVF-PostFiltering, but with more partitions (higher nlist)
so that each partition contains fewer vectors, reducing per-partition load/search time.

Build:   k-means clustering (large nlist) → each partition saved as file on disk.
Query:   find nearest nprobe partitions by centroid distance → load from
         disk → filter by label/predicate → brute-force L2 on remainder.

Usage:
    python SPANN-PostFiltering/run_spann.py --dataset yfcc100m
    python SPANN-PostFiltering/run_spann.py --config config/SPANN-PostFiltering/spann_yfcc100m.json
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_Utils"))
from memory_utils import getCurrentRSS, QueryRSSSampler  # noqa: E402
from predicate import evaluate_predicate  # noqa: E402


# ---------------------------------------------------------------------------
# Index class (same architecture as DiskIVFIndex, larger nlist default)
# ---------------------------------------------------------------------------

class SPANNIndex:
    """Fine-grained disk-backed IVF index with post-filtering."""

    def __init__(
        self,
        nlist: int = 256,
        nprobe: int = 64,
        disk_dir: str | None = None,
    ):
        self.nlist = nlist
        self.nprobe = nprobe
        self.disk_dir = disk_dir

        self.d = 0
        self.n = 0
        self.centroids = None
        self.cluster_sizes = None
        self._built = False

    def build(self, train_vecs: np.ndarray, train_mds: list):
        import faiss

        self.d = train_vecs.shape[1]
        self.n = len(train_vecs)

        if self.disk_dir is None:
            self.disk_dir = "4_Results/SPANN-PostFiltering/disk_data"
        disk = Path(self.disk_dir)
        if disk.exists():
            shutil.rmtree(disk)
        disk.mkdir(parents=True)

        print(f"  Running k-means (nlist={self.nlist}, d={self.d})...")
        t0 = time.perf_counter()
        kmeans = faiss.Kmeans(
            self.d, self.nlist, niter=20, verbose=False,
            spherical=False, seed=42,
        )
        kmeans.train(train_vecs)
        self.centroids = kmeans.centroids.astype(np.float32)
        print(f"  K-means done in {time.perf_counter() - t0:.1f}s")

        print("  Assigning vectors to clusters...")
        t0 = time.perf_counter()
        _, assignments = kmeans.index.search(train_vecs, 1)
        assignments = assignments.ravel()

        cluster_indices = [[] for _ in range(self.nlist)]
        for i, cid in enumerate(assignments):
            cluster_indices[int(cid)].append(i)

        self.cluster_sizes = np.array([len(ci) for ci in cluster_indices], dtype=np.int32)

        print(f"  Saving {self.nlist} clusters to disk...")
        for cid in range(self.nlist):
            idx = np.array(cluster_indices[cid], dtype=np.int32)
            if len(idx) == 0:
                np.save(disk / f"c{cid}_vecs.npy",
                        np.zeros((0, self.d), dtype=np.float32))
                np.save(disk / f"c{cid}_indices.npy",
                        np.zeros(0, dtype=np.int32))
                with open(disk / f"c{cid}_mds.pkl", "wb") as f:
                    pickle.dump([], f)
            else:
                np.save(disk / f"c{cid}_vecs.npy",
                        train_vecs[idx].astype(np.float32))
                np.save(disk / f"c{cid}_indices.npy", idx)
                with open(disk / f"c{cid}_mds.pkl", "wb") as f:
                    pickle.dump([train_mds[i] for i in idx], f)

        np.save(disk / "centroids.npy", self.centroids)
        np.save(disk / "cluster_sizes.npy", self.cluster_sizes)
        print(f"  Saved in {time.perf_counter() - t0:.1f}s")
        self._built = True

    def _get_nearest_clusters(self, x: np.ndarray) -> np.ndarray:
        dists = np.sum((self.centroids - x.astype(np.float32)) ** 2, axis=1)
        n = min(self.nprobe, self.nlist)
        return np.argpartition(dists, n - 1)[:n]

    def _load_cluster(self, cid: int):
        disk = Path(self.disk_dir)
        vecs = np.load(disk / f"c{cid}_vecs.npy").astype(np.float32)
        indices = np.load(disk / f"c{cid}_indices.npy").astype(np.int32)
        with open(disk / f"c{cid}_mds.pkl", "rb") as f:
            mds = pickle.load(f)
        return vecs, indices, mds

    def query(self, x: np.ndarray, k: int, tenant_id: int) -> list[int]:
        tid = int(tenant_id)
        nearest = self._get_nearest_clusters(x)
        all_dists, all_indices = [], []
        for cid in nearest:
            vecs, indices, mds = self._load_cluster(cid)
            if len(vecs) == 0:
                continue
            mask = np.array([tid in md for md in mds], dtype=bool)
            if not mask.any():
                continue
            dists = np.sum((vecs[mask] - x.astype(np.float32)) ** 2, axis=1)
            all_dists.append(dists)
            all_indices.append(indices[mask])
        return self._finalise_topk(all_dists, all_indices, k)

    def query_with_complex_predicate(
        self, x: np.ndarray, k: int, predicate: str
    ) -> list[int]:
        tokens = predicate.split()
        nearest = self._get_nearest_clusters(x)
        all_dists, all_indices = [], []
        for cid in nearest:
            vecs, indices, mds = self._load_cluster(cid)
            if len(vecs) == 0:
                continue
            mask = np.array([evaluate_predicate(tokens, md) for md in mds], dtype=bool)
            if not mask.any():
                continue
            dists = np.sum((vecs[mask] - x.astype(np.float32)) ** 2, axis=1)
            all_dists.append(dists)
            all_indices.append(indices[mask])
        return self._finalise_topk(all_dists, all_indices, k)

    def _finalise_topk(
        self, all_dists: list, all_indices: list, k: int
    ) -> list[int]:
        if not all_dists:
            return [-1] * k
        all_dists = np.concatenate(all_dists)
        all_indices = np.concatenate(all_indices)
        k_act = min(k, len(all_dists))
        top_k = np.argpartition(all_dists, k_act - 1)[:k_act]
        top_k = top_k[np.argsort(all_dists[top_k])]
        result = all_indices[top_k].tolist()
        if len(result) < k:
            result.extend([-1] * (k - len(result)))
        return result

    def get_index_memory_bytes(self) -> int:
        mem = 0
        if self.centroids is not None:
            mem += self.centroids.nbytes
        if self.cluster_sizes is not None:
            mem += self.cluster_sizes.nbytes
        return mem

    def get_disk_bytes(self) -> int:
        total = 0
        for root, _, files in os.walk(self.disk_dir or ""):
            for f in files:
                total += os.path.getsize(Path(root) / f)
        return total


# ---------------------------------------------------------------------------
# Defaults, helpers, main (same pattern as DiskIVF)
# ---------------------------------------------------------------------------

BUILTIN_DEFAULTS = {
    "arxiv": {
        "dataset": "arxiv", "d": 384, "n_labels": 100,
        "nlist": 256, "nprobe": 64,
        "k": 10, "n_queries": 1000, "num_warmup": 20,
    },
    "yfcc100m": {
        "dataset": "yfcc100m", "d": 192, "n_labels": 1000,
        "nlist": 256, "nprobe": 64,
        "k": 10, "n_queries": 1000, "num_warmup": 20,
    },
    "arxiv_small": {
        "dataset": "arxiv_small", "d": 384, "n_labels": 100,
        "nlist": 64, "nprobe": 16,
        "k": 10, "n_queries": 500, "num_warmup": 10,
    },
    "yfcc100m_small": {
        "dataset": "yfcc100m_small", "d": 192, "n_labels": 200,
        "nlist": 64, "nprobe": 16,
        "k": 10, "n_queries": 500, "num_warmup": 10,
    },
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
    parser.add_argument("--output_dir", type=str, default="4_Results/SPANN-PostFiltering")
    parser.add_argument("--disk_dir", type=str, default="4_Results/SPANN-PostFiltering/disk_data")
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument("--nprobe", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config, None)
        dataset = cfg.get("dataset", args.dataset)
    elif args.dataset:
        cfg = load_config(None, args.dataset)
        dataset = args.dataset
    else:
        parser.error("Need --config or --dataset")

    if args.nlist is not None:
        cfg["nlist"] = args.nlist
    if args.nprobe is not None:
        cfg["nprobe"] = args.nprobe

    k = cfg["k"]
    num_warmup = cfg.get("num_warmup", 20)

    print(f"\n{'='*60}\nSPANN-PostFiltering Benchmark: {dataset}\n{'='*60}")
    print(f"  nlist={cfg['nlist']}, nprobe={cfg['nprobe']}")
    print("Loading data...")
    train_vecs, train_mds, query_vecs, query_labels, query_info, sl_gt = \
        load_data(dataset, args.data_dir)
    rss_after_load = getCurrentRSS() / (1024 * 1024)
    print(f"  Train: {train_vecs.shape[0]:,} x {train_vecs.shape[1]}")
    print(f"  SL queries: {len(query_vecs)}")

    print(f"\n{'='*60}\nBuilding index\n{'='*60}")
    index = SPANNIndex(nlist=cfg["nlist"], nprobe=cfg["nprobe"],
                       disk_dir=args.disk_dir)
    rss_before_build = getCurrentRSS() / (1024 * 1024)
    t0 = time.perf_counter()
    index.build(train_vecs, train_mds)
    t_build = time.perf_counter() - t0
    rss_after_build = getCurrentRSS() / (1024 * 1024)
    idx_mem = index.get_index_memory_bytes() / (1024 * 1024)
    disk_mb = index.get_disk_bytes() / (1024 * 1024)
    print(f"  Build: {t_build:.1f}s, in-mem: {idx_mem:.1f} MB, "
          f"disk: {disk_mb:.1f} MB, RSS: {rss_after_build:.1f} MB")

    # ---- SL queries ----
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

    sl_results = {
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "qps": float(1000.0 / np.mean(latencies)),
        "avg_recall": float(np.mean(recalls)),
        "min_recall": float(np.min(recalls)),
        "empty": empty,
        "rss_before_mb": float(rss_before_q),
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
    print(f"\n  SL recall@{k}: {sl_results['avg_recall']:.4f}, "
          f"QPS: {sl_results['qps']:.1f}")

    # ---- CP queries ----
    print(f"\n{'='*60}\nComplex-predicate queries (k={k})\n{'='*60}")
    cp_query_vecs, filters_info, cp_gt_dict = load_cp_data(dataset, args.data_dir)
    cp_filters = filters_info["filters"]
    cp_selectivities = filters_info.get("selectivities", {})
    n_cp = len(cp_query_vecs)
    print(f"  {len(cp_filters)} filters x {n_cp} queries")

    cp_results = {}
    total_cp_time = 0.0
    cp_all_recalls, cp_all_lats = [], []

    for fi, formula in enumerate(cp_filters):
        gt = cp_gt_dict[formula]
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
        sel = cp_selectivities.get(formula, 0)
        cp_results[formula] = {
            "selectivity": float(sel),
            "avg_latency_ms": float(np.mean(lats)),
            "qps": float(1000 / np.mean(lats)) if np.mean(lats) > 0 else 0,
            "avg_recall": float(np.mean(recs)),
            "time_s": float(elapsed),
        }
        cp_all_recalls.extend(recs)
        cp_all_lats.extend(lats)
        if (fi + 1) % 20 == 0:
            print(f"  [{fi + 1}/{len(cp_filters)}] done")

    print(f"  CP recall@{k}: {np.mean(cp_all_recalls):.4f}, "
          f"total: {total_cp_time:.1f}s")

    # ---- Save ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": dataset,
        "index": "SPANN-PostFiltering",
        "nlist": cfg["nlist"], "nprobe": cfg["nprobe"],
        "k": k, "train_size": int(len(train_vecs)),
        "n_sl_queries": len(query_vecs),
        "n_cp_filters": len(cp_filters),
        "n_cp_queries": n_cp,
        "build_time_s": float(t_build),
        "index_memory_mb": float(idx_mem),
        "disk_usage_mb": float(disk_mb),
        "rss_after_load_mb": float(rss_after_load),
        "rss_after_build_mb": float(rss_after_build),        "rss_before_build_mb": float(rss_before_build),
        "single_label": sl_results,
        "complex_predicate": {
            "total_time_s": float(total_cp_time),
            "avg_recall": float(np.mean(cp_all_recalls)) if cp_all_recalls else 0,
            "avg_latency_ms": float(np.mean(cp_all_lats)) if cp_all_lats else 0,
            "per_filter": cp_results,
        },
    }
    out_path = output_dir / f"spann_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults: {out_path.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
