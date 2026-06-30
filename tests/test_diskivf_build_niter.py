"""
Minimal standalone test: DiskIVF build time with custom niter.
Does NOT touch 4_Results/ or shared disk_data_sweep/.
Safe to run alongside run_all_sweeps.sh.

Usage (on WSL):
    python tests/test_diskivf_build_niter.py --dataset yfcc100m --niter 100
    python tests/test_diskivf_build_niter.py --dataset arxiv --niter 100
"""

import argparse
import json
import pickle
import sys
import time
import tempfile
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_Utils"))
from memory_utils import getCurrentRSS  # noqa: E402


def load_data(dataset, data_dir):
    gt_dir = Path(data_dir) / "ground_truth" / dataset
    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        train_mds = pickle.load(f)
    return train_vecs, train_mds


def build_index(train_vecs, train_mds, nlist, niter, disk_dir, max_ppc=256):
    d = train_vecs.shape[1]
    n_total = train_vecs.shape[0]
    n_train = min(n_total, max_ppc * nlist) if max_ppc > 0 else n_total
    t0 = time.perf_counter()

    # ---- K-means (match main experiment, same FAISS API) ----
    print(f"  K-means: nlist={nlist}, niter={niter}, d={d}")
    print(f"  max_points_per_centroid={max_ppc} → trains on ~{n_train:,} / {n_total:,} points")
    kmeans = faiss.Kmeans(d, nlist, niter=niter, verbose=True, gpu=False,
                          max_points_per_centroid=max_ppc)
    kmeans.train(train_vecs)
    centroids = kmeans.centroids.astype(np.float32)
    t_kmeans = time.perf_counter() - t0
    print(f"  K-means done in {t_kmeans:.1f}s")

    # ---- Assign vectors to clusters ----
    t1 = time.perf_counter()
    _, assignments = kmeans.index.search(train_vecs, 1)
    assignments = assignments.ravel()

    cluster_indices = [[] for _ in range(nlist)]
    for i, cid in enumerate(assignments):
        cluster_indices[cid].append(i)
    t_assign = time.perf_counter() - t1
    print(f"  Assignment done in {t_assign:.1f}s")

    # ---- Save to temp disk dir ----
    disk_dir = Path(disk_dir)
    disk_dir.mkdir(parents=True, exist_ok=True)
    t2 = time.perf_counter()
    for cid in range(nlist):
        idxs = np.array(cluster_indices[cid], dtype=np.int32)
        np.save(disk_dir / f"c{cid}_indices.npy", idxs)
        np.save(disk_dir / f"c{cid}_vecs.npy", train_vecs[idxs])
        mds_arr = np.array([train_mds[i] for i in idxs], dtype=object)
        np.save(disk_dir / f"c{cid}_mds.npy", mds_arr)
    t_save = time.perf_counter() - t2
    print(f"  Save {nlist} clusters to disk in {t_save:.1f}s")

    # ---- Centroid index for query-time probe ----
    cent_index = faiss.IndexFlatL2(d)
    cent_index.add(centroids)

    rss = getCurrentRSS() / (1024 * 1024)
    t_total = time.perf_counter() - t0

    print(f"\n  Total build: {t_total:.1f}s")
    print(f"  RSS: {rss:.1f} MB")
    print(f"  Disk dir: {disk_dir.resolve()}")

    return {
        "build_time_s": float(t_total),
        "kmeans_s": float(t_kmeans),
        "assign_s": float(t_assign),
        "save_s": float(t_save),
        "rss_mb": float(rss),
        "disk_dir": str(disk_dir.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["arxiv", "yfcc100m"])
    parser.add_argument("--niter", type=int, required=True)
    parser.add_argument("--nlist", type=int, default=None,
                        help="Override default nlist for this dataset")
    parser.add_argument("--max_ppc", type=int, default=0,
                        help="max_points_per_centroid: 256=FAISS default (~32K pts), 0=all data")
    parser.add_argument("--data_dir", default="1_Data")
    args = parser.parse_args()

    # 同主实验的 nlist 配置
    defaults = {"yfcc100m": 128, "arxiv": 128}
    nlist = args.nlist or defaults[args.dataset]

    # ---- 独立临时目录，绝不碰主实验 ----
    disk_dir = tempfile.mkdtemp(prefix=f"test_diskivf_niter{args.niter}_")

    print(f"\n{'='*60}")
    print(f"DiskIVF Build Test: {args.dataset}  nlist={nlist}  niter={args.niter}  max_ppc={args.max_ppc}")
    print(f"  Disk dir: {disk_dir}")
    print(f"{'='*60}")

    train_vecs, train_mds = load_data(args.dataset, args.data_dir)
    print(f"  Loaded: {train_vecs.shape[0]:,} x {train_vecs.shape[1]}")

    result = build_index(train_vecs, train_mds, nlist, args.niter, disk_dir, max_ppc=args.max_ppc)

    # ---- 保存结果到临时路径 ----
    out_path = Path(disk_dir) / "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result saved: {out_path}")


if __name__ == "__main__":
    main()
