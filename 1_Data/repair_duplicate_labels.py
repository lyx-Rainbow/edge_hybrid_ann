#!/usr/bin/env python3
"""Repair duplicate (vid, tid) access pairs and refresh selectivity metadata.

This script is deliberately dry-run by default.  With --apply it rewrites:

  - train_access.npy      (unique (vid, tid) pairs, sorted)
  - train_mds.pkl         (one sorted unique label list per vector)
  - all_labels.json       (labels with at least one unique assignment)
  - query_info.json       (selectivity and percentile buckets refreshed)
  - metadata.json         (avg labels / sizes refreshed)
  - ground_truth.npy      (optional, exact FAISS recomputation)

It can also patch label-dependent index metadata without rebuilding the
vector index:

  --spann-index-dir DIR   overwrite SPANN metadata.bin
  --diskivf-index-dir DIR overwrite DiskIVF c*_mds.bin files

CP ground truth is not recomputed automatically; rerun the CP pipeline after
this repair.
"""
import argparse
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
K = 10
BUCKET_NAMES = ["1p", "25p", "50p", "75p", "99p"]
BUCKET_PERCENTILES = [1, 25, 50, 75, 99]


def labels_by_vid_from_access(n, access):
    labels = [[] for _ in range(n)]
    for vid, tid in access:
        labels[int(vid)].append(int(tid))
    for lab_list in labels:
        lab_list.sort()
    return labels


def access_from_labels(labels):
    pairs = []
    for vid, lab_list in enumerate(labels):
        for tid in lab_list:
            pairs.append((vid, int(tid)))
    return np.asarray(pairs, dtype=np.int32)


def assign_buckets(query_info, label_counts, n_train):
    sels = {int(lab): count / float(n_train)
            for lab, count in label_counts.items() if count > 0}
    values = np.asarray(sorted(sels.values()), dtype=np.float64)
    targets = np.percentile(values, BUCKET_PERCENTILES)
    changed = 0
    for q in query_info:
        lab = q.get("label")
        if lab is None or int(lab) not in sels:
            q["selectivity"] = 0.0
            q["bucket"] = "unknown"
            continue
        sel = float(sels[int(lab)])
        q["selectivity"] = sel
        idx = int(np.argmin(np.abs(targets - sel)))
        new_bucket = BUCKET_NAMES[idx]
        if q.get("bucket") != new_bucket:
            changed += 1
        q["bucket"] = new_bucket
        q["selectivity_percentile"] = BUCKET_PERCENTILES[idx]
    return changed


def backup_once(path):
    if not path.exists():
        return
    backup = path.with_name(path.name + ".bak_dedup")
    if not backup.exists():
        shutil.copy2(path, backup)


def write_spann_metadata(index_dir, labels):
    path = Path(index_dir) / "metadata.bin"
    backup_once(path)
    with open(path, "wb") as f:
        f.write(np.uint32(len(labels)).tobytes())
        for lab_list in labels:
            arr = np.asarray(lab_list, dtype=np.int32)
            f.write(np.uint32(len(arr)).tobytes())
            if len(arr):
                f.write(arr.tobytes())
    print(f"SPANN metadata written: {path}")


def patch_diskivf_index(index_dir, labels):
    index_dir = Path(index_dir)
    files = sorted(index_dir.glob("c*_vids.bin"))
    for vids_path in files:
        vids = np.fromfile(vids_path, dtype=np.int32)
        mds_path = vids_path.with_name(
            vids_path.name.replace("_vids.bin", "_mds.bin"))
        if not mds_path.exists():
            continue
        backup_once(mds_path)
        with open(mds_path, "wb") as f:
            for vid in vids:
                arr = np.asarray(labels[int(vid)], dtype=np.int32)
                f.write(np.uint32(len(arr)).tobytes())
                if len(arr):
                    f.write(arr.tobytes())
    print(f"DiskIVF mds files patched: {len(files)} clusters in {index_dir}")


def recompute_sl_gt(gt_dir, access, labels, k=10):
    import faiss

    gt_dir = Path(gt_dir)
    train_vecs = np.load(gt_dir / "train_vecs.npy", mmap_mode="r")
    query_vecs = np.load(gt_dir / "query_vecs.npy")
    query_labels = np.load(gt_dir / "query_labels.npy").astype(np.int32)
    n, d = train_vecs.shape
    ground_truth = np.full((len(query_vecs), k), -1, dtype=np.int32)

    by_label = {}
    for vid, lab_list in enumerate(labels):
        for lab in lab_list:
            by_label.setdefault(int(lab), []).append(int(vid))

    for lab, vids in by_label.items():
        qidx = np.where(query_labels == lab)[0]
        if len(qidx) == 0 or not vids:
            continue
        vids = np.asarray(sorted(vids), dtype=np.int64)
        cand = np.ascontiguousarray(train_vecs[vids], dtype=np.float32)
        index = faiss.IndexFlatL2(d)
        index.add(cand)
        k_act = min(k, len(vids))
        _, local = index.search(
            np.ascontiguousarray(query_vecs[qidx], dtype=np.float32), k_act)
        ground_truth[qidx, :k_act] = vids[local]
        print(f"  GT label {lab}: {len(qidx)} queries x {len(vids)} candidates")

    backup_once(gt_dir / "ground_truth.npy")
    np.save(gt_dir / "ground_truth.npy", ground_truth)
    print(f"Saved ground_truth.npy: {gt_dir / 'ground_truth.npy'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--apply", action="store_true",
                        help="write repaired data (default: dry run)")
    parser.add_argument("--recompute-gt", action="store_true",
                        help="exactly recompute SL ground truth with FAISS")
    parser.add_argument("--spann-index-dir", default=None)
    parser.add_argument("--diskivf-index-dir", default=None)
    args = parser.parse_args()

    gt_dir = PROJ_ROOT / "1_Data/ground_truth" / args.dataset
    access_path = gt_dir / "train_access.npy"
    mds_path = gt_dir / "train_mds.pkl"
    meta_path = gt_dir / "metadata.json"
    qinfo_path = gt_dir / "query_info.json"

    access = np.load(access_path).astype(np.int32)
    n = int(access[:, 0].max()) + 1
    if meta_path.exists():
        n = int(json.load(open(meta_path)).get("train_size", n))
    unique = np.unique(access, axis=0)
    extra = len(access) - len(unique)
    print(f"dataset={args.dataset} N={n} pairs={len(access)} "
          f"unique_pairs={len(unique)} extra={extra} ({extra / len(access):.2%})")

    labels = labels_by_vid_from_access(n, unique)
    counts = Counter()
    for lab_list in labels:
        for lab in lab_list:
            counts[int(lab)] += 1
    print(f"labels={len(counts)} avg_labels_per_vec="
          f"{np.mean([len(x) for x in labels]):.4f}")

    query_info = json.load(open(qinfo_path))
    old_buckets = Counter(q.get("bucket") for q in query_info)
    changed = assign_buckets(query_info, counts, n)
    new_buckets = Counter(q.get("bucket") for q in query_info)
    print(f"bucket changes={changed}")
    print(f"old buckets={dict(old_buckets)}")
    print(f"new buckets={dict(new_buckets)}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to write changes.")
        return

    backup_once(access_path)
    backup_once(mds_path)
    backup_once(meta_path)
    backup_once(qinfo_path)
    backup_once(gt_dir / "all_labels.json")

    np.save(access_path, unique)
    with open(mds_path, "wb") as f:
        pickle.dump(labels, f)
    with open(qinfo_path, "w") as f:
        json.dump(query_info, f, indent=2)
    with open(gt_dir / "all_labels.json", "w") as f:
        json.dump(sorted(counts), f)

    if meta_path.exists():
        meta = json.load(open(meta_path))
    else:
        meta = {}
    meta.update({
        "n_labels": len(counts),
        "train_size": n,
        "avg_labels_per_vec": float(np.mean([len(x) for x in labels])),
        "single_label_gt_time_s": 0,
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote access / mds / query_info / all_labels / metadata")
    if args.spann_index_dir:
        write_spann_metadata(args.spann_index_dir, labels)
    if args.diskivf_index_dir:
        patch_diskivf_index(args.diskivf_index_dir, labels)
    if args.recompute_gt:
        recompute_sl_gt(gt_dir, unique, labels, k=K)
    else:
        print("Ground truth unchanged; pass --recompute-gt to refresh SL GT.")


if __name__ == "__main__":
    main()
