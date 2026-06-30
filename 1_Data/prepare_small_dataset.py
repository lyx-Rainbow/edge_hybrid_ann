"""
Create a small subset from existing processed data for fast micro-experiments.

Usage:
    python prepare_small_dataset.py --dataset yfcc100m --n_train 50000 --n_sl_query 500
    python prepare_small_dataset.py --dataset arxiv     --n_train 50000 --n_sl_query 500
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_Utils"))
from predicate import (  # noqa: E402
    build_inverted_index,
    compute_filter_selectivity,
    compute_qualified_indices,
    evaluate_predicate,
)


def load_full(dataset: str, data_dir: str):
    gt_dir = Path(data_dir) / "ground_truth" / dataset
    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        train_mds = pickle.load(f)
    query_vecs = np.load(gt_dir / "query_vecs.npy").astype(np.float32)
    query_labels = np.load(gt_dir / "query_labels.npy").astype(np.int32)
    with open(gt_dir / "query_info.json") as f:
        query_info = json.load(f)
    with open(gt_dir / "metadata.json") as f:
        metadata = json.load(f)
    return train_vecs, train_mds, query_vecs, query_labels, query_info, metadata


def compute_single_label_gt(
    train_vecs, query_vecs, query_labels, label_to_indices, k=10,
):
    n_queries = len(query_labels)
    gt = np.full((n_queries, k), -1, dtype=np.int32)

    # Group queries by label
    label_to_qi = {}
    for i, lab in enumerate(query_labels):
        label_to_qi.setdefault(int(lab), []).append(i)

    for lab, qi_list in label_to_qi.items():
        candidates = label_to_indices.get(lab, np.array([], dtype=np.int32))
        if len(candidates) == 0:
            continue
        cand_vecs = train_vecs[candidates]
        q_vecs = query_vecs[qi_list]
        dists = np.sum((cand_vecs[None] - q_vecs[:, None]) ** 2, axis=2)
        k_act = min(k, len(candidates))
        top = np.argpartition(dists, k_act - 1, axis=1)[:, :k_act]
        for row in range(len(qi_list)):
            order = np.argsort(dists[row, top[row]])
            top[row] = top[row][order]
        for row, qi in enumerate(qi_list):
            gt[qi, :k_act] = candidates[top[row]]
    return gt


def generate_random_filters(all_labels, n_filters=100, seed=42):
    rng = np.random.RandomState(seed)
    labels = np.array(all_labels)
    filters = []

    def sample(n):
        return rng.choice(labels, min(n, len(labels)), replace=False).tolist()

    n_each = max(1, n_filters // 5)
    for _ in range(n_each):
        l = sample(1); filters.append(f"NOT {l[0]}")
    for _ in range(n_each):
        a, b = sample(2); filters.append(f"AND {a} {b}")
    for _ in range(n_each):
        a, b = sample(2); filters.append(f"OR {a} {b}")
    for _ in range(n_each):
        a, b = sample(2); filters.append(f"AND {a} NOT {b}")
    rem = n_filters - len(filters)
    for _ in range(rem):
        a, b, c = sample(3); filters.append(f"OR {a} OR {b} {c}")
    return filters[:n_filters]


def compute_cp_gt(train_vecs, train_mds, cp_query_vecs, filters, k=10):
    BLOCK = 50000
    results = {}
    selectivities = {}
    for fi, formula in enumerate(filters):
        print(f"    [{fi + 1}/{len(filters)}] {formula}", flush=True)
        qualified = compute_qualified_indices(formula, train_mds)
        sel = len(qualified) / len(train_vecs)
        selectivities[formula] = float(sel)

        nq = len(cp_query_vecs)
        gt = np.full((nq, k), -1, dtype=np.int32)
        if len(qualified) == 0:
            results[formula] = gt
            continue

        for qi in range(nq):
            q = cp_query_vecs[qi]
            best_d = np.empty(0, dtype=np.float32)
            best_i = np.empty(0, dtype=np.int32)
            for start in range(0, len(qualified), BLOCK):
                end = min(start + BLOCK, len(qualified))
                blk = qualified[start:end]
                dists = np.sum((train_vecs[blk] - q) ** 2, axis=1)
                keep = min(k, len(dists))
                if keep < len(dists):
                    idx = np.argpartition(dists, keep - 1)[:keep]
                    dists = dists[idx]; blk = blk[idx]
                if len(best_d) == 0:
                    best_d = dists; best_i = blk
                else:
                    md = np.concatenate([best_d, dists])
                    mi = np.concatenate([best_i, blk])
                    keep = min(k, len(md))
                    idx = np.argpartition(md, keep - 1)[:keep]
                    idx = idx[np.argsort(md[idx])]
                    best_d = md[idx]; best_i = mi[idx]
            if len(best_i) > 0:
                gt[qi, :len(best_i)] = best_i
        results[formula] = gt
    return results, selectivities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["arxiv", "yfcc100m"])
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_sl_query", type=int, default=500)
    parser.add_argument("--n_cp_query", type=int, default=40)
    parser.add_argument("--n_filters", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Creating small dataset: {args.dataset}_small")
    print(f"  n_train={args.n_train}, n_sl_query={args.n_sl_query}, "
          f"n_cp_query={args.n_cp_query}, n_filters={args.n_filters}")
    print(f"{'='*60}")

    # Load full data
    print("\nLoading full data...")
    train_vecs, train_mds, query_vecs, query_labels, query_info, metadata = \
        load_full(args.dataset, args.data_dir)
    full_n_train = len(train_vecs)
    d = metadata["dim"]
    n_labels = metadata["n_labels"]
    print(f"  Full train: {full_n_train:,}, queries: {len(query_vecs)}, "
          f"d={d}, labels={n_labels}")

    # Sample train
    rng = np.random.RandomState(args.seed)
    train_idx = rng.choice(full_n_train, args.n_train, replace=False)
    small_train_vecs = train_vecs[train_idx]
    small_train_mds = [train_mds[i] for i in train_idx]
    print(f"  Sampled {args.n_train} train vectors")

    # Remap labels to 0..m-1 for the subset
    active_labels = sorted(set().union(*small_train_mds))
    label_remap = {old: new for new, old in enumerate(active_labels)}
    small_train_mds = [[label_remap[l] for l in md] for md in small_train_mds]
    actual_n_labels = len(active_labels)
    print(f"  Active labels in subset: {actual_n_labels}")

    # Sample single-label queries
    n_sl_avail = min(args.n_sl_query, len(query_vecs))
    sl_idx = rng.choice(len(query_vecs), n_sl_avail, replace=False)
    small_query_vecs = query_vecs[sl_idx]
    # Remap query labels too
    small_query_labels = np.array(
        [label_remap.get(int(query_labels[i]), 0) for i in sl_idx],
        dtype=np.int32,
    )
    # Update query_info with remapped labels
    small_query_info = []
    for i in sl_idx:
        info = dict(query_info[i])
        old_label = info["label"]
        info["label"] = label_remap.get(old_label, 0)
        small_query_info.append(info)
    print(f"  Sampled {n_sl_avail} SL queries")

    # Build inverted index
    print("\nBuilding inverted index...")
    label_to_indices = build_inverted_index(small_train_mds, actual_n_labels)

    # Compute single-label GT
    print("Computing single-label GT...")
    t0 = time.perf_counter()
    sl_gt = compute_single_label_gt(
        small_train_vecs, small_query_vecs, small_query_labels,
        label_to_indices, k=args.k,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Complex-predicate GT
    filters = generate_random_filters(
        list(range(actual_n_labels)), n_filters=args.n_filters, seed=args.seed,
    )
    print(f"\nComputing complex-predicate GT ({len(filters)} filters)...")

    cp_idx = rng.choice(len(query_vecs), min(args.n_cp_query, len(query_vecs)),
                        replace=False)
    cp_query_vecs = query_vecs[cp_idx]

    t0 = time.perf_counter()
    cp_gt, cp_selectivities = compute_cp_gt(
        small_train_vecs, small_train_mds, cp_query_vecs, filters, k=args.k,
    )
    t_cp = time.perf_counter() - t0
    print(f"  Done in {t_cp:.1f}s ({t_cp / 60:.1f} min)")

    # Save
    small_name = f"{args.dataset}_small"
    output_dir = Path(args.data_dir) / "ground_truth" / small_name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_vecs.npy", small_train_vecs)
    with open(output_dir / "train_mds.pkl", "wb") as f:
        pickle.dump(small_train_mds, f)
    np.save(output_dir / "query_vecs.npy", small_query_vecs)
    np.save(output_dir / "query_labels.npy", small_query_labels)
    with open(output_dir / "query_info.json", "w") as f:
        json.dump(small_query_info, f, indent=2)
    np.save(output_dir / "ground_truth.npy", sl_gt)

    # CP outputs
    if len(cp_gt) > 0:
        cp_dir = output_dir / "complex_predicate"
        cp_dir.mkdir(parents=True, exist_ok=True)
        for formula, gt in cp_gt.items():
            safe = formula.replace(" ", "_")
            np.save(cp_dir / f"gt_{safe}.npy", gt)
        with open(cp_dir / "filters.json", "w") as f:
            json.dump({
                "n_filters": len(filters),
                "n_queries": len(cp_query_vecs),
                "filters": sorted(filters),
                "selectivities": {k: float(v) for k, v in cp_selectivities.items()},
            }, f, indent=2)
        np.save(cp_dir / "query_vecs.npy", cp_query_vecs)
        np.save(cp_dir / "query_indices.npy", cp_idx)

    # Metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            **metadata,
            "dataset": small_name,
            "n_labels": actual_n_labels,
            "n_train_sampled": args.n_train,
            "n_sl_query_sampled": n_sl_avail,
            "n_cp_query_sampled": len(cp_query_vecs),
            "n_filters": len(filters),
        }, f, indent=2)

    all_labels = sorted(set().union(*small_train_mds))
    with open(output_dir / "all_labels.json", "w") as f:
        json.dump(all_labels, f)

    print(f"\n{'='*60}")
    print(f"Saved to: {output_dir.resolve()}")
    print(f"  train:        {small_train_vecs.shape}")
    print(f"  SL queries:   {n_sl_avail}")
    print(f"  CP queries:   {len(cp_query_vecs)} x {len(filters)} filters")
    print(f"  GT shape:     {sl_gt.shape}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
