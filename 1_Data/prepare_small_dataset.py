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

_script_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, _script_dir)                       # 1_Data/ (sibling modules)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_Utils"))  # 2_Utils/

from predicate import (  # noqa: E402
    build_inverted_index,
    compute_filter_selectivity,
    compute_qualified_indices,
    evaluate_predicate,
)
from prepare_ann_dataset import (  # noqa: E402
    select_single_label_queries,
    generate_random_filters as gen_filters_ann,
    compute_complex_predicate_gt as compute_cp_gt_ann,
)
from gt_computing import load_arxiv, load_yfcc1m  # noqa: E402


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


# CP GT helpers: use the canonical implementations from prepare_ann_dataset
# (imported as gen_filters_ann / compute_cp_gt_ann at top of file)


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
    parser.add_argument("--from_source", action="store_true",
                        help="Load from raw source files instead of GT cache")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Override output directory name")
    parser.add_argument("--n_query_pool", type=int, default=20000,
                        help="Query pool size (from_source mode)")
    args = parser.parse_args()

    ds_name = args.output_name or f"{args.dataset}_small"
    print(f"\n{'='*60}")
    print(f"Creating dataset: {ds_name}")
    print(f"  n_train={args.n_train}, n_sl_query={args.n_sl_query}, "
          f"n_cp_query={args.n_cp_query}, n_filters={args.n_filters}")
    if args.from_source:
        print(f"  mode=from_source, n_query_pool={args.n_query_pool}")
    print(f"{'='*60}")

    rng = np.random.RandomState(args.seed)

    # ================================================================
    # Load data
    # ================================================================
    if args.from_source:
        # Load from raw source files (e.g. 1_Data/arxiv/*.npy + *.pkl)
        print("\nLoading from source data...")
        if args.dataset == "arxiv":
            train_full, test_full, train_mds_full, test_mds_full, meta = \
                load_arxiv(args.data_dir)
        else:
            train_full, test_full, train_mds_full, test_mds_full, meta = \
                load_yfcc1m(args.data_dir)
        d = meta["dim"]
        n_labels_orig = meta["n_labels"]
        print(f"  Full train: {len(train_full):,}, test: {len(test_full):,}, "
              f"d={d}, labels={n_labels_orig}")

        # Sample train
        train_idx = rng.choice(len(train_full), args.n_train, replace=False)
        small_train_vecs = train_full[train_idx]
        small_train_mds = [train_mds_full[i] for i in train_idx]
        print(f"  Sampled {args.n_train} train vectors")

        # Remap labels
        active_labels = sorted(set().union(*small_train_mds))
        label_remap = {old: new for new, old in enumerate(active_labels)}
        small_train_mds = [[label_remap[l] for l in md] for md in small_train_mds]
        actual_n_labels = len(active_labels)
        print(f"  Active labels in subset: {actual_n_labels}")

        # Query pool: sample from test split
        pool_size = min(args.n_query_pool, len(test_full))
        pool_idx = rng.choice(len(test_full), pool_size, replace=False)
        query_pool_vecs = test_full[pool_idx]
        query_pool_mds = [test_mds_full[i] for i in pool_idx]
        # Remap query pool labels too
        query_pool_mds = [[label_remap.get(l, 0) for l in md]
                          for md in query_pool_mds]
        print(f"  Query pool: {pool_size:,} vectors")

        # Build inverted index
        print("\nBuilding inverted index...")
        label_to_indices = build_inverted_index(small_train_mds, actual_n_labels)

        # Stratified SL query selection
        print(f"Selecting {args.n_sl_query} single-label queries "
              f"(stratified by selectivity)...")
        small_query_vecs, small_query_labels, small_query_info = \
            select_single_label_queries(
                query_pool_vecs, query_pool_mds, small_train_mds,
                label_to_indices, n_queries=args.n_sl_query, seed=args.seed,
            )
        n_sl_avail = len(small_query_labels)
        print(f"  Selected {n_sl_avail} queries")
    else:
        # Legacy mode: load from existing GT directory
        print("\nLoading from existing GT directory...")
        train_vecs, train_mds, query_vecs, query_labels, query_info, metadata = \
            load_full(args.dataset, args.data_dir)
        full_n_train = len(train_vecs)
        d = metadata["dim"]
        n_labels_orig = metadata["n_labels"]
        print(f"  Full train: {full_n_train:,}, queries: {len(query_vecs)}, "
              f"d={d}, labels={n_labels_orig}")

        # Sample train
        train_idx = rng.choice(full_n_train, args.n_train, replace=False)
        small_train_vecs = train_vecs[train_idx]
        small_train_mds = [train_mds[i] for i in train_idx]
        print(f"  Sampled {args.n_train} train vectors")

        # Remap labels
        active_labels = sorted(set().union(*small_train_mds))
        label_remap = {old: new for new, old in enumerate(active_labels)}
        small_train_mds = [[label_remap[l] for l in md] for md in small_train_mds]
        actual_n_labels = len(active_labels)
        print(f"  Active labels in subset: {actual_n_labels}")

        # Legacy: random sampling from existing queries
        n_sl_avail = min(args.n_sl_query, len(query_vecs))
        sl_idx = rng.choice(len(query_vecs), n_sl_avail, replace=False)
        small_query_vecs = query_vecs[sl_idx]
        small_query_labels = np.array(
            [label_remap.get(int(query_labels[i]), 0) for i in sl_idx],
            dtype=np.int32,
        )
        small_query_info = []
        for i in sl_idx:
            info = dict(query_info[i])
            old_label = info["label"]
            info["label"] = label_remap.get(old_label, 0)
            small_query_info.append(info)
        print(f"  Sampled {n_sl_avail} SL queries (random from existing)")

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

    # Complex-predicate GT (use canonical implementation from prepare_ann_dataset)
    filters = gen_filters_ann(
        list(range(actual_n_labels)), n_filters=args.n_filters, seed=args.seed,
    )
    print(f"\nComputing complex-predicate GT ({len(filters)} filters)...")

    # Select CP queries from the FULL query pool (not just the SL-selected subset)
    cp_rng = np.random.RandomState(args.seed + 1)
    if args.from_source:
        _cp_pool = query_pool_vecs
    else:
        _cp_pool = query_vecs  # original full queries from GT directory
    n_cp_avail = min(args.n_cp_query, len(_cp_pool))
    cp_idx = cp_rng.choice(len(_cp_pool), n_cp_avail, replace=False)
    cp_query_vecs = _cp_pool[cp_idx]

    t0 = time.perf_counter()
    cp_gt, cp_selectivities = compute_cp_gt_ann(
        small_train_vecs, small_train_mds, cp_query_vecs, filters, k=args.k,
    )
    t_cp = time.perf_counter() - t0
    print(f"  Done in {t_cp:.1f}s ({t_cp / 60:.1f} min)")

    # Save
    output_dir = Path(args.data_dir) / "ground_truth" / ds_name
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
    meta_out = {
        "dim": int(d),
        "n_labels": actual_n_labels,
        "dataset": ds_name,
        "n_queries": n_sl_avail,
        "k": args.k,
        "train_size": args.n_train,
        "label_method": "natural",
        "avg_labels_per_vec": None,  # natural — varies per vector
        "n_filters": len(filters),
        "n_cp_queries": len(cp_query_vecs),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)

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
