"""
Ground truth computation for Curator benchmark.

Two modes:
  1. Single-label GT  — per (query, label) pair: top-k among training vectors
                         that have THAT SPECIFIC label.
  2. Complex-predicate GT — per filter (e.g. "AND 0 OR NOT 1 2"): top-k among
                         training vectors that satisfy the boolean predicate.

Usage:
    python gt_computing.py --dataset arxiv
    python gt_computing.py --dataset yfcc100m
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_arxiv(data_dir: str):
    data_dir = Path(data_dir) / "arxiv"
    vecs = np.load(data_dir / "embeddings_user100_vec2e6.npy").astype(np.float32)
    with open(data_dir / "processed_user100_vec2e6.pkl", "rb") as f:
        entries = pickle.load(f)

    categories = sorted(set().union(*(e["categories"] for e in entries)))
    cat2id = {c: i for i, c in enumerate(categories)}
    mds = [[cat2id[c] for c in e["categories"]] for e in entries]

    n = len(vecs)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)

    train_vecs = vecs[idx[:n_train]]
    test_vecs  = vecs[idx[n_train:]]
    train_mds  = [mds[i] for i in idx[:n_train]]
    test_mds   = [mds[i] for i in idx[n_train:]]

    meta = {"dim": 384, "n_labels": len(categories), "dataset": "arxiv"}
    return train_vecs, test_vecs, train_mds, test_mds, meta


def load_yfcc1m(data_dir: str):
    data_dir = Path(data_dir) / "yfcc100m"
    vecs = np.load(
        data_dir / "yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy"
    ).astype(np.float32)
    with open(data_dir / "yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl", "rb") as f:
        mds = pickle.load(f)

    n = len(vecs)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)

    train_vecs = vecs[idx[:n_train]]
    test_vecs  = vecs[idx[n_train:]]
    train_mds  = [mds[i] for i in idx[:n_train]]
    test_mds   = [mds[i] for i in idx[n_train:]]

    all_labels = set().union(*train_mds).union(*test_mds)
    n_labels = max(all_labels) + 1

    meta = {"dim": 192, "n_labels": n_labels, "dataset": "yfcc100m"}
    return train_vecs, test_vecs, train_mds, test_mds, meta


# ---------------------------------------------------------------------------
# Inverted index
# ---------------------------------------------------------------------------

def build_inverted_index(train_mds: list, n_labels: int):
    """label -> sorted np.array of training indices."""
    label_to_indices = {i: [] for i in range(n_labels)}
    for idx, labels in enumerate(train_mds):
        for lab in labels:
            label_to_indices[lab].append(idx)
    return {k: np.array(v, dtype=np.int32) for k, v in label_to_indices.items()}


# ---------------------------------------------------------------------------
# Query selection (per-label selectivity)
# ---------------------------------------------------------------------------

def select_single_label_queries(
    test_vecs: np.ndarray,
    test_mds: list,
    train_mds: list,
    label_to_indices: dict,
    n_queries: int = 1000,
    seed: int = 42,
):
    """
    Select n_queries test vectors, each paired with a single label,
    covering diverse per-label selectivity ranges.

    Per-label selectivity = fraction of training vectors having that label.

    Returns:
        query_vecs, query_labels, query_info
    """
    rng = np.random.RandomState(seed)
    n_train = len(train_mds)
    n_labels = len(label_to_indices)

    # Compute per-label selectivity
    label_sel = {}
    for lab, indices in label_to_indices.items():
        label_sel[lab] = len(indices) / n_train

    sels = np.array(list(label_sel.values()))
    max_sel = min(sels.max(), 0.2)
    min_sel = sels.min()

    # Define 5 selectivity buckets
    bounds = np.linspace(min_sel, max_sel, 6)
    buckets = [(bounds[i], bounds[i + 1]) for i in range(5)]
    bucket_names = [f"[{bounds[i]:.4f}, {bounds[i+1]:.4f})" for i in range(5)]

    # Find all valid (test_idx, label) pairs
    candidate_pairs = []
    for i, labels in enumerate(test_mds):
        for lab in labels:
            sel = label_sel.get(lab, 0.0)
            candidate_pairs.append((i, int(lab), sel))

    print(f"  Total candidate (query, label) pairs: {len(candidate_pairs)}")

    n_per_bucket = n_queries // len(buckets)
    selected = []
    used_test_indices = set()

    for (low, high), bname in zip(buckets, bucket_names):
        bucket_candidates = [
            (qi, lab, sel) for qi, lab, sel in candidate_pairs
            if low <= sel < high and qi not in used_test_indices
        ]
        n_pick = min(n_per_bucket, len(bucket_candidates))
        if n_pick > 0:
            idx = rng.choice(len(bucket_candidates), n_pick, replace=False)
            for p in idx:
                qi, lab, sel = bucket_candidates[p]
                selected.append({
                    "query_idx": qi,
                    "label": lab,
                    "selectivity": float(sel),
                    "bucket": bname,
                })
                used_test_indices.add(qi)

    # Fill remaining from any bucket
    shortfall = n_queries - len(selected)
    if shortfall > 0:
        remaining = [
            (qi, lab, sel) for qi, lab, sel in candidate_pairs
            if qi not in used_test_indices
        ]
        n_pick = min(shortfall, len(remaining))
        idx = rng.choice(len(remaining), n_pick, replace=False)
        for p in idx:
            qi, lab, sel = remaining[p]
            selected.append({
                "query_idx": qi,
                "label": lab,
                "selectivity": float(sel),
                "bucket": "overflow",
            })
            used_test_indices.add(qi)

    selected = selected[:n_queries]

    query_indices = [e["query_idx"] for e in selected]
    query_vecs = test_vecs[query_indices]
    query_labels = [e["label"] for e in selected]

    return query_vecs, query_labels, selected


# ---------------------------------------------------------------------------
# Single-label ground truth
# ---------------------------------------------------------------------------

def compute_single_label_gt(
    train_vecs: np.ndarray,
    query_vecs: np.ndarray,
    query_labels: list[int],
    label_to_indices: dict,
    k: int = 10,
):
    """
    For each (query, label) pair, find top-k nearest training vectors
    that have that specific label.

    Organised by label for efficiency: all queries with the same label
    share a single distance computation over that label's candidates.
    """
    n_queries = len(query_labels)
    ground_truth = np.full((n_queries, k), -1, dtype=np.int32)

    # Group queries by label
    label_to_query_indices = {}
    for i, lab in enumerate(query_labels):
        label_to_query_indices.setdefault(lab, []).append(i)

    total = 0
    for lab, qi_list in label_to_query_indices.items():
        candidates = label_to_indices.get(lab, np.array([], dtype=np.int32))
        if len(candidates) == 0:
            continue

        cand_vecs = train_vecs[candidates]  # [n_cand, d]
        q_vecs = query_vecs[qi_list]         # [n_q, d]

        # L2 distances: [n_q, n_cand]
        dists = np.sum(
            (cand_vecs[None, :, :] - q_vecs[:, None, :]) ** 2, axis=2
        )

        k_actual = min(k, len(candidates))
        top_k_local = np.argpartition(dists, k_actual - 1, axis=1)[:, :k_actual]
        # Sort each row
        for row in range(len(qi_list)):
            order = np.argsort(dists[row, top_k_local[row]])
            top_k_local[row] = top_k_local[row][order]

        for row, qi in enumerate(qi_list):
            ground_truth[qi, :k_actual] = candidates[top_k_local[row]]

        total += len(qi_list)

    return ground_truth


# ---------------------------------------------------------------------------
# Complex-predicate utilities
# ---------------------------------------------------------------------------

def evaluate_predicate(tokens: list[str], mds: list[int]) -> bool:
    """Evaluate a boolean formula in Polish notation against one vector's labels."""
    stack = []
    for token in reversed(tokens):
        if token in ("AND", "OR", "NOT"):
            if token == "AND":
                a, b = stack.pop(), stack.pop()
                stack.append(a and b)
            elif token == "OR":
                a, b = stack.pop(), stack.pop()
                stack.append(a or b)
            else:  # NOT
                a = stack.pop()
                stack.append(not a)
        else:
            stack.append(int(token) in mds)
    return stack[0]


def compute_filter_selectivity(
    formula: str, train_mds: list
) -> tuple[float, np.ndarray]:
    """Return (selectivity, qualified_vector_indices) for a filter formula."""
    tokens = formula.split()
    qualified = np.array(
        [i for i, md in enumerate(train_mds)
         if md and evaluate_predicate(tokens, md)],
        dtype=np.int32,
    )
    sel = len(qualified) / len(train_mds)
    return sel, qualified


def generate_random_filters(all_labels: list[int], n_filters: int = 100, seed: int = 42):
    """Generate a diverse set of complex predicate filters."""
    rng = np.random.RandomState(seed)
    labels = np.array(all_labels)
    filters = []

    def sample_labels(n):
        return rng.choice(labels, min(n, len(labels)), replace=False).tolist()

    n_each = max(1, n_filters // 5)

    # NOT filters
    for _ in range(n_each):
        lab = sample_labels(1)
        filters.append(f"NOT {lab[0]}")

    # AND filters
    for _ in range(n_each):
        a, b = sample_labels(2)
        filters.append(f"AND {a} {b}")

    # OR filters
    for _ in range(n_each):
        a, b = sample_labels(2)
        filters.append(f"OR {a} {b}")

    # AND NOT filters
    for _ in range(n_each):
        a, b = sample_labels(2)
        filters.append(f"AND {a} NOT {b}")

    # OR 3 labels — remaining slots
    remaining = n_filters - len(filters)
    for _ in range(remaining):
        a, b, c = sample_labels(3)
        filters.append(f"OR {a} OR {b} {c}")

    return filters[:n_filters]


def compute_complex_predicate_gt(
    train_vecs: np.ndarray,
    train_mds: list,
    query_vecs: np.ndarray,
    filters: list[str],
    k: int = 10,
):
    """
    For each filter, compute top-k among training vectors satisfying the filter,
    for ALL queries.

    Memory-safe: processes one query at a time, training vectors in blocks
    of BLOCK_SIZE, merging into a running top-k.
    """
    BLOCK_SIZE = 50000
    results = {}
    selectivities = {}

    for fi, formula in enumerate(filters):
        print(f"  [{fi + 1}/{len(filters)}] {formula}", flush=True)
        t0 = time.perf_counter()

        sel, qualified = compute_filter_selectivity(formula, train_mds)
        selectivities[formula] = float(sel)

        n_queries = len(query_vecs)
        d = train_vecs.shape[1]

        if len(qualified) == 0:
            results[formula] = np.full((n_queries, k), -1, dtype=np.int32)
            print(f"    selectivity={sel:.4f}, candidates=0, time={time.perf_counter() - t0:.1f}s")
            continue

        gt = np.full((n_queries, k), -1, dtype=np.int32)
        n_candidates = len(qualified)
        k_actual = min(k, n_candidates)

        print(f"    selectivity={sel:.4f}, candidates={n_candidates:,}", flush=True)

        for qi in range(n_queries):
            q_vec = query_vecs[qi]

            # Running top-k: arrays of size at most k
            best_dists = np.empty(0, dtype=np.float32)
            best_indices = np.empty(0, dtype=np.int32)

            for start in range(0, n_candidates, BLOCK_SIZE):
                end = min(start + BLOCK_SIZE, n_candidates)
                block_indices = qualified[start:end]
                block_vecs = train_vecs[block_indices]

                # Distances for this block
                block_dists = np.sum((block_vecs - q_vec) ** 2, axis=1)

                # Keep top-k within this block
                keep = min(k, len(block_dists))
                if keep < len(block_dists):
                    sel_idx = np.argpartition(block_dists, keep - 1)[:keep]
                    block_dists = block_dists[sel_idx]
                    block_indices = block_indices[sel_idx]

                # Merge with running top-k
                if len(best_dists) == 0:
                    best_dists = block_dists
                    best_indices = block_indices
                else:
                    merged_dists = np.concatenate([best_dists, block_dists])
                    merged_indices = np.concatenate([best_indices, block_indices])
                    keep = min(k, len(merged_dists))
                    sel_idx = np.argpartition(merged_dists, keep - 1)[:keep]
                    sort_idx = np.argsort(merged_dists[sel_idx])
                    sel_idx = sel_idx[sort_idx]
                    best_dists = merged_dists[sel_idx]
                    best_indices = merged_indices[sel_idx]

            if len(best_indices) > 0:
                gt[qi, :len(best_indices)] = best_indices

            if (qi + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / (qi + 1) * (n_queries - qi - 1)
                print(f"    query {qi + 1}/{n_queries}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

        results[formula] = gt
        print(f"    done in {time.perf_counter() - t0:.1f}s", flush=True)

    return results, selectivities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute ground truth for Curator benchmark"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["arxiv", "yfcc100m"],
    )
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./ground_truth")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_queries", type=int, default=1000,
                        help="Number of single-label queries")
    parser.add_argument("--n_cp_queries", type=int, default=100,
                        help="Number of queries for complex-predicate GT")
    parser.add_argument("--n_filters", type=int, default=100,
                        help="Number of complex-predicate filters")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loaders = {"arxiv": load_arxiv, "yfcc100m": load_yfcc1m}

    # ================================================================
    # 1. Load dataset
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Ground Truth Computation: {args.dataset}")
    print(f"{'='*60}")

    train_vecs, test_vecs, train_mds, test_mds, meta = loaders[args.dataset](
        args.data_dir
    )
    n_train = len(train_vecs)
    print(f"  Train: {n_train:,} x {meta['dim']}")
    print(f"  Test:  {len(test_vecs):,} x {meta['dim']}")
    print(f"  Labels: {meta['n_labels']}")

    # Build inverted index once (shared)
    print("\n  Building inverted index...")
    t0 = time.perf_counter()
    label_to_indices = build_inverted_index(train_mds, meta["n_labels"])
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ================================================================
    # 2. Select queries (by per-label selectivity)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Selecting {args.n_queries} single-label queries")
    print(f"{'='*60}")

    query_vecs, query_labels, query_info = select_single_label_queries(
        test_vecs, test_mds, train_mds, label_to_indices,
        n_queries=args.n_queries, seed=args.seed,
    )

    from collections import Counter
    bucket_counts = Counter(e["bucket"] for e in query_info)
    print("\n  Query distribution by per-label selectivity:")
    for bname in sorted(bucket_counts.keys()):
        b_entries = [e for e in query_info if e["bucket"] == bname]
        b_sels = [e["selectivity"] for e in b_entries]
        print(f"    {bname:30s}  n={len(b_entries):4d}  "
              f"mean={np.mean(b_sels):.4f}  min={np.min(b_sels):.4f}  max={np.max(b_sels):.4f}")

    # ================================================================
    # 3. Single-label ground truth
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Computing single-label GT (k={args.k})")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    single_label_gt = compute_single_label_gt(
        train_vecs, query_vecs, query_labels, label_to_indices, k=args.k,
    )
    t_sl = time.perf_counter() - t0

    valid = np.sum(single_label_gt >= 0)
    print(f"  Done in {t_sl:.1f}s")
    print(f"  Valid entries: {valid}/{single_label_gt.size} "
          f"({100 * valid / single_label_gt.size:.1f}%)")

    # ================================================================
    # 4. Complex-predicate ground truth (skipped for arxiv — too slow)
    # ================================================================
    if args.dataset == "arxiv":
        print(f"\n{'='*60}")
        print("Complex-predicate GT skipped for arxiv (use --dataset yfcc100m).")
        print(f"{'='*60}")
        t_cp = 0.0
        cp_gt = {}
        cp_selectivities = {}
        cp_query_vecs = np.zeros((0, meta["dim"]), dtype=np.float32)
        cp_indices = np.zeros(0, dtype=np.int32)
        filters = []
        skip_cp = True
    else:
        skip_cp = False

    if not skip_cp:
        print(f"\n{'='*60}")
        print(f"Complex-predicate GT: {args.n_filters} filters x {args.n_cp_queries} queries (k={args.k})")
        print(f"{'='*60}")

        # Select queries for complex predicate (distinct random subset from test)
        cp_rng = np.random.RandomState(args.seed + 1)
        cp_indices = cp_rng.choice(len(test_vecs), args.n_cp_queries, replace=False)
        cp_query_vecs = test_vecs[cp_indices]
        print(f"  Selected {args.n_cp_queries} queries from test set")

        all_labels = sorted(set().union(*train_mds))
        filters = generate_random_filters(all_labels, n_filters=args.n_filters, seed=args.seed)
        print(f"  Generated {len(filters)} filters")

        t0 = time.perf_counter()
        cp_gt, cp_selectivities = compute_complex_predicate_gt(
            train_vecs, train_mds, cp_query_vecs, filters, k=args.k,
        )
        t_cp = time.perf_counter() - t0
        print(f"  Complex-predicate GT done in {t_cp:.1f}s ({t_cp / 60:.1f} min)")

        # Print selectivity distribution
        cp_sels = list(cp_selectivities.values())
        print(f"\n  Filter selectivity stats: min={np.min(cp_sels):.4f}, "
              f"median={np.median(cp_sels):.4f}, mean={np.mean(cp_sels):.4f}, "
              f"max={np.max(cp_sels):.4f}")

    # ================================================================
    # 5. Save
    # ================================================================
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single-label outputs (always saved)
    np.save(output_dir / "ground_truth.npy", single_label_gt)
    np.save(output_dir / "query_vecs.npy", query_vecs)
    np.save(output_dir / "query_labels.npy", np.array(query_labels, dtype=np.int32))
    with open(output_dir / "query_info.json", "w") as f:
        json.dump(query_info, f, indent=2)

    # Complex-predicate outputs (only if computed)
    if not skip_cp:
        cp_dir = output_dir / "complex_predicate"
        cp_dir.mkdir(parents=True, exist_ok=True)
        for formula, gt in cp_gt.items():
            safe_name = formula.replace(" ", "_")
            np.save(cp_dir / f"gt_{safe_name}.npy", gt)
        with open(cp_dir / "filters.json", "w") as f:
            json.dump({
                "n_filters": len(filters),
                "n_queries": args.n_cp_queries,
                "filters": sorted(filters),
                "selectivities": {k: float(v) for k, v in cp_selectivities.items()},
            }, f, indent=2)
        np.save(cp_dir / "query_vecs.npy", cp_query_vecs)
        np.save(cp_dir / "query_indices.npy", cp_indices)

    # Train data (for experiments to reuse the exact split)
    np.save(output_dir / "train_vecs.npy", train_vecs)
    with open(output_dir / "train_mds.pkl", "wb") as f:
        pickle.dump(train_mds, f)

    # Metadata
    meta_out = {
        **meta,
        "n_queries": args.n_queries,
        "k": args.k,
        "train_size": n_train,
        "test_size_original": len(test_vecs),
        "single_label_gt_time_s": float(t_sl),
        "complex_predicate_gt_time_s": float(t_cp),
        "n_filters": len(filters),
        "n_cp_queries": args.n_cp_queries,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    all_labels_list = sorted(
        set().union(*train_mds) | {e["label"] for e in query_info}
    )
    with open(output_dir / "all_labels.json", "w") as f:
        json.dump(all_labels_list, f)

    # Summary
    print(f"\n{'='*60}")
    print(f"Output saved to: {output_dir.resolve()}")
    print(f"{'='*60}")
    print(f"  ground_truth.npy          {single_label_gt.shape} {single_label_gt.dtype}")
    print(f"  query_vecs.npy            {query_vecs.shape}")
    print(f"  query_labels.npy          {len(query_labels)} entries")
    print(f"  query_info.json           {len(query_info)} entries")
    print(f"  train_vecs.npy            {train_vecs.shape}")
    print(f"  train_mds.pkl             {len(train_mds)} entries")
    if not skip_cp:
        print(f"  complex_predicate/        {len(filters)} filters x {args.n_cp_queries} queries")
        for formula in sorted(filters)[:5]:
            gt = cp_gt[formula]
            print(f"    gt_{formula.replace(' ', '_')}.npy  {gt.shape}  sel={cp_selectivities[formula]:.4f}")
        if len(filters) > 5:
            print(f"    ... ({len(filters) - 5} more)")

    print(f"\n  GT computation time: {t_sl + t_cp:.1f}s ({ (t_sl + t_cp) / 60:.1f} min)")
    print(f"  Single-label:  {t_sl:.1f}s")
    if not skip_cp:
        print(f"  Predicate:     {t_cp:.1f}s ({t_cp / 60:.1f} min)")
    else:
        print(f"  Predicate:     skipped")
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
