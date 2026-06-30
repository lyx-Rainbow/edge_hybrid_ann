"""
Complete data preparation pipeline for SIFT1M / GIST1M datasets.

Steps:
  1. Load base vectors from .fvecs files
  2. Synthesize multi-tenant labels via K-means + soft assignment
  3. Split into train / query-pool (or use provided query.fvecs)
  4. Select single-label queries across selectivity buckets
  5. Compute single-label ground truth
  6. (Optional) Compute complex-predicate ground truth
  7. Save to 1_Data/ground_truth/{dataset}/

Usage:
    # Small test run (fast verification)
    python 1_Data/prepare_ann_dataset.py --dataset sift1m --n_train 50000 \
        --n_queries 200 --n_labels 50 --avg_labels 3 --small

    # Full run (user runs manually in WSL terminal)
    python 1_Data/prepare_ann_dataset.py --dataset sift1m
    python 1_Data/prepare_ann_dataset.py --dataset gist1m
"""

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Local imports
from io_utils import read_fvecs, read_ivecs
from synthesize_labels import synthesize_labels_kmeans, synthesize_labels_random


# ---------------------------------------------------------------------------
# Query selection (adapted from gt_computing.py)
# ---------------------------------------------------------------------------

def build_inverted_index(train_mds: list, n_labels: int) -> dict:
    """label -> sorted np.array of training indices."""
    label_to_indices = {i: [] for i in range(n_labels)}
    for idx, labels in enumerate(train_mds):
        for lab in labels:
            label_to_indices[lab].append(idx)
    return {k: np.array(v, dtype=np.int32) for k, v in label_to_indices.items()}


def select_single_label_queries(
    query_pool_vecs: np.ndarray,
    query_pool_mds: list,
    train_mds: list,
    label_to_indices: dict,
    n_queries: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, list[int], list[dict]]:
    """Select (query_vec, label) pairs from the query pool, balancing selectivity.

    Returns:
        query_vecs, query_labels, query_info
    """
    rng = np.random.RandomState(seed)
    n_train = len(train_mds)
    n_labels = len(label_to_indices)

    # Compute per-label selectivity from training set
    label_sel = {}
    for lab, indices in label_to_indices.items():
        label_sel[lab] = len(indices) / n_train

    sels = np.array(list(label_sel.values()))
    if len(sels) == 0:
        raise ValueError("No labels found in training set")
    max_sel = min(sels.max(), 0.5)
    min_sel = max(sels.min(), 0.001)

    # Define 5 selectivity buckets
    bounds = np.linspace(min_sel, max_sel, 6)
    buckets = [(bounds[i], bounds[i + 1]) for i in range(5)]
    bucket_names = [f"[{bounds[i]:.4f}, {bounds[i+1]:.4f})" for i in range(5)]

    # Find all valid (query_pool_idx, label) pairs
    candidate_pairs = []
    for i, labels in enumerate(query_pool_mds):
        for lab in labels:
            sel = label_sel.get(lab, 0.0)
            if sel > 0:
                candidate_pairs.append((i, int(lab), sel))

    print(f"  Total candidate (query, label) pairs: {len(candidate_pairs)}")

    n_per_bucket = max(1, n_queries // len(buckets))
    selected = []
    used_indices = set()

    for (low, high), bname in zip(buckets, bucket_names):
        bucket_candidates = [
            (qi, lab, sel) for qi, lab, sel in candidate_pairs
            if low <= sel < high and qi not in used_indices
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
                used_indices.add(qi)

    # Fill remaining from any bucket
    shortfall = n_queries - len(selected)
    if shortfall > 0:
        remaining = [
            (qi, lab, sel) for qi, lab, sel in candidate_pairs
            if qi not in used_indices
        ]
        n_pick = min(shortfall, len(remaining))
        if n_pick > 0:
            idx = rng.choice(len(remaining), n_pick, replace=False)
            for p in idx:
                qi, lab, sel = remaining[p]
                selected.append({
                    "query_idx": qi,
                    "label": lab,
                    "selectivity": float(sel),
                    "bucket": "overflow",
                })
                used_indices.add(qi)

    selected = selected[:n_queries]

    query_indices = [e["query_idx"] for e in selected]
    query_vecs = query_pool_vecs[query_indices]
    query_labels = [e["label"] for e in selected]

    return query_vecs, query_labels, selected


def compute_single_label_gt(
    train_vecs: np.ndarray,
    query_vecs: np.ndarray,
    query_labels: list[int],
    label_to_indices: dict,
    k: int = 10,
) -> np.ndarray:
    """Compute single-label ground truth: for each (query, label) pair, top-k
    among training vectors that have that label."""
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
        q_vecs = query_vecs[qi_list]        # [n_q, d]

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
        if total % 200 == 0:
            print(f"    processed {total}/{n_queries} queries...", flush=True)

    return ground_truth


# ---------------------------------------------------------------------------
# Complex-predicate GT (adapted from gt_computing.py)
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
                stack.append(not stack.pop())
        else:
            stack.append(int(token) in mds)
    return stack[0]


def generate_random_filters(all_labels: list[int], n_filters: int = 100,
                            seed: int = 42) -> list[str]:
    """Generate a diverse set of complex predicate filters."""
    rng = np.random.RandomState(seed)
    labels = np.array(all_labels)

    def sample_labels(n):
        return rng.choice(labels, min(n, len(labels)), replace=False).tolist()

    n_each = max(1, n_filters // 5)
    filters = []

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
    # OR 3 labels
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
    """Compute top-k among training vectors satisfying each filter, for all queries."""
    BLOCK_SIZE = 50000
    results = {}
    selectivities = {}

    for fi, formula in enumerate(filters):
        print(f"  [{fi + 1}/{len(filters)}] {formula}", flush=True)
        t0 = time.perf_counter()

        # Compute qualified indices
        tokens = formula.split()
        qualified = np.array(
            [i for i, md in enumerate(train_mds)
             if md and evaluate_predicate(tokens, md)],
            dtype=np.int32,
        )
        sel = len(qualified) / len(train_vecs)
        selectivities[formula] = float(sel)

        n_queries = len(query_vecs)

        if len(qualified) == 0:
            results[formula] = np.full((n_queries, k), -1, dtype=np.int32)
            print(f"    selectivity={sel:.4f}, candidates=0, "
                  f"time={time.perf_counter() - t0:.1f}s")
            continue

        gt = np.full((n_queries, k), -1, dtype=np.int32)
        n_candidates = len(qualified)
        print(f"    selectivity={sel:.4f}, candidates={n_candidates:,}", flush=True)

        for qi in range(n_queries):
            q_vec = query_vecs[qi]
            best_dists = np.empty(0, dtype=np.float32)
            best_indices = np.empty(0, dtype=np.int32)

            for start in range(0, n_candidates, BLOCK_SIZE):
                end = min(start + BLOCK_SIZE, n_candidates)
                block_indices = qualified[start:end]
                block_vecs = train_vecs[block_indices]
                block_dists = np.sum((block_vecs - q_vec) ** 2, axis=1)
                keep = min(k, len(block_dists))
                if keep < len(block_dists):
                    sel_idx = np.argpartition(block_dists, keep - 1)[:keep]
                    block_dists = block_dists[sel_idx]
                    block_indices = block_indices[sel_idx]
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
                print(f"    query {qi + 1}/{n_queries}  elapsed={elapsed:.0f}s  "
                      f"eta={eta:.0f}s", flush=True)

        results[formula] = gt
        print(f"    done in {time.perf_counter() - t0:.1f}s", flush=True)

    return results, selectivities


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_dataset(
    dataset: str,
    data_dir: str = "1_Data",
    output_dir: str = "1_Data/ground_truth",
    n_labels: int = 100,
    avg_labels_per_vec: int = 3,
    n_queries: int = 1000,
    k: int = 10,
    seed: int = 42,
    label_method: str = "kmeans",
    skip_cp: bool = False,
    n_cp_queries: int = 100,
    n_filters: int = 100,
    small_mode: bool = False,
    n_train_small: int = 50000,
    n_query_pool_small: int = 5000,
):
    """Main pipeline: prepare a SIFT1M/GIST1M-like dataset for Curator benchmark.

    Parameters
    ----------
    small_mode : bool
        If True, subsample n_train_small training vectors and n_query_pool_small
        query vectors for fast verification.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    ds_dir = data_dir / dataset

    base_file = ds_dir / f"{'sift' if 'sift' in dataset else 'gist'}_base.fvecs"
    query_file = ds_dir / f"{'sift' if 'sift' in dataset else 'gist'}_query.fvecs"
    gt_file = ds_dir / f"{'sift' if 'sift' in dataset else 'gist'}_groundtruth.ivecs"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Preparing dataset: {dataset}")
    print(f"{'='*60}")

    # For small mode, output to a separate dataset name (e.g. "sift1m_small")
    output_dataset = dataset + "_small" if small_mode else dataset

    # Determine read limits for small mode (avoids loading 3.8GB GIST file)
    max_base = n_train_small if small_mode else None
    max_query = n_query_pool_small if small_mode else None

    print(f"\n  Loading base vectors from {base_file}...")
    t0 = time.perf_counter()
    base_vecs = read_fvecs(str(base_file), max_vectors=max_base)
    print(f"  Loaded {base_vecs.shape[0]:,} x {base_vecs.shape[1]} vectors "
          f"in {time.perf_counter() - t0:.1f}s")

    print(f"  Loading query vectors from {query_file}...")
    t0 = time.perf_counter()
    all_query_vecs = read_fvecs(str(query_file), max_vectors=max_query)
    print(f"  Loaded {all_query_vecs.shape[0]:,} x {all_query_vecs.shape[1]} queries "
          f"in {time.perf_counter() - t0:.1f}s")

    d = base_vecs.shape[1]

    # Small mode: already subsampled during read, just adjust counts
    if small_mode:
        n_train = len(base_vecs)
        n_pool = len(all_query_vecs)
        n_queries = min(n_queries, n_pool // 2)
        n_cp_queries = min(n_cp_queries, n_pool // 4)
        n_filters = min(n_filters, n_labels // 2)
        print(f"  [SMALL MODE] Train: {n_train}, Query pool: {n_pool}, "
              f"SL queries: {n_queries}, CP queries: {n_cp_queries}")

    n_train = len(base_vecs)
    n_query_pool = len(all_query_vecs)

    # ------------------------------------------------------------------
    # 2. Synthesize labels
    # ------------------------------------------------------------------
    print(f"\n  Synthesizing labels (method={label_method}, n_labels={n_labels}, "
          f"avg_labels_per_vec={avg_labels_per_vec})...")

    if label_method == "kmeans":
        # Train K-means on base vectors, assign labels to both base and query
        centroids = None
        train_mds = synthesize_labels_kmeans(
            base_vecs, n_labels=n_labels,
            avg_labels_per_vec=avg_labels_per_vec, seed=seed,
        )

        # Assign labels to query vectors using same centroids — but we didn't
        # save them from synthesize_labels_kmeans. Instead, re-cluster queries
        # independently (they use the same label space).
        # Alternative: assign labels to queries the same way.
        # For consistency, train K-means on base, then assign queries to nearest.
        print(f"  Assigning labels to query vectors...")
        import faiss
        # Retrain on base to get centroids for query assignment
        # (We already have labels on base, now need centroids for queries)
        kmeans = faiss.Kmeans(
            d=d, k=n_labels, niter=5, nredo=1, seed=seed,
            verbose=False, max_points_per_centroid=1000, gpu=False,
        )
        kmeans.train(base_vecs)
        centroids = kmeans.centroids

        # Find nearest centroids for each query vector
        index = faiss.IndexFlatL2(d)
        index.add(centroids)
        _, query_assignments = index.search(all_query_vecs, avg_labels_per_vec)
        query_mds = [sorted(row.tolist()) for row in query_assignments]

    elif label_method == "random":
        train_mds = synthesize_labels_random(
            n_train, n_labels=n_labels,
            avg_labels_per_vec=avg_labels_per_vec, seed=seed,
        )
        query_mds = synthesize_labels_random(
            n_query_pool, n_labels=n_labels,
            avg_labels_per_vec=avg_labels_per_vec, seed=seed + 1,
        )
    else:
        raise ValueError(f"Unknown label_method: {label_method}")

    # Use only labels seen in training
    all_train_labels = sorted(set().union(*train_mds))
    actual_n_labels = len(all_train_labels)
    print(f"  Active labels: {actual_n_labels} / {n_labels}")

    # Remap labels to 0..m-1
    label_remap = {old: new for new, old in enumerate(all_train_labels)}
    train_mds = [[label_remap[l] for l in md if l in label_remap] for md in train_mds]
    query_mds = [[label_remap.get(l, 0) for l in md] for md in query_mds]
    n_labels_final = len(all_train_labels)

    # ------------------------------------------------------------------
    # 3. Build inverted index
    # ------------------------------------------------------------------
    print(f"\n  Building inverted index...")
    t0 = time.perf_counter()
    label_to_indices = build_inverted_index(train_mds, n_labels_final)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    for lab in list(range(min(5, n_labels_final))):
        print(f"    label {lab}: {len(label_to_indices[lab]):,} vectors "
              f"(sel={len(label_to_indices[lab]) / n_train:.4f})")

    # ------------------------------------------------------------------
    # 4. Select single-label queries
    # ------------------------------------------------------------------
    print(f"\n  Selecting {n_queries} single-label queries...")
    query_vecs, query_labels, query_info = select_single_label_queries(
        all_query_vecs, query_mds, train_mds, label_to_indices,
        n_queries=n_queries, seed=seed,
    )

    bucket_counts = Counter(e["bucket"] for e in query_info)
    print("\n  Query distribution by per-label selectivity:")
    for bname in sorted(bucket_counts.keys()):
        b_entries = [e for e in query_info if e["bucket"] == bname]
        b_sels = [e["selectivity"] for e in b_entries]
        print(f"    {bname:30s}  n={len(b_entries):4d}  "
              f"mean={np.mean(b_sels):.4f}  min={np.min(b_sels):.4f}  "
              f"max={np.max(b_sels):.4f}")

    # ------------------------------------------------------------------
    # 5. Compute single-label GT
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Computing single-label GT (k={k})")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    single_label_gt = compute_single_label_gt(
        base_vecs, query_vecs, query_labels, label_to_indices, k=k,
    )
    t_sl = time.perf_counter() - t0

    valid = np.sum(single_label_gt >= 0)
    print(f"  Done in {t_sl:.1f}s ({t_sl / 60:.1f} min)")
    print(f"  Valid entries: {valid}/{single_label_gt.size} "
          f"({100 * valid / single_label_gt.size:.1f}%)")

    # ------------------------------------------------------------------
    # 6. Complex-predicate GT (optional)
    # ------------------------------------------------------------------
    t_cp = 0.0
    cp_gt = {}
    cp_selectivities = {}
    cp_query_vecs = np.zeros((0, d), dtype=np.float32)
    cp_query_indices = np.zeros(0, dtype=np.int32)
    filters = []

    if not skip_cp and n_labels_final >= 3:
        print(f"\n{'='*60}")
        print(f"Complex-predicate GT: {n_filters} filters x {n_cp_queries} "
              f"queries (k={k})")
        print(f"{'='*60}")

        cp_rng = np.random.RandomState(seed + 1)
        cp_pool_size = min(n_cp_queries, len(all_query_vecs))
        cp_indices = cp_rng.choice(len(all_query_vecs), cp_pool_size, replace=False)
        cp_query_vecs = all_query_vecs[cp_indices]
        cp_query_indices = cp_indices
        print(f"  Selected {cp_pool_size} queries from query pool")

        all_labels_list = sorted(all_train_labels)
        filters = generate_random_filters(
            all_labels_list, n_filters=n_filters, seed=seed,
        )
        print(f"  Generated {len(filters)} filters")

        t0 = time.perf_counter()
        cp_gt, cp_selectivities = compute_complex_predicate_gt(
            base_vecs, train_mds, cp_query_vecs, filters, k=k,
        )
        t_cp = time.perf_counter() - t0
        print(f"  Complex-predicate GT done in {t_cp:.1f}s ({t_cp / 60:.1f} min)")

        cp_sels = list(cp_selectivities.values())
        if cp_sels:
            print(f"\n  Filter selectivity stats: min={np.min(cp_sels):.4f}, "
                  f"median={np.median(cp_sels):.4f}, mean={np.mean(cp_sels):.4f}, "
                  f"max={np.max(cp_sels):.4f}")
    else:
        skip_cp = True

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    gt_dir = output_dir / output_dataset
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Single-label outputs
    np.save(gt_dir / "train_vecs.npy", base_vecs.astype(np.float32))
    with open(gt_dir / "train_mds.pkl", "wb") as f:
        pickle.dump(train_mds, f)
    np.save(gt_dir / "query_vecs.npy", query_vecs.astype(np.float32))
    np.save(gt_dir / "query_labels.npy", np.array(query_labels, dtype=np.int32))
    with open(gt_dir / "query_info.json", "w") as f:
        json.dump(query_info, f, indent=2)
    np.save(gt_dir / "ground_truth.npy", single_label_gt.astype(np.int32))

    # Complex-predicate outputs
    if not skip_cp and len(cp_gt) > 0:
        cp_dir = gt_dir / "complex_predicate"
        cp_dir.mkdir(parents=True, exist_ok=True)
        for formula, gt in cp_gt.items():
            safe_name = formula.replace(" ", "_")
            np.save(cp_dir / f"gt_{safe_name}.npy", gt)
        with open(cp_dir / "filters.json", "w") as f:
            json.dump({
                "n_filters": len(filters),
                "n_queries": len(cp_query_vecs),
                "filters": sorted(filters),
                "selectivities": {k: float(v) for k, v in cp_selectivities.items()},
            }, f, indent=2)
        np.save(cp_dir / "query_vecs.npy", cp_query_vecs.astype(np.float32))
        np.save(cp_dir / "query_indices.npy", cp_query_indices.astype(np.int32))

    # Metadata
    metadata = {
        "dim": int(d),
        "n_labels": n_labels_final,
        "dataset": output_dataset,
        "n_queries": n_queries,
        "k": k,
        "train_size": n_train,
        "query_pool_size": n_query_pool,
        "single_label_gt_time_s": float(t_sl),
        "complex_predicate_gt_time_s": float(t_cp),
        "n_filters": len(filters),
        "n_cp_queries": len(cp_query_vecs) if not skip_cp else 0,
        "label_method": label_method,
        "avg_labels_per_vec": avg_labels_per_vec,
    }
    if small_mode:
        metadata["small_mode"] = True
        metadata["n_train_original"] = int(
            read_fvecs(str(base_file)).shape[0] if not small_mode else 0
        )
    with open(gt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    all_labels_json = sorted(set().union(*train_mds))
    with open(gt_dir / "all_labels.json", "w") as f:
        json.dump(all_labels_json, f)

    # Summary
    print(f"\n{'='*60}")
    print(f"Output saved to: {gt_dir.resolve()}")
    print(f"{'='*60}")
    print(f"  train_vecs.npy       {base_vecs.shape} {base_vecs.dtype}")
    print(f"  train_mds.pkl        {len(train_mds)} entries")
    print(f"  query_vecs.npy       {query_vecs.shape}")
    print(f"  query_labels.npy     {len(query_labels)} entries")
    print(f"  query_info.json      {len(query_info)} entries")
    print(f"  ground_truth.npy     {single_label_gt.shape} {single_label_gt.dtype}")
    if not skip_cp and len(cp_gt) > 0:
        print(f"  complex_predicate/   {len(filters)} filters x "
              f"{len(cp_query_vecs)} queries")
    print(f"  metadata.json")
    print(f"  all_labels.json")
    if small_mode:
        print(f"\n  ⚠ SMALL MODE: n_train={n_train}, n_queries={n_queries}")
    print(f"\n  GT computation time: {t_sl + t_cp:.1f}s "
          f"({(t_sl + t_cp) / 60:.1f} min)")
    print(f"  Single-label:  {t_sl:.1f}s")
    if not skip_cp:
        print(f"  Predicate:     {t_cp:.1f}s ({t_cp / 60:.1f} min)")
    else:
        print(f"  Predicate:     skipped")
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SIFT1M/GIST1M dataset for Curator benchmark"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["sift1m", "gist1m"],
    )
    parser.add_argument("--data_dir", type=str, default="1_Data")
    parser.add_argument("--output_dir", type=str, default="1_Data/ground_truth")
    parser.add_argument("--n_labels", type=int, default=100)
    parser.add_argument("--avg_labels", type=int, default=3,
                        help="Average labels per vector")
    parser.add_argument("--n_queries", type=int, default=1000,
                        help="Number of single-label queries")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_method", type=str, default="kmeans",
                        choices=["kmeans", "random"])
    parser.add_argument("--skip_cp", action="store_true",
                        help="Skip complex-predicate GT computation")
    parser.add_argument("--n_cp_queries", type=int, default=100)
    parser.add_argument("--n_filters", type=int, default=100)
    parser.add_argument("--small", action="store_true",
                        help="Small mode: subsample data for fast verification")
    parser.add_argument("--n_train_small", type=int, default=50000)
    parser.add_argument("--n_query_pool_small", type=int, default=5000)
    args = parser.parse_args()

    prepare_dataset(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_labels=args.n_labels,
        avg_labels_per_vec=args.avg_labels,
        n_queries=args.n_queries,
        k=args.k,
        seed=args.seed,
        label_method=args.label_method,
        skip_cp=args.skip_cp,
        n_cp_queries=args.n_cp_queries,
        n_filters=args.n_filters,
        small_mode=args.small,
        n_train_small=args.n_train_small,
        n_query_pool_small=args.n_query_pool_small,
    )


if __name__ == "__main__":
    main()
