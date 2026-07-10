"""
Prepare WIT (Wikipedia-based Image Text) dataset for Curator benchmark.

Uses text embeddings (sentence-transformers) + K-means synthetic labels.
Same approach as SIFT1M/GIST1M: K-means clustering on vectors for label synthesis.

Steps:
  1. Parse TSV files, extract text from caption fields
  2. Generate text embeddings via sentence-transformers (all-MiniLM-L6-v2, 384-dim)
  3. Synthesize multi-tenant labels via FAISS K-means + soft assignment
  4. Split into train / query-pool
  5. Select single-label queries across selectivity buckets
  6. Compute single-label ground truth
  7. Save to 1_Data/ground_truth/wit/ and 1_Data/ground_truth/wit_small/

Usage:
    # Small version (fast verification, ~15-20 min, auto-executed)
    python 1_Data/prepare_wit_dataset.py --small

    # Full version (~10.5 hours, user runs manually in WSL terminal)
    python 1_Data/prepare_wit_dataset.py

Requirements:
    pip install sentence-transformers torch faiss-cpu
"""

import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthesize_labels import synthesize_labels_kmeans


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

def parse_wit_tsv(tsv_path: str, max_records: int | None = None) -> list[str]:
    """Parse a WIT TSV file, returning extracted text per record.

    Priority order: caption_reference_description > caption_alt_text_description
    > caption_attribution_description > context_page_description.

    Parameters
    ----------
    tsv_path : str
        Path to the TSV file.
    max_records : int, optional
        If set, read at most this many records (total, not just valid).

    Returns
    -------
    texts : list[str]
        Extracted text per record. Records with completely empty text are skipped.
    """
    texts = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_records is not None and i >= max_records:
                break

            # Extract text
            text = (
                row.get("caption_reference_description", "").strip()
                or row.get("caption_alt_text_description", "").strip()
                or row.get("caption_attribution_description", "").strip()
                or row.get("context_page_description", "").strip()
            )

            if text:
                texts.append(text)

            if (i + 1) % 500000 == 0:
                print(f"    parsed {i + 1:,} records, kept {len(texts):,} with text...",
                      flush=True)

    return texts


# ---------------------------------------------------------------------------
# Embedding generation (batched)
# ---------------------------------------------------------------------------

def generate_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Generate text embeddings using sentence-transformers.

    Parameters
    ----------
    texts : list[str]
    model_name : str
    batch_size : int
    device : str
        "cpu" or "cuda".

    Returns
    -------
    vecs : np.ndarray [n, 384] float32
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model '{model_name}' on {device}...")
    # Try local_files_only first (no network), fall back to online
    import os as _os
    try:
        model = SentenceTransformer(model_name, device=device,
                                     local_files_only=True)
    except Exception:
        print(f"  Local model not found, trying online download...")
        if _os.environ.get("HF_ENDPOINT") is None:
            print(f"  Tip: set HF_ENDPOINT=https://hf-mirror.com for China mirror")
        model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model dimension: {dim}")

    n = len(texts)
    vecs = np.zeros((n, dim), dtype=np.float32)

    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        vecs[start:end] = embeddings

        if (start // batch_size) % 100 == 0 or end == n:
            elapsed = time.perf_counter() - t0
            rate = end / max(elapsed, 0.01)
            eta = (n - end) / max(rate, 0.01)
            print(f"    embedded {end:,}/{n:,} ({100*end/n:.1f}%)  "
                  f"rate={rate:.0f}/s  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)

    print(f"  Embedding complete: {n:,} vectors in {time.perf_counter() - t0:.0f}s")
    return vecs


# ---------------------------------------------------------------------------
# Query selection + GT (same pattern as prepare_ann_dataset.py)
# ---------------------------------------------------------------------------

def build_inverted_index(train_mds: list, n_labels: int) -> dict:
    """label -> sorted np.array of training indices."""
    label_to_indices = {i: [] for i in range(n_labels)}
    for idx, labels in enumerate(train_mds):
        for lab in labels:
            if lab < n_labels:
                label_to_indices[lab].append(idx)
    return {k: np.array(v, dtype=np.int32) for k, v in label_to_indices.items()}


def select_single_label_queries(
    query_pool_vecs: np.ndarray,
    query_pool_mds: list,
    train_mds: list,
    label_to_indices: dict,
    n_queries: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, list[int], list[dict]]:
    """Select (query_vec, label) pairs, balancing selectivity across buckets."""
    rng = np.random.RandomState(seed)
    n_train = len(train_mds)

    # Per-label selectivity
    label_sel = {}
    for lab, indices in label_to_indices.items():
        if len(indices) > 0:
            label_sel[lab] = len(indices) / n_train

    if not label_sel:
        raise ValueError("No labels found in training set")

    sels = np.array(list(label_sel.values()))
    max_sel = min(sels.max(), 0.5)
    min_sel = max(sels.min(), 0.0001)

    # 5 selectivity buckets
    bounds = np.linspace(min_sel, max_sel, 6)
    buckets = [(bounds[i], bounds[i + 1]) for i in range(5)]
    bucket_names = [f"[{bounds[i]:.4f}, {bounds[i+1]:.4f})" for i in range(5)]

    # Find valid (query_idx, label) pairs
    candidate_pairs = []
    for i, labels in enumerate(query_pool_mds):
        for lab in labels:
            sel = label_sel.get(lab, 0.0)
            if sel > 0:
                candidate_pairs.append((i, int(lab), sel))

    print(f"  Candidate (query, label) pairs: {len(candidate_pairs):,}")

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

    # Fill remaining
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
    """Compute top-k among training vectors with matching label."""
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

        cand_vecs = train_vecs[candidates]
        q_vecs = query_vecs[qi_list]

        dists = np.sum((cand_vecs[None, :, :] - q_vecs[:, None, :]) ** 2, axis=2)
        k_actual = min(k, len(candidates))
        top_k_local = np.argpartition(dists, k_actual - 1, axis=1)[:, :k_actual]
        for row in range(len(qi_list)):
            order = np.argsort(dists[row, top_k_local[row]])
            top_k_local[row] = top_k_local[row][order]
        for row, qi in enumerate(qi_list):
            ground_truth[qi, :k_actual] = candidates[top_k_local[row]]

        total += len(qi_list)
        if total % 100 == 0:
            print(f"    GT: {total}/{n_queries} queries processed...", flush=True)

    return ground_truth


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_wit_dataset(
    data_dir: str = "1_Data",
    output_dir: str = "1_Data/ground_truth",
    n_labels: int = 1000,
    avg_labels_per_vec: int = 3,
    n_queries_full: int = 1000,
    n_queries_small: int = 500,
    k: int = 10,
    seed: int = 42,
    small_mode: bool = False,
    n_train_small: int = 50000,
    n_query_pool_small: int = 10000,
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_batch_size: int = 64,
    device: str = "cpu",
    max_train_records: int | None = None,
):
    """Main pipeline for WIT dataset preparation.

    Parameters
    ----------
    small_mode : bool
        If True, create only wit_small (fast verification).
        If False, create both wit_small and wit (full).
    max_train_records : int, optional
        Limit total records read from TSV (for testing).
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    train_tsv = str(data_dir / "wit" / "wit_v1.train.all-00000-of-00010.tsv"
                     / "wit_v1.train.all-00000-of-00010.tsv")

    # ------------------------------------------------------------------
    # 1. Parse TSV and extract texts
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"WIT Dataset Preparation")
    print(f"{'='*60}")
    print(f"  Small mode: {small_mode}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Device: {device}")
    print(f"  n_labels: {n_labels}")
    print(f"  avg_labels_per_vec: {avg_labels_per_vec}")

    # For small mode, only read enough records to get the required subset
    if small_mode and max_train_records is None:
        # Read ~3x needed to get diverse sample after text filtering
        max_read = (n_train_small + n_query_pool_small) * 3
        print(f"  [SMALL MODE] Limiting TSV read to ~{max_read:,} records")
    else:
        max_read = max_train_records

    print(f"\n  Parsing training TSV: {train_tsv}")
    t0 = time.perf_counter()
    texts = parse_wit_tsv(train_tsv, max_records=max_read)
    print(f"  Parsed {len(texts):,} records with text "
          f"in {time.perf_counter() - t0:.1f}s")

    n_total = len(texts)
    if small_mode:
        n_needed = n_train_small + n_query_pool_small
        if n_total < n_needed:
            print(f"  ERROR: Only got {n_total:,} texts, need {n_needed:,}.")
            print(f"  Increase --max_train_records or reduce --n_train_small.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. For small mode: sample texts BEFORE embedding (huge time saver)
    # ------------------------------------------------------------------
    rng = np.random.RandomState(seed)

    if small_mode:
        n_texts_sample = n_train_small + n_query_pool_small
        print(f"\n  [SMALL MODE] Sampling {n_texts_sample:,} texts from "
              f"{n_total:,} before embedding...")
        sample_idx = rng.choice(n_total, n_texts_sample, replace=False)
        texts = [texts[i] for i in sample_idx]
        n_total = len(texts)

    # ------------------------------------------------------------------
    # 3. Generate embeddings (only for sampled texts in small mode)
    # ------------------------------------------------------------------
    print(f"\n  Generating embeddings for {n_total:,} texts...")
    t0 = time.perf_counter()
    all_vecs = generate_embeddings(
        texts,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        device=device,
    )
    d = all_vecs.shape[1]
    t_embed = time.perf_counter() - t0

    # Release text strings to save ~1.5 GB memory (3.7M strings)
    del texts

    # ------------------------------------------------------------------
    # 4. Synthesize K-means labels
    # ------------------------------------------------------------------
    print(f"\n  Synthesizing K-means labels (n_labels={n_labels}, "
          f"avg_labels_per_vec={avg_labels_per_vec})...")
    t0 = time.perf_counter()
    mds = synthesize_labels_kmeans(
        all_vecs, n_labels=n_labels,
        avg_labels_per_vec=avg_labels_per_vec, seed=seed,
    )
    t_kmeans = time.perf_counter() - t0
    print(f"  K-means labeling done in {t_kmeans:.1f}s")

    # Compute actual label count
    all_labels_set = sorted(set().union(*mds))
    actual_n_labels = len(all_labels_set)
    print(f"  Active labels: {actual_n_labels}/{n_labels}")

    # Remap labels to 0..m-1
    label_remap = {old: new for new, old in enumerate(all_labels_set)}
    mds = [[label_remap[l] for l in md] for md in mds]

    # ------------------------------------------------------------------
    # 5. Train/test split
    # ------------------------------------------------------------------
    idx = rng.permutation(n_total)
    n_train_full = int(n_total * 0.8)
    n_query_pool_full = n_total - n_train_full

    train_vecs_full = all_vecs[idx[:n_train_full]]
    query_pool_vecs_full = all_vecs[idx[n_train_full:]]
    train_mds_full = [mds[i] for i in idx[:n_train_full]]
    query_pool_mds_full = [mds[i] for i in idx[n_train_full:]]

    print(f"\n  Train/test split (full):")
    print(f"    Train: {n_train_full:,}, Query pool: {n_query_pool_full:,}")
    print(f"    Vector dimension: {d}")
    print(f"    Embedding time: {t_embed:.0f}s ({t_embed / 60:.1f} min)")
    print(f"    K-means time: {t_kmeans:.1f}s")

    # ------------------------------------------------------------------
    # 5. Create wit_small (always)
    # ------------------------------------------------------------------
    def make_dataset(n_train, n_pool, n_q, output_name):
        print(f"\n{'='*60}")
        print(f"Creating: {output_name} (n_train={n_train})")
        print(f"{'='*60}")

        # Only copy when sampling a subset; use view for full dataset to save memory
        if n_train < len(train_vecs_full):
            train_small = train_vecs_full[:n_train].copy()
        else:
            train_small = train_vecs_full[:n_train]  # view — no extra memory
        train_mds_small = train_mds_full[:n_train]

        pool_size = min(n_pool, len(query_pool_vecs_full))
        if pool_size < len(query_pool_vecs_full):
            query_pool_small = query_pool_vecs_full[:pool_size].copy()
        else:
            query_pool_small = query_pool_vecs_full[:pool_size]  # view
        query_pool_mds_small = query_pool_mds_full[:pool_size]

        n_q = min(n_q, pool_size // 2)

        # Build inverted index
        print(f"  Building inverted index...")
        t0 = time.perf_counter()
        label_to_indices = build_inverted_index(train_mds_small, actual_n_labels)
        active_labels = [k for k, v in label_to_indices.items() if len(v) > 0]
        print(f"  Active labels: {len(active_labels)}/{actual_n_labels} "
              f"in {time.perf_counter() - t0:.1f}s")

        # Select queries
        print(f"  Selecting {n_q} single-label queries...")
        query_vecs, query_labels, query_info = select_single_label_queries(
            query_pool_small, query_pool_mds_small, train_mds_small,
            label_to_indices, n_queries=n_q, seed=seed,
        )

        bucket_counts = Counter(e["bucket"] for e in query_info)
        print("  Query distribution by selectivity:")
        for bname in sorted(bucket_counts.keys()):
            b_entries = [e for e in query_info if e["bucket"] == bname]
            b_sels = [e["selectivity"] for e in b_entries]
            print(f"    {bname:30s}  n={len(b_entries):4d}  "
                  f"mean={np.mean(b_sels):.4f}  min={np.min(b_sels):.4f}  "
                  f"max={np.max(b_sels):.4f}")

        # Compute GT
        print(f"  Computing single-label GT (k={k})...")
        t0 = time.perf_counter()
        sl_gt = compute_single_label_gt(
            train_small, query_vecs, query_labels, label_to_indices, k=k,
        )
        t_gt = time.perf_counter() - t0
        valid = np.sum(sl_gt >= 0)
        print(f"  GT done in {t_gt:.1f}s, valid={valid}/{sl_gt.size} "
              f"({100 * valid / max(sl_gt.size, 1):.1f}%)")

        # Save
        gt_dir = output_dir / output_name
        gt_dir.mkdir(parents=True, exist_ok=True)

        np.save(gt_dir / "train_vecs.npy", train_small.astype(np.float32))
        with open(gt_dir / "train_mds.pkl", "wb") as f:
            pickle.dump(train_mds_small, f)
        np.save(gt_dir / "query_vecs.npy", query_vecs.astype(np.float32))
        np.save(gt_dir / "query_labels.npy", np.array(query_labels, dtype=np.int32))
        with open(gt_dir / "query_info.json", "w") as f:
            json.dump(query_info, f, indent=2)
        np.save(gt_dir / "ground_truth.npy", sl_gt.astype(np.int32))

        with open(gt_dir / "metadata.json", "w") as f:
            json.dump({
                "dim": int(d),
                "n_labels": actual_n_labels,
                "dataset": output_name,
                "n_queries": n_q,
                "k": k,
                "train_size": n_train,
                "query_pool_size": pool_size,
                "embedding_model": embedding_model,
                "embedding_time_s": float(t_embed),
                "kmeans_time_s": float(t_kmeans),
                "label_method": "kmeans",
                "avg_labels_per_vec": avg_labels_per_vec,
            }, f, indent=2)

        all_labels_json = sorted(set().union(*train_mds_small))
        with open(gt_dir / "all_labels.json", "w") as f:
            json.dump(all_labels_json, f)

        print(f"\n  Saved to: {gt_dir.resolve()}")
        print(f"    train_vecs: {train_small.shape}, "
              f"{train_small.nbytes / 1024**2:.1f} MB")
        print(f"    query_vecs: {query_vecs.shape}")
        print(f"    ground_truth: {sl_gt.shape}")

    # Always create small version
    n_train_s = min(n_train_small, n_train_full)
    n_pool_s = min(n_query_pool_small, n_query_pool_full)
    make_dataset(n_train_s, n_pool_s, n_queries_small, "wit_small")

    # ------------------------------------------------------------------
    # 6. Create full version (skip if small_mode)
    # ------------------------------------------------------------------
    if small_mode:
        print(f"\n  [SMALL MODE] Skipping full dataset (wit).")
        print(f"  To create full dataset, run without --small flag.")
    else:
        n_q_full = min(n_queries_full, n_query_pool_full // 2)
        make_dataset(n_train_full, n_query_pool_full, n_q_full, "wit")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"WIT dataset preparation complete!")
    print(f"{'='*60}")
    print(f"  Records with text: {n_total:,}")
    print(f"  Dimension: {d}")
    print(f"  Labels (K-means): {actual_n_labels}")
    print(f"  Embedding time: {t_embed:.0f}s ({t_embed / 60:.1f} min)")
    print(f"  K-means time: {t_kmeans:.1f}s")
    if small_mode:
        print(f"  Created: wit_small")
    else:
        print(f"  Created: wit_small + wit")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare WIT dataset for Curator benchmark"
    )
    parser.add_argument("--data_dir", type=str, default="1_Data")
    parser.add_argument("--output_dir", type=str, default="1_Data/ground_truth")
    parser.add_argument("--n_labels", type=int, default=1000)
    parser.add_argument("--avg_labels_per_vec", type=int, default=3)
    parser.add_argument("--n_queries_full", type=int, default=1000)
    parser.add_argument("--n_queries_small", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small", action="store_true",
                        help="Small mode: create only wit_small (50K vectors)")
    parser.add_argument("--n_train_small", type=int, default=50000)
    parser.add_argument("--n_query_pool_small", type=int, default=10000)
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for embedding: 'cpu' or 'cuda'")
    parser.add_argument("--max_train_records", type=int, default=None,
                        help="Limit total records (for testing)")
    args = parser.parse_args()

    # Auto-detect CUDA
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                args.device = "cpu"
        except ImportError:
            print("torch not available, falling back to CPU")
            args.device = "cpu"

    prepare_wit_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_labels=args.n_labels,
        avg_labels_per_vec=args.avg_labels_per_vec,
        n_queries_full=args.n_queries_full,
        n_queries_small=args.n_queries_small,
        k=args.k,
        seed=args.seed,
        small_mode=args.small,
        n_train_small=args.n_train_small,
        n_query_pool_small=args.n_query_pool_small,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        device=args.device,
        max_train_records=args.max_train_records,
    )


if __name__ == "__main__":
    main()
