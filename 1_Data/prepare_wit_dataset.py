"""
Prepare WIT (Wikipedia-based Image Text) dataset for Curator benchmark.

Uses text embeddings (sentence-transformers) + K-means synthetic labels.
WITH CHECKPOINT SUPPORT: caches intermediate results so restart doesn't recompute.

Checkpoints (in <output_dir>/wit/.cache/):
  .cache_all_vecs.npy   — embedded vectors [N, 384] float32
  .cache_mds.pkl         — label assignments list[list[int]]
  .cache_split.npz       — train/test split indices

If checkpoints exist, processing resumes from the first missing stage.

Usage:
    HF_HUB_OFFLINE=1 python 1_Data/prepare_wit_dataset.py          # full
    HF_HUB_OFFLINE=1 python 1_Data/prepare_wit_dataset.py --small  # small only
"""

import argparse
import csv
import gc
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthesize_labels import synthesize_labels_kmeans


# ===========================================================================
# TSV parsing
# ===========================================================================

def parse_wit_tsv(tsv_path: str, max_records: int | None = None) -> list[str]:
    """Parse a WIT TSV file, returning extracted text per record."""
    texts = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_records is not None and i >= max_records:
                break
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


# ===========================================================================
# Embedding
# ===========================================================================

def generate_embeddings(texts, model_name="all-MiniLM-L6-v2",
                        batch_size=64, device="cpu") -> np.ndarray:
    """Generate text embeddings, returning [n, dim] float32 array."""
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model '{model_name}' on {device}...")
    import os as _os
    try:
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception:
        print(f"  Local model not found, trying online...")
        if _os.environ.get("HF_ENDPOINT") is None:
            print(f"  Tip: set HF_ENDPOINT=https://hf-mirror.com")
        model = SentenceTransformer(model_name, device=device)

    dim = model.get_embedding_dimension() if hasattr(model, 'get_embedding_dimension') \
          else model.get_sentence_embedding_dimension()
    print(f"  Model dimension: {dim}")

    n = len(texts)
    vecs = np.zeros((n, dim), dtype=np.float32)
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        embeddings = model.encode(texts[start:end], batch_size=batch_size,
                                  show_progress_bar=False,
                                  normalize_embeddings=False,
                                  convert_to_numpy=True)
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


# ===========================================================================
# Inverted index
# ===========================================================================

def build_inverted_index(train_mds, n_labels):
    label_to_indices = {i: [] for i in range(n_labels)}
    for idx, labels in enumerate(train_mds):
        for lab in labels:
            if lab < n_labels:
                label_to_indices[lab].append(idx)
    return {k: np.array(v, dtype=np.int32) for k, v in label_to_indices.items()}


# ===========================================================================
# Query selection
# ===========================================================================

def select_single_label_queries(query_pool_vecs, query_pool_mds, train_mds,
                                label_to_indices, n_queries=500, seed=42):
    rng = np.random.RandomState(seed)
    n_train = len(train_mds)

    label_sel = {}
    for lab, indices in label_to_indices.items():
        if len(indices) > 0:
            label_sel[lab] = len(indices) / n_train
    if not label_sel:
        raise ValueError("No labels found")

    sels = np.array(list(label_sel.values()))
    max_sel, min_sel = min(sels.max(), 0.5), max(sels.min(), 0.0001)
    bounds = np.linspace(min_sel, max_sel, 6)
    buckets = [(bounds[i], bounds[i+1]) for i in range(5)]
    bucket_names = [f"[{bounds[i]:.4f}, {bounds[i+1]:.4f})" for i in range(5)]

    candidate_pairs = []
    for i, labels in enumerate(query_pool_mds):
        for lab in labels:
            sel = label_sel.get(lab, 0.0)
            if sel > 0:
                candidate_pairs.append((i, int(lab), sel))
    print(f"  Candidate (query, label) pairs: {len(candidate_pairs):,}")

    n_per_bucket = max(1, n_queries // len(buckets))
    selected, used_indices = [], set()
    for (low, high), bname in zip(buckets, bucket_names):
        bucket = [(qi, lab, sel) for qi, lab, sel in candidate_pairs
                  if low <= sel < high and qi not in used_indices]
        n_pick = min(n_per_bucket, len(bucket))
        if n_pick > 0:
            for p in rng.choice(len(bucket), n_pick, replace=False):
                qi, lab, sel = bucket[p]
                selected.append({"query_idx": qi, "label": lab,
                                 "selectivity": float(sel), "bucket": bname})
                used_indices.add(qi)
    shortfall = n_queries - len(selected)
    if shortfall > 0:
        remaining = [(qi, lab, sel) for qi, lab, sel in candidate_pairs
                     if qi not in used_indices]
        n_pick = min(shortfall, len(remaining))
        if n_pick > 0:
            for p in rng.choice(len(remaining), n_pick, replace=False):
                qi, lab, sel = remaining[p]
                selected.append({"query_idx": qi, "label": lab,
                                 "selectivity": float(sel), "bucket": "overflow"})
                used_indices.add(qi)
    selected = selected[:n_queries]
    query_indices = [e["query_idx"] for e in selected]
    return (query_pool_vecs[query_indices], [e["label"] for e in selected], selected)


# ===========================================================================
# GT computation — MEMORY-OPTIMIZED with candidate batching
# ===========================================================================

def compute_single_label_gt(train_vecs, query_vecs, query_labels,
                            label_to_indices, k=10):
    """Compute top-k GT with batched candidate processing to limit memory."""
    n_queries = len(query_labels)
    ground_truth = np.full((n_queries, k), -1, dtype=np.int32)
    CAND_BATCH = 50000  # process at most 50K candidates at once

    label_to_qi = {}
    for i, lab in enumerate(query_labels):
        label_to_qi.setdefault(lab, []).append(i)

    total = 0
    for lab, qi_list in label_to_qi.items():
        candidates = label_to_indices.get(lab, np.array([], dtype=np.int32))
        if len(candidates) == 0:
            continue

        q_vecs = query_vecs[qi_list]
        n_q = len(qi_list)
        k_act = min(k, len(candidates))

        # Running best: [n_q, k_act] indices and distances
        best_idx = np.full((n_q, k_act), -1, dtype=np.int32)
        best_dst = np.full((n_q, k_act), np.inf, dtype=np.float32)

        for cstart in range(0, len(candidates), CAND_BATCH):
            cend = min(cstart + CAND_BATCH, len(candidates))
            cand_batch = candidates[cstart:cend]
            cand_vecs = train_vecs[cand_batch]  # [batch, d]
            dists = np.sum((cand_vecs[None, :, :] - q_vecs[:, None, :]) ** 2, axis=2)
            # dists: [n_q, batch]

            # Merge with running best
            merged_d = np.concatenate([best_dst, dists], axis=1)
            merged_i = np.concatenate([best_idx, cand_batch[None, :].repeat(n_q, axis=0)], axis=1)
            keep = min(k_act, merged_d.shape[1])
            sel = np.argpartition(merged_d, keep - 1, axis=1)[:, :keep]
            best_dst = np.take_along_axis(merged_d, sel, axis=1)
            best_idx = np.take_along_axis(merged_i, sel, axis=1)

            del cand_vecs, dists, merged_d, merged_i
            gc.collect()

        # Sort final best
        sort_idx = np.argsort(best_dst, axis=1)
        best_dst = np.take_along_axis(best_dst, sort_idx, axis=1)
        best_idx = np.take_along_axis(best_idx, sort_idx, axis=1)

        for row, qi in enumerate(qi_list):
            ground_truth[qi, :k_act] = best_idx[row]

        del best_dst, best_idx, q_vecs
        gc.collect()

        total += n_q
        if total % 200 == 0:
            print(f"    GT: {total}/{n_queries} queries processed...", flush=True)

    return ground_truth


# ===========================================================================
# Main pipeline
# ===========================================================================

def prepare_wit_dataset(data_dir="1_Data", output_dir="1_Data/ground_truth",
                        n_labels=1000, avg_labels_per_vec=3,
                        n_queries_full=1000, n_queries_small=500,
                        k=10, seed=42, small=False,
                        n_train_small=50000, n_query_pool_small=10000,
                        embedding_model="all-MiniLM-L6-v2",
                        embedding_batch_size=64, device="cpu",
                        max_train_records=None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    cache_dir = output_dir / "wit" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_tsv = str(data_dir / "wit" / "wit_v1.train.all-00000-of-00010.tsv"
                     / "wit_v1.train.all-00000-of-00010.tsv")

    rng = np.random.RandomState(seed)

    # ==================================================================
    # Stage 1: TSV parsing
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"WIT Dataset Preparation")
    print(f"{'='*60}")
    print(f"  Small mode: {small}")
    print(f"  Device: {device}")

    if small and max_train_records is None:
        max_read = (n_train_small + n_query_pool_small) * 3
    else:
        max_read = max_train_records

    print(f"\n  Parsing training TSV: {train_tsv}")
    t0 = time.perf_counter()
    texts = parse_wit_tsv(train_tsv, max_records=max_read)
    print(f"  Parsed {len(texts):,} records with text in {time.perf_counter()-t0:.1f}s")
    n_total = len(texts)

    # Small mode: sample before embedding
    if small:
        n_sample = n_train_small + n_query_pool_small
        if n_total < n_sample:
            print(f"ERROR: need {n_sample} texts, got {n_total}")
            sys.exit(1)
        print(f"  [SMALL] Sampling {n_sample:,} texts before embedding...")
        sample_idx = rng.choice(n_total, n_sample, replace=False)
        texts = [texts[i] for i in sample_idx]
        n_total = len(texts)

    # ==================================================================
    # Stage 2: Embedding (CHECKPOINT)
    # ==================================================================
    cache_vecs = cache_dir / ".cache_all_vecs.npy"
    if cache_vecs.exists() and not small:
        print(f"\n  [CHECKPOINT] Loading cached vectors from {cache_vecs}")
        all_vecs = np.load(cache_vecs).astype(np.float32)
        d = all_vecs.shape[1]
        t_embed = -1
        del texts  # free ~1.5 GB on resume
        if all_vecs.shape[0] != n_total:
            print(f"  ERROR: cached vectors ({all_vecs.shape[0]}) != texts ({n_total})")
            print(f"  Delete {cache_dir} and re-run.")
            sys.exit(1)
        print(f"  Loaded {all_vecs.shape[0]:,} x {d} vectors")
    else:
        print(f"\n  Generating embeddings for {n_total:,} texts...")
        t0 = time.perf_counter()
        all_vecs = generate_embeddings(texts, model_name=embedding_model,
                                       batch_size=embedding_batch_size, device=device)
        d = all_vecs.shape[1]
        t_embed = time.perf_counter() - t0
        del texts  # free ~1.5 GB

        if not small:
            print(f"  [CHECKPOINT] Saving vectors to {cache_vecs}")
            np.save(cache_vecs, all_vecs)

    # ==================================================================
    # Stage 3: K-means labels (CHECKPOINT)
    # ==================================================================
    cache_mds = cache_dir / ".cache_mds.pkl"
    if cache_mds.exists() and not small:
        print(f"\n  [CHECKPOINT] Loading cached labels from {cache_mds}")
        with open(cache_mds, "rb") as f:
            mds = pickle.load(f)
        if len(mds) != n_total:
            print(f"  ERROR: cached labels ({len(mds)}) != vectors ({n_total})")
            print(f"  Delete {cache_dir} and re-run.")
            sys.exit(1)
        all_labels_set = sorted(set().union(*mds))
        actual_n_labels = len(all_labels_set)
        t_kmeans = -1
        print(f"  Loaded {len(mds):,} labels, {actual_n_labels} active")
    else:
        print(f"\n  Synthesizing K-means labels (n_labels={n_labels}, "
              f"avg_labels_per_vec={avg_labels_per_vec})...")
        t0 = time.perf_counter()
        mds = synthesize_labels_kmeans(all_vecs, n_labels=n_labels,
                                       avg_labels_per_vec=avg_labels_per_vec, seed=seed)
        t_kmeans = time.perf_counter() - t0
        all_labels_set = sorted(set().union(*mds))
        actual_n_labels = len(all_labels_set)
        print(f"  Active labels: {actual_n_labels}/{n_labels}")
        label_remap = {old: new for new, old in enumerate(all_labels_set)}
        mds = [[label_remap[l] for l in md] for md in mds]

        if not small:
            print(f"  [CHECKPOINT] Saving labels to {cache_mds}")
            with open(cache_mds, "wb") as f:
                pickle.dump(mds, f)

    # ==================================================================
    # Stage 4: Train/test split — MEMORY-CRITICAL
    #
    # Old code created two fancy-index copies (train + pool = +5.3 GB),
    # while all_vecs (5.3 GB) was still alive → peak 10.6 GB → OOM.
    #
    # New approach: permute all_vecs ONCE, then free the original,
    # then take contiguous VIEWS. Net memory unchanged at 5.3 GB.
    # ==================================================================
    cache_split = cache_dir / ".cache_split.npz"
    if cache_split.exists() and not small:
        print(f"\n  [CHECKPOINT] Loading cached split from {cache_split}")
        split = np.load(cache_split)
        train_idx = split["train_idx"]
        pool_idx = split["pool_idx"]
        n_train_full = len(train_idx)
        n_query_pool_full = len(pool_idx)
        # Reconstruct permuted array
        idx_all = np.concatenate([train_idx, pool_idx])
        print(f"  Re-permuting from checkpoint (single copy)...")
        all_vecs_perm = all_vecs[idx_all]
        del all_vecs
        gc.collect()
    else:
        idx_all = rng.permutation(n_total)
        n_train_full = int(n_total * 0.8)
        n_query_pool_full = n_total - n_train_full
        train_idx = idx_all[:n_train_full]
        pool_idx = idx_all[n_train_full:]

        if not small:
            print(f"  [CHECKPOINT] Saving split to {cache_split}")
            np.savez(cache_split, train_idx=train_idx, pool_idx=pool_idx)

        print(f"  Permuting vectors (one copy, then freeing original)...")
        all_vecs_perm = all_vecs[idx_all]
        del all_vecs
        gc.collect()

    # Contiguous VIEWS — zero extra memory
    train_vecs_full = all_vecs_perm[:n_train_full]
    query_pool_vecs_full = all_vecs_perm[n_train_full:]
    train_mds_full = [mds[i] for i in idx_all[:n_train_full]]
    query_pool_mds_full = [mds[i] for i in idx_all[n_train_full:]]

    print(f"\n  Train/test split:")
    print(f"    Train: {n_train_full:,}, Query pool: {n_query_pool_full:,}")
    print(f"    Vector dimension: {d}")
    if t_embed > 0:
        print(f"    Embedding time: {t_embed:.0f}s ({t_embed/60:.1f} min)")
    if t_kmeans > 0:
        print(f"    K-means time: {t_kmeans:.1f}s")

    # ==================================================================
    # Stage 5: Create datasets
    # ==================================================================

    def make_dataset(n_train, n_pool, n_q, output_name):
        print(f"\n{'='*60}")
        print(f"Creating: {output_name} (n_train={n_train})")
        print(f"{'='*60}")

        if n_train < len(train_vecs_full):
            train_vecs = train_vecs_full[:n_train].copy()
        else:
            train_vecs = train_vecs_full[:n_train]
        train_mds_sub = train_mds_full[:n_train]

        pool_size = min(n_pool, len(query_pool_vecs_full))
        if pool_size < len(query_pool_vecs_full):
            pool_vecs = query_pool_vecs_full[:pool_size].copy()
        else:
            pool_vecs = query_pool_vecs_full[:pool_size]
        pool_mds_sub = query_pool_mds_full[:pool_size]

        n_q = min(n_q, pool_size // 2)

        print(f"  Building inverted index...")
        t0 = time.perf_counter()
        label_to_indices = build_inverted_index(train_mds_sub, actual_n_labels)
        active = [k for k, v in label_to_indices.items() if len(v) > 0]
        print(f"  Active labels: {len(active)}/{actual_n_labels} in {time.perf_counter()-t0:.1f}s")

        print(f"  Selecting {n_q} single-label queries...")
        query_vecs, query_labels, query_info = select_single_label_queries(
            pool_vecs, pool_mds_sub, train_mds_sub,
            label_to_indices, n_queries=n_q, seed=seed)

        bc = Counter(e["bucket"] for e in query_info)
        print("  Query distribution:")
        for b in sorted(bc.keys()):
            be = [e for e in query_info if e["bucket"] == b]
            bs = [e["selectivity"] for e in be]
            print(f"    {b:30s}  n={len(be):4d}  mean={np.mean(bs):.4f}")

        print(f"  Computing GT (k={k}) [memory-optimized]...")
        t0 = time.perf_counter()
        gt = compute_single_label_gt(train_vecs, query_vecs, query_labels,
                                     label_to_indices, k=k)
        t_gt = time.perf_counter() - t0
        valid = np.sum(gt >= 0)
        print(f"  GT done in {t_gt:.1f}s, valid={valid}/{gt.size} ({100*valid/max(gt.size,1):.1f}%)")

        # Save
        gt_dir = output_dir / output_name
        gt_dir.mkdir(parents=True, exist_ok=True)
        np.save(gt_dir / "train_vecs.npy", train_vecs.astype(np.float32))
        with open(gt_dir / "train_mds.pkl", "wb") as f:
            pickle.dump(train_mds_sub, f)
        np.save(gt_dir / "query_vecs.npy", query_vecs.astype(np.float32))
        np.save(gt_dir / "query_labels.npy", np.array(query_labels, dtype=np.int32))
        with open(gt_dir / "query_info.json", "w") as f:
            json.dump(query_info, f, indent=2)
        np.save(gt_dir / "ground_truth.npy", gt.astype(np.int32))
        with open(gt_dir / "metadata.json", "w") as f:
            json.dump({"dim": int(d), "n_labels": actual_n_labels,
                       "dataset": output_name, "n_queries": n_q, "k": k,
                       "train_size": n_train, "query_pool_size": pool_size,
                       "embedding_model": embedding_model,
                       "label_method": "kmeans",
                       "avg_labels_per_vec": avg_labels_per_vec}, f, indent=2)
        all_lbls = sorted(set().union(*train_mds_sub))
        with open(gt_dir / "all_labels.json", "w") as f:
            json.dump(all_lbls, f)
        print(f"\n  Saved: {gt_dir.resolve()}")
        print(f"    train_vecs: {train_vecs.shape}, {train_vecs.nbytes/1024**2:.1f} MB")
        print(f"    query_vecs: {query_vecs.shape}")
        print(f"    ground_truth: {gt.shape}")

        # Free memory
        del train_vecs, pool_vecs, gt
        gc.collect()

    # Always create small
    n_ts = min(n_train_small, n_train_full)
    n_ps = min(n_query_pool_small, n_query_pool_full)
    make_dataset(n_ts, n_ps, n_queries_small, "wit_small")

    if small:
        print(f"\n  [SMALL MODE] Skipping wit (full).")
    else:
        make_dataset(n_train_full, n_query_pool_full, n_queries_full, "wit")

    print(f"\n{'='*60}")
    print(f"WIT dataset preparation complete!")
    print(f"  Records: {n_total:,}  |  Dim: {d}  |  Labels: {actual_n_labels}")
    if not small:
        print(f"  Checkpoints cached in: {cache_dir}")
    print(f"{'='*60}\n")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="Prepare WIT dataset for Curator benchmark")
    p.add_argument("--data_dir", type=str, default="1_Data")
    p.add_argument("--output_dir", type=str, default="1_Data/ground_truth")
    p.add_argument("--n_labels", type=int, default=1000)
    p.add_argument("--avg_labels_per_vec", type=int, default=3)
    p.add_argument("--n_queries_full", type=int, default=1000)
    p.add_argument("--n_queries_small", type=int, default=500)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--small", action="store_true")
    p.add_argument("--n_train_small", type=int, default=50000)
    p.add_argument("--n_query_pool_small", type=int, default=10000)
    p.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--embedding_batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_train_records", type=int, default=None)
    args = p.parse_args()

    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"

    prepare_wit_dataset(**vars(args))


if __name__ == "__main__":
    main()
