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
from synthesize_labels import synthesize_labels_kmeans, synthesize_labels_random
from prepare_ann_dataset import (  # noqa: E402
    generate_random_filters,
    compute_complex_predicate_gt,
)


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
                        max_train_records=None,
                        label_method="kmeans",
                        output_name=None,
                        skip_cp=False,
                        n_cp_queries=100, n_filters=50,
                        distribution="hierarchical",
                        n_coarse=None, n_fine=None,
                        coarse_sel_min=0.10, coarse_sel_max=0.30,
                        fine_sel_min=0.001, fine_sel_max=0.03,
                        zipf_alpha=2.0,
                        coarse_labels_per_vec=1, fine_labels_per_vec=2):
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
    # Stage 2: Embedding (CHECKPOINT — used by kmeans mode; random mode
    #           loads a subset from the existing cache via mmap)
    # ==================================================================
    cache_vecs = cache_dir / ".cache_all_vecs.npy"
    t_embed = -1

    if label_method == "random":
        # Random labels don't need K-means.  Load the required subset
        # from the existing embedding cache (which covers all ~2.97M vectors).
        if not cache_vecs.exists():
            print(f"  ERROR: Cache {cache_vecs} not found.  Run kmeans mode "
                  f"first to generate embeddings, then re-run with random labels.")
            sys.exit(1)
        needed = n_train_small + n_query_pool_small if small else n_total
        print(f"\n  [RANDOM] Loading {needed:,} vectors from cache (mmap)...")
        all_vecs_mmap = np.load(cache_vecs, mmap_mode='r')
        if needed > all_vecs_mmap.shape[0]:
            print(f"  ERROR: cache has {all_vecs_mmap.shape[0]:,} vectors, "
                  f"need {needed:,}")
            sys.exit(1)
        all_vecs = all_vecs_mmap[:needed].copy()
        d = all_vecs.shape[1]
        n_total = needed  # override n_total to match loaded subset
        t_embed = -1
        del texts
        print(f"  Loaded {all_vecs.shape[0]:,} x {d} vectors (mmap subset)")
    else:
        if cache_vecs.exists() and not small:
            print(f"\n  [CHECKPOINT] Loading cached vectors from {cache_vecs}")
            all_vecs = np.load(cache_vecs).astype(np.float32)
            d = all_vecs.shape[1]
            t_embed = -1
            del texts  # free ~1.5 GB on resume
            if all_vecs.shape[0] != n_total:
                print(f"  ERROR: cached vectors ({all_vecs.shape[0]}) != texts "
                      f"({n_total})")
                print(f"  Delete {cache_dir} and re-run.")
                sys.exit(1)
            print(f"  Loaded {all_vecs.shape[0]:,} x {d} vectors")
        else:
            print(f"\n  Generating embeddings for {n_total:,} texts...")
            t0 = time.perf_counter()
            all_vecs = generate_embeddings(
                texts, model_name=embedding_model,
                batch_size=embedding_batch_size, device=device,
            )
            d = all_vecs.shape[1]
            t_embed = time.perf_counter() - t0
            del texts  # free ~1.5 GB
            if not small:
                print(f"  [CHECKPOINT] Saving vectors to {cache_vecs}")
                np.save(cache_vecs, all_vecs)

    # ==================================================================
    # Stage 3: Labels (K-means or Random)
    # ==================================================================
    if label_method == "random":
        print(f"\n  Generating random labels (distribution={distribution}, "
              f"n_labels={n_labels}, avg_labels_per_vec={avg_labels_per_vec})...")
        t0 = time.perf_counter()
        mds = synthesize_labels_random(
            n_total, n_labels=n_labels,
            avg_labels_per_vec=avg_labels_per_vec, seed=seed,
            distribution=distribution,
            n_coarse=n_coarse, n_fine=n_fine,
            coarse_sel_min=coarse_sel_min, coarse_sel_max=coarse_sel_max,
            fine_sel_min=fine_sel_min, fine_sel_max=fine_sel_max,
            zipf_alpha=zipf_alpha,
            coarse_labels_per_vec=coarse_labels_per_vec,
            fine_labels_per_vec=fine_labels_per_vec,
        )
        t_kmeans = time.perf_counter() - t0
        all_labels_set = sorted(set().union(*mds))
        actual_n_labels = len(all_labels_set)
        print(f"  Active labels: {actual_n_labels}/{n_labels}")
        label_remap = {old: new for new, old in enumerate(all_labels_set)}
        mds = [[label_remap[l] for l in md] for md in mds]
    else:
        cache_mds = cache_dir / ".cache_mds.pkl"
        if cache_mds.exists() and not small:
            print(f"\n  [CHECKPOINT] Loading cached labels from {cache_mds}")
            with open(cache_mds, "rb") as f:
                mds = pickle.load(f)
            if len(mds) != n_total:
                print(f"  ERROR: cached labels ({len(mds)}) != vectors "
                      f"({n_total})")
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
            mds = synthesize_labels_kmeans(
                all_vecs, n_labels=n_labels,
                avg_labels_per_vec=avg_labels_per_vec, seed=seed,
            )
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
    # ==================================================================
    cache_split = cache_dir / ".cache_split.npz"

    if label_method == "random" and small and output_name:
        # Random labels + custom size: use exact sizes, simple sequential split
        # (vectors came from mmap in arbitrary but fixed TSV order; labels are
        # random, so sequential split is statistically valid)
        n_train_full = min(n_train_small, n_total)
        n_query_pool_full = min(n_query_pool_small, n_total - n_train_full)
        print(f"\n  Train/pool split (sequential, exact): "
              f"train={n_train_full:,}, pool={n_query_pool_full:,}")
        idx_all = np.arange(n_train_full + n_query_pool_full)
        rng.shuffle(idx_all)
        train_idx = idx_all[:n_train_full]
        pool_idx = idx_all[n_train_full:]
        print(f"  Permuting vectors (mmap subset, single copy)...")
        all_vecs_perm = all_vecs[idx_all]
        del all_vecs
        gc.collect()
    else:
        # Original logic with checkpoint
        if cache_split.exists() and not small:
            print(f"\n  [CHECKPOINT] Loading cached split from {cache_split}")
            split = np.load(cache_split)
            train_idx = split["train_idx"]
            pool_idx = split["pool_idx"]
            n_train_full = len(train_idx)
            n_query_pool_full = len(pool_idx)
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

    def make_dataset(n_train, n_pool, n_q, ds_name):
        print(f"\n{'='*60}")
        print(f"Creating: {ds_name} (n_train={n_train})")
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

        # Complex-predicate GT
        t_cp_wit = 0.0
        cp_gt = {}
        cp_sels = {}
        filters = []
        cp_qvecs = np.zeros((0, d), dtype=np.float32)
        cp_idx = np.zeros(0, dtype=np.int32)
        if not skip_cp and actual_n_labels >= 3:
            print(f"\n  Computing complex-predicate GT: {n_filters} filters x "
                  f"{n_cp_queries} queries (k={k})")
            cp_rng = np.random.RandomState(seed + 1)
            n_cp_avail = min(n_cp_queries, len(pool_vecs))
            cp_idx = cp_rng.choice(len(pool_vecs), n_cp_avail, replace=False)
            cp_qvecs = pool_vecs[cp_idx]
            all_labels_sorted = sorted(set().union(*train_mds_sub))
            _wit_sels = {
                lab: len(label_to_indices[lab]) / n_train
                for lab in all_labels_sorted
            }
            filters = generate_random_filters(
                all_labels_sorted, n_filters=n_filters, seed=seed,
                label_selectivities=_wit_sels,
            )
            t0 = time.perf_counter()
            cp_gt, cp_sels = compute_complex_predicate_gt(
                train_vecs, train_mds_sub, cp_qvecs, filters, k=k,
            )
            t_cp_wit = time.perf_counter() - t0
            print(f"  CP GT done in {t_cp_wit:.1f}s ({t_cp_wit/60:.1f} min)")
            cp_sels_list = list(cp_sels.values())
            if cp_sels_list:
                print(f"  Filter selectivity: min={np.min(cp_sels_list):.4f}, "
                      f"median={np.median(cp_sels_list):.4f}, "
                      f"max={np.max(cp_sels_list):.4f}")

        # Save
        gt_dir = output_dir / ds_name
        gt_dir.mkdir(parents=True, exist_ok=True)
        np.save(gt_dir / "train_vecs.npy", train_vecs.astype(np.float32))
        with open(gt_dir / "train_mds.pkl", "wb") as f:
            pickle.dump(train_mds_sub, f)
        np.save(gt_dir / "query_vecs.npy", query_vecs.astype(np.float32))
        np.save(gt_dir / "query_labels.npy", np.array(query_labels, dtype=np.int32))
        with open(gt_dir / "query_info.json", "w") as f:
            json.dump(query_info, f, indent=2)
        np.save(gt_dir / "ground_truth.npy", gt.astype(np.int32))

        # CP outputs
        if not skip_cp and len(cp_gt) > 0:
            cp_dir = gt_dir / "complex_predicate"
            cp_dir.mkdir(parents=True, exist_ok=True)
            for formula, gt_cp in cp_gt.items():
                safe_name = formula.replace(" ", "_")
                np.save(cp_dir / f"gt_{safe_name}.npy", gt_cp)
            with open(cp_dir / "filters.json", "w") as f:
                json.dump({
                    "n_filters": len(filters),
                    "n_queries": len(cp_qvecs),
                    "filters": sorted(filters),
                    "selectivities": {k: float(v) for k, v in cp_sels.items()},
                }, f, indent=2)
            np.save(cp_dir / "query_vecs.npy", cp_qvecs.astype(np.float32))
            np.save(cp_dir / "query_indices.npy", cp_idx.astype(np.int32))

        # Metadata
        meta = {
            "dim": int(d), "n_labels": actual_n_labels,
            "dataset": ds_name, "n_queries": n_q, "k": k,
            "train_size": n_train, "query_pool_size": pool_size,
            "embedding_model": embedding_model,
            "label_method": label_method,
            "avg_labels_per_vec": avg_labels_per_vec,
            "single_label_gt_time_s": float(t_gt),
            "complex_predicate_gt_time_s": float(t_cp_wit),
            "n_filters": n_filters if not skip_cp else 0,
            "n_cp_queries": n_cp_queries if not skip_cp else 0,
        }
        if label_method == "random":
            meta["distribution"] = distribution
            meta["zipf_alpha"] = zipf_alpha
            if distribution == "hierarchical":
                _avg_cs = (coarse_sel_min + coarse_sel_max) / 2.0
                _actual_nc = n_coarse or max(2, int(2.0 * coarse_labels_per_vec / _avg_cs))
                meta["n_coarse"] = min(_actual_nc, n_labels)
                meta["n_fine"] = n_fine or (n_labels - meta["n_coarse"])
                meta["coarse_sel_min"] = coarse_sel_min
                meta["coarse_sel_max"] = coarse_sel_max
                meta["fine_sel_min"] = fine_sel_min
                meta["fine_sel_max"] = fine_sel_max
        with open(gt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        all_lbls = sorted(set().union(*train_mds_sub))
        with open(gt_dir / "all_labels.json", "w") as f:
            json.dump(all_lbls, f)
        print(f"\n  Saved: {gt_dir.resolve()}")
        print(f"    train_vecs: {train_vecs.shape}, {train_vecs.nbytes/1024**2:.1f} MB")
        print(f"    query_vecs: {query_vecs.shape}")
        print(f"    ground_truth: {gt.shape}")
        if not skip_cp and len(cp_gt) > 0:
            print(f"    complex_predicate: {len(filters)} filters × {len(cp_qvecs)} queries")

        del train_vecs, pool_vecs, gt
        gc.collect()

    # Always create the requested dataset
    if output_name:
        # Explicit output name → generate exactly one dataset
        n_ts = min(n_train_small, n_train_full) if small else n_train_full
        n_ps = min(n_query_pool_small, n_query_pool_full) if small else n_query_pool_full
        n_q = n_queries_small if small else n_queries_full
        make_dataset(n_ts, n_ps, n_q, output_name)
    else:
        # Legacy behaviour: always create small, optionally create full
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
    p.add_argument("--label_method", type=str, default="kmeans",
                   choices=["kmeans", "random"])
    p.add_argument("--output_name", type=str, default=None,
                   help="Override output directory name")
    p.add_argument("--skip_cp", action="store_true",
                   help="Skip complex-predicate GT")
    p.add_argument("--n_cp_queries", type=int, default=100)
    p.add_argument("--n_filters", type=int, default=50)
    p.add_argument("--distribution", type=str, default="hierarchical",
                   choices=["uniform", "skewed", "hierarchical"])
    p.add_argument("--n_coarse", type=int, default=None)
    p.add_argument("--n_fine", type=int, default=None)
    p.add_argument("--coarse_sel_min", type=float, default=0.10)
    p.add_argument("--coarse_sel_max", type=float, default=0.30)
    p.add_argument("--fine_sel_min", type=float, default=0.001)
    p.add_argument("--fine_sel_max", type=float, default=0.03)
    p.add_argument("--zipf_alpha", type=float, default=2.0)
    p.add_argument("--coarse_labels_per_vec", type=int, default=1)
    p.add_argument("--fine_labels_per_vec", type=int, default=2)
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
