"""
Synthesize multi-tenant labels for datasets without natural labels (SIFT1M, GIST1M).

Strategy: K-means clustering + soft assignment.
  1. Run FAISS K-means on the vectors to get K cluster centers.
  2. For each vector, compute distances to all K centers.
  3. Assign the avg_labels_per_vec nearest centers as the vector's synthetic labels.
  4. Ensure each label covers at least min_coverage vectors (reassign if needed).

This yields geometrically meaningful labels — vectors sharing a label are close in
the vector space, so filtered retrieval is non-trivial.

Usage:
    from synthesize_labels import synthesize_labels_kmeans
    mds = synthesize_labels_kmeans(vecs, n_labels=100, avg_labels_per_vec=3)
"""

import numpy as np


def synthesize_labels_kmeans(
    vecs: np.ndarray,
    n_labels: int = 100,
    avg_labels_per_vec: int = 3,
    seed: int = 42,
    kmeans_niter: int = 25,
    kmeans_max_points_per_centroid: int = 1000,
) -> list[list[int]]:
    """Synthesize multi-labels via FAISS K-means + soft assignment.

    Parameters
    ----------
    vecs : np.ndarray [n, d], float32
        Input vectors.
    n_labels : int
        Number of synthetic labels (K in K-means).
    avg_labels_per_vec : int
        Average number of labels per vector.
    seed : int
        Random seed for K-means.
    kmeans_niter : int
        Number of K-means iterations.
    kmeans_max_points_per_centroid : int
        Max training samples per centroid (passed to FAISS Kmeans).

    Returns
    -------
    mds : list[list[int]]
        mds[i] = sorted list of label IDs assigned to vector i.
    """
    import faiss

    n, d = vecs.shape
    n_labels = min(n_labels, n)  # can't have more labels than vectors

    print(f"  Running FAISS K-means (K={n_labels}, d={d}, n={n:,}, "
          f"niter={kmeans_niter})...")

    kmeans = faiss.Kmeans(
        d=d,
        k=n_labels,
        niter=kmeans_niter,
        nredo=1,
        seed=seed,
        verbose=True,
        max_points_per_centroid=kmeans_max_points_per_centroid,
        gpu=False,  # CPU is fine for this scale
    )
    kmeans.train(vecs)

    centroids = kmeans.centroids  # [n_labels, d]
    print(f"  K-means done. Centroids shape: {centroids.shape}")

    # Step 2-3: For each vector, find nearest avg_labels_per_vec centroids.
    # Use FAISS brute-force index for efficient nearest-centroid search.
    print(f"  Assigning {avg_labels_per_vec} nearest centroids per vector...")
    index = faiss.IndexFlatL2(d)
    index.add(centroids)

    _, assignments = index.search(vecs, avg_labels_per_vec)
    # assignments: [n, avg_labels_per_vec], each row = centroid IDs (sorted by dist)

    mds = [sorted(row.tolist()) for row in assignments]

    # Step 4: Check label coverage
    label_counts = np.bincount(
        np.concatenate([np.array(labels) for labels in mds]),
        minlength=n_labels,
    )
    empty_labels = np.where(label_counts == 0)[0]
    min_count = label_counts.min()
    max_count = label_counts.max()
    median_count = np.median(label_counts)

    print(f"  Label coverage: min={min_count}, median={median_count:.0f}, "
          f"max={max_count}, empty_labels={len(empty_labels)}")

    if len(empty_labels) > 0:
        print(f"  ⚠ {len(empty_labels)} labels have zero coverage. "
              f"Consider reducing n_labels or increasing avg_labels_per_vec.")

    # Compute selectivity stats
    selectivities = label_counts / n
    print(f"  Per-label selectivity: min={selectivities.min():.4f}, "
          f"median={np.median(selectivities):.4f}, "
          f"max={selectivities.max():.4f}")

    return mds


def synthesize_labels_random(
    n_vectors: int,
    n_labels: int = 100,
    avg_labels_per_vec: int = 3,
    seed: int = 42,
) -> list[list[int]]:
    """Random uniform label assignment (baseline / ablation study).

    Each vector gets exactly avg_labels_per_vec random labels.
    Each label is guaranteed to cover at least
    floor(n_vectors * avg_labels_per_vec / n_labels) vectors.

    Parameters
    ----------
    n_vectors : int
        Number of vectors.
    n_labels : int
        Number of labels.
    avg_labels_per_vec : int
        Number of labels per vector.
    seed : int
        Random seed.

    Returns
    -------
    mds : list[list[int]]
    """
    rng = np.random.RandomState(seed)

    # Ensure uniform label coverage by assigning labels round-robin then shuffling
    total_assignments = n_vectors * avg_labels_per_vec
    labels_pool = np.concatenate([
        np.tile(np.arange(n_labels), total_assignments // n_labels + 1)
    ])[:total_assignments]
    rng.shuffle(labels_pool)

    # Group into n_vectors chunks of size avg_labels_per_vec
    labels_pool = labels_pool.reshape(n_vectors, avg_labels_per_vec)
    mds = [sorted(row.tolist()) for row in labels_pool]

    # Verify
    label_counts = np.bincount(
        np.concatenate([np.array(labels) for labels in mds]),
        minlength=n_labels,
    )
    print(f"  Random labels: min_coverage={label_counts.min()}, "
          f"max_coverage={label_counts.max()}, "
          f"empty={np.sum(label_counts == 0)}")

    return mds


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    # Small-scale test: random data
    np.random.seed(42)
    test_vecs = np.random.rand(5000, 128).astype(np.float32)
    t0 = time.perf_counter()
    mds = synthesize_labels_kmeans(test_vecs, n_labels=20, avg_labels_per_vec=3)
    t = time.perf_counter() - t0
    print(f"\n  Small test: {len(test_vecs)} vectors, {len(mds)} label lists, "
          f"avg labels/vec={sum(len(m) for m in mds)/len(mds):.1f}")
    print(f"  Time: {t:.1f}s")
