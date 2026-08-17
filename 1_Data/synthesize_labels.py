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


# ---------------------------------------------------------------------------
# Random label synthesis with controllable selectivity distributions
# ---------------------------------------------------------------------------

def _uniform_random_labels(
    n_vectors: int, n_labels: int, avg_labels_per_vec: int, rng: np.random.RandomState,
) -> list[list[int]]:
    """Round-robin shuffle assignment — backward-compatible uniform mode."""
    total_assignments = n_vectors * avg_labels_per_vec
    labels_pool = np.concatenate([
        np.tile(np.arange(n_labels), total_assignments // n_labels + 1)
    ])[:total_assignments]
    rng.shuffle(labels_pool)
    labels_pool = labels_pool.reshape(n_vectors, avg_labels_per_vec)
    mds = [sorted(row.tolist()) for row in labels_pool]

    label_counts = np.bincount(
        np.concatenate([np.array(l) for l in mds]), minlength=n_labels,
    )
    print(f"  Random labels (uniform): min={label_counts.min()}, "
          f"max={label_counts.max()}, empty={np.sum(label_counts == 0)}")
    return mds


def _generate_zipf_selectivities(
    n_labels: int,
    total_expected: float,
    sel_min: float,
    sel_max: float,
    alpha: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate per-label selectivity values following a Zipf-like distribution.

    The raw Zipf counts are normalised to [sel_min, sel_max] and scaled so that
    sum(sels) ≈ total_expected (the expected number of labels per vector from
    this group).

    Returns float64 array of length n_labels.
    """
    raw = rng.zipf(alpha, size=n_labels).astype(np.float64)
    rmin, rmax = raw.min(), raw.max()
    if rmax - rmin < 1e-12:
        normed = np.full(n_labels, 0.5, dtype=np.float64)
    else:
        normed = (raw - rmin) / (rmax - rmin)
    sels = sel_min + normed * (sel_max - sel_min)
    # Scale so sum matches the desired total expected per-vector assignments
    current_sum = sels.sum()
    if current_sum > 1e-12:
        sels = sels * (total_expected / current_sum)
    sels = np.clip(sels, max(sel_min, 1e-4), min(sel_max, 1.0))
    # Re-normalise after clipping
    post_sum = sels.sum()
    if post_sum > 1e-12:
        sels = sels * (total_expected / post_sum)
    return sels


def _assign_bernoulli(
    n_vectors: int,
    label_ids: np.ndarray,
    selectivities: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Independent Bernoulli assignment: each label i is assigned to each vector
    with probability selectivities[i].

    Returns a boolean matrix of shape [n_vectors, len(label_ids)].
    """
    n_labels = len(label_ids)
    # Generate uniform random matrix and threshold
    rand = rng.random_sample((n_vectors, n_labels))
    return rand < selectivities[np.newaxis, :]


def _fill_zero_label_vectors(
    assignment: np.ndarray,
    label_ids: np.ndarray,
    selectivities: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Post-process: for vectors that got 0 labels, assign labels with the
    largest deficit (target count minus actual count)."""
    n_vectors, n_labels = assignment.shape
    zero_mask = ~assignment.any(axis=1)
    zero_count = zero_mask.sum()
    if zero_count == 0:
        return assignment

    target_counts = selectivities * n_vectors
    actual_counts = assignment.sum(axis=0)
    deficits = np.maximum(0, target_counts - actual_counts)
    if deficits.sum() < 1:
        # No meaningful deficit — just give each zero-vector a random label
        for idx in np.where(zero_mask)[0]:
            lab = rng.choice(label_ids)
            assignment[idx, lab] = True
        return assignment

    deficit_probs = deficits / deficits.sum()
    for idx in np.where(zero_mask)[0]:
        lab = rng.choice(label_ids, p=deficit_probs)
        assignment[idx, lab] = True
        deficits[lab] -= 1
        if deficits[lab] <= 0:
            deficits[lab] = 0
            if deficits.sum() < 1:
                break
    # Any remaining zero-vectors get a random label
    still_zero = ~assignment.any(axis=1)
    for idx in np.where(still_zero)[0]:
        assignment[idx, rng.choice(label_ids)] = True
    return assignment


def _bernoulli_to_mds(
    assignment: np.ndarray, label_ids: np.ndarray,
) -> list[list[int]]:
    """Convert a boolean assignment matrix to sorted list-of-lists format."""
    mds = []
    for row in assignment:
        labs = label_ids[row].tolist()
        mds.append(sorted(labs))
    return mds


def _hierarchical_random_labels(
    n_vectors: int,
    n_labels: int,
    avg_labels_per_vec: int,
    n_coarse: int | None,
    n_fine: int | None,
    coarse_sel_min: float,
    coarse_sel_max: float,
    fine_sel_min: float,
    fine_sel_max: float,
    zipf_alpha: float,
    # coarse_labels_per_vec / fine_labels_per_vec are NOT used as hard
    # constraints — coarse sels are applied raw (without scaling), and
    # fine labels fill the remaining "budget".
    coarse_labels_per_vec: int,
    fine_labels_per_vec: int,
    rng: np.random.RandomState,
) -> list[list[int]]:
    """Hierarchical random label assignment.

    Coarse (high-selectivity) labels are assigned via independent Bernoulli
    trials with **raw** probabilities in [coarse_sel_min, coarse_sel_max],
    without scaling.  This guarantees a wide selectivity spread (e.g. 15–50 %
    for 5 coarse labels with 100-label config).

    Fine (low-selectivity) labels use Zipf-distributed probabilities and fill
    the remaining per-vector budget to reach approximately
    ``avg_labels_per_vec`` total labels per vector.
    """
    # Auto-compute label counts so that expected coarse labels per vector
    # ≈ coarse_labels_per_vec (default 1).  Formula:
    #   n_coarse ≈ coarse_labels_per_vec / avg(coarse_sel)
    # Multiplied by 2 to ensure enough AND coarse-coarse pairs.
    if n_coarse is None:
        avg_cs = (coarse_sel_min + coarse_sel_max) / 2.0
        n_coarse = max(2, int(2.0 * coarse_labels_per_vec / avg_cs))
    n_coarse = min(n_coarse, n_labels)
    if n_fine is None:
        n_fine = n_labels - n_coarse

    print(f"  Hierarchical labels: n_coarse={n_coarse}, n_fine={n_fine}, "
          f"coarse_sel=[{coarse_sel_min}, {coarse_sel_max}], "
          f"fine_sel=[{fine_sel_min}, {fine_sel_max}], zipf_alpha={zipf_alpha}")

    coarse_ids = np.arange(n_coarse, dtype=np.int32)
    fine_ids = np.arange(n_coarse, n_labels, dtype=np.int32)

    # ------------------------------------------------------------------
    # 1. Coarse labels — raw Bernoulli, NO scaling
    # ------------------------------------------------------------------
    coarse_sels = np.linspace(coarse_sel_min, coarse_sel_max, n_coarse,
                              dtype=np.float64)
    print(f"  Coarse raw sels: [{coarse_sels.min():.4f}, {coarse_sels.max():.4f}], "
          f"expected per-vec={coarse_sels.sum():.2f}")
    print(f"  Assigning coarse labels (Bernoulli, n_coarse={n_coarse})...")
    coarse_assign = _assign_bernoulli(n_vectors, coarse_ids, coarse_sels, rng)
    coarse_counts = coarse_assign.sum(axis=1).astype(np.float64)
    avg_coarse = coarse_counts.mean()
    print(f"  Coarse: actual avg/vec={avg_coarse:.2f}")

    # ------------------------------------------------------------------
    # 2. Fine labels — fill remaining budget
    # ------------------------------------------------------------------
    remaining_budget = max(0.0, float(avg_labels_per_vec) - avg_coarse)
    print(f"  Fine budget: {remaining_budget:.2f} labels/vec (target={avg_labels_per_vec})")

    if remaining_budget > 0.01 and n_fine > 0:
        fine_sels = _generate_zipf_selectivities(
            n_fine, remaining_budget, fine_sel_min, fine_sel_max, zipf_alpha, rng,
        )
        print(f"  Fine sels: [{fine_sels.min():.4f}, {fine_sels.max():.4f}], "
              f"sum={fine_sels.sum():.2f}")
        print(f"  Assigning fine labels (Bernoulli, n_fine={n_fine})...")
        fine_assign = _assign_bernoulli(n_vectors, fine_ids, fine_sels, rng)
    else:
        fine_assign = np.zeros((n_vectors, n_fine), dtype=bool)
        print(f"  Fine labels skipped (budget ≤ 0 or n_fine=0)")

    # ------------------------------------------------------------------
    # 3. Combine and post-process
    # ------------------------------------------------------------------
    all_ids = np.arange(n_labels, dtype=np.int32)
    if remaining_budget > 0.01 and n_fine > 0:
        full_assign = np.concatenate([coarse_assign, fine_assign], axis=1)
        all_sels = np.concatenate([coarse_sels, fine_sels])
    else:
        # Fine labels skipped — pad with zeros so shapes match
        fine_assign_zero = np.zeros((n_vectors, n_fine), dtype=bool)
        full_assign = np.concatenate([coarse_assign, fine_assign_zero], axis=1)
        all_sels = np.concatenate([coarse_sels, np.zeros(n_fine, dtype=np.float64)])

    full_assign = _fill_zero_label_vectors(full_assign, all_ids, all_sels, rng)
    mds = _bernoulli_to_mds(full_assign, all_ids)

    # ------------------------------------------------------------------
    # 4. Statistics
    # ------------------------------------------------------------------
    label_counts = np.bincount(
        np.concatenate([np.array(l) for l in mds]), minlength=n_labels,
    )
    sels = label_counts / n_vectors
    lpv = [len(m) for m in mds]
    print(f"  Hierarchical done: "
          f"min_sel={sels[sels>0].min():.4f}, "
          f"max_sel={sels.max():.4f}, "
          f"ratio={sels.max()/max(sels[sels>0].min(),1e-6):.0f}:1, "
          f"avg_labels/vec={np.mean(lpv):.1f}, "
          f"zero={sum(1 for m in mds if len(m)==0)}")
    if n_coarse > 0:
        c_sels = sels[:n_coarse]
        if len(c_sels) > 0:
            print(f"  Coarse actual: min={c_sels.min():.4f}, "
                  f"max={c_sels.max():.4f}")
    if n_fine > 0:
        f_sels = sels[n_coarse:]
        f_nonzero = f_sels[f_sels > 0]
        if len(f_nonzero) > 0:
            print(f"  Fine actual:   min={f_nonzero.min():.4f}, "
                  f"max={f_nonzero.max():.4f}")
        else:
            print(f"  Fine actual:   all zero (budget consumed by coarse)")
    return mds


def _skewed_random_labels(
    n_vectors: int,
    n_labels: int,
    avg_labels_per_vec: int,
    sel_min: float,
    sel_max: float,
    zipf_alpha: float,
    rng: np.random.RandomState,
) -> list[list[int]]:
    """Single-tier Zipf-skewed random labels (all labels in one pool)."""
    sels = _generate_zipf_selectivities(
        n_labels, avg_labels_per_vec, sel_min, sel_max, zipf_alpha, rng,
    )
    all_ids = np.arange(n_labels, dtype=np.int32)
    print(f"  Assigning skewed labels (Bernoulli, n_labels={n_labels})...")
    assign = _assign_bernoulli(n_vectors, all_ids, sels, rng)
    assign = _fill_zero_label_vectors(assign, all_ids, sels, rng)
    mds = _bernoulli_to_mds(assign, all_ids)

    label_counts = np.bincount(
        np.concatenate([np.array(l) for l in mds]), minlength=n_labels,
    )
    final_sels = label_counts / n_vectors
    label_counts_per_vec = [len(m) for m in mds]
    print(f"  Skewed labels: min_sel={final_sels[final_sels > 0].min():.4f}, "
          f"max_sel={final_sels.max():.4f}, "
          f"ratio={final_sels.max() / max(final_sels[final_sels > 0].min(), 1e-6):.0f}:1, "
          f"avg_labels/vec={np.mean(label_counts_per_vec):.1f}, "
          f"zero_label_vecs={sum(1 for m in mds if len(m) == 0)}")
    return mds


def synthesize_labels_random(
    n_vectors: int,
    n_labels: int = 100,
    avg_labels_per_vec: int = 3,
    seed: int = 42,
    distribution: str = "hierarchical",
    n_coarse: int | None = None,
    n_fine: int | None = None,
    coarse_sel_min: float = 0.15,
    coarse_sel_max: float = 0.40,
    fine_sel_min: float = 0.001,
    fine_sel_max: float = 0.05,
    zipf_alpha: float = 1.5,
    coarse_labels_per_vec: int = 1,
    fine_labels_per_vec: int = 2,
) -> list[list[int]]:
    """Random label assignment with controllable selectivity distribution.

    Supports three modes:
      - ``"uniform"``: round-robin shuffle (backward-compatible, all labels
        have roughly equal coverage).
      - ``"skewed"``: single-tier Zipf distribution giving a skewed per-label
        selectivity profile in [sel_min, sel_max].
      - ``"hierarchical"`` (default): two-tier structure with a small set of
        coarse (high-selectivity) labels and a larger set of fine
        (low-selectivity) labels.  Coarse labels are spread uniformly;
        fine labels follow a Zipf distribution.  Assignments are independent
        Bernoulli trials, producing natural inter-label correlation.

    Parameters
    ----------
    n_vectors : int
        Number of vectors to label.
    n_labels : int
        Total number of labels.
    avg_labels_per_vec : int
        Target average number of labels per vector.
    seed : int
        Random seed for reproducibility.
    distribution : str
        ``"uniform"`` | ``"skewed"`` | ``"hierarchical"``.
    n_coarse : int | None
        Number of coarse labels (hierarchical mode).  Defaults to
        ``int(n_labels * 0.2)``.
    n_fine : int | None
        Number of fine labels (hierarchical mode).  Defaults to
        ``n_labels - n_coarse``.
    coarse_sel_min, coarse_sel_max : float
        Selectivity range for coarse labels (e.g. 0.15–0.40).
    fine_sel_min, fine_sel_max : float
        Selectivity range for fine labels (e.g. 0.001–0.05).
    zipf_alpha : float
        Zipf skew parameter (higher = more skewed).  Used by ``"skewed"``
        and ``"hierarchical"`` (fine labels).
    coarse_labels_per_vec : int
        Expected number of coarse labels per vector (hierarchical mode).
    fine_labels_per_vec : int
        Expected number of fine labels per vector (hierarchical mode).

    Returns
    -------
    mds : list[list[int]]
        mds[i] = sorted list of label IDs assigned to vector i.
    """
    rng = np.random.RandomState(seed)

    if distribution == "uniform":
        return _uniform_random_labels(n_vectors, n_labels, avg_labels_per_vec, rng)

    # Validate that expected counts sum to target
    actual_avg = coarse_labels_per_vec + fine_labels_per_vec if distribution == "hierarchical" else avg_labels_per_vec
    if actual_avg != avg_labels_per_vec:
        print(f"  ⚠ coarse_lpv ({coarse_labels_per_vec}) + fine_lpv "
              f"({fine_labels_per_vec}) = {actual_avg} ≠ "
              f"avg_labels_per_vec ({avg_labels_per_vec}); using actual sum")

    if distribution == "skewed":
        return _skewed_random_labels(
            n_vectors, n_labels, avg_labels_per_vec,
            fine_sel_min, fine_sel_max, zipf_alpha, rng,
        )
    elif distribution == "hierarchical":
        return _hierarchical_random_labels(
            n_vectors, n_labels, avg_labels_per_vec,
            n_coarse, n_fine,
            coarse_sel_min, coarse_sel_max,
            fine_sel_min, fine_sel_max,
            zipf_alpha,
            coarse_labels_per_vec, fine_labels_per_vec,
            rng,
        )
    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            f"Choose 'uniform', 'skewed', or 'hierarchical'."
        )


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
