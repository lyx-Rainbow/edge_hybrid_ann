"""
Basic usage example for Curator index.

This example demonstrates:
1. Creating a Curator index
2. Training the index
3. Adding vectors with tenant access control
4. Querying with tenant filtering
5. Complex predicate support
"""

import faiss
import numpy as np

# Import the Python wrapper (adjust path as needed)
# from curator_solo.python.curator import Curator
# Or if you've installed the wrapper:
from curator import Curator


def main():
    # Parameters
    d = 128           # vector dimension
    nlist = 16        # number of clusters (branch factor)
    n_vectors = 10000  # number of vectors
    n_tenants = 10    # number of tenants
    k = 10            # top-k results

    print("=" * 60)
    print("Curator Index - Basic Usage Example")
    print("=" * 60)

    # 1. Create index
    print("\n[1] Creating Curator index...")
    index = Curator(
        d=d,
        nlist=nlist,
        bf_capacity=1000,
        bf_error_rate=0.001,
        max_sl_size=128,
        clus_niter=20,
        max_leaf_size=128,
        nprobe=4,
        prune_thres=1.6,
        variance_boost=0.2,
        search_ef=32,
        beam_size=2,
        use_temp_index_caching=True,
        pq_M=16,
        pq_nbits=8,
        pq_enabled=True,
        pq_use_adc_rerank=False,
    )
    print(f"  Index created with d={d}, nlist={nlist}")

    # 2. Generate random data
    print("\n[2] Generating random data...")
    np.random.seed(42)
    X = np.random.randn(n_vectors, d).astype(np.float32)
    print(f"  Generated {n_vectors} vectors of dimension {d}")

    # 3. Train the index (k-means clustering for the tree)
    print("\n[3] Training the index...")
    index.train(X.flatten())  # faiss expects flat array
    print("  Training complete")

    # 4. Add vectors to the index
    print("\n[4] Adding vectors...")
    for i in range(n_vectors):
        index.create(X[i], label=i)
    print(f"  Added {n_vectors} vectors")

    # 5. Grant access: each tenant gets access to a random subset
    print("\n[5] Granting tenant access...")
    tenant_vectors = {t: set() for t in range(n_tenants)}
    for i in range(n_vectors):
        # Each vector is accessible by 1-3 random tenants
        n_access = np.random.randint(1, 4)
        tenant_ids = np.random.choice(n_tenants, size=n_access, replace=False)
        for tid in tenant_ids:
            index.grant_access(i, tid)
            tenant_vectors[tid].add(i)
    print(f"  Access granted for {n_tenants} tenants")

    # 6. Finalize (flush to flash storage + PQ training)
    print("\n[6] Finalizing index (PQ training + flash storage)...")
    index.flush()
    print("  Index finalized")

    # 7. Query with tenant filtering
    print("\n[7] Querying with tenant filtering...")
    query = np.random.randn(d).astype(np.float32)
    tenant_id = 0
    results = index.query(query, k=k, tenant_id=tenant_id)
    print(f"  Tenant {tenant_id} query top-{k} results: {results}")
    print(f"  (Tenant {tenant_id} has access to {len(tenant_vectors[tenant_id])} vectors)")

    # 8. Query with complex predicate
    print("\n[8] Querying with complex predicate...")
    mask_0 = np.zeros(n_vectors, dtype=bool)
    mask_0[list(tenant_vectors[0])] = True
    mask_1 = np.zeros(n_vectors, dtype=bool)
    mask_1[list(tenant_vectors[1])] = True

    # Find vectors accessible by both tenant 0 AND tenant 1
    mask_and = mask_0 & mask_1
    qualified_labels = np.where(mask_and)[0].astype(np.uint32)

    if len(qualified_labels) > 0:
        # Option A: Build a tenant for the predicate, then query
        label = index.index_filter(predicate="AND_0_1", qualified_labels=qualified_labels)
        if label >= 0:
            results = index.query_with_complex_predicate(query, k=k, predicate="AND_0_1")
            print(f"  AND(0,1) results: {results}")
    else:
        print("  No vectors qualify for AND(0,1)")

    # 9. Memory usage
    print("\n[9] Memory usage...")
    mem_bytes = index.get_index_memory_bytes()
    print(f"  Total index memory: {mem_bytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
