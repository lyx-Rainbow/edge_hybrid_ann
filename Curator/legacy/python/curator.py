from typing import Any

import faiss
import numpy as np

try:
    from .base import Index
except ImportError:
    from base import Index

Metadata = list[list[int]]


def _evaluate_predicate(tokens: list[str], mds: list[int]) -> bool:
    """Evaluate a boolean formula in Polish notation.

    Example: tokens for 'AND 1 OR NOT 0 2' -> ['AND', '1', 'OR', 'NOT', '0', '2']
    A variable (like '1') is considered True if it is in mds.
    """
    stack = []

    def eval_token(token):
        try:
            return int(token) in mds
        except ValueError:
            raise ValueError(f"Malformed token: {token}")

    for token in reversed(tokens):
        if token in ("AND", "OR", "NOT"):
            if token == "AND":
                a = stack.pop()
                b = stack.pop()
                stack.append(a and b)
            elif token == "OR":
                a = stack.pop()
                b = stack.pop()
                stack.append(a or b)
            else:  # NOT
                a = stack.pop()
                stack.append(not a)
        else:
            stack.append(eval_token(token))

    if len(stack) != 1:
        raise ValueError("Malformed expression")
    return stack[0]


def compute_qualified_labels(filter_str: str, train_mds: list[list[int]]) -> np.ndarray:
    """Compute the list of vector IDs that satisfy the given filter.

    Parameters
    ----------
    filter_str : str
        Filter string in Polish notation (e.g., "OR 1 2")
    train_mds : list[list[int]]
        List of access lists for each training vector

    Returns
    -------
    np.ndarray
        Array of vector IDs that satisfy the filter
    """
    tokens = filter_str.split()
    qualified_ids = []
    for i, md in enumerate(train_mds):
        if md and _evaluate_predicate(tokens, md):
            qualified_ids.append(i)
    return np.array(qualified_ids, dtype=np.uint32)


class Curator(Index):
    """Curator index"""

    def __init__(
        self,
        d: int,
        nlist: int,
        bf_capacity: int = 1000,
        bf_error_rate: float = 0.001,
        max_sl_size: int = 128,
        clus_niter: int = 20,
        max_leaf_size: int = 128,
        nprobe: int = 1200,
        prune_thres: float = 1.6,
        variance_boost: float = 0.2,
        search_ef: int = 128,
        beam_size: int = 1,
        use_temp_index_caching: bool = True,
        pq_M: int = 16,
        pq_nbits: int = 8,
        pq_enabled: bool = True,
        pq_use_adc_rerank: bool = False,
        pq_rerank_topk_factor: int = 4,
        use_flash_storage: bool = True,
        disk_cache_prefix: str | None = None,
        persist_pq_codes: bool = True,
        persist_temp_indexes: bool = True,
    ):
        """Initialize Curator index.

        Parameters
        ----------
        d : int
            Dimensionality of the vectors.
        nlist : int
            Number of cells/buckets in the inverted file.
        bf_capacity : int, optional
            The capacity of the Bloom filter.
        bf_error_rate : float, optional
            The error rate of the Bloom filter.
        use_temp_index_caching : bool, optional
            If True, build_index_for_filter will cache temporary index structures
            in memory (instead of using direct tenant-based indexing), providing
            an efficient middle-ground between memory and functionality. This is
            enabled by default. For disk-backed persistence, also provide
            disk_cache_prefix. Defaults to True.
        disk_cache_prefix : str | None, optional
            Prefix for disk-backed cache files (PQ codes, temp indexes). If
            provided, enables optional disk caching on top of the in-memory
            temp index structures. Defaults to None (disk caching disabled).
        """
        super().__init__()

        self.d = d
        self.nlist = nlist
        self.bf_capacity = bf_capacity
        self.bf_error_rate = bf_error_rate
        self.max_sl_size = max_sl_size
        self.clus_niter = clus_niter
        self.max_leaf_size = max_leaf_size
        self.nprobe = nprobe
        self.prune_thres = prune_thres
        self.variance_boost = variance_boost
        self.search_ef = search_ef
        self.beam_size = beam_size
        self.use_temp_index_caching = use_temp_index_caching
        self.pq_M = pq_M
        self.pq_nbits = pq_nbits
        self.pq_enabled = pq_enabled
        self.pq_use_adc_rerank = pq_use_adc_rerank
        self.pq_rerank_topk_factor = pq_rerank_topk_factor
        self.use_flash_storage = use_flash_storage
        self.persist_pq_codes = persist_pq_codes
        self.persist_temp_indexes = persist_temp_indexes

        self.index = faiss.MultiTenantIndexIVFHierarchical(
            self.d,
            self.nlist,
            faiss.METRIC_L2,
            self.bf_capacity,
            self.bf_error_rate,
            self.max_sl_size,
            self.clus_niter,
            self.max_leaf_size,
            self.nprobe,
            self.prune_thres,
            self.variance_boost,
            self.search_ef,
            self.beam_size,
            self.use_temp_index_caching,
        )
        self.index.set_pq_config(
            self.pq_M,
            self.pq_nbits,
            self.pq_enabled,
            self.pq_use_adc_rerank,
            self.pq_rerank_topk_factor,
        )
        self.index.set_use_flash_storage(self.use_flash_storage)

        if disk_cache_prefix is not None:
            flash_path = f"{disk_cache_prefix}/vectors.bin"
            self.index.set_flash_storage_path(flash_path)
            if self.persist_pq_codes:
                pq_path = f"{disk_cache_prefix}/pq_codes.bin"
                self.index.set_pq_codes_path(pq_path)
                self.index.set_persist_pq_codes(True)

        # Map indexed filters to their assigned labels
        self.filter_to_label = dict()

        # Map indexed filters to their precomputed bitmaps
        self.filter_to_bitmap = dict()

        # Map predicate filters to their sorted internal vector IDs for optimized search
        self.filter_to_sorted_vids = dict()

    @property
    def params(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "nlist": self.nlist,
            "bf_capacity": self.bf_capacity,
            "bf_error_rate": self.bf_error_rate,
            "max_sl_size": self.max_sl_size,
            "clus_niter": self.clus_niter,
            "max_leaf_size": self.max_leaf_size,
            "pq_M": self.pq_M,
            "pq_nbits": self.pq_nbits,
            "pq_enabled": self.pq_enabled,
            "pq_use_adc_rerank": self.pq_use_adc_rerank,
            "pq_rerank_topk_factor": self.pq_rerank_topk_factor,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "nprobe": self.nprobe,
            "prune_thres": self.prune_thres,
            "variance_boost": self.variance_boost,
            "search_ef": self.search_ef,
            "beam_size": self.beam_size,
        }

    @search_params.setter
    def search_params(self, params: dict[str, Any]) -> None:
        if "nprobe" in params:
            self.nprobe = params["nprobe"]
            self.index.nprobe = self.nprobe
        if "prune_thres" in params:
            self.prune_thres = params["prune_thres"]
            self.index.prune_thres = self.prune_thres
        if "variance_boost" in params:
            self.variance_boost = params["variance_boost"]
            self.index.variance_boost = self.variance_boost
        if "search_ef" in params:
            self.search_ef = params["search_ef"]
            self.index.search_ef = self.search_ef
        if "beam_size" in params:
            self.beam_size = params["beam_size"]
            self.index.beam_size = self.beam_size

    def train(
        self, X: np.ndarray, tenant_ids: Metadata | None = None, **train_params
    ) -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.index.train(len(X), faiss.swig_ptr(X), 0)  # type: ignore

    def create(self, x: np.ndarray, label: int) -> None:
        vector = np.ascontiguousarray(x[None], dtype=np.float32)
        labels = np.array([label], dtype=np.int64)
        self.index.add_vector_with_ids(  # type: ignore
            1,
            faiss.swig_ptr(vector),
            faiss.cast_integer_to_idx_t_ptr(labels.ctypes.data),
        )

    def grant_access(self, label: int, tenant_id: int) -> None:
        self.index.grant_access(label, tenant_id)  # type: ignore

    def flush(self) -> None:
        """Persist raw vectors to flash and finalize the index.

        After bulk ``create`` / ``grant_access``, call this to:
        1. Train PQ codebook (if enabled) and encode residuals
        2. Write full-precision vectors to flash storage in leaf order
        3. Free the in-memory raw vector buffer

        Must be called once after the build phase is complete.
        """
        if not self.index.flash_finalized:
            self.index.finalize_flash_storage()

    def compact(self) -> None:
        """No-op: flash storage uses fixed-size regions per leaf and does
        not need compaction. Kept for API compatibility.
        """
        pass

    def get_page_file_size_bytes(self) -> int:
        """Return the size of the flash storage file in bytes, or 0."""
        if self.index.flash_fp is not None:
            import os
            if os.path.exists(self.index.flash_storage_path):
                return os.path.getsize(self.index.flash_storage_path)
        return 0

    def clear_temp_index_cache(self) -> None:
        """Drop all entries from the in-memory temp-index caches."""
        self.index.cached_temp_indexes.clear()
        self.index.cached_qualified_vecs.clear()

    def set_rerank_params(self, enabled: bool, topk_factor: int) -> None:
        """Toggle ADC rerank on/off and adjust the candidate multiplier at
        query time.  Does not invalidate PQ codes — safe to call between
        queries on the same index.
        """
        self.index.set_rerank_params(enabled, topk_factor)  # type: ignore

    def get_rerank_enabled(self) -> bool:
        return self.index.get_rerank_enabled()  # type: ignore

    def get_rerank_topk_factor(self) -> int:
        return self.index.get_rerank_topk_factor()  # type: ignore

    def save_to_file(self, path: str) -> None:
        """Binary-serialize the full index state to a file."""
        raise NotImplementedError(
            "save_to_file is not yet implemented for Curator"
        )

    @classmethod
    def load_from_file(cls, path: str, **kwargs) -> "Curator":
        """Instantiate a Curator from a binary file."""
        raise NotImplementedError(
            "load_from_file is not yet implemented for Curator"
        )

    def delete_vector(self, label: int) -> None:
        try:
            self.index.remove_vector(label)  # type: ignore
        except RuntimeError as e:
            msg = str(e)
            if "not supported" in msg:
                print(
                    f"Warning: remove_vector not supported for label {label}; skipping physical removal."
                )
                return
            raise

    def revoke_access(self, label: int, tenant_id: int) -> None:
        self.index.revoke_access(label, tenant_id)  # type: ignore

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        vector = np.ascontiguousarray(x[None], dtype=np.float32)
        distances = np.empty((1, k), dtype=np.float32)
        ids = np.empty((1, k), dtype=np.int64)
        self.index.search(  # type: ignore
            1,
            faiss.swig_ptr(vector),
            k,
            -1 if tenant_id is None else int(tenant_id),
            faiss.swig_ptr(distances),
            faiss.cast_integer_to_idx_t_ptr(ids.ctypes.data),
            None,
        )
        top_ids = ids
        return top_ids[0].tolist()

    """ Complex predicate support """

    def query_with_complex_predicate(
        self, x: np.ndarray, k: int, predicate: str
    ) -> list[int]:
        if predicate in self.filter_to_label:
            return self.query(x, k, self.filter_to_label[predicate])
        elif predicate in self.filter_to_bitmap:
            return self.search_with_bitmap_filter(
                x, k, self.filter_to_bitmap[predicate]
            )
        else:
            raise ValueError(
                f"No index or bitmap found for predicate {predicate}"
            )

    def index_filter(
        self,
        predicate: str,
        train_mds: list[list[int]] | None = None,
        *,
        qualified_labels: np.ndarray | None = None,
    ) -> int:
        """Build index for a given predicate. Returns the assigned tenant label.

        Either ``train_mds`` or ``qualified_labels`` must be provided.
        Returns ``-1`` if no vectors match the filter predicate.
        """
        if qualified_labels is None:
            if train_mds is None:
                raise ValueError("Either train_mds or qualified_labels must be provided")
            qualified_labels = compute_qualified_labels(predicate, train_mds)

        qualified_labels = np.ascontiguousarray(qualified_labels, dtype=np.uint32)
        if len(qualified_labels) == 0:
            # No vectors qualify for this filter; store an empty bitmap fallback
            # so that query_with_complex_predicate returns empty results instead of
            # raising an error.
            self.filter_to_bitmap[predicate] = qualified_labels
            return -1

        self.index.build_index_for_filter(  # type: ignore[attr-defined]
            qualified_labels,
            len(qualified_labels),
            predicate,
        )
        label = self.index.get_filter_label(predicate)
        self.filter_to_label[predicate] = label
        return label

    def build_filter_bitmap(self, predicate: str, train_mds: list[list[int]]) -> None:
        """Build a tenant index for a given predicate."""
        self.index_filter(predicate=predicate, train_mds=train_mds)

    def build_filter_sorted_vids(
        self, predicate: str, train_mds: list[list[int]]
    ) -> None:
        """Build a tenant index for the given predicate.

        Uses the main tree search path (build_index_for_filter + tenant-based
        query) instead of the temporary index path, because the temp index
        distance computation produces incorrect rankings on this build.
        """
        self.index_filter(predicate=predicate, train_mds=train_mds)

    def search_with_bitmap_filter(
        self,
        x: np.ndarray,
        k: int,
        qualified_labels: np.ndarray | list[int],
    ) -> list[int]:
        """Search using bitmap filter with explicit list of qualified labels."""
        qualified_labels = np.ascontiguousarray(qualified_labels, dtype=np.uint32)
        if len(qualified_labels) == 0:
            return [-1] * k

        x = np.ascontiguousarray(x[None], dtype=np.float32)
        D = np.empty((1, k), dtype=np.float32)
        I = np.empty((1, k), dtype=np.int64)

        self.index.search_with_bitmap_filter(  # type: ignore[attr-defined]
            1,
            faiss.swig_ptr(x),
            k,
            faiss.cast_integer_to_uint32_ptr(qualified_labels.ctypes.data),
            len(qualified_labels),
            faiss.swig_ptr(D),
            faiss.cast_integer_to_idx_t_ptr(I.ctypes.data),
        )
        return I[0].tolist()

    def get_last_search_profile(self) -> dict[str, Any]:
        """Get profiling data from the last search call.

        Covers both bitmap_filter and standard single-label query paths.
        Fields are populated depending on which query path was last used.

        Returns
        -------
        dict[str, Any]
        """
        return {
            # bitmap_filter path
            "preproc_time_ms": self.index.get_last_preproc_time_ms(),  # type: ignore
            "sort_time_ms": self.index.get_last_sort_time_ms(),  # type: ignore
            "build_temp_index_time_ms": self.index.get_last_build_temp_index_time_ms(),  # type: ignore
            "search_time_ms": self.index.get_last_search_time_ms(),  # type: ignore
            "qualified_labels_count": self.index.get_last_qualified_labels_count(),  # type: ignore
            "temp_nodes_count": self.index.get_last_temp_nodes_count(),  # type: ignore
            # standard single-label query path
            "beam_search_time_ms": self.index.get_last_beam_search_time_ms(),  # type: ignore
            "beam_layers_visited": self.index.get_last_beam_layers_visited(),  # type: ignore
            "beam_nodes_scored": self.index.get_last_beam_nodes_scored(),  # type: ignore
            "frontier_search_time_ms": self.index.get_last_frontier_search_time_ms(),  # type: ignore
            "frontier_nodes_popped": self.index.get_last_frontier_nodes_popped(),  # type: ignore
            "frontier_shortlists_scanned": self.index.get_last_frontier_shortlists_scanned(),  # type: ignore
            "frontier_children_expanded": self.index.get_last_frontier_children_expanded(),  # type: ignore
            "pq_table_build_time_ms": self.index.get_last_pq_table_build_time_ms(),  # type: ignore
            "pq_distance_compute_time_ms": self.index.get_last_pq_distance_compute_time_ms(),  # type: ignore
            "exact_distance_compute_time_ms": self.index.get_last_exact_distance_compute_time_ms(),  # type: ignore
            "candidate_merge_time_ms": self.index.get_last_candidate_merge_time_ms(),  # type: ignore
            "rerank_time_ms": self.index.get_last_rerank_time_ms(),  # type: ignore
            "rerank_count": self.index.get_last_rerank_count(),  # type: ignore
            "total_search_time_ms": self.index.get_last_total_search_time_ms(),  # type: ignore
        }

    def query_unfiltered(self, x: np.ndarray, k: int) -> list[int]:
        vector = np.ascontiguousarray(x[None], dtype=np.float32)
        distances = np.empty((1, k), dtype=np.float32)
        ids = np.empty((1, k), dtype=np.int64)
        self.index.search(  # type: ignore
            1,
            faiss.swig_ptr(vector),
            k,
            -1,
            faiss.swig_ptr(distances),
            faiss.cast_integer_to_idx_t_ptr(ids.ctypes.data),
            None,
        )
        return ids[0].tolist()

    def batch_query(
        self, X: np.ndarray, k: int, access_lists: list[list[int]], num_threads: int = 1
    ) -> list[list[int]]:
        raise NotImplementedError(
            "Batch querying is not supported for Curator"
        )

    def enable_stats_tracking(self, enable: bool = True) -> None:
        self.index.set_profiling_enabled(enable)

    def get_search_stats(self) -> dict[str, Any]:
        """Get search statistics. For compatibility, returns last search profile data."""
        return self.get_last_search_profile()

    def get_cached_temp_index_memory_usage(self) -> int:
        """Get total memory usage of all cached temporary indexes in bytes.

        Only relevant when use_temp_index_caching=True.

        Returns
        -------
        int
            Total memory usage in bytes
        """
        return self.index.get_cached_temp_index_memory_usage()  # type: ignore

    def get_index_memory_bytes(self) -> int:
        """Deterministic total memory in bytes, counting only index-owned data
        (TreeNode, shortlists, vector_indices, centroids, bloom filters,
        PQ codes, cached temp indexes).

        This is NOT RSS and does not include Python interpreter overhead,
        OS page cache, or GC artifacts. Use for reproducible before/after
        comparisons.
        """
        return self.index.get_total_memory_bytes()  # type: ignore

    def get_memory_breakdown(self) -> dict[str, Any]:
        """Component-level memory breakdown.

        Returns a dictionary with per-component byte counts:
        - tree: num_nodes, node_attrs, centroids
        - bloom_filters
        - shortlists: overhead, payload
        - vector_indices
        - id_allocators: vector, tenant
        - pq: codebook, codes
        - flash_index, raw_vectors_buffer
        - temp_cache: index, qualified_vecs
        - total (matches get_index_memory_bytes)
        """
        bd = self.index.get_memory_breakdown()  # type: ignore
        return {
            "tree": {
                "num_nodes": bd.num_tree_nodes,
                "node_attrs_bytes": bd.tree_node_attrs_bytes,
                "centroids_bytes": bd.centroids_bytes,
            },
            "bloom_filters_bytes": bd.bloom_filter_bytes,
            "shortlists": {
                "overhead_bytes": bd.shortlists_overhead_bytes,
                "payload_bytes": bd.shortlists_payload_bytes,
            },
            "vector_indices_bytes": bd.vector_indices_bytes,
            "id_allocators": {
                "vector_allocator_bytes": bd.id_allocator_bytes,
                "tenant_allocator_bytes": bd.tenant_id_allocator_bytes,
            },
            "pq": {
                "codebook_bytes": bd.pq_codebook_bytes,
                "codes_bytes": bd.pq_codes_bytes,
            },
            "flash_index_bytes": bd.flash_index_bytes,
            "raw_vectors_buffer_bytes": bd.raw_vectors_buffer_bytes,
            "temp_cache": {
                "index_bytes": bd.temp_index_cache_bytes,
                "qualified_vecs_bytes": bd.temp_qualified_vecs_bytes,
            },
            "total_bytes": bd.total_bytes,
        }

    def query_with_profile(
        self, x: np.ndarray, k: int, tenant_id: int | None = None
    ) -> tuple[list[int], dict[str, Any]]:
        """Query with automatic profiling enabled.

        Returns (result_ids, profile_dict) where profile_dict is
        the full search profile from get_last_search_profile().
        """
        self.enable_stats_tracking(True)
        result = self.query(x, k, tenant_id)
        profile = self.get_last_search_profile()
        return result, profile
