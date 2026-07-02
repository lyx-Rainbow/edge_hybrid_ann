// profiling.h — Search profiling data structures (header-only)
#pragma once

#include <cstddef>
#include <cstring>

namespace curator {

// ============================================================================
// SearchProfile — per-query timing and statistics
// WARNING: NOT thread-safe. Multi-threaded parallel query mode causes data races.
// This issue exists in the original code and is intentionally NOT fixed.
// ============================================================================
struct SearchProfile {
    char query_type[32] = "";

    // Standard single-tenant path
    double beam_search_ms = 0;
    int    beam_layers_visited = 0;
    int    beam_nodes_scored = 0;
    double frontier_search_ms = 0;
    int    frontier_nodes_popped = 0;
    int    frontier_shortlists_scanned = 0;
    int    frontier_children_expanded = 0;
    double pq_table_build_ms = 0;
    double pq_distance_compute_ms = 0;
    double exact_distance_compute_ms = 0;
    double candidate_merge_ms = 0;
    double rerank_ms = 0;
    int    rerank_count = 0;
    double total_search_ms = 0;

    // Bitmap filter path
    double preproc_ms = 0;
    double sort_ms = 0;
    double build_temp_index_ms = 0;
    double search_ms = 0;
    size_t qualified_labels_count = 0;
    size_t temp_nodes_count = 0;

    void reset() {
        std::memset(this, 0, sizeof(*this));
    }
};

// ============================================================================
// MemoryBreakdown — component-level memory accounting
// ============================================================================
struct MemoryBreakdown {
    size_t num_tree_nodes = 0;
    size_t tree_node_attrs_bytes = 0;
    size_t centroids_bytes = 0;
    size_t bloom_filter_bytes = 0;
    size_t shortlists_overhead_bytes = 0;
    size_t shortlists_payload_bytes = 0;
    size_t vector_indices_bytes = 0;
    size_t id_allocator_bytes = 0;
    size_t tenant_id_allocator_bytes = 0;
    size_t pq_codebook_bytes = 0;
    size_t pq_cache_bytes = 0;       // PQ block cache current usage
    size_t flash_index_bytes = 0;
    size_t raw_vectors_buffer_bytes = 0;
    size_t temp_index_cache_bytes = 0;
    size_t temp_qualified_vecs_bytes = 0;
    size_t total_bytes = 0;
};

} // namespace curator
