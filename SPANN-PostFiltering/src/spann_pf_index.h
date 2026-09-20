// spann_pf_index.h — SPTAG-based SPANN index with PostFiltering wrapper
//
// Architecture:
//   - Build: trains a SPTAG SPANN index (head BKT + disk posting lists),
//            then stores label metadata in a separate binary file for PostFilter.
//   - Search (single-label): calls SPTAG SearchIndex with overfetch (k' > k),
//            then PostFilters the results by label, sorts, and returns top-k.
//   - Search (complex-predicate): same overfetch + PostFilter flow but
//            evaluates a prefix-notation boolean expression per candidate.
//   - Search (unfiltered): direct SPTAG SearchIndex with k' = k, no PostFilter.
//
// Thread safety: when batch_query is enabled (OpenMP parallel queries),
// SPTAG internal threads are set to 1 to avoid nested parallelism.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "config.h"

// Forward declare SPTAG types — keep SPTAG headers out of public header
namespace SPTAG {
class VectorIndex;
}

class SPANNPostFilterIndex {
public:
    explicit SPANNPostFilterIndex(const SPANNConfig& cfg);
    ~SPANNPostFilterIndex();

    // ── Build ──
    // vectors:       [n × d] float32, row-major contiguous
    // access_pairs:  [n_pairs × 2] int32, each row = (vid, tid)
    void build(size_t n, const float* vectors,
               const int32_t* access_pairs, size_t n_pairs);

    // ── Single-label search ──
    // Returns top-k vectors that have `tenant_id` as a label.
    // Uses SPTAG overfetch + PostFilter (collect→sort→topk).
    void search(const float* query, size_t k, int32_t tenant_id,
                float* distances, int32_t* labels) const;

    // ── Complex-predicate search ──
    // predicate is a prefix-notation boolean expression (e.g. "AND 0 NOT 1").
    // Uses same overfetch + PostFilter flow with predicate::evaluate().
    void search_with_predicate(const float* query, size_t k,
                               const std::string& predicate,
                               float* distances, int32_t* labels) const;

    // ── Unfiltered search ──
    // Direct SPTAG search, no PostFilter — fastest path.
    void search_unfiltered(const float* query, size_t k,
                           float* distances, int32_t* labels) const;

    // ── Info ──
    size_t memory_bytes() const;
    size_t disk_bytes() const;
    size_t ntotal() const { return ntotal_; }
    size_t dim() const { return d_; }

    // ── Persistence (load pre-built index) ──
    void load(const std::string& index_dir, const std::string& metadata_path);

    // ── Re-configure SPTAG search parameters (for CLI search-mode sweep) ──
    void configure_spann_parameters();

    // ── Set search-time parameters only (safe after load — no build re-trigger) ──
    void set_search_parameters(size_t max_check, double overfetch_factor);

private:
    SPANNConfig cfg_;
    size_t ntotal_ = 0;
    size_t d_ = 0;

    // SPTAG index instance (created as SPANN<float> via factory)
    std::shared_ptr<SPTAG::VectorIndex> spann_index_;

    // ── Label metadata (for PostFiltering) ──
    // vid_to_labels_[vid] = sorted label list for vector vid
    std::vector<std::vector<int32_t>> vid_to_labels_;

    // Label frequency stats (for adaptive overfetch factor)
    std::vector<size_t> label_counts_;
    size_t total_label_assignments_ = 0;

    // ── Internal methods ──

    // Compute overfetch k' based on label selectivity.
    // Uses adaptive formula when overfetch_adaptive=true and tenant_id is valid.
    // Falls back to fixed factor when label stats are unavailable.
    size_t compute_overfetch_k(size_t k, int32_t tenant_id) const;

    // Write label metadata to binary file.
    void save_metadata(const std::string& path) const;

    // Read label metadata from binary file.
    void load_metadata(const std::string& path);

    // (moved to public for CLI search-mode reconfiguration after load)
};
