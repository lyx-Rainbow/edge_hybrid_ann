// prefiltering_index.h — Brute-force filtered search index (all data in memory)
//
// Architecture:
//   - Build: stores full training vectors + builds bidirectional label index
//     (label→vids for single-label lookup, vid→labels for predicate evaluation)
//   - Search (single-label): O(1) inverted-index lookup → brute-force L2 on candidates
//   - Search (complex-predicate): scan all vectors → evaluate predicate → L2 on qualified
//   - Search (unfiltered): brute-force L2 on all vectors
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "config.h"

class PreFilteringIndex {
public:
    explicit PreFilteringIndex(const PreFilteringConfig& cfg);
    ~PreFilteringIndex();

    // ── Build ──
    // vectors:       [n × d] float32, row-major contiguous
    // access_pairs:  [n_pairs × 2] int32, each row = (vid, tid)
    //
    // d is taken from cfg.d (auto-detected from .npy shape by main.cpp).
    // n_labels is auto-detected as max(tid) + 1 from access_pairs.
    void build(size_t n, const float* vectors,
               const int32_t* access_pairs, size_t n_pairs);

    // ── Single-label search ──
    // Returns top-k vectors (by L2 distance) that have `tenant_id` as a label.
    // If tenant_id is out of range or has no vectors, fills with -1.
    void search(const float* query, size_t k, int32_t tenant_id,
                float* distances, int32_t* labels) const;

    // ── Complex-predicate search ──
    // predicate is a prefix-notation boolean expression (e.g. "AND 0 NOT 1").
    // Iterates ALL training vectors to find those matching the predicate,
    // then performs brute-force L2 on the qualified set.
    void search_with_predicate(const float* query, size_t k,
                               const std::string& predicate,
                               float* distances, int32_t* labels) const;

    // ── Unfiltered search ──
    // Brute-force L2 on ALL training vectors (no label filtering).
    void search_unfiltered(const float* query, size_t k,
                           float* distances, int32_t* labels) const;

    // ── Info ──
    size_t memory_bytes() const;
    size_t ntotal() const { return ntotal_; }
    size_t dim() const { return d_; }
    size_t n_labels() const { return n_labels_; }

private:
    PreFilteringConfig cfg_;
    size_t d_ = 0;
    size_t ntotal_ = 0;
    size_t n_labels_ = 0;

    // Bidirectional label index
    std::vector<float> train_vecs_;                       // [N × d] full training vectors
    std::vector<std::vector<int32_t>> label_to_vids_;     // label → vid list (inverted, single-label)
    std::vector<std::vector<int32_t>> vid_to_labels_;     // vid → label list (forward, predicate eval)

    // External-scan mode: raw vectors live on disk and are read in small chunks.
    FILE* vector_fp_ = nullptr;

    // Core search helper: brute-force L2 among candidates
    void search_candidates(const float* query, size_t k,
                           const std::vector<int32_t>& candidates,
                           float* distances, int32_t* labels) const;
    void search_candidates_external(const float* query, size_t k,
                                    const std::vector<int32_t>& candidates,
                                    float* distances, int32_t* labels) const;
    void simulate_chunked_io(size_t n_cands) const;
};
