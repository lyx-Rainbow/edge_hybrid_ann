// prefiltering_index.cpp — PreFilteringIndex implementation
#include "prefiltering_index.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "distance.h"
#include "predicate.h"

// ============================================================================
// Constructor
// ============================================================================
PreFilteringIndex::PreFilteringIndex(const PreFilteringConfig& cfg) : cfg_(cfg) {}

// ============================================================================
// Build
// ============================================================================
void PreFilteringIndex::build(size_t n, const float* vectors,
                               const int32_t* access_pairs, size_t n_pairs) {
    d_ = cfg_.d;
    ntotal_ = n;

    // ── 1. Copy training vectors ──
    train_vecs_.assign(vectors, vectors + n * d_);

    // ── 2. Discover n_labels from access_pairs ──
    int32_t max_label = -1;
    for (size_t i = 0; i < n_pairs; i++) {
        int32_t tid = access_pairs[i * 2 + 1];
        if (tid > max_label) max_label = tid;
    }
    n_labels_ = static_cast<size_t>(max_label + 1);
    if (cfg_.n_labels > 0 && cfg_.n_labels > n_labels_) {
        n_labels_ = cfg_.n_labels;  // config can only enlarge, not shrink
    }

    // ── 3. Allocate index structures ──
    label_to_vids_.resize(n_labels_);
    vid_to_labels_.resize(n);

    // ── 4. Build both directions from access_pairs ──
    for (size_t i = 0; i < n_pairs; i++) {
        int32_t vid = access_pairs[i * 2];
        int32_t tid = access_pairs[i * 2 + 1];

        THROW_IF_NOT_FMT(vid >= 0 && static_cast<size_t>(vid) < n,
                         "vid %d out of range [0, %zu)", vid, n);
        THROW_IF_NOT_FMT(tid >= 0 && static_cast<size_t>(tid) < n_labels_,
                         "tid %d out of range [0, %zu)", tid, n_labels_);

        label_to_vids_[tid].push_back(vid);
        vid_to_labels_[vid].push_back(tid);
    }

    // ── 5. Sort inner vectors for deterministic order ──
    for (size_t i = 0; i < n_labels_; i++) {
        std::sort(label_to_vids_[i].begin(), label_to_vids_[i].end());
    }
    for (size_t i = 0; i < n; i++) {
        std::sort(vid_to_labels_[i].begin(), vid_to_labels_[i].end());
    }
}

// ============================================================================
// Single-label search
// ============================================================================
void PreFilteringIndex::search(const float* query, size_t k, int32_t tenant_id,
                                float* distances, int32_t* labels) const {
    // Out-of-range tenant_id → empty result
    if (tenant_id < 0 || static_cast<size_t>(tenant_id) >= n_labels_) {
        for (size_t i = 0; i < k; i++) {
            labels[i] = -1;
            distances[i] = std::numeric_limits<float>::max();
        }
        return;
    }

    const auto& candidates = label_to_vids_[tenant_id];
    search_candidates(query, k, candidates, distances, labels);
}

// ============================================================================
// Complex-predicate search
// ============================================================================
void PreFilteringIndex::search_with_predicate(
        const float* query, size_t k,
        const std::string& predicate,
        float* distances, int32_t* labels) const {
    // Tokenize once
    auto tokens = predicate::tokenize(predicate);

    // Scan ALL vectors, collect those matching the predicate
    std::vector<int32_t> candidates;
    candidates.reserve(ntotal_ / 4);  // typical selectivity heuristic
    for (size_t vid = 0; vid < ntotal_; vid++) {
        const auto& vlabels = vid_to_labels_[vid];
        if (predicate::evaluate(tokens, vlabels.data(),
                                vlabels.data() + vlabels.size())) {
            candidates.push_back(static_cast<int32_t>(vid));
        }
    }

    search_candidates(query, k, candidates, distances, labels);
}

// ============================================================================
// Unfiltered search
// ============================================================================
void PreFilteringIndex::search_unfiltered(const float* query, size_t k,
                                           float* distances, int32_t* labels) const {
    // All vectors are candidates
    std::vector<int32_t> candidates(ntotal_);
    for (size_t i = 0; i < ntotal_; i++) {
        candidates[i] = static_cast<int32_t>(i);
    }
    search_candidates(query, k, candidates, distances, labels);
}

// ============================================================================
// search_candidates — brute-force L2 among candidates, return top-k
// ============================================================================
void PreFilteringIndex::search_candidates(
        const float* query, size_t k,
        const std::vector<int32_t>& candidates,
        float* distances, int32_t* labels) const {
    const size_t n_cands = candidates.size();

    // Empty candidate set → all -1
    if (n_cands == 0) {
        for (size_t i = 0; i < k; i++) {
            labels[i] = -1;
            distances[i] = std::numeric_limits<float>::max();
        }
        return;
    }

    // Compute L2 distances for all candidates
    std::vector<std::pair<float, int32_t>> dists;
    dists.reserve(n_cands);

    // Use 4-way unrolled L2 for batches of 4
    size_t i = 0;
    for (; i + 3 < n_cands; i += 4) {
        const float* y0 = train_vecs_.data() + static_cast<size_t>(candidates[i]) * d_;
        const float* y1 = train_vecs_.data() + static_cast<size_t>(candidates[i+1]) * d_;
        const float* y2 = train_vecs_.data() + static_cast<size_t>(candidates[i+2]) * d_;
        const float* y3 = train_vecs_.data() + static_cast<size_t>(candidates[i+3]) * d_;

        float d0, d1, d2, d3;
        distance::l2_sqr_4way(query, y0, y1, y2, y3, d_, d0, d1, d2, d3);

        dists.emplace_back(d0, candidates[i]);
        dists.emplace_back(d1, candidates[i+1]);
        dists.emplace_back(d2, candidates[i+2]);
        dists.emplace_back(d3, candidates[i+3]);
    }
    // Remainder
    for (; i < n_cands; i++) {
        const float* y = train_vecs_.data() + static_cast<size_t>(candidates[i]) * d_;
        float d = distance::l2_sqr(query, y, d_);
        dists.emplace_back(d, candidates[i]);
    }

    // Partial sort: top-k
    size_t k_act = std::min(k, n_cands);
    std::partial_sort(dists.begin(), dists.begin() + static_cast<long>(k_act),
                      dists.end());

    // Output
    for (size_t j = 0; j < k; j++) {
        if (j < k_act) {
            labels[j] = dists[j].second;
            distances[j] = dists[j].first;
        } else {
            labels[j] = -1;
            distances[j] = std::numeric_limits<float>::max();
        }
    }
}

// ============================================================================
// Memory accounting
// ============================================================================
size_t PreFilteringIndex::memory_bytes() const {
    size_t mem = 0;

    // train_vecs_
    mem += train_vecs_.capacity() * sizeof(float);

    // label_to_vids_: vector overhead + contents
    mem += label_to_vids_.capacity() * sizeof(std::vector<int32_t>);
    for (const auto& v : label_to_vids_) {
        mem += v.capacity() * sizeof(int32_t);
    }

    // vid_to_labels_: vector overhead + contents
    mem += vid_to_labels_.capacity() * sizeof(std::vector<int32_t>);
    for (const auto& v : vid_to_labels_) {
        mem += v.capacity() * sizeof(int32_t);
    }

    return mem;
}
