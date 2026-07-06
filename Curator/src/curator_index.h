// curator_index.h — CuratorIndex: main orchestrator class
#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "config.h"
#include "flash_store.h"
#include "pq_codec.h"
#include "profiling.h"
#include "tree_node.h"

namespace curator {

// Fwd
struct TempIndexNode;

class CuratorIndex {
public:
    explicit CuratorIndex(const CuratorConfig& cfg);
    ~CuratorIndex();

    // ── Build ──
    void train(size_t n, const float* x);
    void add_vector(const float* x, ext_vid_t label);
    void grant_access(ext_vid_t label, ext_lid_t tenant);
    // Grant access to multiple vids for the same tenant (batch, for filter index building)
    void batch_grant_access(const std::vector<int_vid_t>& vids, int_lid_t tid);
    void flush(); // Train PQ + finalize flash storage

    // ── Query (single-query interfaces; batch by caller) ──
    void search(const float* x, size_t k, ext_lid_t tenant,
                float* distances, ext_vid_t* labels) const;
    void search_unfiltered(const float* x, size_t k,
                           float* distances, ext_vid_t* labels) const;
    void search_with_bitmap(const float* x, size_t k,
                            const ext_vid_t* qualified, size_t n_qualified,
                            float* distances, ext_vid_t* labels) const;
    void search_with_bitmap(const float* x, size_t k,
                            const int_vid_t* sorted_qualified, size_t n_qualified,
                            float* distances, ext_vid_t* labels) const;

    // ── Management ──
    bool revoke_access(ext_vid_t label, ext_lid_t tenant);
    bool remove_vector(ext_vid_t label); // stub: NOT_SUPPORTED
    ext_lid_t build_filter_index(const std::string& predicate,
                                  const ext_vid_t* qualified, size_t n);
    ext_lid_t build_filter_index(const std::string& predicate,
                                  const int_vid_t* qualified, size_t n);
    ext_lid_t get_filter_label(const std::string& predicate) const;
    std::vector<int_vid_t> find_all_qualified_vecs(const std::string& filter) const;

    // ── Runtime parameter tuning ──
    void set_rerank_params(bool enabled, size_t topk_factor);
    bool get_rerank_enabled() const;
    size_t get_rerank_topk_factor() const;

    // ── ID mapping export (debug/Python) ──
    void get_label_to_vid_mapping(const ext_vid_t* labels, size_t n,
                                   int_vid_t* vids_out) const;

    // ── Memory ──
    size_t memory_bytes() const;
    MemoryBreakdown memory_breakdown() const;
    void print_tree_info() const;

    // ── PQ cache statistics ──
    PQBlockCache::Stats pq_cache_stats() const;
    size_t pq_cache_bytes() const;

    // ── Profiling ──
    void enable_profiling(bool on) const;
    const SearchProfile& last_profile() const;

    // ── Cache ──
    size_t cached_temp_index_memory() const;
    void clear_temp_index_cache();

    // ── Accessors ──
    const CuratorConfig& config() const { return cfg_; }
    size_t ntotal() const { return ntotal_; }
    size_t d() const { return cfg_.d; }
    const TreeNode* root() const { return root_; }
    const VectorIdAllocator& vid_map() const { return vid_map_; }
    const TenantIdAllocator& tid_map() const { return tid_map_; }

private:
    CuratorConfig cfg_;
    TreeNode* root_ = nullptr;
    size_t ntotal_ = 0;

    // ID mappings
    VectorIdAllocator vid_map_;
    TenantIdAllocator tid_map_;

    // Raw vector buffer (before flush)
    std::vector<float> raw_buffer_;
    std::unordered_map<int_vid_t, size_t> vid_to_buf_offset_;
    std::unordered_map<int_vid_t, int_vid_t> vid_to_leaf_id_;
    std::unordered_map<int_vid_t, size_t> vid_to_local_idx_;
    std::vector<int_vid_t> seq_to_vid_;
    std::unordered_map<int_vid_t, size_t> vid_to_seq_;

    // Flash store metadata (populated at flush)
    std::unordered_map<int_vid_t, size_t> leaf_node_id_to_seq_;
    size_t num_leaves_ = 0;
    size_t leaf_region_size_ = 0;
    bool flash_finalized_ = false;

    // Submodules
    mutable PQCodec pq_;  // mutable: LRU cache state updated during const search
    FlashStore flash_;

    // Complex predicate auxiliary
    std::unordered_map<std::string, ext_lid_t> filter_to_label_;

    // Temp index cache
    std::unordered_map<int_lid_t, std::vector<TempIndexNode>> temp_indexes_;
    std::unordered_map<int_lid_t, std::vector<int_vid_t>> qualified_cache_;

    // Profiling
    mutable SearchProfile profile_;
    mutable bool profiling_on_ = false;

    // ── Private helpers ──
    void grant_access_impl(TreeNode* node, int_vid_t vid, int_lid_t tid);
    float compute_vector_distance(const float* query, int_vid_t vid) const;
    std::string convert_complex_predicate(const std::string& filter) const;
    bool get_cached_temp_index_data(int_lid_t tid,
                                     const std::vector<TempIndexNode>*& nodes,
                                     const std::vector<int_vid_t>*& vids) const;

    // Search helpers
    void search_one(const float* x, size_t k, int_lid_t tid,
                    float* distances, int_vid_t* labels) const;

    // Diagnostics
    void sanity_check() const;
};

} // namespace curator
