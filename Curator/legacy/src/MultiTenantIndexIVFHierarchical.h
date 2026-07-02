#pragma once

#include <stdint.h>
#include <cstdio>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <faiss/BloomFilter.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/MultiTenantIndexIVFFlat.h>
#include <faiss/complex_predicate.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

using ext_vid_t = label_t; // external vector ID
using int_vid_t = vid_t;   // internal vector ID
using ext_lid_t = tid_t;   // external label/tenant ID
using int_lid_t = tid_t;   // internal label/tenant ID

/*
 * We define the following constraints for the index because later we will
 * use vector IDs (of type vid_t) to encode the location of vectors in the
 * index. This leads to gaps in the vector ID space, which requires us to
 * use a wider type for vector IDs than necessary.
 */
constexpr size_t CURATOR_MAX_BRANCH_FACTOR_LOG2 = 6;
constexpr size_t CURATOR_MAX_LEAF_SIZE_LOG2 = 10;
constexpr size_t CURATOR_MAX_TREE_DEPTH =
        (sizeof(int_vid_t) * 8 - CURATOR_MAX_LEAF_SIZE_LOG2) /
        CURATOR_MAX_BRANCH_FACTOR_LOG2;
constexpr size_t CURATOR_MAX_BRANCH_FACTOR = 1
        << CURATOR_MAX_BRANCH_FACTOR_LOG2;
constexpr size_t CURATOR_MAX_LEAF_SIZE = 1 << CURATOR_MAX_LEAF_SIZE_LOG2;

template <typename ExtLabel, typename IntLabel>
struct IdAllocator {
    /*
     * This class is used to manage the mapping between external _label_s and
     * internal _id_s (non-negative integers). Internal ids are allocated in a
     * contiguous manner starting from 0.
     */

    static const IntLabel INVALID_ID;

    std::unordered_set<IntLabel> free_list;
    std::unordered_map<ExtLabel, IntLabel> label_to_id;
    std::vector<ExtLabel> id_to_label;

    IntLabel allocate_id(ExtLabel label);

    ExtLabel allocate_reserved_label() {
        // count down from the maximum possible label value to avoid conflicts
        // with user-provided labels
        for (ExtLabel label = std::numeric_limits<ExtLabel>::max();
             label != std::numeric_limits<ExtLabel>::min();
             label--) {
            if (!has_label(label)) {
                return label;
            }
        }

        FAISS_THROW_MSG("No available reserved label");
    }

    void free_id(ExtLabel label);

    bool has_label(ExtLabel label) const {
        return label_to_id.find(label) != label_to_id.end();
    }

    const IntLabel get_id(ExtLabel label) const {
        auto it = label_to_id.find(label);
        FAISS_THROW_IF_NOT_MSG(it != label_to_id.end(), "label does not exist");
        return it->second;
    }

    const IntLabel get_or_create_id(ExtLabel label) {
        if (has_label(label)) {
            return get_id(label);
        } else {
            return allocate_id(label);
        }
    }

    const ExtLabel get_label(IntLabel id) const {
        if (id >= id_to_label.size() || id_to_label[id] == INVALID_ID) {
            FAISS_THROW_MSG("id does not exist");
        }
        return id_to_label[id];
    }
};

// Unlike IdAllocator, IdMapping is not responsible for managing the allocation
// of internal IDs. Useful when we want a customize allocation strategy.
template <typename ExtLabel, typename IntLabel>
struct IdMapping {
    std::unordered_map<ExtLabel, IntLabel> label_to_id;
    std::unordered_map<IntLabel, ExtLabel> id_to_label;

    void add_mapping(ExtLabel label, IntLabel id) {
        label_to_id[label] = id;
        id_to_label[id] = label;
    }

    void remove_mapping(ExtLabel label) {
        FAISS_THROW_IF_NOT_MSG(has_label(label), "label does not exist");
        id_to_label.erase(label_to_id[label]);
        label_to_id.erase(label);
    }

    bool has_label(ExtLabel label) const {
        return label_to_id.find(label) != label_to_id.end();
    }

    bool has_id(IntLabel id) const {
        return id_to_label.find(id) != id_to_label.end();
    }

    const IntLabel get_id(ExtLabel label) const {
        FAISS_THROW_IF_NOT_MSG(has_label(label), "label does not exist");
        return label_to_id.at(label);
    }

    const ExtLabel get_label(IntLabel id) const {
        FAISS_THROW_IF_NOT_MSG(has_id(id), "id does not exist");
        return id_to_label.at(id);
    }
};

using VectorIdAllocator = IdMapping<ext_vid_t, int_vid_t>;
using TenantIdAllocator = IdAllocator<ext_lid_t, int_lid_t>;

// Forward declaration for complex_predicate namespace
namespace complex_predicate {

struct TempIndexNode {
    int start, end;
    std::vector<int> children;
    float* centroid; // Non-owning pointer to TreeNode's centroid
};

} // namespace complex_predicate

struct RunningMean {
    int n;
    double sum;

    RunningMean() : sum(0.0), n(0) {}

    void add(double x) {
        sum += x;
        n++;
    }

    void remove(double x) {
        FAISS_ASSERT_MSG(n > 0, "no elements to remove");

        sum -= x;
        n--;

        if (n == 0) {
            // reset the sum to avoid numerical issues
            sum = 0.0;
        }
    }

    double get_mean() const {
        if (n > 0) {
            return sum / n;
        } else {
            return 0.0;
        }
    }
};

template <typename T>
struct SortedList {
    std::vector<T> data;

    SortedList() {}

    SortedList(const std::vector<T>& data_) : data(data_) {
        std::sort(data.begin(), data.end());
    }

    SortedList(const SortedList& other) : data(other.data) {}

    SortedList(SortedList&& other) noexcept : data(std::move(other.data)) {}

    SortedList& operator=(const SortedList& other) {
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }

    SortedList& operator=(SortedList&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
        }
        return *this;
    }

    typename std::vector<T>::iterator begin() {
        return data.begin();
    }

    typename std::vector<T>::iterator end() {
        return data.end();
    }

    typename std::vector<T>::const_iterator begin() const {
        return data.begin();
    }

    typename std::vector<T>::const_iterator end() const {
        return data.end();
    }

    void insert(const T& item) {
        auto it = std::lower_bound(data.begin(), data.end(), item);
        data.insert(it, item);
    }

    void erase(const T& item) {
        auto it = std::lower_bound(data.begin(), data.end(), item);
        FAISS_ASSERT_MSG(it != data.end() && *it == item, "item not found");
        data.erase(it);
    }

    bool contains(const T& item) const {
        return std::binary_search(data.begin(), data.end(), item);
    }

    size_t size() const {
        return data.size();
    }

    SortedList merge(const SortedList& other) {
        SortedList result;
        std::merge(
                data.begin(),
                data.end(),
                other.data.begin(),
                other.data.end(),
                std::back_inserter(result.data));
        return result;
    }
};

using ShortList = SortedList<int_vid_t>;

struct TreeNode {
    /* information about the tree structure */
    size_t level;      // the level of this node in the tree
    size_t sibling_id; // the id of this node among its siblings
    TreeNode* parent;
    std::vector<TreeNode*> children;
    int_vid_t node_id;

    /* information about the cluster */
    float* centroid;
    RunningMean variance;

    /* available for all nodes */
    size_t bf_capacity;
    float bf_false_pos;
    bloom_filter bf;
    std::unordered_map<int_lid_t, ShortList> shortlists;

    /* only for leaf nodes */
    ShortList vector_indices; // vectors assigned to this leaf node

    TreeNode(
            size_t level,
            size_t sibling_id,
            TreeNode* parent,
            float* centroid,
            size_t d,
            size_t bf_capacity,
            float bf_false_pos);

    ~TreeNode() {
        free(centroid);
        for (TreeNode* child : children) {
            delete child;
        }
    }

    bloom_filter init_bloom_filter() const {
        bloom_parameters bf_params;
        bf_params.projected_element_count = bf_capacity;
        bf_params.false_positive_probability = bf_false_pos;
        bf_params.random_seed = 0xA5A5A5A5;
        bf_params.compute_optimal_parameters();
        return bloom_filter(bf_params);
    }

    bloom_filter recompute_bloom_filter() const {
        auto bf = init_bloom_filter();

        for (const auto& kv : shortlists) {
            bf.insert(kv.first);
        }

        for (const auto& child : children) {
            bf |= child->bf;
        }

        return bf;
    }
};

struct MultiTenantIndexIVFHierarchical : MultiTenantIndex {
    /* construction parameters */
    size_t bf_capacity;
    float bf_false_pos;
    size_t max_sl_size;
    size_t n_clusters;
    size_t clus_niter;
    size_t max_leaf_size;

    /* search parameters */
    size_t nprobe;
    float prune_thres;
    float variance_boost;

    /* main data structures */
    TreeNode* tree_root;
    VectorIdAllocator id_allocator;
    TenantIdAllocator tid_allocator;

    // --- PQ (Product Quantization) in-memory compressed representation ---
    size_t pq_M = 0;
    size_t pq_nbits = 0;
    bool pq_enabled = false;
    bool pq_use_adc_rerank = false;
    size_t pq_rerank_topk_factor = 4;
    std::vector<float> pq_codebook;                       // M × (1<<nbits) × (d/M) floats
    std::vector<std::vector<uint8_t>> vid_to_pq_code;     // indexed by sequence number
    std::vector<int_vid_t> seq_to_vid;                    // sequence number → vid
    std::unordered_map<int_vid_t, size_t> vid_to_seq;     // vid → sequence index for PQ lookup

    // --- Flash storage (full-precision raw vectors) ---
    std::string flash_storage_path;
    mutable FILE* flash_fp = nullptr;
    size_t leaf_region_size = 0;                          // max_leaf_size × d × sizeof(float)
    std::unordered_map<int_vid_t, size_t> leaf_node_id_to_seq; // leaf node_id → DFS leaf seq
    size_t num_leaves = 0;
    bool flash_finalized = false;
    bool use_flash_storage = true;    // if false, keep vectors in memory (no flash write)

    // --- PQ external storage (disk-backed PQ codes) ---
    bool persist_pq_codes = false;
    std::string pq_codes_path;
    mutable FILE* pq_codes_fp = nullptr;                  // file handle for disk-based PQ code access
    mutable void* pq_codes_mmap = nullptr;                // mmap'd region of PQ codes file
    mutable size_t pq_codes_mmap_size = 0;                // size of mmap'd region

    // --- Temporary buffer during build (replaces IndexFlat* storage) ---
    std::vector<float> raw_vectors_buffer;                // ntotal × d floats
    std::unordered_map<int_vid_t, size_t> vid_to_buffer_offset; // vid → offset in buffer
    std::unordered_map<int_vid_t, int_vid_t> vid_to_leaf_node_id; // vid → leaf node_id
    std::unordered_map<int_vid_t, size_t> vid_to_local_index;   // vid → position within leaf

    // --- Backward-compat: optional external IndexFlat storage (owned externally) ---
    IndexFlat* storage = nullptr;

    /* auxiliary data structures */
    std::unordered_map<std::string, ext_lid_t> filter_to_label;

    /* indexing strategy control */
    bool use_temp_index_caching;

    /* cached temporary indexes for filters */
    std::unordered_map<ext_lid_t, std::vector<complex_predicate::TempIndexNode>> cached_temp_indexes;
    std::unordered_map<ext_lid_t, std::vector<int_vid_t>> cached_qualified_vecs;

    /* ────────────────────────────────────────────
     * Memory breakdown (component-level accounting)
     * ──────────────────────────────────────────── */
    struct MemoryBreakdown {
        // Tree structure
        size_t num_tree_nodes = 0;
        size_t tree_node_attrs_bytes = 0;
        size_t centroids_bytes = 0;

        // Bloom Filters
        size_t bloom_filter_bytes = 0;

        // Short lists
        size_t shortlists_overhead_bytes = 0;   // unordered_map internal overhead
        size_t shortlists_payload_bytes = 0;    // actual int_vid_t data

        // Vector indices (leaf-level)
        size_t vector_indices_bytes = 0;

        // ID allocators
        size_t id_allocator_bytes = 0;
        size_t tenant_id_allocator_bytes = 0;

        // PQ (Product Quantization)
        size_t pq_codebook_bytes = 0;
        size_t pq_codes_bytes = 0;              // vid_to_pq_code total payload

        // Flash / buffer
        size_t flash_index_bytes = 0;           // offset maps etc.
        size_t raw_vectors_buffer_bytes = 0;    // 0 after flush

        // Temp index cache
        size_t temp_index_cache_bytes = 0;
        size_t temp_qualified_vecs_bytes = 0;

        // Grand total
        size_t total_bytes = 0;
    };

    /* profiling data structures */
    struct SearchProfilingData {
        // ---- bitmap_filter path ----
        double preproc_time_ms = 0.0;
        double sort_time_ms = 0.0;
        double build_temp_index_time_ms = 0.0;
        double search_time_ms = 0.0;            // bitmap-filter search phase
        size_t qualified_labels_count = 0;
        size_t temp_nodes_count = 0;

        // ---- standard single-label query path ----
        double beam_search_time_ms = 0.0;
        int    beam_layers_visited = 0;
        int    beam_nodes_scored = 0;

        double frontier_search_time_ms = 0.0;
        int    frontier_nodes_popped = 0;
        int    frontier_shortlists_scanned = 0;
        int    frontier_children_expanded = 0;

        double pq_table_build_time_ms = 0.0;
        double pq_distance_compute_time_ms = 0.0;
        double exact_distance_compute_time_ms = 0.0;

        double candidate_merge_time_ms = 0.0;
        double rerank_time_ms = 0.0;
        int    rerank_count = 0;

        double total_search_time_ms = 0.0;

        // Query type discriminator: "standard" | "bitmap_filter" | "unfiltered" | "temp_index"
        char   query_type[32] = "";
    };

    mutable SearchProfilingData last_search_profile;
    mutable bool enable_profiling = false;

    /* experimental */
    size_t search_ef;
    size_t beam_size;

    /* optimization mode control */
    mutable bool use_optimized_search = false;

    MultiTenantIndexIVFHierarchical(
            size_t d,
            size_t n_clusters,
            MetricType metric = METRIC_L2,
            size_t bf_capacity = 1000,
            float bf_false_pos = 0.01,
            size_t max_sl_size = 128,
            size_t clus_niter = 20,
            size_t max_leaf_size = 128,
            size_t nprobe = 3000,
            float prune_thres = 1.6,
            float variance_boost = 0.4,
            size_t search_ef = 0,
            size_t beam_size = 2,
            bool use_temp_index_caching = false);

    MultiTenantIndexIVFHierarchical(
            IndexFlat* storage,
            size_t n_clusters,
            size_t bf_capacity = 1000,
            float bf_false_pos = 0.01,
            size_t max_sl_size = 128,
            size_t clus_niter = 20,
            size_t max_leaf_size = 128,
            size_t nprobe = 3000,
            float prune_thres = 1.6,
            float variance_boost = 0.4,
            size_t search_ef = 0,
            size_t beam_size = 2,
            bool use_temp_index_caching = false);

    ~MultiTenantIndexIVFHierarchical() override;

    /*
     * PQ (Product Quantization) and Flash Storage configuration
     */

    void set_pq_config(
            size_t M,
            size_t nbits,
            bool enabled,
            bool use_adc_rerank,
            size_t rerank_topk_factor);

    void train_pq_codebook();

    void set_flash_storage_path(const std::string& path);

    void finalize_flash_storage();

    void load_vector_from_flash(int_vid_t vid, float* out) const;

    void load_vectors_from_flash_batched(
            const std::vector<int_vid_t>& vids,
            std::vector<float>& vectors_out) const;

    size_t get_total_memory_bytes() const;

    void set_rerank_params(bool enabled, size_t topk_factor);
    bool get_rerank_enabled() const;
    size_t get_rerank_topk_factor() const;

    void set_use_flash_storage(bool use);

    // --- PQ external storage (Goal 1) ---
    void set_pq_codes_path(const std::string& path);
    void set_persist_pq_codes(bool persist);
    void write_pq_codes_to_disk();
    void load_pq_codes_from_disk();
    void free_pq_codes();
    bool pq_codes_in_memory() const;
    bool pq_codes_on_disk() const;
    const uint8_t* get_pq_code_by_seq(size_t seq_idx) const;

    /*
     * API functions
     */

    void train(idx_t n, const float* x, ext_lid_t tid) override;

    void train_helper(TreeNode* node, idx_t n, const float* x);

    void add_vector_with_ids(idx_t n, const float* x, const idx_t* labels)
            override;

    void grant_access(idx_t label, ext_lid_t tid) override;

    void grant_access_helper(TreeNode* node, int_vid_t vid, int_lid_t tid);

    bool remove_vector(idx_t label) override;

    bool revoke_access(idx_t label, ext_lid_t tid) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            ext_lid_t tid,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void search_with_bitmap_filter(
            idx_t n,
            const float* x,
            idx_t k,
            const ext_vid_t* qualified_labels,
            size_t qualified_labels_size,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    // Optimized version that accepts sorted internal vector IDs directly
    // Skips preprocessing (external->internal ID conversion) and sorting phases
    void search_with_bitmap_filter_optimized(
            idx_t n,
            const float* x,
            idx_t k,
            const int_vid_t* sorted_qualified_vids,
            size_t sorted_qualified_vids_size,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    void search_one(
            const float* x,
            idx_t k,
            int_lid_t tid,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    void search_one(
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    void add_vector(idx_t n, const float* x) override {
        FAISS_THROW_MSG("add_vector not supported");
    }

    void reset() override {
        FAISS_THROW_MSG("reset not supported");
    }

    /*
     * Helper functions
     */

    TreeNode* assign_vec_to_leaf(const float* x);

    std::vector<idx_t> get_vector_path(ext_vid_t label) const;

    void split_short_list(TreeNode* node, int_lid_t tid);

    bool merge_short_list(TreeNode* node, int_lid_t tid);

    void locate_vector(ext_vid_t label) const;

    void print_tree_info() const;

    std::string convert_complex_predicate(const std::string& filter) const;

    std::vector<int_vid_t> find_all_qualified_vecs(
            const std::string& filter) const;

    void batch_grant_access(const std::vector<int_vid_t>& vids, int_lid_t tid);

    void build_index_for_filter(const ext_vid_t* qualified_labels, size_t qualified_labels_size, const std::string& filter_key);

    ext_lid_t get_filter_label(const std::string& filter_key) const;

    // Get mapping from external labels to internal vector IDs for Python interface
    void get_label_to_vid_mapping(const ext_vid_t* labels, size_t labels_size, int_vid_t* vids) const;

    // Check if a tenant ID corresponds to a cached temp index and get its data
    bool get_cached_temp_index_data(ext_lid_t tid, std::vector<int_vid_t>*& qualified_vecs, std::vector<complex_predicate::TempIndexNode>*& temp_nodes) const;

    // Calculate total memory usage of all cached temporary indexes
    size_t get_cached_temp_index_memory_usage() const;

    // Component-level memory breakdown (structured version of memory_usage)
    MemoryBreakdown get_memory_breakdown() const;

    void sanity_check() const;

    TreeNode* find_assigned_leaf(ext_vid_t label) const;

    void memory_usage() const;

    // Utility method for efficient distance computation
    float compute_vector_distance(const float* query, int_vid_t vid) const;

    // Enable/disable profiling for search_with_bitmap_filter calls
    void set_profiling_enabled(bool enabled) const {
        enable_profiling = enabled;
    }

    // Get profiling data from last search call (covers all query paths)
    SearchProfilingData get_last_search_profile() const {
        return last_search_profile;
    }

    // Individual profiling metric accessors (SWIG-friendly)
    // ---- bitmap_filter fields ----
    double get_last_preproc_time_ms() const {
        return last_search_profile.preproc_time_ms;
    }
    double get_last_sort_time_ms() const {
        return last_search_profile.sort_time_ms;
    }
    double get_last_build_temp_index_time_ms() const {
        return last_search_profile.build_temp_index_time_ms;
    }
    double get_last_search_time_ms() const {
        return last_search_profile.search_time_ms;
    }
    size_t get_last_qualified_labels_count() const {
        return last_search_profile.qualified_labels_count;
    }
    size_t get_last_temp_nodes_count() const {
        return last_search_profile.temp_nodes_count;
    }

    // ---- standard query fields ----
    double get_last_beam_search_time_ms() const {
        return last_search_profile.beam_search_time_ms;
    }
    int get_last_beam_layers_visited() const {
        return last_search_profile.beam_layers_visited;
    }
    int get_last_beam_nodes_scored() const {
        return last_search_profile.beam_nodes_scored;
    }
    double get_last_frontier_search_time_ms() const {
        return last_search_profile.frontier_search_time_ms;
    }
    int get_last_frontier_nodes_popped() const {
        return last_search_profile.frontier_nodes_popped;
    }
    int get_last_frontier_shortlists_scanned() const {
        return last_search_profile.frontier_shortlists_scanned;
    }
    int get_last_frontier_children_expanded() const {
        return last_search_profile.frontier_children_expanded;
    }
    double get_last_pq_table_build_time_ms() const {
        return last_search_profile.pq_table_build_time_ms;
    }
    double get_last_pq_distance_compute_time_ms() const {
        return last_search_profile.pq_distance_compute_time_ms;
    }
    double get_last_exact_distance_compute_time_ms() const {
        return last_search_profile.exact_distance_compute_time_ms;
    }
    double get_last_candidate_merge_time_ms() const {
        return last_search_profile.candidate_merge_time_ms;
    }
    double get_last_rerank_time_ms() const {
        return last_search_profile.rerank_time_ms;
    }
    int get_last_rerank_count() const {
        return last_search_profile.rerank_count;
    }
    double get_last_total_search_time_ms() const {
        return last_search_profile.total_search_time_ms;
    }

    // Optimization mode control
    void set_optimized_search_enabled(bool enabled) const {
        use_optimized_search = enabled;
    }

    bool get_optimized_search_enabled() const {
        return use_optimized_search;
    }
};

namespace complex_predicate {

void build_temp_index_for_filter(
        const MultiTenantIndexIVFHierarchical* index,
        const std::vector<int_vid_t>& sorted_qualified_vecs,
        std::vector<TempIndexNode>& nodes);

void search_temp_index(
        const MultiTenantIndexIVFHierarchical* index,
        const std::vector<int_vid_t>& qualified_vecs,
        const std::vector<TempIndexNode>& nodes,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params);

} // namespace complex_predicate

} // namespace faiss
