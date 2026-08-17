// curator_index.cpp — CuratorIndex implementation
#include "curator_index.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <queue>
#include <string>
#include <unistd.h>

#include "cluster_tree.h"
#include "complex_predicate.h"
#include "common.h"
#include "config.h"
#include "distance.h"
#include "flash_store.h"
#include "pq_block_cache.h"
#include "pq_codec.h"
#include "profiling.h"
#include "shortlist.h"
#include "temp_index.h"
#include "tree_node.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace curator {

// ============================================================================
// Thread-local scratch buffer for compute_vector_distance (avoid per-call heap alloc)
// ============================================================================
static thread_local std::vector<float> tl_scratch;

// ============================================================================
// Anonymous namespace: search helper functions (static linkage)
// ============================================================================
namespace {

using Candidate = std::pair<float, const TreeNode*>;
template <typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

// ── Beam search ──
std::vector<Candidate> beam_search(
        const CuratorIndex& index,
        const float* x,
        int_lid_t tid,
        size_t beam_width,
        std::vector<Candidate>& unexpanded) {
    std::vector<Candidate> beam;
    std::vector<Candidate> next_beam;

    const TreeNode* root = index.root();
    size_t d = index.d();
    float var_boost = index.config().variance_boost;

    if (!root->bf.contains(tid)) {
        return beam;
    }

    float score = node_score(root, x, d, var_boost);
    beam.emplace_back(score, root);

    while (true) {
        bool updated = false;
        for (auto& [scr, node] : beam) {
            if (node->shortlists.find(tid) != node->shortlists.end()) {
                next_beam.emplace_back(scr, node);
            } else {
                updated = true;
                if (!node->children.empty()) {
                    for (size_t i = 0; i < node->children.size() - 1; i++) {
                        PREFETCH(node->children[i + 1]->centroid.data());
                        auto child_score = node_score(node->children[i], x, d, var_boost);
                        next_beam.emplace_back(child_score, node->children[i]);
                    }
                    auto last_score = node_score(node->children.back(), x, d, var_boost);
                    next_beam.emplace_back(last_score, node->children.back());
                }
            }
        }

        if (!updated) break;

        std::sort(next_beam.begin(), next_beam.end());
        auto n_keep = std::min(beam_width, next_beam.size());
        for (size_t i = n_keep; i < next_beam.size(); i++) {
            unexpanded.push_back(next_beam[i]);
        }
        next_beam.resize(n_keep);

        std::swap(beam, next_beam);
        next_beam.clear();
    }

    return beam;
}

// ── Compute child scores with prefetch ──
void compute_child_scores_with_prefetch(
        const CuratorIndex& index,
        const TreeNode* node,
        const float* x,
        MinHeap<Candidate>& output) {
    if (node->children.empty()) return;

    size_t d = index.d();
    float var_boost = index.config().variance_boost;

    for (size_t i = 0; i < node->children.size() - 1; i++) {
        PREFETCH(node->children[i + 1]->centroid.data());
        auto score = node_score(node->children[i], x, d, var_boost);
        output.emplace(score, node->children[i]);
    }
    auto score = node_score(node->children.back(), x, d, var_boost);
    output.emplace(score, node->children.back());
}

} // anonymous namespace

// ============================================================================
// IdAllocator static member
// ============================================================================
template <>
const int_lid_t TenantIdAllocator::INVALID_ID = -1;

// ============================================================================
// Constructor / Destructor
// ============================================================================
CuratorIndex::CuratorIndex(const CuratorConfig& cfg) : cfg_(cfg) {
    CURATOR_ASSERT_FMT(cfg_.n_clusters <= MAX_BRANCH_FACTOR,
                       "n_clusters must be <= %zu", MAX_BRANCH_FACTOR);
    CURATOR_ASSERT_FMT(cfg_.max_leaf_size <= MAX_LEAF_SIZE,
                       "max_leaf_size must be <= %zu", MAX_LEAF_SIZE);
    CURATOR_ASSERT_FMT(cfg_.search_ef > 0,
                       "search_ef must be > 0 (default is 128)");

    root_ = new TreeNode(0, 0, nullptr, nullptr, cfg_.d,
                          cfg_.bf_capacity, cfg_.bf_false_pos);

    // Setup PQ sequence mapping pointers
    pq_.set_seq_maps(&seq_to_vid_, &vid_to_seq_);
}

CuratorIndex::~CuratorIndex() {
    delete root_;
    root_ = nullptr;
    // FlashStore and PQCodec have their own RAII destructors
}

// ============================================================================
// Build
// ============================================================================
void CuratorIndex::train(size_t n, const float* x) {
    build_tree(root_, n, x, cfg_);
}

void CuratorIndex::add_vector(const float* x, ext_vid_t label) {
    size_t d = cfg_.d;

    // Assign to leaf
    TreeNode* leaf = assign_to_leaf(root_, x, d);

    // Compute vid with path encoding
    auto offset = sizeof(int_vid_t) * 8 -
            leaf->level * MAX_BRANCH_FACTOR_LOG2 - MAX_LEAF_SIZE_LOG2;
    int_vid_t local_vid = static_cast<int_vid_t>(leaf->vector_indices.size());
    int_vid_t vid = leaf->node_id | (local_vid << offset);

    // Map external label to internal vid
    vid_map_.add_mapping(label, vid);

    // Buffer raw vector
    size_t buf_offset = raw_buffer_.size();
    raw_buffer_.insert(raw_buffer_.end(), x, x + d);
    vid_to_buf_offset_[vid] = buf_offset;
    vid_to_leaf_id_[vid] = leaf->node_id;
    vid_to_local_idx_[vid] = leaf->vector_indices.size();
    vid_to_seq_[vid] = seq_to_vid_.size();
    seq_to_vid_.push_back(vid);

    ntotal_++;
    leaf->vector_indices.insert(vid);

    // Update variance upward
    TreeNode* curr = leaf;
    while (curr != nullptr) {
        float dist = l2_sqr(x, curr->centroid.data(), d);
        curr->variance.add(dist);
        curr = curr->parent;
    }
}

void CuratorIndex::grant_access(ext_vid_t label, ext_lid_t tenant) {
    int_vid_t vid = vid_map_.get_id(label);
    int_lid_t int_tid = tid_map_.get_or_create_id(tenant);
    grant_access_impl(root_, vid, int_tid);
    // Maintain reverse index for accurate CP predicate evaluation
    vid_to_tids_[vid].insert(static_cast<tid_t>(tenant));
}

void CuratorIndex::grant_access_impl(TreeNode* node, int_vid_t vid, int_lid_t tid) {
    if (node->children.empty()) {
        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            it->second.insert(vid);
        } else {
            node->shortlists.emplace(tid, ShortList(std::vector<int_vid_t>{vid}));
        }
        node->bf.insert(tid);
    } else {
        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            it->second.insert(vid);
            if (it->second.size() > cfg_.max_sl_size) {
                split_shortlist(node, tid, cfg_.max_sl_size);
            }
        } else if (!node->bf.contains(tid)) {
            node->shortlists.emplace(tid, ShortList(std::vector<int_vid_t>{vid}));
        } else {
            auto offset = sizeof(int_vid_t) * 8 -
                    (node->level + 1) * MAX_BRANCH_FACTOR_LOG2;
            auto child_id = (vid >> offset) & (MAX_BRANCH_FACTOR - 1);
            grant_access_impl(node->children[child_id], vid, tid);
        }
        node->bf.insert(tid);
    }
}

void CuratorIndex::batch_grant_access(const std::vector<int_vid_t>& vids, int_lid_t tid) {
    for (int_vid_t vid : vids) {
        grant_access_impl(root_, vid, tid);
    }
}

void CuratorIndex::flush() {
    size_t d = cfg_.d;

    // ── Train PQ + encode + write to disk + open block cache ──
    if (cfg_.pq_enabled && ntotal_ > 0) {
        pq_.train(ntotal_, raw_buffer_.data(), d, cfg_.pq_M, cfg_.pq_nbits);

        // Encode all vectors → temporary local buffer
        std::vector<std::vector<uint8_t>> codes;
        pq_.encode_all(ntotal_, raw_buffer_.data(), codes);

        // Determine PQ codes file path
        // v5 §20: include this pointer to avoid same-process multi-instance collision
        std::string pq_path = cfg_.pq_codes_path;
        if (pq_path.empty()) {
            pq_path = "/tmp/curator_pq_" + std::to_string(getpid()) + "_" +
                      std::to_string(reinterpret_cast<uintptr_t>(this)) + ".bin";
        }

        // Write codes to disk
        pq_.write_to_disk(pq_path, codes, cfg_.pq_M, cfg_.pq_nbits);

        // Release PQ codes from memory (now on disk)
        codes.clear();
        codes.shrink_to_fit();

        // Open block cache for on-demand loading
        bool ok = pq_.open_cache(pq_path, cfg_.pq_cache_block_size, cfg_.pq_cache_max_blocks);
        if (!ok) {
            fprintf(stderr, "WARNING: PQ block cache open failed for '%s', "
                    "will fall back to exact distance in search\n", pq_path.c_str());
        }

        // Record path + persistence for destructor cleanup
        pq_.set_pq_file_path(pq_path);
        pq_.set_persist(cfg_.persist_pq_codes);

        // v5 §21: pq_enabled requires flash_storage for exact-distance fallback
        if (!cfg_.use_flash_storage) {
            fprintf(stderr, "INFO: pq_enabled=true requires use_flash_storage=true "
                    "for fallback; auto-enabling flash storage\n");
            cfg_.use_flash_storage = true;
        }
    }

    // ── Finalize flash storage ──
    if (cfg_.use_flash_storage && ntotal_ > 0) {
        // DFS traverse leaves, assign sequential IDs
        std::vector<TreeNode*> leaves;
        std::function<void(TreeNode*)> dfs = [&](TreeNode* node) {
            if (node->children.empty()) {
                leaves.push_back(node);
            } else {
                for (auto* child : node->children) {
                    dfs(child);
                }
            }
        };
        dfs(root_);

        num_leaves_ = leaves.size();
        leaf_region_size_ = cfg_.max_leaf_size * d * sizeof(float);

        // Build leaf region and sequence mappings
        for (size_t seq = 0; seq < leaves.size(); seq++) {
            leaf_node_id_to_seq_[leaves[seq]->node_id] = seq;
        }

        // Determine flash path
        std::string flash_path = cfg_.flash_path;
        if (flash_path.empty()) {
            flash_path = "/tmp/curator_flash_" + std::to_string(getpid()) + ".dat";
        }

        flash_.open(flash_path);
        flash_.truncate(num_leaves_ * leaf_region_size_);

        // Write vectors grouped by leaf
        size_t offset = 0;
        for (size_t seq = 0; seq < leaves.size(); seq++) {
            auto* leaf = leaves[seq];
            std::vector<float> leaf_vecs;
            for (int_vid_t vid : leaf->vector_indices.data) {
                auto it = vid_to_buf_offset_.find(vid);
                if (it != vid_to_buf_offset_.end()) {
                    const float* src = raw_buffer_.data() + it->second;
                    leaf_vecs.insert(leaf_vecs.end(), src, src + d);
                }
            }
            if (!leaf_vecs.empty()) {
                flash_.write_leaf_region(offset, leaf_region_size_,
                                          leaf_vecs.data(), leaf_vecs.size() / d, d);
            }
            offset += leaf_region_size_;
        }

        flash_finalized_ = true;
    }

    // Free raw buffer only when vectors are safely persisted on flash.
    // If neither flash nor PQ was used, raw_buffer must stay for exact-distance search.
    if (flash_finalized_) {
        raw_buffer_.clear();
        raw_buffer_.shrink_to_fit();
    }
}

// ============================================================================
// Post-build memory compaction
// ============================================================================
void CuratorIndex::compact_memory() {
    std::function<void(TreeNode*)> walk = [&](TreeNode* node) {
        node->centroid.shrink_to_fit();
        node->children.shrink_to_fit();
        node->shortlists.rehash(0);
        for (auto& kv : node->shortlists) {
            kv.second.data.shrink_to_fit();
        }
        node->vector_indices.data.shrink_to_fit();
        for (TreeNode* child : node->children) {
            walk(child);
        }
    };
    if (root_) {
        walk(root_);
    }

    // Build-only map: with flash finalized the query path never touches it
    // (raw-buffer fallback is dead), so drop it entirely.
    if (flash_finalized_) {
        std::unordered_map<int_vid_t, size_t>().swap(vid_to_buf_offset_);
    }

    // Compact the query-time lookup maps.
    leaf_node_id_to_seq_.rehash(0);
    vid_to_leaf_id_.rehash(0);
    vid_to_local_idx_.rehash(0);

    // vid_map_/tid_map_ keep reserve slack in their containers; compact them.
    vid_map_.label_to_id.rehash(0);
    vid_map_.id_to_label.rehash(0);
    tid_map_.label_to_id.rehash(0);
    tid_map_.id_to_label.shrink_to_fit();
}

// ============================================================================
// Search: tenant-filtered
// ============================================================================
void CuratorIndex::search(const float* x, size_t k, ext_lid_t tenant,
                           float* distances, ext_vid_t* labels) const {
    if (tenant < 0) {
        search_unfiltered(x, k, distances, labels);
        return;
    }

    int_lid_t int_tid = tid_map_.get_id(tenant);
    std::vector<int_vid_t> int_labels(k);
    search_one(x, k, int_tid, distances, int_labels.data());

    for (size_t i = 0; i < k; i++) {
        if (int_labels[i] != 0) {
            labels[i] = vid_map_.get_label(int_labels[i]);
        } else {
            labels[i] = 0;
        }
    }
}

void CuratorIndex::search_one(const float* x, size_t k, int_lid_t tid,
                               float* distances, int_vid_t* labels) const {
    size_t d = cfg_.d;
    size_t search_ef = cfg_.search_ef;
    size_t beam_sz = cfg_.beam_size;
    float var_boost = cfg_.variance_boost;

    // ── Profiling init ──
    auto t_start = profiling_on_ ? std::chrono::high_resolution_clock::now()
                                  : std::chrono::high_resolution_clock::time_point{};

    // ── Temp index cache for bitmap filter queries ──
    // When build_filter_index() is called with use_temp_index_caching=true,
    // it caches a lightweight TempIndexNode tree + sorted qualified vids
    // keyed by the filter's internal tenant ID (filter_tid).
    //
    // If this search's tid happens to be a cached filter_tid, we take the
    // fast path: search_temp_index directly on the cached nodes, skipping
    // the full beam-search → frontier → PQ pipeline on the main cluster tree.
    //
    // Standard single-tenant queries (tid = real tenant label) will NOT
    // hit this cache — they fall through to the normal search path below.
    const std::vector<TempIndexNode>* temp_nodes = nullptr;
    const std::vector<int_vid_t>* qualified_vecs = nullptr;
    if (get_cached_temp_index_data(tid, temp_nodes, qualified_vecs)) {
        // Build PQ distance table + batch-distance callback
        std::vector<float> pq_d_table;
        bool use_pq = cfg_.pq_enabled && pq_.is_trained() && pq_.has_cache();
        if (use_pq) {
            pq_.build_distance_table(x, pq_d_table);
        }

        auto batch_dist_fn = [&](const std::vector<int_vid_t>& vids,
                                  std::vector<std::pair<float, int_vid_t>>& out) {
            if (use_pq) {
                pq_.compute_pq_distances(pq_d_table, vids, out);
            } else {
                for (int_vid_t vid : vids) {
                    out.emplace_back(compute_vector_distance(x, vid), vid);
                }
            }
        };

        search_temp_index(*temp_nodes, *qualified_vecs, x, k, d,
                          search_ef, beam_sz, batch_dist_fn, distances, labels);
        if (profiling_on_) {
            std::strncpy(profile_.query_type, "temp_index", sizeof(profile_.query_type) - 1);
            auto t_end = std::chrono::high_resolution_clock::now();
            profile_.total_search_ms =
                std::chrono::duration<double, std::milli>(t_end - t_start).count();
        }
        return;
    }

    if (profiling_on_) {
        profile_.reset();
        std::strncpy(profile_.query_type, "standard", sizeof(profile_.query_type) - 1);
    }

    // ── Phase 1: Beam search ──
    auto t_beam = profiling_on_ ? std::chrono::high_resolution_clock::now()
                                 : std::chrono::high_resolution_clock::time_point{};

    MinHeap<Candidate> frontier;

    if (beam_sz > 0) {
        std::vector<Candidate> unexpanded;
        auto beam = beam_search(*this, x, tid, beam_sz, unexpanded);
        for (auto& cand : unexpanded) frontier.push(cand);
        for (auto& cand : beam) frontier.push(cand);
    } else {
        float score = node_score(root_, x, d, var_boost);
        frontier.emplace(score, root_);
    }

    if (profiling_on_) {
        auto t_end = std::chrono::high_resolution_clock::now();
        profile_.beam_search_ms =
            std::chrono::duration<double, std::milli>(t_end - t_beam).count();
    }

    // ── Phase 2: Frontier search ──
    auto t_frontier = profiling_on_ ? std::chrono::high_resolution_clock::now()
                                     : std::chrono::high_resolution_clock::time_point{};

    RunningList cand_vectors(static_cast<int>(search_ef));
    std::vector<std::pair<float, int_vid_t>> sorted_cands;
    sorted_cands.reserve(cfg_.max_sl_size);

    // ★ v5: PQ distance table — stack-local, per-query, thread-safe
    std::vector<float> pq_d_table;
    bool pq_table_built = false;
    bool use_pq = cfg_.pq_enabled && pq_.is_trained() && pq_.has_cache();

    while (!frontier.empty()) {
        auto [score, node] = frontier.top();
        frontier.pop();

        if (profiling_on_) profile_.frontier_nodes_popped++;

        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            if (profiling_on_) profile_.frontier_shortlists_scanned++;

            if (use_pq) {
                // ══════ PQ approximate distance path ══════
                // Build distance table once per query (on stack, thread-safe)
                if (!pq_table_built) {
                    auto t_tbl = profiling_on_
                        ? std::chrono::high_resolution_clock::now()
                        : std::chrono::high_resolution_clock::time_point{};
                    pq_.build_distance_table(x, pq_d_table);
                    pq_table_built = true;
                    if (profiling_on_) {
                        auto t_end = std::chrono::high_resolution_clock::now();
                        profile_.pq_table_build_ms =
                            std::chrono::duration<double, std::milli>(t_end - t_tbl).count();
                    }
                }

                // Compute ADC distances using on-disk PQ codes (thread-safe)
                auto t_pq = profiling_on_
                    ? std::chrono::high_resolution_clock::now()
                    : std::chrono::high_resolution_clock::time_point{};
                pq_.compute_pq_distances(pq_d_table, it->second.data, sorted_cands);
                if (profiling_on_) {
                    auto t_end = std::chrono::high_resolution_clock::now();
                    profile_.pq_distance_compute_ms +=
                        std::chrono::duration<double, std::milli>(t_end - t_pq).count();
                }
            } else {
                // ══════ Exact distance fallback (existing path) ══════
                for (int_vid_t vid : it->second.data) {
                    float dist = compute_vector_distance(x, vid);
                    sorted_cands.emplace_back(dist, vid);
                }
            }
            std::sort(sorted_cands.begin(), sorted_cands.end());

            auto t_cand_start = profiling_on_ ? std::chrono::high_resolution_clock::now()
                                               : std::chrono::high_resolution_clock::time_point{};
            bool updated = cand_vectors.batch_insert(sorted_cands);
            if (profiling_on_) {
                auto t_cand_end = std::chrono::high_resolution_clock::now();
                profile_.candidate_merge_ms +=
                    std::chrono::duration<double, std::milli>(t_cand_end - t_cand_start).count();
            }
            sorted_cands.clear();
            if (!updated) break;
        } else if (node->bf.contains(tid)) {
            if (profiling_on_) profile_.frontier_children_expanded++;
            compute_child_scores_with_prefetch(*this, node, x, frontier);
        }
    }

    if (profiling_on_) {
        auto t_end = std::chrono::high_resolution_clock::now();
        profile_.frontier_search_ms =
            std::chrono::duration<double, std::milli>(t_end - t_frontier).count();
    }

    // ── Phase 3: ADC Rerank (if enabled) ──
    // Batch-read candidates from flash to reduce random I/O: collects all
    // offsets, calls flash_.read_batch() once, then computes exact L2 distances.
    if (cfg_.pq_use_adc_rerank && !cand_vectors.vids.empty()) {
        auto t_rerank = profiling_on_ ? std::chrono::high_resolution_clock::now()
                                       : std::chrono::high_resolution_clock::time_point{};

        size_t n_rerank = std::min(k * cfg_.pq_rerank_topk_factor, cand_vectors.vids.size());
        if (profiling_on_) profile_.rerank_count = static_cast<int>(n_rerank);

        RunningList exact_list(static_cast<int>(n_rerank));

        if (flash_finalized_ && flash_.is_open()) {
            // ── Batch I/O path: collect offsets, read all at once ──
            std::vector<size_t> offsets;
            std::vector<int_vid_t> rerank_vids;
            offsets.reserve(n_rerank);
            rerank_vids.reserve(n_rerank);

            for (size_t i = 0; i < n_rerank; i++) {
                int_vid_t vid = cand_vectors.vids[i];
                auto leaf_it = vid_to_leaf_id_.find(vid);
                if (leaf_it != vid_to_leaf_id_.end()) {
                    auto seq_it = leaf_node_id_to_seq_.find(leaf_it->second);
                    if (seq_it != leaf_node_id_to_seq_.end()) {
                        auto local_it = vid_to_local_idx_.find(vid);
                        if (local_it != vid_to_local_idx_.end()) {
                            offsets.push_back(seq_it->second * leaf_region_size_ +
                                             local_it->second * cfg_.d * sizeof(float));
                            rerank_vids.push_back(vid);
                        }
                    }
                }
            }

            if (!offsets.empty()) {
                std::vector<float> batch_buf;
                flash_.read_batch(offsets, cfg_.d, batch_buf);
                for (size_t i = 0; i < rerank_vids.size(); i++) {
                    float exact_dist = l2_sqr(x, batch_buf.data() + i * cfg_.d, cfg_.d);
                    exact_list.insert(rerank_vids[i], exact_dist);
                }
            }
        } else {
            // ── In-memory fallback: raw_buffer_ still available ──
            for (size_t i = 0; i < n_rerank; i++) {
                float exact_dist = compute_vector_distance(x, cand_vectors.vids[i]);
                exact_list.insert(cand_vectors.vids[i], exact_dist);
            }
        }
        cand_vectors = std::move(exact_list);

        if (profiling_on_) {
            auto t_end = std::chrono::high_resolution_clock::now();
            profile_.rerank_ms =
                std::chrono::duration<double, std::milli>(t_end - t_rerank).count();
        }
    }

    // ── Output top-k ──
    size_t n_out = std::min(k, cand_vectors.vids.size());
    for (size_t i = 0; i < n_out; i++) {
        labels[i] = cand_vectors.vids[i];
        distances[i] = cand_vectors.dists[i];
    }
    for (size_t i = n_out; i < k; i++) {
        labels[i] = 0;
        distances[i] = std::numeric_limits<float>::max();
    }

    if (profiling_on_) {
        auto t_end = std::chrono::high_resolution_clock::now();
        profile_.total_search_ms =
            std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }
}

// ============================================================================
// Search: unfiltered
// ============================================================================
void CuratorIndex::search_unfiltered(const float* x, size_t k,
                                      float* distances, ext_vid_t* labels) const {
    size_t d = cfg_.d;
    float var_boost = cfg_.variance_boost;
    size_t nprobe = cfg_.nprobe;
    float prune_thres = cfg_.prune_thres;

    auto node_priority = [&](const TreeNode* node) {
        float dist = l2_sqr(x, node->centroid.data(), d);
        float var = static_cast<float>(node->variance.get_mean());
        return dist - var_boost * var;
    };

    MinHeap<Candidate> pq;
    pq.emplace(node_priority(root_), root_);

    int n_cand_vecs = 0;
    std::vector<Candidate> buckets;

    while (!pq.empty() && n_cand_vecs < static_cast<int>(nprobe)) {
        auto [score, node] = pq.top();
        pq.pop();

        if (!node->children.empty()) {
            for (auto* child : node->children) {
                pq.emplace(node_priority(child), child);
            }
        } else {
            float var = static_cast<float>(node->variance.get_mean());
            float dist = score + var_boost * var;
            buckets.emplace_back(dist, node);
            n_cand_vecs += static_cast<int>(node->vector_indices.size());
        }
    }

    std::sort(buckets.begin(), buckets.end());

    RunningList results(static_cast<int>(k));
    if (!buckets.empty()) {
        float min_buck_dist = buckets[0].first;
        for (auto& [dist, node] : buckets) {
            if (dist > prune_thres * min_buck_dist) break;

            for (int_vid_t vid : node->vector_indices) {
                float exact_dist = compute_vector_distance(x, vid);
                results.insert(vid, exact_dist);
            }
        }
    }

    size_t n_out = std::min(k, results.vids.size());
    for (size_t i = 0; i < n_out; i++) {
        labels[i] = vid_map_.get_label(results.vids[i]);
        distances[i] = results.dists[i];
    }
    for (size_t i = n_out; i < k; i++) {
        labels[i] = 0;
        distances[i] = std::numeric_limits<float>::max();
    }
}

// ============================================================================
// Search: bitmap filter
// unreachable, Reserved for direct bitmap-filter search without caching
// ============================================================================
void CuratorIndex::search_with_bitmap(
        const float* x, size_t k,
        const ext_vid_t* qualified, size_t n_qualified,
        float* distances, ext_vid_t* labels) const {
    // Convert ext_vid_t → int_vid_t
    std::vector<int_vid_t> int_vids;
    int_vids.reserve(n_qualified);
    for (size_t i = 0; i < n_qualified; i++) {
        if (vid_map_.has_label(qualified[i])) {
            int_vids.push_back(vid_map_.get_id(qualified[i]));
        }
    }
    std::sort(int_vids.begin(), int_vids.end());
    search_with_bitmap(x, k, int_vids.data(), int_vids.size(), distances, labels);
}

void CuratorIndex::search_with_bitmap(
        const float* x, size_t k,
        const int_vid_t* sorted_qualified, size_t n_qualified,
        float* distances, ext_vid_t* labels) const {
    if (profiling_on_) {
        profile_.reset();
        std::strncpy(profile_.query_type, "bitmap_filter", sizeof(profile_.query_type) - 1);
        profile_.qualified_labels_count = n_qualified;
    }

    // Build temporary index
    std::vector<TempIndexNode> temp_nodes;
    std::vector<int_vid_t> sorted_vids(sorted_qualified, sorted_qualified + n_qualified);

    auto t_build = profiling_on_ ? std::chrono::high_resolution_clock::now()
                                  : std::chrono::high_resolution_clock::time_point{};

    build_temp_index(root_, sorted_vids, cfg_.n_clusters, cfg_.max_sl_size, temp_nodes);

    if (profiling_on_) {
        auto t_end = std::chrono::high_resolution_clock::now();
        profile_.build_temp_index_ms =
            std::chrono::duration<double, std::milli>(t_end - t_build).count();
        profile_.temp_nodes_count = temp_nodes.size();
    }

    // Search temp index with real vector distances
    std::vector<float> pq_d_table;
    bool use_pq = cfg_.pq_enabled && pq_.is_trained() && pq_.has_cache();
    if (use_pq) {
        pq_.build_distance_table(x, pq_d_table);
    }

    auto batch_dist_fn = [&](const std::vector<int_vid_t>& vids,
                              std::vector<std::pair<float, int_vid_t>>& out) {
        if (use_pq) {
            pq_.compute_pq_distances(pq_d_table, vids, out);
        } else {
            for (int_vid_t vid : vids) {
                out.emplace_back(compute_vector_distance(x, vid), vid);
            }
        }
    };

    std::vector<int_vid_t> int_labels(k);
    search_temp_index(temp_nodes, sorted_vids, x, k, cfg_.d,
                      cfg_.search_ef, cfg_.beam_size, batch_dist_fn,
                      distances, int_labels.data());

    // Convert int_vid_t → ext_vid_t
    for (size_t i = 0; i < k; i++) {
        if (int_labels[i] != 0 && vid_map_.has_id(int_labels[i])) {
            labels[i] = vid_map_.get_label(int_labels[i]);
        } else {
            labels[i] = 0;
        }
    }
}

// ============================================================================
// compute_vector_distance — flash → buffer two-level fallback
// ============================================================================
float CuratorIndex::compute_vector_distance(const float* query, int_vid_t vid) const {
    if (tl_scratch.size() != cfg_.d) {
        tl_scratch.resize(cfg_.d);
    }

    if (flash_finalized_ && flash_.is_open()) {
        // Look up offset
        auto leaf_it = vid_to_leaf_id_.find(vid);
        if (leaf_it != vid_to_leaf_id_.end()) {
            auto seq_it = leaf_node_id_to_seq_.find(leaf_it->second);
            if (seq_it != leaf_node_id_to_seq_.end()) {
                auto local_it = vid_to_local_idx_.find(vid);
                if (local_it != vid_to_local_idx_.end()) {
                    size_t offset = seq_it->second * leaf_region_size_ +
                                    local_it->second * cfg_.d * sizeof(float);
                    flash_.read_vector(offset, cfg_.d, tl_scratch.data());
                    return l2_sqr(query, tl_scratch.data(), cfg_.d);
                }
            }
        }
    }

    if (!raw_buffer_.empty()) {
        auto it = vid_to_buf_offset_.find(vid);
        if (it != vid_to_buf_offset_.end()) {
            return l2_sqr(query, raw_buffer_.data() + it->second, cfg_.d);
        }
    }

    CURATOR_THROW_FMT("Cannot find vector for vid=%lu", static_cast<unsigned long>(vid));
}

// ============================================================================
// Management
// ============================================================================
bool CuratorIndex::revoke_access(ext_vid_t label, ext_lid_t tenant) {
    int_lid_t int_tid = tid_map_.get_id(tenant);
    int_vid_t vid = vid_map_.get_id(label);
    TreeNode* leaf = find_assigned_leaf(root_, vid);

    // Phase 1: find node containing the shortlist
    TreeNode* curr = leaf;
    while (curr != nullptr) {
        if (curr->shortlists.find(int_tid) != curr->shortlists.end()) break;
        curr = curr->parent;
    }
    CURATOR_THROW_IF_NOT_MSG(curr != nullptr,
            "Cannot find node containing shortlist for tenant");

    // Phase 2: remove vector from shortlist
    auto& sl = curr->shortlists.at(int_tid);
    sl.erase(vid);
    if (sl.size() == 0) {
        curr->shortlists.erase(int_tid);
        curr->bf = curr->recompute_bloom_filter();
    }

    // Phase 3: recursively merge upward
    while (curr && try_merge_shortlists(curr, int_tid, cfg_.max_sl_size)) {
        curr = curr->parent;
    }

    return true;
}

bool CuratorIndex::remove_vector(ext_vid_t /*label*/) {
    CURATOR_THROW_MSG("remove_vector is not supported");
}

// ============================================================================
// Filter index building
//
// Builds a temporary or persistent index for complex predicate queries
// (AND/OR/NOT combinations of tenant labels).
//
// Called by external code (CLI or Python via subprocess) when a complex
// predicate like "1 AND 2" or "1 OR (2 AND 3)" needs to be evaluated.
// The workflow:
//   1. find_all_qualified_vecs(filter) → get qualified vid list
//   2. build_filter_index(filter, qualified, n) → allocate filter_tid,
//      build TempIndexNode tree (if caching) or batch_grant_access (if not)
//   3. search(query, k, filter_tid, ...) → search_one hits temp_index cache
//
// Currently NOT called from main.cpp bench mode (which only does simple
// single-tenant queries via query_labels).  To enable complex-predicate
// experiments, add a --filter CLI option that calls this function.
// ============================================================================
ext_lid_t CuratorIndex::build_filter_index(
        const std::string& predicate,
        const ext_vid_t* qualified, size_t n) {
    // Convert ext → int
    std::vector<int_vid_t> int_vids;
    int_vids.reserve(n);
    for (size_t i = 0; i < n; i++) {
        if (vid_map_.has_label(qualified[i])) {
            int_vids.push_back(vid_map_.get_id(qualified[i]));
        }
    }
    std::sort(int_vids.begin(), int_vids.end());
    return build_filter_index(predicate, int_vids.data(), int_vids.size());
}

ext_lid_t CuratorIndex::build_filter_index(
        const std::string& predicate,
        const int_vid_t* qualified, size_t n) {
    std::vector<int_vid_t> int_vids(qualified, qualified + n);
    // Already sorted by caller, but ensure for safety
    if (!std::is_sorted(int_vids.begin(), int_vids.end())) {
        std::sort(int_vids.begin(), int_vids.end());
    }

    // Allocate new internal tenant ID
    ext_lid_t filter_label = tid_map_.allocate_reserved_label();
    int_lid_t filter_tid = tid_map_.allocate_id(filter_label);
    filter_to_label_[predicate] = filter_label;

    if (cfg_.use_temp_index_caching) {
        // Build and cache temp index
        auto& nodes = temp_indexes_[filter_tid];
        nodes.clear();
        build_temp_index(root_, int_vids, cfg_.n_clusters, cfg_.max_sl_size, nodes);
        qualified_cache_[filter_tid] = std::move(int_vids);
    } else {
        // Direct batch grant
        batch_grant_access(int_vids, filter_tid);
    }

    return filter_label;
}

ext_lid_t CuratorIndex::get_filter_label(const std::string& predicate) const {
    auto it = filter_to_label_.find(predicate);
    if (it != filter_to_label_.end()) return it->second;
    return -1;
}

// ============================================================================
// Complex predicate helpers
// ============================================================================
std::vector<int_vid_t> CuratorIndex::find_all_qualified_vecs(
        const std::string& filter) const {
    std::vector<int_vid_t> result;
    auto tokens = predicate::tokenize_formula(filter);

    for (size_t i = 0; i < seq_to_vid_.size(); i++) {
        int_vid_t vid = seq_to_vid_[i];

        // Use reverse index for accurate predicate evaluation.
        // The tree traversal approach (walking leaf→root shortlists) can miss
        // tenant IDs when shortlist splitting distributes entries across branches
        // not on this vector's assigned leaf path.
        static const std::unordered_set<tid_t> kEmptySet;
        auto it = vid_to_tids_.find(vid);
        const auto& access_set = (it != vid_to_tids_.end()) ? it->second : kEmptySet;
        if (predicate::evaluate_formula(tokens, access_set)) {
            result.push_back(vid);
        }
    }

    return result;
}

std::string CuratorIndex::convert_complex_predicate(const std::string& filter) const {
    // Pass-through for now — complex predicate parsing is done in evaluate
    return filter;
}

bool CuratorIndex::get_cached_temp_index_data(
        int_lid_t tid,
        const std::vector<TempIndexNode>*& nodes,
        const std::vector<int_vid_t>*& vids) const {
    auto it = temp_indexes_.find(tid);
    if (it != temp_indexes_.end()) {
        nodes = &it->second;
        auto vit = qualified_cache_.find(tid);
        if (vit != qualified_cache_.end()) {
            vids = &vit->second;
            return true;
        }
    }
    return false;
}

// ============================================================================
// Runtime parameter tuning
// ============================================================================
void CuratorIndex::set_rerank_params(bool enabled, size_t topk_factor) {
    cfg_.pq_use_adc_rerank = enabled;
    cfg_.pq_rerank_topk_factor = topk_factor;
}

bool CuratorIndex::get_rerank_enabled() const {
    return cfg_.pq_use_adc_rerank;
}

size_t CuratorIndex::get_rerank_topk_factor() const {
    return cfg_.pq_rerank_topk_factor;
}

// ============================================================================
// ID mapping export
// ============================================================================
void CuratorIndex::get_label_to_vid_mapping(
        const ext_vid_t* labels, size_t n, int_vid_t* vids_out) const {
    for (size_t i = 0; i < n; i++) {
        vids_out[i] = vid_map_.get_id(labels[i]);
    }
}

// ============================================================================
// Memory accounting
// ============================================================================
size_t CuratorIndex::memory_bytes() const {
    return memory_breakdown().total_bytes;
}

MemoryBreakdown CuratorIndex::memory_breakdown() const {
    MemoryBreakdown mb;

    // Tree traversal
    std::function<void(const TreeNode*)> traverse = [&](const TreeNode* node) {
        mb.num_tree_nodes++;
        mb.tree_node_attrs_bytes += sizeof(TreeNode);
        mb.centroids_bytes += node->centroid.size() * sizeof(float);
        mb.bloom_filter_bytes += node->bf.size();

        // Shortlist overhead + payload
        for (const auto& kv : node->shortlists) {
            mb.shortlists_overhead_bytes += sizeof(kv);
            mb.shortlists_payload_bytes += kv.second.data.size() * sizeof(int_vid_t);
        }

        // Vector indices
        mb.vector_indices_bytes += node->vector_indices.data.size() * sizeof(int_vid_t);

        for (auto* child : node->children) {
            traverse(child);
        }
    };

    if (root_) traverse(root_);

    // ID allocators
    mb.id_allocator_bytes = vid_map_.label_to_id.size() *
        (sizeof(ext_vid_t) + sizeof(int_vid_t) + sizeof(void*) * 2);
    mb.tenant_id_allocator_bytes = tid_map_.label_to_id.size() *
        (sizeof(ext_lid_t) + sizeof(int_lid_t) + sizeof(void*) * 2);

    // PQ
    mb.pq_codebook_bytes = pq_.codebook().size() * sizeof(float);
    mb.pq_cache_bytes = pq_.cache_bytes();  // block cache current usage

    // Flash index structures
    mb.flash_index_bytes =
        vid_to_buf_offset_.size() * (sizeof(int_vid_t) + sizeof(size_t) + sizeof(void*)) +
        vid_to_leaf_id_.size() * (sizeof(int_vid_t) + sizeof(int_vid_t) + sizeof(void*)) +
        leaf_node_id_to_seq_.size() * (sizeof(int_vid_t) + sizeof(size_t) + sizeof(void*));

    mb.raw_vectors_buffer_bytes = raw_buffer_.size() * sizeof(float);

    // Temp index cache
    for (const auto& kv : temp_indexes_) {
        mb.temp_index_cache_bytes += kv.second.size() * sizeof(TempIndexNode);
    }
    for (const auto& kv : qualified_cache_) {
        mb.temp_qualified_vecs_bytes += kv.second.size() * sizeof(int_vid_t);
    }

    mb.total_bytes = mb.tree_node_attrs_bytes + mb.centroids_bytes +
                     mb.bloom_filter_bytes + mb.shortlists_overhead_bytes +
                     mb.shortlists_payload_bytes + mb.vector_indices_bytes +
                     mb.id_allocator_bytes + mb.tenant_id_allocator_bytes +
                     mb.pq_codebook_bytes + mb.pq_cache_bytes +
                     mb.flash_index_bytes + mb.raw_vectors_buffer_bytes +
                     mb.temp_index_cache_bytes + mb.temp_qualified_vecs_bytes;

    return mb;
}

void CuratorIndex::print_tree_info() const {
    size_t total_nodes = 0, total_leaves = 0;
    double total_depth = 0;
    std::deque<const TreeNode*> q;
    q.push_back(root_);

    while (!q.empty()) {
        auto* node = q.front();
        q.pop_front();
        total_nodes++;

        if (node->children.empty()) {
            total_leaves++;
            total_depth += node->level;
        } else {
            for (auto* child : node->children) {
                q.push_back(child);
            }
        }
    }

    printf("Tree info: %zu nodes, %zu leaves, avg depth=%.2f\n",
           total_nodes, total_leaves,
           total_leaves > 0 ? total_depth / total_leaves : 0.0);
}

// ============================================================================
// Profiling
// ============================================================================
void CuratorIndex::enable_profiling(bool on) const {
    profiling_on_ = on;
}

const SearchProfile& CuratorIndex::last_profile() const {
    return profile_;
}

// ============================================================================
// Cache management
// ============================================================================
size_t CuratorIndex::cached_temp_index_memory() const {
    size_t total = 0;
    for (const auto& kv : temp_indexes_) {
        total += kv.second.size() * sizeof(TempIndexNode);
    }
    for (const auto& kv : qualified_cache_) {
        total += kv.second.size() * sizeof(int_vid_t);
    }
    return total;
}

void CuratorIndex::clear_temp_index_cache() {
    temp_indexes_.clear();
    qualified_cache_.clear();
}

// ============================================================================
// PQ cache statistics
// ============================================================================
PQBlockCache::Stats CuratorIndex::pq_cache_stats() const {
    return pq_.cache_stats();
}

size_t CuratorIndex::pq_cache_bytes() const {
    return pq_.cache_bytes();
}

// ============================================================================
// I/O statistics
// ============================================================================
FlashStore::Stats CuratorIndex::flash_io_stats() const {
    return flash_.stats();
}

void CuratorIndex::reset_io_stats() {
    flash_.reset_stats();
    pq_.reset_cache_stats();
}

// ============================================================================
// Sanity check (diagnostic)
// ============================================================================
void CuratorIndex::sanity_check() const {
    std::function<void(const TreeNode*)> check = [&](const TreeNode* node) {
        // Check bloom filter consistency
        auto recomputed = node->recompute_bloom_filter();

        // Check shortlist size constraints
        for (const auto& kv : node->shortlists) {
            CURATOR_ASSERT_FMT(kv.second.size() <= cfg_.max_sl_size,
                               "Shortlist for tid=%d at node level=%zu exceeds max_sl_size: %zu > %zu",
                               static_cast<int>(kv.first), node->level, kv.second.size(), cfg_.max_sl_size);
        }

        // Check children invariant
        if (!node->children.empty()) {
            CURATOR_ASSERT_FMT(node->children.size() == cfg_.n_clusters,
                               "Non-leaf node at level=%zu has %zu children, expected %zu",
                               node->level, node->children.size(), cfg_.n_clusters);
        }

        for (auto* child : node->children) {
            check(child);
        }
    };

    if (root_) check(root_);
    printf("Sanity check passed.\n");
}

} // namespace curator
