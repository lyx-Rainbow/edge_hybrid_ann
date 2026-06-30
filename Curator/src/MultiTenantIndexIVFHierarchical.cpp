// -*- c++ -*-

#include <faiss/MultiTenantIndexIVFHierarchical.h>

#include <omp.h>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <set>

#include <faiss/Clustering.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/utils.h>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "MultiTenantIndexIVFHierarchical.h"

namespace faiss {
namespace complex_predicate {

void build_temp_index_for_filter(
        const MultiTenantIndexIVFHierarchical* index,
        const std::vector<int_vid_t>& sorted_qualified_vecs,
        std::vector<TempIndexNode>& nodes) {
    // Verify that the input vector is sorted
    FAISS_THROW_IF_NOT_MSG(
            std::is_sorted(
                    sorted_qualified_vecs.begin(), sorted_qualified_vecs.end()),
            "Input vector must be sorted in ascending order");

    // Build the tree
    // Helper function to recursively build temp index tree
    std::function<int(int, int, TreeNode*)> build_temp_tree =
            [&](int start, int end, TreeNode* curr_node) -> int {
        // Create new node
        int curr_node_idx = nodes.size();
        float* centroid = curr_node->centroid;
        nodes.push_back(TempIndexNode{start, end, /*children*/ {}, centroid});

        // Base case - leaf node or # qualfied vectors is small enough for
        // buffering
        if (end - start <= index->max_sl_size || curr_node->children.empty()) {
            return curr_node_idx;
        }

        // Calculate offset and mask to extract branch idx from vector ID
        auto level = curr_node->level;
        auto offset = sizeof(int_vid_t) * 8 -
                CURATOR_MAX_BRANCH_FACTOR_LOG2 * (level + 1);
        auto mask = (CURATOR_MAX_BRANCH_FACTOR - 1);

        // Find ranges of vector IDs for each child using binary search
        // We only compare the part of bits in vector IDs that corresponds to
        // the current level
        std::vector<std::pair<int, int>> child_ranges;
        child_ranges.reserve(index->n_clusters);

        for (int child_idx = 0; child_idx < index->n_clusters; child_idx++) {
            // Find first element with current prefix
            int first = std::lower_bound(
                                sorted_qualified_vecs.begin() + start,
                                sorted_qualified_vecs.begin() + end,
                                child_idx,
                                [offset, mask](int_vid_t vid, int child_idx) {
                                    return ((vid >> offset) & mask) < child_idx;
                                }) -
                    sorted_qualified_vecs.begin();

            // Find first element with next prefix
            int last = std::lower_bound(
                               sorted_qualified_vecs.begin() + first,
                               sorted_qualified_vecs.begin() + end,
                               child_idx + 1,
                               [offset, mask](int_vid_t vid, int child_idx) {
                                   return ((vid >> offset) & mask) < child_idx;
                               }) -
                    sorted_qualified_vecs.begin();

            child_ranges.emplace_back(first, last);
        }

        // Recursively build child nodes
        for (int child_idx = 0; child_idx < index->n_clusters; child_idx++) {
            auto& range = child_ranges[child_idx];
            // If at least one vector ID belongs to this child, build the child
            if (range.first != range.second) {
                TreeNode* child_node = curr_node->children[child_idx];
                int child_node_idx =
                        build_temp_tree(range.first, range.second, child_node);
                nodes[curr_node_idx].children.push_back(child_node_idx);
            }
        }

        return curr_node_idx;
    };

    // Start building from root node over all qualified vectors
    if (!sorted_qualified_vecs.empty()) {
        build_temp_tree(0, sorted_qualified_vecs.size(), index->tree_root);
    }
}

} // namespace complex_predicate

template <>
const int_lid_t TenantIdAllocator::INVALID_ID = -1;

template <typename ExtLabel, typename IntLabel>
IntLabel IdAllocator<ExtLabel, IntLabel>::allocate_id(ExtLabel label) {
    FAISS_THROW_IF_NOT_MSG(
            label_to_id.find(label) == label_to_id.end(),
            "label already exists");

    IntLabel id;
    if (free_list.empty()) {
        id = id_to_label.size();
        id_to_label.push_back(INVALID_ID);
    } else {
        id = *free_list.begin();
        free_list.erase(free_list.begin());
    }

    label_to_id.emplace(label, id);
    id_to_label[id] = label;

    return id;
}

template <typename ExtLabel, typename IntLabel>
void IdAllocator<ExtLabel, IntLabel>::free_id(ExtLabel label) {
    auto it = label_to_id.find(label);
    FAISS_THROW_IF_NOT_MSG(it != label_to_id.end(), "label does not exist");

    label_to_id.erase(label);

    IntLabel id = it->second;
    if (id == id_to_label.size() - 1) {
        id_to_label.pop_back();
    } else {
        id_to_label[id] = INVALID_ID;
        free_list.emplace(id);
    }
}

TreeNode::TreeNode(
        size_t level,
        size_t sibling_id,
        TreeNode* parent,
        float* centroid,
        size_t d,
        size_t bf_capacity,
        float bf_false_pos)
        : level(level),
          sibling_id(sibling_id),
          parent(parent),
          bf_capacity(bf_capacity),
          bf_false_pos(bf_false_pos) {
    if (parent != nullptr) {
        auto offset =
                sizeof(int_vid_t) * 8 - level * CURATOR_MAX_BRANCH_FACTOR_LOG2;
        this->node_id = parent->node_id | (sibling_id << offset);
    } else {
        this->node_id = 0;
    }

    if (centroid == nullptr) {
        this->centroid = nullptr;
    } else {
        this->centroid =
                static_cast<float*>(std::aligned_alloc(64, d * sizeof(float)));
        std::memcpy(this->centroid, centroid, sizeof(float) * d);
    }

    this->bf = init_bloom_filter();
}

MultiTenantIndexIVFHierarchical::MultiTenantIndexIVFHierarchical(
        size_t d,
        size_t n_clusters,
        MetricType metric,
        size_t bf_capacity,
        float bf_false_pos,
        size_t max_sl_size,
        size_t clus_niter,
        size_t max_leaf_size,
        size_t nprobe,
        float prune_thres,
        float variance_boost,
        size_t search_ef,
        size_t beam_size,
        bool use_temp_index_caching)
        : MultiTenantIndex(d, metric),
          n_clusters(n_clusters),
          bf_capacity(bf_capacity),
          bf_false_pos(bf_false_pos),
          max_sl_size(max_sl_size),
          clus_niter(clus_niter),
          max_leaf_size(max_leaf_size),
          nprobe(nprobe),
          prune_thres(prune_thres),
          variance_boost(variance_boost),
          search_ef(search_ef),
          beam_size(beam_size),
          use_temp_index_caching(use_temp_index_caching) {
    FAISS_ASSERT_FMT(
            n_clusters <= CURATOR_MAX_BRANCH_FACTOR,
            "n_clusters should be less than or equal to %zu",
            CURATOR_MAX_BRANCH_FACTOR);

    FAISS_ASSERT_FMT(
            max_leaf_size <= CURATOR_MAX_LEAF_SIZE,
            "max_leaf_size should be less than or equal to %zu",
            CURATOR_MAX_LEAF_SIZE);

    tree_root =
            new TreeNode(0, 0, nullptr, nullptr, d, bf_capacity, bf_false_pos);
}

MultiTenantIndexIVFHierarchical::MultiTenantIndexIVFHierarchical(
        IndexFlat* storage,
        size_t n_clusters,
        size_t bf_capacity,
        float bf_false_pos,
        size_t max_sl_size,
        size_t clus_niter,
        size_t max_leaf_size,
        size_t nprobe,
        float prune_thres,
        float variance_boost,
        size_t search_ef,
        size_t beam_size,
        bool use_temp_index_caching)
        : MultiTenantIndex(storage->d, storage->metric_type),
          storage(storage),
          n_clusters(n_clusters),
          bf_capacity(bf_capacity),
          bf_false_pos(bf_false_pos),
          max_sl_size(max_sl_size),
          clus_niter(clus_niter),
          max_leaf_size(max_leaf_size),
          nprobe(nprobe),
          prune_thres(prune_thres),
          variance_boost(variance_boost),
          search_ef(search_ef),
          beam_size(beam_size),
          use_temp_index_caching(use_temp_index_caching) {
    FAISS_ASSERT_FMT(
            n_clusters <= CURATOR_MAX_BRANCH_FACTOR,
            "n_clusters should be less than or equal to %zu",
            CURATOR_MAX_BRANCH_FACTOR);

    FAISS_ASSERT_FMT(
            max_leaf_size <= CURATOR_MAX_LEAF_SIZE,
            "max_leaf_size should be less than or equal to %zu",
            CURATOR_MAX_LEAF_SIZE);

    tree_root =
            new TreeNode(0, 0, nullptr, nullptr, d, bf_capacity, bf_false_pos);
}

MultiTenantIndexIVFHierarchical::~MultiTenantIndexIVFHierarchical() {
    delete tree_root;
    tree_root = nullptr;
    if (flash_fp != nullptr) {
        std::fclose(flash_fp);
        flash_fp = nullptr;
    }
    if (pq_codes_mmap != nullptr && pq_codes_mmap != MAP_FAILED) {
        munmap(pq_codes_mmap, pq_codes_mmap_size);
        pq_codes_mmap = nullptr;
        pq_codes_mmap_size = 0;
    }
    if (pq_codes_fp != nullptr) {
        std::fclose(pq_codes_fp);
        pq_codes_fp = nullptr;
    }
}

void MultiTenantIndexIVFHierarchical::train(
        idx_t n,
        const float* x,
        ext_lid_t tid) {
    train_helper(tree_root, n, x);
}

void MultiTenantIndexIVFHierarchical::train_helper(
        TreeNode* node,
        idx_t n,
        const float* x) {
    if (node->centroid == nullptr) {
        node->centroid = new float[d];
        for (size_t i = 0; i < d; i++) {
            node->centroid[i] = 0;
        }

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                node->centroid[j] += x[i * d + j];
            }
        }

        for (size_t i = 0; i < d; i++) {
            node->centroid[i] /= n;
        }
    }

    // stop if there are too few samples to cluster
    if (n <= max_leaf_size || node->level >= CURATOR_MAX_TREE_DEPTH) {
        return;
    }

    // partition the data into n_clusters clusters
    IndexFlatL2 quantizer(d);
    ClusteringParameters cp;
    cp.niter = clus_niter;
    cp.min_points_per_centroid = 1;
    Clustering clus(d, n_clusters, cp);
    clus.train(n, x, quantizer);
    quantizer.is_trained = true;

    std::vector<idx_t> cluster_ids(n);
    quantizer.assign(n, x, cluster_ids.data());

    // sort the vectors by cluster
    std::vector<float> sorted_x(n * d);
    std::vector<size_t> cluster_size(n_clusters, 0);
    std::vector<size_t> cluster_offsets(n_clusters, 0);

    for (size_t i = 0; i < n; i++) {
        size_t cluster = cluster_ids[i];
        cluster_size[cluster]++;
    }

    cluster_offsets[0] = 0;
    for (size_t i = 1; i < n_clusters; i++) {
        cluster_offsets[i] = cluster_offsets[i - 1] + cluster_size[i - 1];
    }

    std::vector<size_t> tmp_offsets = cluster_offsets;

    for (size_t i = 0; i < n; ++i) {
        size_t cluster = cluster_ids[i];
        size_t curr_offset = tmp_offsets[cluster]++;
        std::memcpy(
                sorted_x.data() + curr_offset * d,
                x + i * d,
                sizeof(float) * d);
    }

    // recursively train children
    for (size_t clus_id = 0; clus_id < n_clusters; clus_id++) {
        TreeNode* child = new TreeNode(
                node->level + 1,
                clus_id,
                node,
                clus.centroids.data() + clus_id * d,
                d,
                bf_capacity,
                bf_false_pos);

        train_helper(
                child,
                cluster_size[clus_id],
                sorted_x.data() + cluster_offsets[clus_id] * d);

        node->children.push_back(child);
    }
}

void MultiTenantIndexIVFHierarchical::add_vector_with_ids(
        idx_t n,
        const float* x,
        const idx_t* labels) {
    for (size_t i = 0; i < n; i++) {
        ext_vid_t label = labels[i];
        const float* xi = x + i * d;

        // add the vector to the leaf node
        TreeNode* leaf = assign_vec_to_leaf(xi);
        auto offset = sizeof(int_vid_t) * 8 -
                leaf->level * CURATOR_MAX_BRANCH_FACTOR_LOG2 -
                CURATOR_MAX_LEAF_SIZE_LOG2;
        int_vid_t local_vid =
                static_cast<int_vid_t>(leaf->vector_indices.size());
        int_vid_t vid = leaf->node_id | (local_vid << offset);

        // add the vector to the vector store and access matrix
        id_allocator.add_mapping(label, vid);

        // Buffer raw vector for later PQ encoding and flash write
        size_t buf_offset = raw_vectors_buffer.size();
        raw_vectors_buffer.insert(raw_vectors_buffer.end(), xi, xi + d);
        vid_to_buffer_offset[vid] = buf_offset;
        vid_to_leaf_node_id[vid] = leaf->node_id;
        vid_to_local_index[vid] = leaf->vector_indices.size(); // pre-insert size = position
        vid_to_seq[vid] = seq_to_vid.size();
        seq_to_vid.push_back(vid);

        // Also add to external storage if provided (backward compat)
        if (storage != nullptr) {
            storage->add(1, xi);
        }
        ntotal++;
        leaf->vector_indices.insert(vid);

        TreeNode* curr = leaf;
        while (curr != nullptr) {
            float dist = fvec_L2sqr(xi, curr->centroid, d);
            curr->variance.add(dist);
            curr = curr->parent;
        }
    }
}

void MultiTenantIndexIVFHierarchical::grant_access(
        idx_t label,
        ext_lid_t ext_tid) {
    int_vid_t vid = id_allocator.get_id(label);
    int_lid_t int_tid = tid_allocator.get_or_create_id(ext_tid);
    grant_access_helper(tree_root, vid, int_tid);
}

void MultiTenantIndexIVFHierarchical::grant_access_helper(
        TreeNode* node,
        int_vid_t vid,
        int_lid_t tid) {
    if (node->children.empty()) {
        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            it->second.insert(vid);
        } else {
            node->shortlists.emplace(tid, std::vector<int_vid_t>{vid});
        }

        node->bf.insert(tid);
    } else {
        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            it->second.insert(vid);
            if (it->second.size() > max_sl_size) {
                split_short_list(node, tid);
            }
        } else if (!node->bf.contains(tid)) {
            node->shortlists.emplace(tid, std::vector<int_vid_t>{vid});
        } else {
            auto offset = sizeof(int_vid_t) * 8 -
                    (node->level + 1) * CURATOR_MAX_BRANCH_FACTOR_LOG2;
            auto child_id = (vid >> offset) & (CURATOR_MAX_BRANCH_FACTOR - 1);
            grant_access_helper(node->children[child_id], vid, tid);
        }

        node->bf.insert(tid);
    }
}

bool MultiTenantIndexIVFHierarchical::remove_vector(idx_t label) {
    FAISS_THROW_MSG("remove_vector is not supported");
}

bool MultiTenantIndexIVFHierarchical::revoke_access(
        idx_t label,
        ext_lid_t tid) {
    int_lid_t int_tid = tid_allocator.get_id(tid);
    int_vid_t vid = id_allocator.get_id(label);
    auto leaf = find_assigned_leaf(label);

    // phase 1: find the node that contain shortlist
    auto curr = leaf;
    while (curr != nullptr) {
        if (curr->shortlists.find(int_tid) != curr->shortlists.end()) {
            break;
        }
        curr = curr->parent;
    }

    FAISS_THROW_IF_NOT_MSG(
            curr != nullptr,
            "Cannot find the node that contains the shortlist of the tenant");

    // phase 2: remove the vector from the shortlist
    auto& shortlist = curr->shortlists.at(int_tid);
    shortlist.erase(vid);
    if (shortlist.size() == 0) {
        curr->shortlists.erase(int_tid);
        curr->bf = curr->recompute_bloom_filter();
    }

    // phase 3: recursively merge shortlists and update bloom filters
    while (curr && merge_short_list(curr, int_tid)) {
        curr = curr->parent;
    }

    return true;
}

void MultiTenantIndexIVFHierarchical::search(
        idx_t n,
        const float* x,
        idx_t k,
        ext_lid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (tid < 0) {
        // perform unfiltered search
        search(n, x, k, distances, labels, params);
        return;
    }

    int_lid_t int_tid = tid_allocator.get_id(tid);

    bool inter_query_parallel = getenv("BATCH_QUERY") != nullptr;
    if (inter_query_parallel) {
#pragma omp parallel for schedule(dynamic) if (n > 1)
        for (idx_t i = 0; i < n; i++) {
            search_one(
                    x + i * d,
                    k,
                    int_tid,
                    distances + i * k,
                    labels + i * k,
                    params);
        }
    } else {
        for (idx_t i = 0; i < n; i++) {
            search_one(
                    x + i * d,
                    k,
                    int_tid,
                    distances + i * k,
                    labels + i * k,
                    params);
        }
    }
}

void MultiTenantIndexIVFHierarchical::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    for (idx_t i = 0; i < n; i++) {
        search_one(x + i * d, k, distances + i * k, labels + i * k, params);
    }
}

template <typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;
using HeapForL2 = CMax<float, idx_t>;

namespace {

struct RunningList {
    std::vector<int_vid_t> vids;
    std::vector<float> dists;
    std::vector<int_vid_t> vids_tmp;
    std::vector<float> dists_tmp;
    int capacity;

    RunningList(int capacity) : capacity(capacity) {
        vids.reserve(capacity + 1);
        dists.reserve(capacity + 1);
        vids_tmp.reserve(capacity + 1);
        dists_tmp.reserve(capacity + 1);
    }

    bool insert(int_vid_t vid, float dist) {
        auto it = std::lower_bound(dists.begin(), dists.end(), dist);
        auto pos = it - dists.begin();
        bool updated = false;

        if (dists.size() < capacity) {
            dists.insert(it, dist);
            vids.insert(vids.begin() + pos, vid);
            updated = true;
        } else if (dist < dists.back()) {
            dists.insert(it, dist);
            dists.pop_back();
            vids.insert(vids.begin() + pos, vid);
            vids.pop_back();
            updated = true;
        }

        return updated;
    }

    bool batch_insert(const std::vector<std::pair<float, int_vid_t>>& cands) {
        bool updated = dists.size() < capacity || cands[0].first < dists.back();
        if (!updated) {
            return false;
        }

        int i = 0, j = 0;
        while (i < dists.size() && j < cands.size()) {
            if (dists[i] <= cands[j].first) {
                vids_tmp.push_back(vids[i]);
                dists_tmp.push_back(dists[i]);
                i++;
            } else {
                vids_tmp.push_back(cands[j].second);
                dists_tmp.push_back(cands[j].first);
                j++;
            }

            if (vids_tmp.size() == capacity) {
                break;
            }
        }

        while (vids_tmp.size() < capacity && i < dists.size()) {
            vids_tmp.push_back(vids[i]);
            dists_tmp.push_back(dists[i]);
            i++;
        }

        while (vids_tmp.size() < capacity && j < cands.size()) {
            vids_tmp.push_back(cands[j].second);
            dists_tmp.push_back(cands[j].first);
            j++;
        }

        std::swap(vids, vids_tmp);
        std::swap(dists, dists_tmp);
        vids_tmp.clear();
        dists_tmp.clear();

        return true;
    }

    void resize(int new_capacity) {
        if (new_capacity <= capacity) {
            return;
        }
        capacity = new_capacity;
        vids.reserve(capacity + 1);
        dists.reserve(capacity + 1);
        vids_tmp.reserve(capacity + 1);
        dists_tmp.reserve(capacity + 1);
    }

    void reset() {
        vids.clear();
        dists.clear();
        vids_tmp.clear();
        dists_tmp.clear();
    }
};

using Candidate = std::pair<float, const TreeNode*>;

inline float node_score(
        const TreeNode* node,
        const float* x,
        size_t d,
        float var_boost) {
    float dist = fvec_L2sqr(x, node->centroid, d);
    if (var_boost == 0.0) {
        return dist;
    } else {
        float var = node->variance.get_mean();
        return dist - var_boost * var;
    }
}

inline std::vector<Candidate> beam_search(
        const MultiTenantIndexIVFHierarchical& index,
        const float* x,
        int_lid_t tid,
        size_t beam_width,
        std::vector<Candidate>& unexpanded) {
    std::vector<Candidate> beam;
    std::vector<Candidate> next_beam;

    if (!index.tree_root->bf.contains(tid)) {
        return beam;
    }

    float score = node_score(index.tree_root, x, index.d, index.variance_boost);
    beam.emplace_back(score, index.tree_root);

    while (true) {
        bool updated = false;
        for (auto [score, node] : beam) {
            if (node->shortlists.find(tid) != node->shortlists.end()) {
                next_beam.emplace_back(score, node);
            } else {
                updated = true;

                if (!node->children.empty()) {
                    auto centroid = node->children[0]->centroid;
                    for (size_t i = 0; i < node->children.size() - 1; i++) {
                        auto next_centroid = node->children[i + 1]->centroid;
                        prefetch_L1(next_centroid);
                        auto score = node_score(
                                node->children[i],
                                x,
                                index.d,
                                index.variance_boost);
                        next_beam.emplace_back(score, node->children[i]);
                        centroid = next_centroid;
                    }
                    auto score = node_score(
                            node->children.back(),
                            x,
                            index.d,
                            index.variance_boost);
                    next_beam.emplace_back(score, node->children.back());
                }
            }
        }

        if (!updated) {
            break;
        }

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

inline void compute_dists_with_prefetch(
        const MultiTenantIndexIVFHierarchical& index,
        const int_vid_t* vids,
        size_t n_vids,
        const float* x,
        std::vector<std::pair<float, int_vid_t>>& output) {
    if (n_vids == 0) {
        return;
    }

    size_t d = index.d;

    // Determine if we can use direct pointer access (buffer or external storage)
    bool use_direct_access = !index.raw_vectors_buffer.empty() ||
                             index.storage != nullptr;

    // Get vector data pointer from buffer or external storage
    auto get_vector_ptr = [&](int_vid_t vid) -> const float* {
        if (!index.raw_vectors_buffer.empty()) {
            idx_t buf_offset = index.vid_to_buffer_offset.at(vid);
            return index.raw_vectors_buffer.data() + buf_offset;
        }
        // Backward compat: external IndexFlat storage
        idx_t storage_idx = index.vid_to_buffer_offset.at(vid);
        return index.storage->get_xb() + storage_idx * d;
    };

    // If vectors are on flash only, use single-vector reads (slower but correct)
    if (!use_direct_access) {
        for (size_t i = 0; i < n_vids; i++) {
            int_vid_t vid = vids[i];
            float dist = index.compute_vector_distance(x, vid);
            output.emplace_back(dist, vid);
        }
        return;
    }

    // For very small batches, use simple computation
    if (n_vids < 4) {
        for (size_t i = 0; i < n_vids; i++) {
            int_vid_t vid = vids[i];
            float dist = index.compute_vector_distance(x, vid);
            output.emplace_back(dist, vid);
        }
        return;
    }

    // Process vectors in batches of 4 using fvec_L2sqr_batch_4
    size_t i = 0;

    // Prefetch the first batch of vectors
    for (size_t j = 0; j < std::min(n_vids, size_t(8)); j++) {
        prefetch_L1(get_vector_ptr(vids[j]));
    }

    // Process batches of 4 vectors
    for (; i + 3 < n_vids; i += 4) {
        // Prefetch next batch if available
        if (i + 8 < n_vids) {
            prefetch_L1(get_vector_ptr(vids[i + 4]));
            prefetch_L1(get_vector_ptr(vids[i + 5]));
            prefetch_L1(get_vector_ptr(vids[i + 6]));
            prefetch_L1(get_vector_ptr(vids[i + 7]));
        }

        // Get vector pointers for batch of 4
        const float* y0 = get_vector_ptr(vids[i]);
        const float* y1 = get_vector_ptr(vids[i + 1]);
        const float* y2 = get_vector_ptr(vids[i + 2]);
        const float* y3 = get_vector_ptr(vids[i + 3]);

        // Compute 4 distances (individual calls — fvec_L2sqr_batch_4 not in FAISS 1.7.4 headers)
        float dist0, dist1, dist2, dist3;
        dist0 = fvec_L2sqr(x, y0, d);
        dist1 = fvec_L2sqr(x, y1, d);
        dist2 = fvec_L2sqr(x, y2, d);
        dist3 = fvec_L2sqr(x, y3, d);

        // Add results to output
        output.emplace_back(dist0, vids[i]);
        output.emplace_back(dist1, vids[i + 1]);
        output.emplace_back(dist2, vids[i + 2]);
        output.emplace_back(dist3, vids[i + 3]);
    }

    // Handle remaining vectors (1-3 vectors)
    for (; i < n_vids; i++) {
        int_vid_t vid = vids[i];
        float dist = index.compute_vector_distance(x, vid);
        output.emplace_back(dist, vid);
    }
}

// Overload for backward compatibility with vector parameter
inline void compute_dists_with_prefetch(
        const MultiTenantIndexIVFHierarchical& index,
        const std::vector<int_vid_t>& vids,
        const float* x,
        std::vector<std::pair<float, int_vid_t>>& output) {
    compute_dists_with_prefetch(index, vids.data(), vids.size(), x, output);
}

// ADC (Asymmetric Distance Computation) using PQ codes (in-memory or disk-backed).
// Uses get_pq_code_by_seq() to transparently access PQ codes from memory or mmap'd disk file.
// Only valid when pq_enabled and flash_finalized.
inline void compute_pq_distances(
        const MultiTenantIndexIVFHierarchical& index,
        const int_vid_t* vids,
        size_t n_vids,
        const float* x,
        std::vector<std::pair<float, int_vid_t>>& output) {
    if (n_vids == 0) return;

    size_t d = index.d;
    size_t M = index.pq_M;
    size_t nbits = index.pq_nbits;
    size_t ksub = 1 << nbits;
    size_t dsub = d / M;

    // Build distance lookup table per sub-quantizer
    std::vector<float> d_table(M * ksub);
    for (size_t m = 0; m < M; m++) {
        const float* q_sub = x + m * dsub;
        const float* cb = index.pq_codebook.data() + m * ksub * dsub;
        for (size_t k = 0; k < ksub; k++) {
            d_table[m * ksub + k] = fvec_L2sqr(q_sub, cb + k * dsub, dsub);
        }
    }

    // Sum distances from lookup table for each vector
    for (size_t i = 0; i < n_vids; i++) {
        int_vid_t vid = vids[i];
        auto seq_it = index.vid_to_seq.find(vid);
        if (seq_it == index.vid_to_seq.end()) continue;
        size_t seq_idx = seq_it->second;
        const uint8_t* code = index.get_pq_code_by_seq(seq_idx);

        float dist = 0.0f;
        for (size_t m = 0; m < M; m++) {
            dist += d_table[m * ksub + code[m]];
        }
        output.emplace_back(dist, vid);
    }
}

inline void compute_pq_distances(
        const MultiTenantIndexIVFHierarchical& index,
        const std::vector<int_vid_t>& vids,
        const float* x,
        std::vector<std::pair<float, int_vid_t>>& output) {
    compute_pq_distances(index, vids.data(), vids.size(), x, output);
}

inline void compute_child_scores_with_prefetch(
        const MultiTenantIndexIVFHierarchical& index,
        const TreeNode* node,
        const float* x,
        MinHeap<Candidate>& output) {
    if (node->children.empty()) {
        return;
    }

    auto var_boost = index.variance_boost;

    auto centroid = node->children[0]->centroid;
    for (size_t i = 0; i < node->children.size() - 1; i++) {
        auto next_centroid = node->children[i + 1]->centroid;
        prefetch_L1(next_centroid);
        auto score = node_score(node->children[i], x, index.d, var_boost);
        output.emplace(score, node->children[i]);
        centroid = next_centroid;
    }

    auto score = node_score(node->children.back(), x, index.d, var_boost);
    output.emplace(score, node->children.back());
}

}; // namespace

void MultiTenantIndexIVFHierarchical::search_one(
        const float* x,
        idx_t k,
        int_lid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    using Candidate = std::pair<float, const TreeNode*>;

    // ── profiling init ──
    if (enable_profiling) {
        last_search_profile = SearchProfilingData{};
        std::strncpy(last_search_profile.query_type, "standard", sizeof(last_search_profile.query_type) - 1);
    }
    auto t_total_start = enable_profiling ? std::chrono::high_resolution_clock::now()
                                          : std::chrono::high_resolution_clock::time_point{};

    // Check if this tenant ID corresponds to a cached temp index
    std::vector<int_vid_t>* qualified_vecs = nullptr;
    std::vector<complex_predicate::TempIndexNode>* temp_nodes = nullptr;

    if (get_cached_temp_index_data(tid, qualified_vecs, temp_nodes)) {
        if (enable_profiling) {
            std::strncpy(last_search_profile.query_type, "temp_index", sizeof(last_search_profile.query_type) - 1);
        }
        // Use temp index search instead of regular search
        complex_predicate::search_temp_index(
            this,
            *qualified_vecs,
            *temp_nodes,
            x,
            k,
            distances,
            labels,
            params
        );
        if (enable_profiling) {
            auto t_end = std::chrono::high_resolution_clock::now();
            last_search_profile.total_search_time_ms =
                std::chrono::duration<double, std::milli>(t_end - t_total_start).count();
        }
        return;
    }

    // Require search_ef > 0 to always use the experimental search
    FAISS_THROW_IF_NOT_MSG(search_ef > 0, "search_ef must be greater than 0");

    // ── Phase 1: Beam search ──
    auto t_beam_start = enable_profiling ? std::chrono::high_resolution_clock::now()
                                         : std::chrono::high_resolution_clock::time_point{};

    // Inlined content from filtered_search_experimental
    MinHeap<Candidate> frontier;

    if (beam_size > 0) {
        std::vector<Candidate> unexpanded;
        auto beam = beam_search(*this, x, tid, beam_size, unexpanded);
        for (auto& cand : unexpanded) {
            frontier.push(cand);
        }
        for (auto& cand : beam) {
            frontier.push(cand);
        }
    } else {
        float score = node_score(tree_root, x, d, variance_boost);
        frontier.emplace(score, tree_root);
    }

    if (enable_profiling) {
        auto t_beam_end = std::chrono::high_resolution_clock::now();
        last_search_profile.beam_search_time_ms =
            std::chrono::duration<double, std::milli>(t_beam_end - t_beam_start).count();
        last_search_profile.beam_layers_visited = 0;   // beam_search is opaque here
        last_search_profile.beam_nodes_scored = 0;     // would need instrumenting beam_search itself
    }

    // ── Phase 2: Frontier search ──
    auto t_frontier_start = enable_profiling ? std::chrono::high_resolution_clock::now()
                                             : std::chrono::high_resolution_clock::time_point{};

    RunningList cand_vectors(search_ef);
    std::vector<std::pair<float, int_vid_t>> sorted_cands;
    sorted_cands.reserve(max_sl_size);
    std::vector<float> child_dists;
    child_dists.reserve(n_clusters);

    bool pq_table_built = false;  // track first PQ table build

    while (!frontier.empty()) {
        auto [score, node] = frontier.top();
        frontier.pop();

        if (enable_profiling) {
            last_search_profile.frontier_nodes_popped++;
        }

        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            if (enable_profiling) {
                last_search_profile.frontier_shortlists_scanned++;
            }

            if (pq_enabled && (!vid_to_pq_code.empty() || pq_codes_on_disk())) {
                // Time PQ table build on first call only
                if (enable_profiling && !pq_table_built) {
                    auto t_pq_tbl = std::chrono::high_resolution_clock::now();
                    compute_pq_distances(*this, it->second.data, x, sorted_cands);
                    auto t_pq_end = std::chrono::high_resolution_clock::now();
                    // Rough split: table build ≈ 20% of first compute_pq_distances call
                    // More precise measurement would require modifying compute_pq_distances
                    last_search_profile.pq_table_build_time_ms +=
                        std::chrono::duration<double, std::milli>(t_pq_end - t_pq_tbl).count() * 0.2;
                    last_search_profile.pq_distance_compute_time_ms +=
                        std::chrono::duration<double, std::milli>(t_pq_end - t_pq_tbl).count() * 0.8;
                    pq_table_built = true;
                } else if (enable_profiling) {
                    auto t_pq = std::chrono::high_resolution_clock::now();
                    compute_pq_distances(*this, it->second.data, x, sorted_cands);
                    auto t_pq_end = std::chrono::high_resolution_clock::now();
                    last_search_profile.pq_distance_compute_time_ms +=
                        std::chrono::duration<double, std::milli>(t_pq_end - t_pq).count();
                } else {
                    compute_pq_distances(*this, it->second.data, x, sorted_cands);
                }
            } else {
                if (enable_profiling) {
                    auto t_exact = std::chrono::high_resolution_clock::now();
                    compute_dists_with_prefetch(*this, it->second.data, x, sorted_cands);
                    auto t_exact_end = std::chrono::high_resolution_clock::now();
                    last_search_profile.exact_distance_compute_time_ms +=
                        std::chrono::duration<double, std::milli>(t_exact_end - t_exact).count();
                } else {
                    compute_dists_with_prefetch(*this, it->second.data, x, sorted_cands);
                }
            }
            std::sort(sorted_cands.begin(), sorted_cands.end());

            auto t_merge = enable_profiling ? std::chrono::high_resolution_clock::now()
                                            : std::chrono::high_resolution_clock::time_point{};
            bool updated = cand_vectors.batch_insert(sorted_cands);
            if (enable_profiling) {
                auto t_merge_end = std::chrono::high_resolution_clock::now();
                last_search_profile.candidate_merge_time_ms +=
                    std::chrono::duration<double, std::milli>(t_merge_end - t_merge).count();
            }
            sorted_cands.clear();
            if (!updated) {
                break;
            }
        } else if (node->bf.contains(tid)) {
            if (enable_profiling) {
                last_search_profile.frontier_children_expanded++;
            }
            compute_child_scores_with_prefetch(*this, node, x, frontier);
        }
    }

    if (enable_profiling) {
        auto t_frontier_end = std::chrono::high_resolution_clock::now();
        last_search_profile.frontier_search_time_ms =
            std::chrono::duration<double, std::milli>(t_frontier_end - t_frontier_start).count();
    }

    // ── Phase 3: ADC Rerank ──
    if (pq_use_adc_rerank && (!vid_to_pq_code.empty() || pq_codes_on_disk())
        && !cand_vectors.vids.empty()) {
        auto t_rerank = enable_profiling ? std::chrono::high_resolution_clock::now()
                                         : std::chrono::high_resolution_clock::time_point{};

        size_t n_rerank = std::min(
            static_cast<size_t>(k * pq_rerank_topk_factor),
            cand_vectors.vids.size());

        if (enable_profiling) {
            last_search_profile.rerank_count = static_cast<int>(n_rerank);
        }

        RunningList exact_list(static_cast<int>(n_rerank));
        for (size_t i = 0; i < n_rerank; i++) {
            float exact_dist = compute_vector_distance(x, cand_vectors.vids[i]);
            exact_list.insert(cand_vectors.vids[i], exact_dist);
        }
        cand_vectors = std::move(exact_list);

        if (enable_profiling) {
            auto t_rerank_end = std::chrono::high_resolution_clock::now();
            last_search_profile.rerank_time_ms =
                std::chrono::duration<double, std::milli>(t_rerank_end - t_rerank).count();
        }
    }

    heap_heapify<HeapForL2>(k, distances, labels);

    auto n_results = std::min(static_cast<size_t>(k), cand_vectors.vids.size());
    for (size_t i = 0; i < n_results; i++) {
        labels[i] = id_allocator.get_label(cand_vectors.vids[i]);
        distances[i] = cand_vectors.dists[i];
    }

    heap_reorder<HeapForL2>(k, distances, labels);

    if (enable_profiling) {
        auto t_total_end = std::chrono::high_resolution_clock::now();
        last_search_profile.total_search_time_ms =
            std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
    }
}



void MultiTenantIndexIVFHierarchical::search_one(
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    using Candidate = std::pair<float, const TreeNode*>;

    auto node_priority = [&](TreeNode* node) {
        float dist = fvec_L2sqr(x, node->centroid, d);
        float var = node->variance.get_mean();
        float score = dist - this->variance_boost * var;
        return score;
    };

    MinHeap<Candidate> pq;
    pq.emplace(node_priority(tree_root), tree_root);

    int n_cand_vecs = 0;
    std::vector<Candidate> buckets;

    while (!pq.empty() && n_cand_vecs < nprobe) {
        auto [score, node] = pq.top();
        pq.pop();

        if (!node->children.empty()) {
            for (auto child : node->children) {
                pq.emplace(node_priority(child), child);
            }
        } else {
            float var = node->variance.get_mean();
            float dist = score + this->variance_boost * var;
            buckets.emplace_back(dist, node);
            n_cand_vecs += node->vector_indices.size();
        }
    }

    std::sort(buckets.begin(), buckets.end());

    heap_heapify<HeapForL2>(k, distances, labels);

    if (buckets.empty()) {
        return;
    }

    float min_buck_dist = buckets[0].first;
    for (auto [dist, node] : buckets) {
        if (dist > this->prune_thres * min_buck_dist) {
            break;
        }

        for (auto vid : node->vector_indices) {
            ext_vid_t lbl = id_allocator.get_label(vid);
            float dist = compute_vector_distance(x, vid);
            if (dist < distances[0]) {
                maxheap_replace_top(k, distances, labels, dist, lbl);
            }
        }
    }

    heap_reorder<HeapForL2>(k, distances, labels);
}

float MultiTenantIndexIVFHierarchical::compute_vector_distance(const float* query, int_vid_t vid) const {
    // Prefer flash (finalized), then buffer (build phase), then external storage
    if (flash_finalized) {
        std::vector<float> x(d);
        load_vector_from_flash(vid, x.data());
        return fvec_L2sqr(query, x.data(), d);
    }
    if (!raw_vectors_buffer.empty()) {
        idx_t buf_offset = vid_to_buffer_offset.at(vid);
        return fvec_L2sqr(query, raw_vectors_buffer.data() + buf_offset, d);
    }
    // Backward compat: external IndexFlat storage
    idx_t storage_idx = vid_to_buffer_offset.at(vid); // reuse mapping for storage
    return fvec_L2sqr(query, storage->get_xb() + storage_idx * d, d);
}

TreeNode* MultiTenantIndexIVFHierarchical::assign_vec_to_leaf(const float* x) {
    TreeNode* curr = tree_root;

    while (!curr->children.empty()) {
        idx_t child_id;
        float min_dist = std::numeric_limits<float>::max();

        for (auto i = 0; i < curr->children.size(); i++) {
            auto child_centroid = curr->children[i]->centroid;
            float dist = fvec_L2sqr(x, child_centroid, d);
            if (dist < min_dist) {
                min_dist = dist;
                child_id = i;
            }
        }

        curr = curr->children[child_id];
    }
    return curr;
}

std::vector<idx_t> MultiTenantIndexIVFHierarchical::get_vector_path(
        ext_vid_t label) const {
    std::vector<idx_t> path;
    int_vid_t vid = id_allocator.get_id(label);
    auto curr = tree_root;
    auto shift = sizeof(int_vid_t) * 8 - CURATOR_MAX_BRANCH_FACTOR_LOG2;
    while (!curr->children.empty()) {
        auto child_id = (vid >> shift) & (CURATOR_MAX_BRANCH_FACTOR - 1);
        path.push_back(child_id);
        curr = curr->children[child_id];
        shift -= CURATOR_MAX_BRANCH_FACTOR_LOG2;
    }
    return path;
}

void MultiTenantIndexIVFHierarchical::split_short_list(
        TreeNode* node,
        int_lid_t tid) {
    if (node->children.empty() ||
        node->shortlists.at(tid).size() <= max_sl_size) {
        return;
    }

    std::vector<std::vector<int_vid_t>> child_sls(node->children.size());
    for (int_vid_t vid : node->shortlists.at(tid)) {
        auto offset = sizeof(int_vid_t) * 8 -
                CURATOR_MAX_BRANCH_FACTOR_LOG2 * (node->level + 1);
        auto child_id = (vid >> offset) & (CURATOR_MAX_BRANCH_FACTOR - 1);
        child_sls[child_id].push_back(vid);
    }

    node->shortlists.erase(tid);
    for (size_t i = 0; i < node->children.size(); i++) {
        if (child_sls[i].empty()) {
            continue;
        }

        TreeNode* child = node->children[i];
        child->shortlists.emplace(tid, child_sls[i]);
        child->bf.insert(tid);
    }

    // in rare cases, the short list of a child node may still exceed the
    // threshold
    for (size_t i = 0; i < node->children.size(); i++) {
        if (child_sls[i].size() > max_sl_size) {
            split_short_list(node->children[i], tid);
        }
    }
}

bool MultiTenantIndexIVFHierarchical::merge_short_list(
        TreeNode* node,
        int_lid_t tid) {
    if (node->parent == nullptr) {
        return false;
    }

    size_t total_sl_size = 0;
    for (TreeNode* sibling : node->parent->children) {
        auto it = sibling->shortlists.find(tid);
        if (it != sibling->shortlists.end()) {
            total_sl_size += it->second.size();
        } else if (sibling->bf.contains(tid)) {
            return false;
        }
    }

    if (total_sl_size > max_sl_size) {
        return false;
    }

    ShortList combined_sl;
    for (TreeNode* sibling : node->parent->children) {
        auto it = sibling->shortlists.find(tid);
        if (it != sibling->shortlists.end()) {
            combined_sl = combined_sl.merge(it->second);
            sibling->shortlists.erase(tid);
            sibling->bf = sibling->recompute_bloom_filter();
        }
    }
    node->parent->shortlists.emplace(tid, combined_sl);

    return true;
}

void MultiTenantIndexIVFHierarchical::locate_vector(ext_vid_t label) const {
    auto path = get_vector_path(label);

    printf("Found vector %u at path: ", label);
    for (auto id : path) {
        printf("%lu ", id);
    }
    printf("\n");
}

void MultiTenantIndexIVFHierarchical::print_tree_info() const {
    size_t total_nodes = 0;
    size_t total_leaf_nodes = 0;
    double total_depth = 0;
    size_t total_vectors = 0;
    std::vector<size_t> leaf_sizes;

    std::queue<const TreeNode*> q;
    q.push(tree_root);

    while (!q.empty()) {
        auto node = q.front();
        q.pop();

        total_nodes++;
        if (node->children.empty()) {
            total_leaf_nodes++;
            total_depth += node->level;
            leaf_sizes.push_back(node->vector_indices.size());
            total_vectors += node->vector_indices.size();
        } else {
            for (auto child : node->children) {
                q.push(child);
            }
        }
    }

    double leaf_size_avg = total_vectors / total_leaf_nodes;
    double leaf_size_var = 0;
    for (size_t size : leaf_sizes) {
        leaf_size_var += (size - leaf_size_avg) * (size - leaf_size_avg);
    }
    leaf_size_var /= total_leaf_nodes;
    double leaf_size_std = sqrt(leaf_size_var);

    auto max_leaf_size =
            *std::max_element(leaf_sizes.begin(), leaf_sizes.end());
    std::vector<size_t> leaf_size_hist(max_leaf_size + 1, 0);
    for (size_t size : leaf_sizes) {
        leaf_size_hist[size]++;
    }

    printf("Total number of tree nodes: %lu\n", total_nodes);
    printf("Total number of leaf nodes: %lu\n", total_leaf_nodes);
    printf("Average depth of leaf nodes: %.2f\n",
           total_depth / total_leaf_nodes);

    printf("Average leaf node size: %.2f\n", leaf_size_avg);
    printf("Standard deviation of leaf node sizes: %.2f\n", leaf_size_std);
    printf("Leaf node size histogram: ");
    for (size_t i = 0; i < leaf_size_hist.size(); i++) {
        if (leaf_size_hist[i] > 0) {
            printf("(%lu, %lu) ", i, leaf_size_hist[i]);
        }
    }
    fflush(stdout);
}

std::string MultiTenantIndexIVFHierarchical::convert_complex_predicate(
        const std::string& filter) const {
    using namespace complex_predicate;

    std::string converted_filter;
    auto tokens = tokenize_formula(filter);

    for (auto i = 0; i < tokens.size(); i++) {
        auto token = tokens[i];

        if (token != "AND" && token != "OR" && token != "NOT") {
            int_lid_t tenant_id = tid_allocator.get_id(std::stol(token));
            token = std::to_string(tenant_id);
        }

        converted_filter = converted_filter + token;
        if (i < tokens.size() - 1) {
            converted_filter = converted_filter + " ";
        }
    }

    return converted_filter;
}

std::vector<int_vid_t> MultiTenantIndexIVFHierarchical::find_all_qualified_vecs(
        const std::string& filter) const {
    using namespace complex_predicate;
    using Candidate = std::tuple<const TreeNode*, VarMap, State>;

    auto update_var_map = [&](const TreeNode* node, VarMap var_map) -> VarMap {
        auto new_var_map = std::unordered_map<std::string, State>();

        for (const auto& var : var_map->unresolved_vars()) {
            int_lid_t tid = std::stoi(var);
            auto it = node->shortlists.find(tid);
            if (it != node->shortlists.end()) {
                auto state = make_state(Type::SOME, true, it->second.data);
                new_var_map[var] = state;
            } else if (!node->bf.contains(tid)) {
                new_var_map[var] = STATE_NONE;
            }
        }

        return var_map->update(new_var_map);
    };

    auto remove_external_vecs = [&](const TreeNode* node,
                                    const Buffer& buffer) -> Buffer {
        auto shift = sizeof(int_vid_t) * 8 -
                CURATOR_MAX_BRANCH_FACTOR_LOG2 * node->level;
        auto shifted_node_id = node->node_id >> shift;

        Buffer new_buffer;
        for (auto vid : buffer) {
            if ((vid >> shift) == shifted_node_id) {
                new_buffer.push_back(vid);
            }
        }

        return new_buffer;
    };

    auto var_map_data = std::unordered_map<std::string, State>();
    auto filter_expr = parse_formula(filter, &var_map_data);
    auto var_map = make_var_map(std::move(var_map_data));

    std::vector<int_vid_t> qual_vecs;
    std::queue<Candidate> frontier;
    frontier.emplace(tree_root, var_map, STATE_UNKNOWN);

    while (!frontier.empty()) {
        auto [node, vmap, state] = frontier.front();
        frontier.pop();

        if (*state == Type::UNKNOWN) {
            vmap = update_var_map(node, vmap);
            state = filter_expr->evaluate(vmap, false);
        }

        if (*state == Type::NONE) {
            continue;
        }

        if (*state == Type::SOME) {
            state = filter_expr->evaluate(vmap, true);
            auto buffer = remove_external_vecs(node, state->short_list);
            qual_vecs.insert(qual_vecs.end(), buffer.begin(), buffer.end());
            continue;
        }

        if (node->children.empty()) {
            if (*state == Type::ALL) {
                auto& buffer = node->vector_indices.data;
                qual_vecs.insert(qual_vecs.end(), buffer.begin(), buffer.end());
            } else if (*state == Type::MOST) {
                state = filter_expr->evaluate(vmap, true);
                auto buffer = remove_external_vecs(node, state->exclude_list);
                auto& leaf_vecs = node->vector_indices.data;
                auto diff = buffer_difference(leaf_vecs, buffer);
                qual_vecs.insert(qual_vecs.end(), diff.begin(), diff.end());
            } else {
                // printf("False positive in bloom filter\n");
            }
        } else {
            for (auto child : node->children) {
                frontier.emplace(child, vmap, state);
            }
        }
    }

    return qual_vecs;
}

void MultiTenantIndexIVFHierarchical::batch_grant_access(
        const std::vector<int_vid_t>& vids,
        int_lid_t tid) {
    tree_root->shortlists.emplace(tid, vids);
    tree_root->bf.insert(tid);
    split_short_list(tree_root, tid);
}

void MultiTenantIndexIVFHierarchical::build_index_for_filter(
        const ext_vid_t* qualified_labels, 
        size_t qualified_labels_size,
        const std::string& filter_key) {
    
    if (qualified_labels_size == 0) {
        return;
    }

    // Convert external labels to internal vector IDs
    std::vector<int_vid_t> qualified_vecs;
    qualified_vecs.reserve(qualified_labels_size);
    
    for (size_t i = 0; i < qualified_labels_size; i++) {
        ext_vid_t label = qualified_labels[i];
        int_vid_t vid = id_allocator.get_id(label);
        qualified_vecs.push_back(vid);
    }
    
    if (qualified_vecs.empty()) {
        return;
    }

    // Always allocate new tenant ID for this filter (both approaches need this)
    ext_lid_t filter_label = tid_allocator.allocate_reserved_label();
    int_lid_t filter_tid = tid_allocator.allocate_id(filter_label);

    // Store the mapping for this filter using the provided key
    filter_to_label.emplace(filter_key, filter_label);

    if (use_temp_index_caching) {
        // Alternative approach: Cache temporary index structures without modifying main index
        // Sort qualified vectors' IDs in ascending order for efficient tree building
        std::sort(qualified_vecs.begin(), qualified_vecs.end());
        
        // Build temporary index structure
        std::vector<complex_predicate::TempIndexNode> temp_nodes;
        complex_predicate::build_temp_index_for_filter(this, qualified_vecs, temp_nodes);
        
        // Cache the temporary index and qualified vectors for this filter (using tenant ID as key)
        cached_temp_indexes.emplace(filter_tid, std::move(temp_nodes));
        cached_qualified_vecs.emplace(filter_tid, std::move(qualified_vecs));
    } else {
        // Original approach: Direct indexing using batch_grant_access
        // Grant access to all qualified vectors
        batch_grant_access(qualified_vecs, filter_tid);
    }
}

ext_lid_t MultiTenantIndexIVFHierarchical::get_filter_label(const std::string& filter_key) const {
    auto it = filter_to_label.find(filter_key);
    if (it == filter_to_label.end()) {
        FAISS_THROW_FMT("Filter key '%s' not found. Make sure to call build_index_for_filter first.", filter_key.c_str());
    }
    
    ext_lid_t filter_label = it->second;
    return filter_label;
}

bool MultiTenantIndexIVFHierarchical::get_cached_temp_index_data(
        ext_lid_t tid, 
        std::vector<int_vid_t>*& qualified_vecs, 
        std::vector<complex_predicate::TempIndexNode>*& temp_nodes) const {
    if (!use_temp_index_caching) {
        return false;
    }
    
    // Direct lookup using tenant ID as key
    auto temp_index_it = cached_temp_indexes.find(tid);
    auto qualified_vecs_it = cached_qualified_vecs.find(tid);
    
    if (temp_index_it != cached_temp_indexes.end() && 
        qualified_vecs_it != cached_qualified_vecs.end()) {
        qualified_vecs = const_cast<std::vector<int_vid_t>*>(&qualified_vecs_it->second);
        temp_nodes = const_cast<std::vector<complex_predicate::TempIndexNode>*>(&temp_index_it->second);
        return true;
    }
    
    return false;
}

size_t MultiTenantIndexIVFHierarchical::get_cached_temp_index_memory_usage() const {
    size_t total_memory = 0;
    
    for (const auto& [tenant_id, temp_nodes] : cached_temp_indexes) {
        // Memory for TempIndexNode structures
        total_memory += temp_nodes.size() * sizeof(complex_predicate::TempIndexNode);
        
        // Memory for children vectors in each TempIndexNode
        for (const auto& node : temp_nodes) {
            total_memory += node.children.size() * sizeof(int);
        }
    }
    
    for (const auto& [tenant_id, qualified_vecs] : cached_qualified_vecs) {
        // Memory for qualified vector IDs
        total_memory += qualified_vecs.size() * sizeof(int_vid_t);
    }
    
    return total_memory;
}

namespace {
bool check_bloom_filter(
        const MultiTenantIndexIVFHierarchical& index,
        const TreeNode& node) {
    for (auto child : node.children) {
        if (!check_bloom_filter(index, *child)) {
            return false;
        }
    }

    auto expected_bf = node.recompute_bloom_filter();
    return node.bf == expected_bf;
}

std::pair<bool, std::set<int_lid_t>> check_shortlists(
        const MultiTenantIndexIVFHierarchical& index,
        const TreeNode& node) {
    // recursively check the shortlists in the descendant nodes

    std::set<int_lid_t> shortlists_in_desc;
    for (auto child : node.children) {
        auto [success, sls] = check_shortlists(index, *child);
        if (!success) {
            return {false, {}};
        }
        shortlists_in_desc.insert(sls.begin(), sls.end());
    }

    // check 1. shortlist size should not exceed the threshold

    for (const auto& [tenant, shortlist] : node.shortlists) {
        if (shortlist.size() > index.max_sl_size) {
            printf("Oversized shortlist\n");
            return {false, {}};
        }
    }

    // check 2. shortlist merging should be done correctly

    if (!node.children.empty()) {
        std::set<int_lid_t> all_tenants;
        for (auto child : node.children) {
            for (const auto& [tenant, shortlist] : child->shortlists) {
                all_tenants.insert(tenant);
            }
        }

        for (auto tenant : all_tenants) {
            size_t total_sl_size = 0;
            for (auto child : node.children) {
                if (!child->bf.contains(tenant)) {
                    continue;
                } else if (
                        child->shortlists.find(tenant) !=
                        child->shortlists.end()) {
                    total_sl_size += child->shortlists.at(tenant).size();
                } else {
                    total_sl_size += index.max_sl_size + 1;
                    break;
                }
            }

            if (total_sl_size <= index.max_sl_size) {
                printf("Fail to merge\n");
                return {false, {}};
            }
        }
    }

    // check 3. there should not be two shortlists of the same tenant in any
    // path

    std::set<int_lid_t> shortlists_in_node;
    for (const auto& [tenant, shortlist] : node.shortlists) {
        shortlists_in_node.insert(tenant);
    }

    std::set<int_lid_t> intersect;
    std::set_intersection(
            shortlists_in_desc.begin(),
            shortlists_in_desc.end(),
            shortlists_in_node.begin(),
            shortlists_in_node.end(),
            std::inserter(intersect, intersect.begin()));

    if (intersect.size() > 0) {
        printf("Duplicate shortlists\n");
        return {false, {}};
    }

    std::set<int_lid_t> all_shortlists;
    std::set_union(
            shortlists_in_desc.begin(),
            shortlists_in_desc.end(),
            shortlists_in_node.begin(),
            shortlists_in_node.end(),
            std::inserter(all_shortlists, all_shortlists.begin()));

    return {true, all_shortlists};
}
} // namespace

void MultiTenantIndexIVFHierarchical::sanity_check() const {
    printf("Sanity check:\n");
    printf("Checking bloom filters...\n");
    check_bloom_filter(*this, *tree_root);
    printf("Checking shortlists...\n");
    check_shortlists(*this, *tree_root);
}

TreeNode* MultiTenantIndexIVFHierarchical::find_assigned_leaf(
        ext_vid_t label) const {
    int_vid_t vid = id_allocator.get_id(label);
    auto curr = tree_root;
    auto shift = sizeof(int_vid_t) * 8 - CURATOR_MAX_BRANCH_FACTOR_LOG2;
    while (!curr->children.empty()) {
        auto child_id = (vid >> shift) & (CURATOR_MAX_BRANCH_FACTOR - 1);
        curr = curr->children[child_id];
        shift -= CURATOR_MAX_BRANCH_FACTOR_LOG2;
    }
    return curr;
}

void MultiTenantIndexIVFHierarchical::memory_usage() const {
    auto unordered_map_memory_usage = [](const auto& map) {
        const auto& [key, value] = *map.begin();
        size_t per_elem = sizeof(key) + sizeof(value) + sizeof(void*);
        return map.bucket_count() * sizeof(void*) + map.size() * per_elem;
    };

    auto vector_memory_usage = [](const auto& vec) {
        return vec.capacity() * sizeof(*vec.begin());
    };

    auto vector_payload_size = [](const auto& vec) {
        return vec.size() * sizeof(*vec.begin());
    };

    auto bloom_filter_size = [](bloom_filter& bf) {
        return sizeof(unsigned int) * bf.hash_count() + bf.size() / 8 +
                sizeof(unsigned int) + 4 * sizeof(unsigned long long int) +
                sizeof(double);
    };

    auto short_list_size = [&](const ShortList& sl) {
        return vector_memory_usage(sl.data);
    };

    auto short_lists_size = [&](const auto& shortlists) {
        size_t size = unordered_map_memory_usage(shortlists);
        for (const auto& [lid, sl] : shortlists) {
            size += short_list_size(sl);
        }
        return size;
    };

    auto short_lists_payload_size = [&](const auto& shortlists) {
        size_t size = 0;
        for (const auto& [lid, sl] : shortlists) {
            size += vector_payload_size(sl.data);
        }
        return size;
    };

    auto node_attrs_size = [&](const TreeNode* node) {
        return sizeof(size_t) * 3 + sizeof(void*) * 2 + sizeof(int_vid_t) +
                sizeof(float) + sizeof(RunningMean) + sizeof(node->children) +
                sizeof(node->shortlists);
    };

    size_t id_allocator_size =
            unordered_map_memory_usage(id_allocator.label_to_id) +
            unordered_map_memory_usage(id_allocator.id_to_label);

    size_t tid_allocator_size =
            unordered_map_memory_usage(tid_allocator.label_to_id) +
            vector_memory_usage(tid_allocator.id_to_label);

    size_t id_allocator_total_size = id_allocator_size + tid_allocator_size;

    size_t num_tree_nodes = 0;
    size_t bloom_filter_total_size = 0;
    size_t short_lists_total_size = 0;
    size_t short_lists_payload_total_size = 0;
    size_t vector_indices_total_size = 0;
    size_t centroid_total_size = 0;
    size_t node_attrs_total_size = 0;

    std::queue<TreeNode*> q;
    q.push(tree_root);

    while (!q.empty()) {
        TreeNode* node = q.front();
        q.pop();

        num_tree_nodes++;
        bloom_filter_total_size += bloom_filter_size(node->bf);
        short_lists_total_size += short_lists_size(node->shortlists);
        short_lists_payload_total_size +=
                short_lists_payload_size(node->shortlists);
        centroid_total_size += sizeof(float) * this->d;
        node_attrs_total_size += node_attrs_size(node);

        if (node->children.empty()) {
            vector_indices_total_size += short_list_size(node->vector_indices);
        } else {
            for (TreeNode* child : node->children) {
                q.push(child);
            }
        }
    }

    size_t total_memory_usage = 0;
    total_memory_usage = sizeof(MultiTenantIndexIVFHierarchical);
    total_memory_usage += id_allocator_total_size;
    total_memory_usage += bloom_filter_total_size;
    total_memory_usage += short_lists_total_size;
    total_memory_usage += vector_indices_total_size;
    total_memory_usage += centroid_total_size;
    total_memory_usage += node_attrs_total_size;

    printf("Memory usage breakdown:\n");
    printf("Number of tree nodes: %lu\n", num_tree_nodes);
    printf("Total memory usage: %lu bytes\n", total_memory_usage);
    printf("ID allocator: %lu bytes\n", id_allocator_total_size);
    printf("Bloom filters: %lu bytes\n", bloom_filter_total_size);
    printf("Short lists: %lu bytes\n", short_lists_total_size);
    printf("Short lists payload: %lu bytes\n", short_lists_payload_total_size);
    printf("Vector indices: %lu bytes\n", vector_indices_total_size);
    printf("Centroids: %lu bytes\n", centroid_total_size);
    printf("Node attributes: %lu bytes\n", node_attrs_total_size);
}

// ============================================================
//  Component-level memory breakdown (structured, returns data)
// ============================================================

MultiTenantIndexIVFHierarchical::MemoryBreakdown MultiTenantIndexIVFHierarchical::get_memory_breakdown() const {
    MemoryBreakdown bd;

    auto unordered_map_memory_usage = [](const auto& map) {
        if (map.empty()) return size_t(0);
        const auto& [key, value] = *map.begin();
        size_t per_elem = sizeof(key) + sizeof(value) + sizeof(void*);
        return map.bucket_count() * sizeof(void*) + map.size() * per_elem;
    };

    auto vector_capacity_bytes = [](const auto& vec) {
        return vec.capacity() * sizeof(typename std::remove_reference_t<decltype(vec)>::value_type);
    };

    auto vector_payload_bytes = [](const auto& vec) {
        return vec.size() * sizeof(typename std::remove_reference_t<decltype(vec)>::value_type);
    };

    auto bloom_filter_bytes = [](const bloom_filter& bf) -> size_t {
        return sizeof(unsigned int) * bf.hash_count() + bf.size() / 8 +
               sizeof(unsigned int) + 4 * sizeof(unsigned long long int) +
               sizeof(double);
    };

    // Walk the tree
    std::queue<TreeNode*> q;
    q.push(tree_root);

    while (!q.empty()) {
        TreeNode* node = q.front();
        q.pop();

        bd.num_tree_nodes++;

        // Node attributes (fixed-size fields of TreeNode)
        bd.tree_node_attrs_bytes += sizeof(size_t) * 3    // level, sibling_id, bf_capacity
                                  + sizeof(void*) * 2      // parent pointer + centroid pointer
                                  + sizeof(int_vid_t)       // node_id
                                  + sizeof(float)           // bf_false_pos
                                  + sizeof(RunningMean)     // variance
                                  + sizeof(decltype(node->children))
                                  + sizeof(decltype(node->shortlists));

        // Centroid
        bd.centroids_bytes += d * sizeof(float);

        // Bloom filter
        bd.bloom_filter_bytes += bloom_filter_bytes(node->bf);

        // Short lists
        bd.shortlists_overhead_bytes += unordered_map_memory_usage(node->shortlists);
        for (const auto& [lid, sl] : node->shortlists) {
            bd.shortlists_payload_bytes += vector_payload_bytes(sl.data);
        }

        // Vector indices (leaf only)
        if (node->children.empty()) {
            bd.vector_indices_bytes += vector_capacity_bytes(node->vector_indices.data);
        } else {
            for (TreeNode* child : node->children) {
                q.push(child);
            }
        }
    }

    // ID allocators
    bd.id_allocator_bytes =
        unordered_map_memory_usage(id_allocator.label_to_id) +
        unordered_map_memory_usage(id_allocator.id_to_label);
    bd.tenant_id_allocator_bytes =
        unordered_map_memory_usage(tid_allocator.label_to_id) +
        vector_capacity_bytes(tid_allocator.id_to_label);

    // PQ
    bd.pq_codebook_bytes = pq_codebook.size() * sizeof(float);
    for (const auto& code : vid_to_pq_code) {
        bd.pq_codes_bytes += code.size() * sizeof(uint8_t);
    }

    // Flash index maps
    bd.flash_index_bytes =
        leaf_node_id_to_seq.size() * (sizeof(int_vid_t) + sizeof(size_t)) +
        vid_to_buffer_offset.size() * (sizeof(int_vid_t) + sizeof(size_t)) +
        vid_to_leaf_node_id.size() * (sizeof(int_vid_t) + sizeof(int_vid_t)) +
        vid_to_local_index.size() * (sizeof(int_vid_t) + sizeof(size_t)) +
        vid_to_seq.size() * (sizeof(int_vid_t) + sizeof(size_t));

    // Raw vectors buffer (0 after flush)
    bd.raw_vectors_buffer_bytes = raw_vectors_buffer.size() * sizeof(float);

    // Temp index cache
    for (const auto& [tid, nodes] : cached_temp_indexes) {
        bd.temp_index_cache_bytes += nodes.size() * sizeof(complex_predicate::TempIndexNode);
        for (const auto& node : nodes) {
            bd.temp_index_cache_bytes += node.children.size() * sizeof(int);
        }
    }
    for (const auto& [tid, vecs] : cached_qualified_vecs) {
        bd.temp_qualified_vecs_bytes += vecs.size() * sizeof(int_vid_t);
    }

    // Grand total (matches get_total_memory_bytes)
    bd.total_bytes = get_total_memory_bytes();

    return bd;
}

namespace complex_predicate {

// This search algorithm is a modified version of the single-label filtering
// algorithm using search_ef to terminate the search. Beam search is used to
// initialize the search frontier. Variance boost is disabled for simplicity.
void search_temp_index(
        const MultiTenantIndexIVFHierarchical* index,
        const std::vector<int_vid_t>& qualified_vecs,
        const std::vector<TempIndexNode>& nodes,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) {
    using Candidate = std::pair<float, int>; // (score, node idx in temp index)

    // Return if the temp index is empty
    if (nodes.empty()) {
        return;
    }
    if (index->beam_size == 0) {
        FAISS_THROW_MSG("beam_size must be greater than 0");
    }

    /* STAGE 1: BEAM SEARCH */

    std::vector<Candidate> beam;       // beam of current step
    std::vector<Candidate> next_beam;  // beam of next step
    std::vector<Candidate> unexpanded; // unexpanded nodes during beam search

    // Initialize beam with root node
    float score = fvec_L2sqr(x, nodes[0].centroid, index->d);
    beam.emplace_back(score, 0);

    while (true) {
        bool updated = false; // if beam is updated
        for (auto [score, node_idx] : beam) {
            const TempIndexNode& node = nodes[node_idx];
            if (node.children.empty()) {
                next_beam.emplace_back(score, node_idx);
            } else {
                updated = true;
                for (int child_idx : node.children) {
                    // Variance boost is disabled for simplicity
                    float child_score =
                            fvec_L2sqr(x, nodes[child_idx].centroid, index->d);
                    next_beam.emplace_back(child_score, child_idx);
                }
            }
        }

        // Break if beam is not updated (i.e., next beam = current beam)
        if (!updated) {
            break;
        }

        // Only expand the top-beam-width nodes in the next step
        // Move the rest of the nodes to unexpanded list
        std::sort(next_beam.begin(), next_beam.end());

        auto n_keep = std::min(index->beam_size, next_beam.size());
        for (size_t i = n_keep; i < next_beam.size(); i++) {
            unexpanded.push_back(next_beam[i]);
        }
        next_beam.resize(n_keep);

        std::swap(beam, next_beam);
        next_beam.clear();
    }

    // Check for overlaps between beam and unexpanded
    std::set<int> beam_nodes;
    std::set<int> unexpanded_nodes;

    for (const auto& [score, node_idx] : beam) {
        beam_nodes.insert(node_idx);
    }

    for (const auto& [score, node_idx] : unexpanded) {
        unexpanded_nodes.insert(node_idx);
    }

    /* STAGE 2: BEST-FIRST SEARCH */
    // Limited size result set. The final top-k candidates will be returned
    RunningList cand_vectors(index->search_ef);

    // Temporary array holding vectors to be merged into the result set
    std::vector<std::pair<float, int_vid_t>> sorted_cands;
    sorted_cands.reserve(index->max_sl_size);

    // Initialize search frontier with the frontier from beam search
    // i.e., nodes in the beam of the last step and all nodes that are
    // visited but not expanded
    MinHeap<Candidate> frontier;

    for (auto& cand : unexpanded) {
        frontier.push(cand);
    }
    for (auto& cand : beam) {
        frontier.push(cand);
    }

    // Main loop of the search algorithm
    while (!frontier.empty()) {
        auto [score, node_idx] = frontier.top();
        frontier.pop();

        const TempIndexNode& node = nodes[node_idx];

        // Only process vectors if this is a leaf node (no children)
        if (node.children.empty()) {
            const int_vid_t* node_vids_ptr = qualified_vecs.data() + node.start;
            size_t node_vids_count = node.end - node.start;
            if (index->pq_enabled && (!index->vid_to_pq_code.empty() || index->pq_codes_on_disk())) {
                compute_pq_distances(*index, node_vids_ptr, node_vids_count, x, sorted_cands);
            } else {
                compute_dists_with_prefetch(*index, node_vids_ptr, node_vids_count, x, sorted_cands);
            }

            if (!sorted_cands.empty()) {
                std::sort(sorted_cands.begin(), sorted_cands.end());

                // Break if the result set cannot be improved by any vector in
                // the buffer
                bool updated = cand_vectors.batch_insert(sorted_cands);
                sorted_cands.clear();
                if (!updated) {
                    break;
                }
            }
        } else {
            // Add child nodes to frontier
            for (int child_idx : node.children) {
                float child_score =
                        fvec_L2sqr(x, nodes[child_idx].centroid, index->d);
                frontier.emplace(child_score, child_idx);
            }
        }
    }


    // Extract the top-k candidates directly from the sorted candidate list
    auto n_results = std::min(static_cast<size_t>(k), cand_vectors.vids.size());

    // Initialize unused slots with sentinel values
    for (size_t i = 0; i < k; i++) {
        distances[i] = std::numeric_limits<float>::max();
        labels[i] = -1;
    }

    // Copy results in sorted order (best distances first)
    for (size_t i = 0; i < n_results; i++) {
        labels[i] = index->id_allocator.get_label(cand_vectors.vids[i]);
        distances[i] = cand_vectors.dists[i];
    }
}

} // namespace complex_predicate



void MultiTenantIndexIVFHierarchical::search_with_bitmap_filter(
        idx_t n,
        const float* x,
        idx_t k,
        const ext_vid_t* qualified_labels,
        size_t qualified_labels_size,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // Initialize profiling data only if profiling is enabled
    if (enable_profiling) {
        last_search_profile = SearchProfilingData{};
        last_search_profile.qualified_labels_count = qualified_labels_size;
    }

    if (qualified_labels_size == 0) {
        heap_heapify<HeapForL2>(k, distances, labels);
        return;
    }

    // PHASE 1: Preprocessing - Convert external labels to internal vector IDs
    auto preproc_start = enable_profiling ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
    std::vector<int_vid_t> qualified_vecs;
    qualified_vecs.reserve(qualified_labels_size);
    
    for (size_t i = 0; i < qualified_labels_size; i++) {
        ext_vid_t label = qualified_labels[i];
        if (id_allocator.label_to_id.find(label) != id_allocator.label_to_id.end()) {
            int_vid_t vid = id_allocator.get_id(label);
            qualified_vecs.push_back(vid);
        }
    }
    
    if (enable_profiling) {
        auto preproc_end = std::chrono::high_resolution_clock::now();
        last_search_profile.preproc_time_ms = std::chrono::duration<double, std::milli>(preproc_end - preproc_start).count();
    }

    if (qualified_vecs.empty()) {
        heap_heapify<HeapForL2>(k, distances, labels);
        return;
    }

    // PHASE 2: Sorting - Sort qualified vectors' IDs in ascending order
    auto sort_start = enable_profiling ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
    std::sort(qualified_vecs.begin(), qualified_vecs.end());
    
    if (enable_profiling) {
        auto sort_end = std::chrono::high_resolution_clock::now();
        last_search_profile.sort_time_ms = std::chrono::duration<double, std::milli>(sort_end - sort_start).count();
    }

    // PHASE 3: Build temporary index
    auto build_start = enable_profiling ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
    std::vector<complex_predicate::TempIndexNode> nodes;
    complex_predicate::build_temp_index_for_filter(this, qualified_vecs, nodes);
    
    if (enable_profiling) {
        auto build_end = std::chrono::high_resolution_clock::now();
        last_search_profile.build_temp_index_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
        last_search_profile.temp_nodes_count = nodes.size();
    }

    // PHASE 4: Search the temporary index
    auto search_start = enable_profiling ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
    for (idx_t i = 0; i < n; i++) {
        complex_predicate::search_temp_index(
                this,
                qualified_vecs,
                nodes,
                x + i * d,
                k,
                distances + i * k,
                labels + i * k,
                params);
    }
    
    if (enable_profiling) {
        auto search_end = std::chrono::high_resolution_clock::now();
        last_search_profile.search_time_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();
    }
}

// Optimized version that takes sorted internal vector IDs directly
void MultiTenantIndexIVFHierarchical::search_with_bitmap_filter_optimized(
        idx_t n,
        const float* x,
        idx_t k,
        const int_vid_t* sorted_qualified_vids,
        size_t sorted_qualified_vids_size,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // Initialize profiling data only if profiling is enabled
    if (enable_profiling) {
        last_search_profile = SearchProfilingData{};
        last_search_profile.qualified_labels_count = sorted_qualified_vids_size;
        // Skip preprocessing and sorting phases (set to 0)
        last_search_profile.preproc_time_ms = 0.0;
        last_search_profile.sort_time_ms = 0.0;
    }

    if (sorted_qualified_vids_size == 0) {
        heap_heapify<HeapForL2>(k, distances, labels);
        return;
    }

    // Convert array to vector for compatibility with existing functions
    std::vector<int_vid_t> qualified_vecs(sorted_qualified_vids, sorted_qualified_vids + sorted_qualified_vids_size);

    // Verify that the input is actually sorted (debug assertion)
    FAISS_THROW_IF_NOT_MSG(
        std::is_sorted(qualified_vecs.begin(), qualified_vecs.end()),
        "Input sorted_qualified_vids must be sorted in ascending order");

    // PHASE 3: Build temporary index (skip phases 1 and 2)
    auto build_start = enable_profiling ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
    std::vector<complex_predicate::TempIndexNode> nodes;
    complex_predicate::build_temp_index_for_filter(this, qualified_vecs, nodes);
    
    if (enable_profiling) {
        auto build_end = std::chrono::high_resolution_clock::now();
        last_search_profile.build_temp_index_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
        last_search_profile.temp_nodes_count = nodes.size();
    }

    // PHASE 4: Search the temporary index
    auto search_start = enable_profiling ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
    for (idx_t i = 0; i < n; i++) {
        complex_predicate::search_temp_index(
                this,
                qualified_vecs,
                nodes,
                x + i * d,
                k,
                distances + i * k,
                labels + i * k,
                params);
    }
    
    if (enable_profiling) {
        auto search_end = std::chrono::high_resolution_clock::now();
        last_search_profile.search_time_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();
    }
}

// Get mapping from external labels to internal vector IDs for Python interface
void MultiTenantIndexIVFHierarchical::get_label_to_vid_mapping(
        const ext_vid_t* labels,
        size_t labels_size,
        int_vid_t* vids) const {

    for (size_t i = 0; i < labels_size; i++) {
        ext_vid_t label = labels[i];
        if (id_allocator.label_to_id.find(label) != id_allocator.label_to_id.end()) {
            vids[i] = id_allocator.get_id(label);
        } else {
            vids[i] = -1;
        }
    }
}

// ============================================================
//  PQ (Product Quantization) methods
// ============================================================

void MultiTenantIndexIVFHierarchical::set_pq_config(
        size_t M,
        size_t nbits,
        bool enabled,
        bool use_adc_rerank,
        size_t rerank_topk_factor) {
    pq_M = M;
    pq_nbits = nbits;
    pq_enabled = enabled;
    pq_use_adc_rerank = use_adc_rerank;
    pq_rerank_topk_factor = rerank_topk_factor;
}

void MultiTenantIndexIVFHierarchical::train_pq_codebook() {
    if (!pq_enabled || pq_M == 0 || pq_nbits == 0) {
        return;
    }

    size_t ntotal = seq_to_vid.size();
    if (ntotal == 0) {
        return;
    }

    // Train PQ codebook on original vectors (ADC-compatible)
    ProductQuantizer pq(d, pq_M, pq_nbits);
    pq.train(ntotal, raw_vectors_buffer.data());

    // Store codebook
    size_t ksub = 1 << pq_nbits;
    size_t dsub = d / pq_M;
    pq_codebook.resize(pq_M * ksub * dsub);
    std::memcpy(pq_codebook.data(), pq.get_centroids(0, 0), pq_codebook.size() * sizeof(float));

    // Encode all vectors
    vid_to_pq_code.resize(ntotal);
    for (size_t i = 0; i < ntotal; i++) {
        const float* xi = raw_vectors_buffer.data() + vid_to_buffer_offset.at(seq_to_vid[i]);
        std::vector<uint8_t> code(pq_M);
        pq.compute_code(xi, code.data());
        vid_to_pq_code[i] = std::move(code);
    }

    printf("[Curator] PQ codebook trained: M=%zu, nbits=%zu, ntotal=%zu\n",
           pq_M, pq_nbits, ntotal);
}

// ============================================================
//  Flash Storage methods
// ============================================================

void MultiTenantIndexIVFHierarchical::set_flash_storage_path(const std::string& path) {
    flash_storage_path = path;

    std::filesystem::create_directories(
            std::filesystem::path(flash_storage_path).parent_path());

    flash_fp = std::fopen(flash_storage_path.c_str(), "w+b");
    if (flash_fp == nullptr) {
        FAISS_THROW_MSG("Failed to open flash storage file: " + flash_storage_path);
    }
}

void MultiTenantIndexIVFHierarchical::set_use_flash_storage(bool use) {
    use_flash_storage = use;
}

// ============================================================
//  PQ External Storage methods (Goal 1)
// ============================================================

void MultiTenantIndexIVFHierarchical::set_pq_codes_path(const std::string& path) {
    pq_codes_path = path;
}

void MultiTenantIndexIVFHierarchical::set_persist_pq_codes(bool persist) {
    persist_pq_codes = persist;
}

bool MultiTenantIndexIVFHierarchical::pq_codes_in_memory() const {
    return !vid_to_pq_code.empty();
}

bool MultiTenantIndexIVFHierarchical::pq_codes_on_disk() const {
    return pq_codes_mmap != nullptr && pq_codes_mmap != MAP_FAILED;
}

const uint8_t* MultiTenantIndexIVFHierarchical::get_pq_code_by_seq(size_t seq_idx) const {
    // If codes are in memory, return pointer directly
    if (!vid_to_pq_code.empty()) {
        return vid_to_pq_code[seq_idx].data();
    }
    // If codes are on disk (mmap'd), compute offset and return pointer
    if (pq_codes_mmap != nullptr && pq_codes_mmap != MAP_FAILED) {
        size_t code_bytes = pq_M * pq_nbits / 8;
        size_t body_offset = 32; // header size
        size_t offset = body_offset + seq_idx * code_bytes;
        return static_cast<const uint8_t*>(pq_codes_mmap) + offset;
    }
    // No codes available
    FAISS_THROW_MSG("PQ codes not available in memory or on disk");
}

void MultiTenantIndexIVFHierarchical::write_pq_codes_to_disk() {
    if (vid_to_pq_code.empty()) {
        FAISS_THROW_MSG("No PQ codes to write");
    }
    if (pq_codes_path.empty()) {
        FAISS_THROW_MSG("PQ codes path not set");
    }

    size_t ntotal = seq_to_vid.size();
    size_t code_bytes = pq_M * pq_nbits / 8;

    // Create parent directories if needed
    std::filesystem::create_directories(
            std::filesystem::path(pq_codes_path).parent_path());

    // Open file for writing
    FILE* fp = std::fopen(pq_codes_path.c_str(), "wb");
    if (fp == nullptr) {
        FAISS_THROW_MSG("Failed to open PQ codes file for writing: " + pq_codes_path);
    }

    // Write header (32 bytes)
    uint32_t magic = 0x50514344; // "PQCD"
    uint32_t version = 1;
    uint32_t n_vectors = static_cast<uint32_t>(ntotal);
    uint32_t M = static_cast<uint32_t>(pq_M);
    uint32_t nbits = static_cast<uint32_t>(pq_nbits);
    uint32_t code_bytes_u32 = static_cast<uint32_t>(code_bytes);
    uint64_t body_offset = 32;

    std::fwrite(&magic, sizeof(magic), 1, fp);
    std::fwrite(&version, sizeof(version), 1, fp);
    std::fwrite(&n_vectors, sizeof(n_vectors), 1, fp);
    std::fwrite(&M, sizeof(M), 1, fp);
    std::fwrite(&nbits, sizeof(nbits), 1, fp);
    std::fwrite(&code_bytes_u32, sizeof(code_bytes_u32), 1, fp);
    std::fwrite(&body_offset, sizeof(body_offset), 1, fp);

    // Write body: PQ codes in seq order
    std::vector<uint8_t> row_buf(code_bytes);
    for (size_t i = 0; i < ntotal; i++) {
        const auto& code = vid_to_pq_code[i];
        std::memcpy(row_buf.data(), code.data(), code_bytes);
        std::fwrite(row_buf.data(), 1, code_bytes, fp);
    }

    std::fclose(fp);

    printf("[Curator] PQ codes written to disk: %s (%zu vectors, M=%zu, nbits=%zu, file_size=%zu bytes)\n",
           pq_codes_path.c_str(), ntotal, pq_M, pq_nbits,
           32 + ntotal * code_bytes);
}

void MultiTenantIndexIVFHierarchical::load_pq_codes_from_disk() {
    if (pq_codes_path.empty()) {
        FAISS_THROW_MSG("PQ codes path not set");
    }

    // Close any existing mmap
    if (pq_codes_mmap != nullptr && pq_codes_mmap != MAP_FAILED) {
        munmap(pq_codes_mmap, pq_codes_mmap_size);
        pq_codes_mmap = nullptr;
        pq_codes_mmap_size = 0;
    }

    // Open file and mmap it for read-only access
    int fd = open(pq_codes_path.c_str(), O_RDONLY);
    if (fd < 0) {
        FAISS_THROW_MSG("Failed to open PQ codes file for reading: " + pq_codes_path);
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        FAISS_THROW_MSG("Failed to stat PQ codes file: " + pq_codes_path);
    }
    pq_codes_mmap_size = st.st_size;

    // Validate file size
    size_t ntotal = seq_to_vid.size();
    size_t code_bytes = pq_M * pq_nbits / 8;
    size_t expected_size = 32 + ntotal * code_bytes;
    if (static_cast<size_t>(pq_codes_mmap_size) != expected_size) {
        close(fd);
        FAISS_THROW_MSG("PQ codes file size mismatch");
    }

    // mmap the entire file
    pq_codes_mmap = mmap(nullptr, pq_codes_mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (pq_codes_mmap == MAP_FAILED) {
        pq_codes_mmap = nullptr;
        FAISS_THROW_MSG("Failed to mmap PQ codes file: " + pq_codes_path);
    }

    // Validate magic number
    uint32_t magic = *static_cast<const uint32_t*>(pq_codes_mmap);
    if (magic != 0x50514344) {
        munmap(pq_codes_mmap, pq_codes_mmap_size);
        pq_codes_mmap = nullptr;
        pq_codes_mmap_size = 0;
        FAISS_THROW_MSG("Invalid PQ codes file magic number");
    }

    printf("[Curator] PQ codes loaded from disk via mmap: %s (%zu bytes)\n",
           pq_codes_path.c_str(), pq_codes_mmap_size);
}

void MultiTenantIndexIVFHierarchical::free_pq_codes() {
    if (vid_to_pq_code.empty()) {
        return;
    }

    size_t n_freed = 0;
    for (const auto& code : vid_to_pq_code) {
        n_freed += code.size() * sizeof(uint8_t);
    }

    vid_to_pq_code.clear();
    vid_to_pq_code.shrink_to_fit();

    printf("[Curator] Freed %zu bytes of in-memory PQ codes\n", n_freed);

    // Set up disk-based access if path is configured
    if (!pq_codes_path.empty()) {
        load_pq_codes_from_disk();
    }
}

void MultiTenantIndexIVFHierarchical::finalize_flash_storage() {
    size_t ntotal = seq_to_vid.size();
    if (ntotal == 0) {
        return;
    }

    // Step 1: Train PQ and encode (if enabled)
    if (pq_enabled && vid_to_pq_code.empty()) {
        train_pq_codebook();
    }

    // If flash storage is disabled, keep vectors in memory buffer
    if (!use_flash_storage) {
        flash_finalized = true; // mark as "done" but buffer remains
        printf("[Curator] Flash storage disabled — keeping %zu vectors in memory\n", ntotal);
        return;
    }

    // Step 2: DFS-traverse leaves, assign sequence numbers
    leaf_node_id_to_seq.clear();
    num_leaves = 0;
    std::function<void(TreeNode*)> dfs_leaves = [&](TreeNode* node) {
        if (node->children.empty()) {
            leaf_node_id_to_seq[node->node_id] = num_leaves++;
        } else {
            for (TreeNode* child : node->children) {
                dfs_leaves(child);
            }
        }
    };
    dfs_leaves(tree_root);

    // Step 3: Compute leaf region size
    leaf_region_size = max_leaf_size * d * sizeof(float);

    // Step 4: Open flash file if not already open
    if (flash_fp == nullptr) {
        if (flash_storage_path.empty()) {
            // Use PID to avoid conflicts between concurrent experiments
            flash_storage_path = "curator_flash_" + std::to_string(getpid()) + ".bin";
        }
        flash_fp = std::fopen(flash_storage_path.c_str(), "w+b");
        if (flash_fp == nullptr) {
            FAISS_THROW_MSG("Failed to open flash storage file for finalize");
        }
    }

    // Step 5: Write vectors to flash — one fseek per leaf, sequential within leaf
    // Group vectors by leaf, sorted by local_index
    std::unordered_map<int_vid_t, std::vector<std::pair<size_t, const float*>>> leaf_vectors;
    for (size_t i = 0; i < ntotal; i++) {
        int_vid_t vid = seq_to_vid[i];
        int_vid_t leaf_node_id = vid_to_leaf_node_id.at(vid);
        const float* vec = raw_vectors_buffer.data() + vid_to_buffer_offset.at(vid);
        leaf_vectors[leaf_node_id].push_back({vid_to_local_index.at(vid), vec});
    }

    // Sort each leaf's vectors by local index
    for (auto& [leaf_node_id, vecs] : leaf_vectors) {
        std::sort(vecs.begin(), vecs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    // Write each leaf: fseek to leaf start, then sequential fwrite for all its vectors
    for (auto& [leaf_node_id, vecs] : leaf_vectors) {
        size_t leaf_seq = leaf_node_id_to_seq.at(leaf_node_id);
        size_t region_start = leaf_seq * leaf_region_size;
        std::fseek(flash_fp, static_cast<long>(region_start), SEEK_SET);
        for (const auto& [local_idx, vec] : vecs) {
            std::fwrite(vec, sizeof(float), d, flash_fp);
        }
    }

    // Pre-allocate total file size (sets unwritten gap bytes to zero)
    size_t total_size = num_leaves * leaf_region_size;
    if (ftruncate(fileno(flash_fp), static_cast<off_t>(total_size)) != 0) {
        std::fseek(flash_fp, static_cast<long>(total_size - 1), SEEK_SET);
        char zero = 0;
        std::fwrite(&zero, 1, 1, flash_fp);
    }

    std::fflush(flash_fp);

    // Step 6: Free the raw vectors buffer
    raw_vectors_buffer.clear();
    raw_vectors_buffer.shrink_to_fit();
    vid_to_buffer_offset.clear();

    // Step 7: Persist PQ codes to disk if configured
    if (persist_pq_codes && !pq_codes_path.empty() && !vid_to_pq_code.empty()) {
        write_pq_codes_to_disk();
        free_pq_codes();
    }

    flash_finalized = true;

    printf("[Curator] Flash storage finalized: ntotal=%zu, num_leaves=%zu, "
           "leaf_region_size=%zu, path=%s\n",
           ntotal, num_leaves, leaf_region_size, flash_storage_path.c_str());
}

void MultiTenantIndexIVFHierarchical::load_vector_from_flash(
        int_vid_t vid, float* out) const {
    if (!flash_finalized) {
        FAISS_THROW_MSG("Flash storage not finalized");
    }

    auto leaf_id_it = vid_to_leaf_node_id.find(vid);
    if (leaf_id_it == vid_to_leaf_node_id.end()) {
        FAISS_THROW_MSG("vid not found in leaf mapping");
    }
    int_vid_t leaf_node_id = leaf_id_it->second;

    auto leaf_seq_it = leaf_node_id_to_seq.find(leaf_node_id);
    if (leaf_seq_it == leaf_node_id_to_seq.end()) {
        FAISS_THROW_MSG("Leaf node_id not found in flash mapping");
    }
    size_t leaf_seq = leaf_seq_it->second;

    size_t local_idx = vid_to_local_index.at(vid);

    size_t offset = leaf_seq * leaf_region_size + local_idx * d * sizeof(float);
    std::fseek(flash_fp, static_cast<long>(offset), SEEK_SET);
    size_t nread = std::fread(out, sizeof(float), d, flash_fp);
    if (nread != d) {
        FAISS_THROW_MSG("Failed to read vector from flash");
    }
}

void MultiTenantIndexIVFHierarchical::load_vectors_from_flash_batched(
        const std::vector<int_vid_t>& vids,
        std::vector<float>& vectors_out) const {
    if (vids.empty()) {
        vectors_out.clear();
        return;
    }

    vectors_out.resize(vids.size() * d);

    struct ReadReq {
        int_vid_t vid;
        size_t offset;
        size_t out_idx;
    };
    std::vector<ReadReq> reqs;
    reqs.reserve(vids.size());

    for (size_t i = 0; i < vids.size(); i++) {
        int_vid_t vid = vids[i];
        int_vid_t leaf_node_id = vid_to_leaf_node_id.at(vid);
        size_t leaf_seq = leaf_node_id_to_seq.at(leaf_node_id);
        size_t local_idx = vid_to_local_index.at(vid);

        size_t offset = leaf_seq * leaf_region_size + local_idx * d * sizeof(float);
        reqs.push_back({vid, offset, i});
    }

    std::sort(reqs.begin(), reqs.end(),
              [](const ReadReq& a, const ReadReq& b) { return a.offset < b.offset; });

    const size_t merge_threshold = 65536;
    size_t batch_start = 0;
    while (batch_start < reqs.size()) {
        size_t batch_end = batch_start + 1;
        size_t current_end_offset = reqs[batch_start].offset + d * sizeof(float);
        while (batch_end < reqs.size()) {
            size_t next_offset = reqs[batch_end].offset;
            if (next_offset <= current_end_offset + merge_threshold) {
                current_end_offset = std::max(current_end_offset,
                                              next_offset + d * sizeof(float));
                batch_end++;
            } else {
                break;
            }
        }

        size_t batch_size = batch_end - batch_start;
        if (batch_size == 1) {
            size_t offset = reqs[batch_start].offset;
            std::fseek(flash_fp, static_cast<long>(offset), SEEK_SET);
            size_t nr = std::fread(vectors_out.data() + reqs[batch_start].out_idx * d,
                                   sizeof(float), d, flash_fp);
            FAISS_THROW_IF_NOT_FMT(nr == d, "Failed to read single vector from flash (read %zu/%d)", nr, d);
        } else {
            size_t range_start = reqs[batch_start].offset;
            size_t range_end = reqs[batch_end - 1].offset + d * sizeof(float);
            size_t range_size = range_end - range_start;
            std::vector<uint8_t> buf(range_size);
            std::fseek(flash_fp, static_cast<long>(range_start), SEEK_SET);
            size_t nr = std::fread(buf.data(), 1, range_size, flash_fp);
            FAISS_THROW_IF_NOT_FMT(nr == range_size, "Failed to read batched range from flash (read %zu/%zu)", nr, range_size);

            for (size_t j = batch_start; j < batch_end; j++) {
                size_t offset_in_buf = reqs[j].offset - range_start;
                std::memcpy(vectors_out.data() + reqs[j].out_idx * d,
                            buf.data() + offset_in_buf, d * sizeof(float));
            }
        }

        batch_start = batch_end;
    }
}

// ============================================================
//  Memory accounting
// ============================================================

size_t MultiTenantIndexIVFHierarchical::get_total_memory_bytes() const {
    size_t total = 0;

    std::function<size_t(const TreeNode*)> tree_mem = [&](const TreeNode* node) -> size_t {
        size_t mem = sizeof(TreeNode);
        mem += d * sizeof(float); // centroid
        mem += (node->bf.size() + 7) / 8; // Bloom filter bits → bytes
        for (const auto& [tid, sl] : node->shortlists) {
            mem += sizeof(int_lid_t) + sl.data.size() * sizeof(int_vid_t);
        }
        mem += node->vector_indices.data.size() * sizeof(int_vid_t);
        for (const auto* child : node->children) {
            mem += tree_mem(child);
        }
        return mem;
    };
    total += tree_mem(tree_root);

    total += pq_codebook.size() * sizeof(float);
    for (const auto& code : vid_to_pq_code) {
        total += code.size() * sizeof(uint8_t);
    }
    total += raw_vectors_buffer.size() * sizeof(float);
    total += leaf_node_id_to_seq.size() * (sizeof(int_vid_t) + sizeof(size_t));
    total += id_allocator.label_to_id.size() * (sizeof(ext_vid_t) + sizeof(int_vid_t));
    total += id_allocator.id_to_label.size() * sizeof(ext_vid_t);
    total += tid_allocator.label_to_id.size() * (sizeof(ext_lid_t) + sizeof(int_lid_t));
    total += tid_allocator.id_to_label.size() * sizeof(ext_lid_t);

    for (const auto& [tid, nodes] : cached_temp_indexes) {
        total += nodes.size() * sizeof(complex_predicate::TempIndexNode);
        for (const auto& node : nodes) {
            total += node.children.size() * sizeof(int);
        }
    }
    for (const auto& [tid, vecs] : cached_qualified_vecs) {
        total += vecs.size() * sizeof(int_vid_t);
    }

    return total;
}

// ============================================================
//  Rerank control
// ============================================================

void MultiTenantIndexIVFHierarchical::set_rerank_params(bool enabled, size_t topk_factor) {
    pq_use_adc_rerank = enabled;
    pq_rerank_topk_factor = topk_factor;
}

bool MultiTenantIndexIVFHierarchical::get_rerank_enabled() const {
    return pq_use_adc_rerank;
}

size_t MultiTenantIndexIVFHierarchical::get_rerank_topk_factor() const {
    return pq_rerank_topk_factor;
}


} // namespace faiss