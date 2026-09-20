// cluster_tree.cpp — Hierarchical clustering tree implementation
#include "cluster_tree.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "common.h"
#include "config.h" // CuratorConfig defined here (no dependency on curator_index.h)
#include "kmeans.h"
#include "tree_node.h"

namespace curator {

void build_tree(TreeNode* node, size_t n, const float* x, const CuratorConfig& cfg) {
    size_t d = cfg.d;

    // Root node: compute global mean as centroid
    if (node->centroid.empty()) {
        node->centroid.resize(d, 0.0f);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                node->centroid[j] += x[i * d + j];
            }
        }
        for (size_t j = 0; j < d; j++) {
            node->centroid[j] /= static_cast<float>(n);
        }
    }

    // Stop if too few samples or max depth reached
    if (n <= cfg.max_leaf_size || node->level >= MAX_TREE_DEPTH) {
        return;
    }

    // Guard: if we have fewer vectors than clusters, K-means would truncate
    // n_clusters inside kmeans() (line 48), causing result.centroids to have
    // fewer entries than cfg.n_clusters.  The subsequent loop below would then
    // read past the end of result.centroids when clus_id >= n.
    // Simply stop recursion here — splitting into n < k branches is meaningless.
    if (n < cfg.n_clusters) {
        return;
    }

    // K-means clustering
    KMeansConfig kcfg;
    kcfg.niter = static_cast<int>(cfg.clus_niter);
    kcfg.seed = 1234;
    auto result = kmeans(static_cast<int>(d), static_cast<int>(n), x,
                          static_cast<int>(cfg.n_clusters), kcfg);

    // Sort vectors by cluster assignment
    std::vector<size_t> cluster_size(cfg.n_clusters, 0);
    std::vector<size_t> cluster_offsets(cfg.n_clusters, 0);

    for (size_t i = 0; i < n; i++) {
        cluster_size[result.assignments[i]]++;
    }

    cluster_offsets[0] = 0;
    for (size_t i = 1; i < cfg.n_clusters; i++) {
        cluster_offsets[i] = cluster_offsets[i - 1] + cluster_size[i - 1];
    }

    std::vector<float> sorted_x(n * d);
    std::vector<size_t> tmp_offsets = cluster_offsets;

    for (size_t i = 0; i < n; i++) {
        size_t cluster = result.assignments[i];
        size_t pos = tmp_offsets[cluster]++;
        std::memcpy(sorted_x.data() + pos * d, x + i * d, d * sizeof(float));
    }

    // Recursively build children
    for (size_t clus_id = 0; clus_id < cfg.n_clusters; clus_id++) {
        TreeNode* child = new TreeNode(
                node->level + 1,
                clus_id,
                node,
                result.centroids.data() + clus_id * d,
                d,
                node->bf_capacity,
                node->bf_false_pos);

        build_tree(child, cluster_size[clus_id],
                   sorted_x.data() + cluster_offsets[clus_id] * d, cfg);

        node->children.push_back(child);
    }

    // Invariant: all non-leaf nodes have exactly n_clusters children
    CURATOR_ASSERT_FMT(node->children.size() == cfg.n_clusters,
                       "build_tree: invariant violated: %zu children != %zu n_clusters",
                       node->children.size(), cfg.n_clusters);
}

TreeNode* assign_to_leaf(TreeNode* root, const float* x, size_t d) {
    TreeNode* curr = root;
    while (!curr->children.empty()) {
        int best_child = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < curr->children.size(); i++) {
            float dist = l2_sqr(x, curr->children[i]->centroid.data(), d);
            if (dist < best_dist) {
                best_dist = dist;
                best_child = static_cast<int>(i);
            }
        }
        curr = curr->children[best_child];
    }
    return curr;
}

std::vector<int_vid_t> get_vector_path(TreeNode* root, ext_vid_t label,
                                        const VectorIdAllocator& vid_map) {
    std::vector<int_vid_t> path;
    if (!vid_map.has_label(label)) return path;

    int_vid_t vid = vid_map.get_id(label);
    path.push_back(vid);

    TreeNode* curr = root;
    while (!curr->children.empty()) {
        size_t level = curr->level + 1;
        auto offset = sizeof(int_vid_t) * 8 - level * MAX_BRANCH_FACTOR_LOG2;
        auto mask = (int_vid_t{1} << MAX_BRANCH_FACTOR_LOG2) - 1;
        size_t child_id = static_cast<size_t>((vid >> offset) & mask);

        if (child_id >= curr->children.size()) break;
        curr = curr->children[child_id];
        path.push_back(curr->node_id);
    }
    return path;
}

TreeNode* find_assigned_leaf(TreeNode* root, int_vid_t vid) {
    TreeNode* curr = root;
    while (!curr->children.empty()) {
        size_t level = curr->level + 1;
        auto offset = sizeof(int_vid_t) * 8 - level * MAX_BRANCH_FACTOR_LOG2;
        auto mask = (MAX_BRANCH_FACTOR - 1);
        size_t child_id = static_cast<size_t>((vid >> offset) & mask);

        if (child_id >= curr->children.size()) return curr;
        curr = curr->children[child_id];
    }
    return curr;
}

} // namespace curator
