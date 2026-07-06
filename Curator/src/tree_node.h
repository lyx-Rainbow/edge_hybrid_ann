// tree_node.h — TreeNode data structure for Curator hierarchical clustering tree
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "bloom_filter.h"
#include "common.h"
#include "distance.h"

namespace curator {

struct TreeNode {
    /* tree structure */
    size_t level;
    size_t sibling_id;
    TreeNode* parent;
    std::vector<TreeNode*> children;
    int_vid_t node_id;

    /* cluster information */
    // Do NOT modify (push_back/resize) after construction — TempIndexNode
    // holds non-owning pointers (centroid.data()) into this vector.
    std::vector<float> centroid;
    RunningMean variance;

    /* bloom filter + shortlists (all nodes) */
    size_t bf_capacity;
    float bf_false_pos;
    bloom_filter bf;
    std::unordered_map<int_lid_t, ShortList> shortlists;

    /* leaf nodes only */
    ShortList vector_indices;

    TreeNode(
            size_t level,
            size_t sibling_id,
            TreeNode* parent,
            const float* centroid_ptr,
            size_t d,
            size_t bf_capacity,
            float bf_false_pos)
            : level(level),
              sibling_id(sibling_id),
              parent(parent),
              bf_capacity(bf_capacity),
              bf_false_pos(bf_false_pos) {
        // Compute node_id from path
        if (parent != nullptr) {
            auto offset = sizeof(int_vid_t) * 8 - level * MAX_BRANCH_FACTOR_LOG2;
            this->node_id = parent->node_id | (sibling_id << offset);
        } else {
            this->node_id = 0;
        }

        // Copy centroid (or leave empty for root pre-initialization)
        if (centroid_ptr != nullptr) {
            centroid.assign(centroid_ptr, centroid_ptr + d);
        }

        this->bf = init_bloom_filter();
    }

    ~TreeNode() {
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
        auto new_bf = init_bloom_filter();
        for (const auto& kv : shortlists) {
            new_bf.insert(kv.first);
        }
        for (const auto& child : children) {
            new_bf |= child->bf;
        }
        return new_bf;
    }
};

// ============================================================================
// node_score — distance from query to node centroid, minus variance boost
// Defined here (not in distance.h) because it depends on TreeNode layout
// ============================================================================
inline float node_score(const TreeNode* node, const float* x, size_t d, float var_boost) {
    float dist = l2_sqr(x, node->centroid.data(), d);
    if (var_boost == 0.0f) {
        return dist;
    }
    return dist - var_boost * static_cast<float>(node->variance.get_mean());
}

} // namespace curator
