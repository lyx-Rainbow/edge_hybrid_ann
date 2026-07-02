// temp_index.h — Lightweight temporary index for bitmap-filter search
#pragma once

#include <cstddef>
#include <vector>

#include "common.h"

namespace curator {

struct TreeNode; // forward declaration

// Lightweight node mirroring TreeNode structure for bitmap-filter search.
// centroid is a NON-OWNING pointer to the main cluster tree's TreeNode::centroid.
// Lifetime: valid for the lifetime of the main cluster tree.
struct TempIndexNode {
    int start, end;
    std::vector<int> children;
    const float* centroid; // Non-owning pointer to TreeNode::centroid
};

// Build a temporary index tree from sorted qualified vids.
// Depends on invariant: all non-leaf TreeNodes have exactly n_clusters children.
void build_temp_index(
        const TreeNode* root,
        const std::vector<int_vid_t>& sorted_qualified_vecs,
        size_t n_clusters,
        size_t max_sl_size,
        std::vector<TempIndexNode>& nodes);

// Search the temporary index for top-k results
void search_temp_index(
        const std::vector<TempIndexNode>& nodes,
        const std::vector<int_vid_t>& qualified_vecs,
        const float* query,
        size_t k,
        size_t d,
        size_t search_ef,
        size_t beam_size,
        float* distances,
        int_vid_t* labels);

} // namespace curator
