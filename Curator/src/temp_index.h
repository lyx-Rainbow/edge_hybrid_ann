// temp_index.h — Lightweight temporary index for bitmap-filter search
#pragma once

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "common.h"

namespace curator {

struct TreeNode; // forward declaration

// Lightweight node mirroring TreeNode structure for bitmap-filter search.
// centroid is a NON-OWNING pointer to the main cluster tree's TreeNode::centroid.
// Lifetime: valid for the lifetime of the main cluster tree.
// Note: TreeNode::centroid (std::vector<float>) must not be modified after
// construction, as TempIndexNode holds raw pointers into it.
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

// Batch distance computation callback type.
// Takes a list of vids and fills output with (distance, vid) pairs.
// Caller (CuratorIndex) provides the actual implementation based on
// PQ-codec availability and flash/buffer state.
using BatchDistanceFn = std::function<void(
    const std::vector<int_vid_t>& vids,
    std::vector<std::pair<float, int_vid_t>>& out)>;

// Search the temporary index for top-k results.
// compute_distances: callback to compute real vector distances (PQ or exact).
//   Previously used node-centroid distance as a proxy, which caused
//   significant recall degradation for bitmap-filter queries.
void search_temp_index(
        const std::vector<TempIndexNode>& nodes,
        const std::vector<int_vid_t>& qualified_vecs,
        const float* query,
        size_t k,
        size_t d,
        size_t search_ef,
        size_t beam_size,
        BatchDistanceFn compute_distances,
        float* distances,
        int_vid_t* labels);

} // namespace curator
