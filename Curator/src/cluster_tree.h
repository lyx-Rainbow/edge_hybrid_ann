// cluster_tree.h — Hierarchical clustering tree build and operations
#pragma once

#include <cstddef>
#include <vector>

#include "common.h"
#include "tree_node.h"

namespace curator {

struct CuratorConfig; // fwd

// Recursively build the clustering tree via K-means
void build_tree(TreeNode* root, size_t n, const float* x, const CuratorConfig& cfg);

// Assign a single vector to its closest leaf by descent
TreeNode* assign_to_leaf(TreeNode* root, const float* x, size_t d);

// Get the path of vector IDs from root to leaf (for a given label)
std::vector<int_vid_t> get_vector_path(TreeNode* root, ext_vid_t label,
                                       const VectorIdAllocator& vid_map);

// Decode a vid's path bits and find the assigned leaf node
TreeNode* find_assigned_leaf(TreeNode* root, int_vid_t vid);

} // namespace curator
