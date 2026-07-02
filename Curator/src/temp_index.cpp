// temp_index.cpp — Temporary index for bitmap-filter search
#include "temp_index.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <vector>

#include "common.h"
#include "distance.h"
#include "tree_node.h"

namespace curator {

//TODO: sorted_qualified_vecs是怎么得到的？具体定义在哪个文件里？
void build_temp_index(
        const TreeNode* root,
        const std::vector<int_vid_t>& sorted_qualified_vecs,
        size_t n_clusters,
        size_t max_sl_size,
        std::vector<TempIndexNode>& nodes) {
    CURATOR_THROW_IF_NOT_MSG(
            std::is_sorted(sorted_qualified_vecs.begin(), sorted_qualified_vecs.end()),
            "Input must be sorted in ascending order");

    std::function<int(int, int, const TreeNode*)> build =
            [&](int start, int end, const TreeNode* curr_node) -> int {
        int curr_idx = static_cast<int>(nodes.size());
        nodes.push_back(TempIndexNode{start, end, {}, curr_node->centroid.data()});

        // Base case: small enough or leaf node
        if (end - start <= static_cast<int>(max_sl_size) || curr_node->children.empty()) {
            return curr_idx;
        }

        // Extract branch index from vector ID bits
        auto level = curr_node->level;
        auto offset = sizeof(int_vid_t) * 8 -
                MAX_BRANCH_FACTOR_LOG2 * (level + 1);
        auto mask = (MAX_BRANCH_FACTOR - 1);

        // Binary-search ranges for each child prefix
        std::vector<std::pair<int, int>> child_ranges;
        child_ranges.reserve(n_clusters);

        for (size_t child_idx = 0; child_idx < n_clusters; child_idx++) {
            int first = static_cast<int>(
                    std::lower_bound(
                            sorted_qualified_vecs.begin() + start,
                            sorted_qualified_vecs.begin() + end,
                            child_idx,
                            [offset, mask](int_vid_t vid, size_t cid) {
                                return ((vid >> offset) & mask) < cid;
                            }) - sorted_qualified_vecs.begin());

            int last = static_cast<int>(
                    std::lower_bound(
                            sorted_qualified_vecs.begin() + first,
                            sorted_qualified_vecs.begin() + end,
                            child_idx + 1,
                            [offset, mask](int_vid_t vid, size_t cid) {
                                return ((vid >> offset) & mask) < cid;
                            }) - sorted_qualified_vecs.begin());

            child_ranges.emplace_back(first, last);
        }

        // Recurse into non-empty children
        for (size_t child_idx = 0; child_idx < n_clusters; child_idx++) {
            auto& range = child_ranges[child_idx];
            if (range.first != range.second) {
                const TreeNode* child_node = curr_node->children[child_idx];
                int child_idx2 = build(range.first, range.second, child_node);
                nodes[curr_idx].children.push_back(child_idx2);
            }
        }

        return curr_idx;
    };

    if (!sorted_qualified_vecs.empty()) {
        build(0, static_cast<int>(sorted_qualified_vecs.size()), root);
    }
}

void search_temp_index(
        const std::vector<TempIndexNode>& nodes,
        const std::vector<int_vid_t>& qualified_vecs,
        const float* query,
        size_t k,
        size_t d,
        size_t search_ef,
        size_t beam_size,
        float* distances,
        int_vid_t* labels) {
    if (nodes.empty()) return;

    using Candidate = std::pair<float, int>; // (score, node_idx)
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> frontier;
    RunningList results(static_cast<int>(search_ef));

    // Initialize: score root node (variance_boost = 0)
    float root_score = l2_sqr(query, nodes[0].centroid, d);
    frontier.emplace(root_score, 0);

    // Beam search on temp index
    std::vector<Candidate> beam;
    std::vector<Candidate> next_beam;

    if (beam_size > 0 && nodes[0].children.empty() == false) {
        beam.emplace_back(root_score, 0);

        while (true) {
            //TODO: 评分时为什么不使用tree_node.h中的node_score()函数？因为temp_index没有variance信息吗？
            bool updated = false;
            for (auto& [score, node_idx] : beam) {
                const auto& node = nodes[node_idx];

                // Check if this is a "leaf" in temp index (no children or small range)
                if (node.children.empty() || (node.end - node.start) <= static_cast<int>(search_ef)) {
                    next_beam.emplace_back(score, node_idx);
                } else {
                    updated = true;
                    for (int child_idx : node.children) {
                        const auto& child = nodes[child_idx];
                        float child_score = l2_sqr(query, child.centroid, d);
                        next_beam.emplace_back(child_score, child_idx);
                    }
                }
            }

            if (!updated) break;

            std::sort(next_beam.begin(), next_beam.end());
            auto n_keep = std::min(beam_size, next_beam.size());
            for (size_t i = n_keep; i < next_beam.size(); i++) {
                frontier.push(next_beam[i]);
            }
            next_beam.resize(n_keep);

            std::swap(beam, next_beam);
            next_beam.clear();
        }

        // Push beam results to frontier
        for (auto& cand : beam) {
            frontier.push(cand);
        }
    } else {
        frontier.emplace(root_score, 0);
    }
    //TODO: 这里的beam search和frontier search的区别是什么？为什么beam search不直接收集qualified_vecs，而是继续扩展节点？可能是因为beam search只保留了部分节点，qualified_vecs可能不完整？

    // Frontier search: pop nodes, collect candidates
    while (!frontier.empty()) {
        auto [score, node_idx] = frontier.top();
        frontier.pop();

        const auto& node = nodes[node_idx];

        // Expand children if any
        for (int child_idx : node.children) {
            const auto& child = nodes[child_idx];
            float child_score = l2_sqr(query, child.centroid, d);
            frontier.emplace(child_score, child_idx);
        }

        // Collect qualified vectors from this node's range
        if (node.children.empty() || (node.end - node.start) <= static_cast<int>(search_ef)) {
            for (int i = node.start; i < node.end; i++) {
                int_vid_t vid = qualified_vecs[i];
                // NOTE: compute_vector_distance depends on CuratorIndex state
                // For temp_index, we use a simplified approach:
                // The caller (CuratorIndex) should handle exact distance computation
                // Here we just insert with a placeholder distance of 0
                // (Real implementation requires access to raw vectors)
                results.insert(vid, score); // Use node score as approximate distance
            }
        }
    }

    // Output top-k
    size_t n_out = std::min(k, results.dists.size());
    for (size_t i = 0; i < n_out; i++) {
        distances[i] = results.dists[i];
        labels[i] = results.vids[i];
    }
    // Pad remaining with sentinel values
    for (size_t i = n_out; i < k; i++) {
        distances[i] = std::numeric_limits<float>::max();
        labels[i] = 0;
    }
}

} // namespace curator
