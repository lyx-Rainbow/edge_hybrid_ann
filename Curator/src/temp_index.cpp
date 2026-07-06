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
        BatchDistanceFn compute_distances,
        float* distances,
        int_vid_t* labels) {
    if (nodes.empty()) return;

    using Candidate = std::pair<float, int>; // (score, node_idx)
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> frontier;
    RunningList results(static_cast<int>(search_ef));

    // Initialize: score root node.
    // variance_boost = 0 for temp_index (no RunningMean variance data available),
    // so node_score() degenerates to plain l2_sqr — use it directly.
    float root_score = l2_sqr(query, nodes[0].centroid, d);
    frontier.emplace(root_score, 0);

    // ── Phase 1: Beam search (navigation only — no vector collection) ──
    std::vector<Candidate> beam;
    std::vector<Candidate> next_beam;

    if (beam_size > 0 && nodes[0].children.empty() == false) {
        beam.emplace_back(root_score, 0);

        while (true) {
            bool updated = false;
            for (auto& [score, node_idx] : beam) {
                const auto& node = nodes[node_idx];

                // Terminal node: keep in beam for Phase 2, do NOT expand further.
                // Terminal condition: no children (true leaf) OR range is small
                // enough that all vids can be scanned (≤ search_ef).
                if (node.children.empty() ||
                    (node.end - node.start) <= static_cast<int>(search_ef)) {
                    next_beam.emplace_back(score, node_idx);
                } else {
                    updated = true;
                    // Expand children — score them by centroid distance.
                    // variance_boost=0 here (TempIndexNode has no variance),
                    // so plain l2_sqr is used instead of node_score().
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
            // Nodes beyond beam_width are pushed to frontier for Phase 2
            for (size_t i = n_keep; i < next_beam.size(); i++) {
                frontier.push(next_beam[i]);
            }
            next_beam.resize(n_keep);

            std::swap(beam, next_beam);
            next_beam.clear();
        }

        // Push beam results to frontier for Phase 2
        for (auto& cand : beam) {
            frontier.push(cand);
        }
    } else {
        frontier.emplace(root_score, 0);
    }
    // Phase 1 (beam search) and Phase 2 (frontier search) are complementary:
    // - Beam search navigates depth-first with limited width, finding the most
    //   promising regions. Nodes not in the top-beam_width are pushed to frontier.
    // - Frontier search explores the full priority-ordered space, collecting
    //   vectors from terminal nodes and expanding non-terminal ones.
    // This two-phase design ensures both depth-priority (fast descent) and
    // breadth-completeness (no promising region missed).

    // ── Phase 2: Frontier search (collect vectors with REAL distances) ──
    while (!frontier.empty()) {
        auto [score, node_idx] = frontier.top();
        frontier.pop();

        const auto& node = nodes[node_idx];

        bool is_terminal = node.children.empty() ||
                           (node.end - node.start) <= static_cast<int>(search_ef);

        if (is_terminal) {
            // ★ Terminal node: collect all vids in this range and compute
            //   real distances via the callback (PQ or exact).
            //   Do NOT expand children — they are subsets of this range
            //   and would cause redundant double-collection.
            std::vector<int_vid_t> node_vids;
            node_vids.reserve(node.end - node.start);
            for (int i = node.start; i < node.end; i++) {
                node_vids.push_back(qualified_vecs[i]);
            }

            // ★ Compute real vector distances (PQ ADC or exact L2)
            std::vector<std::pair<float, int_vid_t>> node_dists;
            compute_distances(node_vids, node_dists);
            std::sort(node_dists.begin(), node_dists.end());
            results.batch_insert(node_dists);
        } else {
            // ★ Non-terminal node: expand children, do NOT collect vids.
            //   The range is too large to scan directly — push deeper.
            for (int child_idx : node.children) {
                const auto& child = nodes[child_idx];
                float child_score = l2_sqr(query, child.centroid, d);
                frontier.emplace(child_score, child_idx);
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
