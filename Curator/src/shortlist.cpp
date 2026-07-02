// shortlist.cpp — Short list split and merge operations
#include "shortlist.h"

#include "common.h"
#include "tree_node.h"

namespace curator {

void split_shortlist(TreeNode* node, int_lid_t tid, size_t max_sl_size) {
    auto it = node->shortlists.find(tid);
    if (it == node->shortlists.end()) return;
    if (it->second.size() <= max_sl_size) return;

    // For each vid in the shortlist, determine which child it belongs to
    // and push it down
    auto level = node->level;
    auto offset = sizeof(int_vid_t) * 8 - (level + 1) * MAX_BRANCH_FACTOR_LOG2;
    auto mask = (MAX_BRANCH_FACTOR - 1);

    std::vector<ShortList> child_sls(node->children.size());
    for (int_vid_t vid : it->second.data) {
        size_t child_id = static_cast<size_t>((vid >> offset) & mask);
        if (child_id < child_sls.size()) {
            child_sls[child_id].insert(vid);
        }
    }

    // Replace node's shortlist with child-distributed entries
    it->second.data.clear();
    for (size_t c = 0; c < child_sls.size(); c++) {
        if (!child_sls[c].data.empty()) {
            auto& child_node = node->children[c];
            auto child_it = child_node->shortlists.find(tid);
            if (child_it != child_node->shortlists.end()) {
                // Merge
                child_it->second = child_it->second.merge(child_sls[c]);
            } else {
                child_node->shortlists.emplace(tid, std::move(child_sls[c]));
            }
            child_node->bf.insert(tid);
        }
    }

    // Remove empty shortlist from this node
    if (it->second.data.empty()) {
        node->shortlists.erase(it);
    }

    // Recompute this node's bloom filter
    node->bf = node->recompute_bloom_filter();
}

bool try_merge_shortlists(TreeNode* node, int_lid_t tid, size_t max_sl_size) {
    // Sum up all child shortlists for this tenant
    size_t total = 0;
    for (const auto* child : node->children) {
        auto it = child->shortlists.find(tid);
        if (it != child->shortlists.end()) {
            total += it->second.size();
        }
    }

    if (total == 0) return false;
    if (total > max_sl_size) return false;

    // Merge into parent
    ShortList merged;
    for (auto* child : node->children) {
        auto it = child->shortlists.find(tid);
        if (it != child->shortlists.end()) {
            merged = merged.merge(it->second);
            child->shortlists.erase(it);
            // Recompute child's bloom filter
            child->bf = child->recompute_bloom_filter();
        }
    }

    node->shortlists[tid] = std::move(merged);
    node->bf.insert(tid);

    return true;
}

} // namespace curator
