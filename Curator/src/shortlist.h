// shortlist.h — Short list split and merge operations
#pragma once

#include <cstddef>

#include "common.h"

namespace curator {

struct TreeNode; // fwd

// Push exceeding short-list entries to children when max_sl_size is exceeded
void split_shortlist(TreeNode* node, int_lid_t tid, size_t max_sl_size);

// Try to merge children's short lists for the same tenant into parent
// Returns true if merge was performed (caller should cascade upward)
bool try_merge_shortlists(TreeNode* node, int_lid_t tid, size_t max_sl_size);

} // namespace curator
