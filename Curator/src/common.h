// common.h — Public infrastructure: type aliases, constants, utility templates, exception macros
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// Prefetch macro — replaces FAISS prefetch_L1
// ============================================================================
#define PREFETCH(ptr) __builtin_prefetch((ptr), 0, 3)

// ============================================================================
// Exception macros — replaces FAISS_THROW_* / FAISS_ASSERT_*
// ============================================================================
#define CURATOR_THROW_MSG(MSG) throw std::runtime_error(MSG)

#define CURATOR_THROW_IF_NOT(COND, MSG) \
    do { if (!(COND)) CURATOR_THROW_MSG(MSG); } while (0)

#define CURATOR_THROW_IF_NOT_MSG CURATOR_THROW_IF_NOT

#define CURATOR_THROW_FMT(MSG, ...)                                      \
    do {                                                                  \
        char _buf[1024];                                                  \
        std::snprintf(_buf, sizeof(_buf), MSG, ##__VA_ARGS__);           \
        throw std::runtime_error(std::string(_buf));                      \
    } while (0)

#define CURATOR_THROW_IF_NOT_FMT(COND, MSG, ...) \
    do { if (!(COND)) CURATOR_THROW_FMT(MSG, ##__VA_ARGS__); } while (0)

#define CURATOR_ASSERT_MSG(COND, MSG) \
    do { if (!(COND)) CURATOR_THROW_MSG(MSG); } while (0)

#define CURATOR_ASSERT_FMT(COND, MSG, ...) \
    do { if (!(COND)) CURATOR_THROW_FMT(MSG, ##__VA_ARGS__); } while (0)

namespace curator {

// ============================================================================
// Type aliases
// ============================================================================
using ext_vid_t = uint32_t;  // external vector ID
using int_vid_t = uint64_t;  // internal vector ID (path-encoded)
using ext_lid_t = int16_t;   // external label/tenant ID
using int_lid_t = int16_t;   // internal label/tenant ID (contiguous)
using tid_t = int16_t;       // retained alias for complex_predicate template compatibility
using Buffer = std::vector<int_vid_t>;

// ============================================================================
// Compile-time constants (matching original CURATOR_MAX_* )
// ============================================================================
constexpr size_t MAX_BRANCH_FACTOR_LOG2 = 6;
constexpr size_t MAX_LEAF_SIZE_LOG2 = 10;
constexpr size_t MAX_TREE_DEPTH =
        (sizeof(int_vid_t) * 8 - MAX_LEAF_SIZE_LOG2) / MAX_BRANCH_FACTOR_LOG2;
constexpr size_t MAX_BRANCH_FACTOR = 1 << MAX_BRANCH_FACTOR_LOG2;
constexpr size_t MAX_LEAF_SIZE = 1 << MAX_LEAF_SIZE_LOG2;

// ============================================================================
// RunningMean — dynamic mean tracker
// ============================================================================
struct RunningMean {
    int n = 0;
    double sum = 0.0;

    void add(double x) {
        sum += x;
        n++;
    }

    void remove(double x) {
        CURATOR_ASSERT_MSG(n > 0, "no elements to remove");
        sum -= x;
        n--;
        if (n == 0) sum = 0.0;
    }

    double get_mean() const {
        return (n > 0) ? sum / n : 0.0;
    }
};

// ============================================================================
// SortedList<T> — ordered list with binary-search insert/erase/contains
// ============================================================================
template <typename T>
struct SortedList {
    std::vector<T> data;

    SortedList() = default;
    SortedList(const std::vector<T>& data_) : data(data_) {
        std::sort(data.begin(), data.end());
    }
    SortedList(const SortedList&) = default;
    SortedList(SortedList&&) noexcept = default;
    SortedList& operator=(const SortedList&) = default;
    SortedList& operator=(SortedList&&) noexcept = default;

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }

    void insert(const T& item) {
        auto it = std::lower_bound(data.begin(), data.end(), item);
        data.insert(it, item);
    }

    void erase(const T& item) {
        auto it = std::lower_bound(data.begin(), data.end(), item);
        CURATOR_ASSERT_MSG(it != data.end() && *it == item, "item not found");
        data.erase(it);
    }

    bool contains(const T& item) const {
        return std::binary_search(data.begin(), data.end(), item);
    }

    size_t size() const { return data.size(); }

    SortedList merge(const SortedList& other) const {
        SortedList result;
        std::merge(data.begin(), data.end(),
                   other.data.begin(), other.data.end(),
                   std::back_inserter(result.data));
        return result;
    }
};

using ShortList = SortedList<int_vid_t>;

// ============================================================================
// RunningList — top-K ordered candidate set (O(k) insert)
// Shared by curator_index (tenant/unfiltered search) and temp_index
// ============================================================================
struct RunningList {
    std::vector<int_vid_t> vids;
    std::vector<float> dists;
    std::vector<int_vid_t> vids_tmp;
    std::vector<float> dists_tmp;
    int capacity;

    explicit RunningList(int capacity) : capacity(capacity) {
        vids.reserve(capacity + 1);
        dists.reserve(capacity + 1);
        vids_tmp.reserve(capacity + 1);
        dists_tmp.reserve(capacity + 1);
    }

    // Insert single (vid, dist) pair in sorted order; returns true if list updated
    bool insert(int_vid_t vid, float dist) {
        auto it = std::lower_bound(dists.begin(), dists.end(), dist);
        auto pos = it - dists.begin();
        bool updated = false;

        if (dists.size() < static_cast<size_t>(capacity)) {
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

    // Batch insert pre-sorted (by distance ascending) candidates via merge
    bool batch_insert(const std::vector<std::pair<float, int_vid_t>>& cands) {
        if (cands.empty()) return false;
        bool updated = (dists.size() < static_cast<size_t>(capacity)) ||
                       (cands[0].first < dists.back());
        if (!updated) return false;

        size_t i = 0, j = 0;
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
            //TODO: 下行的break操作是否会导致在dists和cands均有多出来的元素时，多出来的元素插入后无法保证大于capacity部分有序？
            if (vids_tmp.size() == static_cast<size_t>(capacity)) break;
        }
        while (vids_tmp.size() < static_cast<size_t>(capacity) && i < dists.size()) {
            vids_tmp.push_back(vids[i]);
            dists_tmp.push_back(dists[i]);
            i++;
        }
        while (vids_tmp.size() < static_cast<size_t>(capacity) && j < cands.size()) {
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
        if (new_capacity <= capacity) return;
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

// ============================================================================
// IdAllocator<ExtLabel, IntLabel> — contiguous internal-ID allocator
// ============================================================================
template <typename ExtLabel, typename IntLabel>
struct IdAllocator {
    static const IntLabel INVALID_ID;

    //TODO: 下面三个变量的含义是什么？
    std::unordered_set<IntLabel> free_list;
    std::unordered_map<ExtLabel, IntLabel> label_to_id;
    std::vector<ExtLabel> id_to_label;

    IntLabel allocate_id(ExtLabel label) {
        CURATOR_THROW_IF_NOT_MSG(
                label_to_id.find(label) == label_to_id.end(),
                "label already exists");

        IntLabel id;
        if (free_list.empty()) {
            id = static_cast<IntLabel>(id_to_label.size());
            id_to_label.push_back(INVALID_ID);
        } else {
            id = *free_list.begin();
            free_list.erase(free_list.begin());
        }

        label_to_id.emplace(label, id);
        id_to_label[id] = label;
        return id;
    }

    ExtLabel allocate_reserved_label() {
        for (ExtLabel label = std::numeric_limits<ExtLabel>::max();
             label != std::numeric_limits<ExtLabel>::min();
             label--) {
            if (!has_label(label)) return label;
        }
        CURATOR_THROW_MSG("No available reserved label");
    }

    void free_id(ExtLabel label) {
        auto it = label_to_id.find(label);
        CURATOR_THROW_IF_NOT_MSG(it != label_to_id.end(), "label does not exist");
        label_to_id.erase(label);
        IntLabel id = it->second;
        if (id == static_cast<IntLabel>(id_to_label.size() - 1)) {
            id_to_label.pop_back();
        } else {
            id_to_label[id] = INVALID_ID;
            free_list.emplace(id);
        }
    }

    bool has_label(ExtLabel label) const {
        return label_to_id.find(label) != label_to_id.end();
    }

    const IntLabel get_id(ExtLabel label) const {
        auto it = label_to_id.find(label);
        CURATOR_THROW_IF_NOT_MSG(it != label_to_id.end(), "label does not exist");
        return it->second;
    }

    const IntLabel get_or_create_id(ExtLabel label) {
        if (has_label(label)) return get_id(label);
        return allocate_id(label);
    }

    const ExtLabel get_label(IntLabel id) const {
        if (id >= static_cast<IntLabel>(id_to_label.size()) || id_to_label[id] == INVALID_ID) {
            CURATOR_THROW_MSG("id does not exist");
        }
        return id_to_label[id];
    }
};

// ============================================================================
// IdMapping<ExtLabel, IntLabel> — bidirectional ID mapping (no allocation)
// ============================================================================
template <typename ExtLabel, typename IntLabel>
struct IdMapping {
    std::unordered_map<ExtLabel, IntLabel> label_to_id;
    std::unordered_map<IntLabel, ExtLabel> id_to_label;

    void add_mapping(ExtLabel label, IntLabel id) {
        label_to_id[label] = id;
        id_to_label[id] = label;
    }

    void remove_mapping(ExtLabel label) {
        CURATOR_THROW_IF_NOT_MSG(has_label(label), "label does not exist");
        id_to_label.erase(label_to_id[label]);
        label_to_id.erase(label);
    }

    bool has_label(ExtLabel label) const {
        return label_to_id.find(label) != label_to_id.end();
    }

    bool has_id(IntLabel id) const {
        return id_to_label.find(id) != id_to_label.end();
    }

    const IntLabel get_id(ExtLabel label) const {
        CURATOR_THROW_IF_NOT_MSG(has_label(label), "label does not exist");
        return label_to_id.at(label);
    }

    const ExtLabel get_label(IntLabel id) const {
        CURATOR_THROW_IF_NOT_MSG(has_id(id), "id does not exist");
        return id_to_label.at(id);
    }
};

using VectorIdAllocator = IdMapping<ext_vid_t, int_vid_t>;
using TenantIdAllocator = IdAllocator<ext_lid_t, int_lid_t>;

} // namespace curator
