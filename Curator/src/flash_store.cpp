// flash_store.cpp — Flash disk storage implementation (pread/pwrite-based I/O)
//
// Uses pread/pwrite instead of fseek+fread/fwrite to avoid potential offset
// truncation on platforms where long is 32-bit.  Offsets use off_t which is
// guaranteed 64-bit when _FILE_OFFSET_BITS=64 is defined (set in CMakeLists.txt).
// This is consistent with pq_block_cache.cpp which also uses pread.
#include "flash_store.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>

namespace curator {

bool FlashStore::open(const std::string& path) {
    close();

    // Create parent directories if needed
    std::filesystem::path p(path);
    auto parent = p.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent)) {
        std::filesystem::create_directories(parent);
    }

    fp_ = std::fopen(path.c_str(), "w+b");
    if (!fp_) {
        throw std::runtime_error("FlashStore: cannot open file: " + path +
                                 " (" + std::strerror(errno) + ")");
    }
    return true;
}

void FlashStore::close() {
    if (fp_ != nullptr) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
}

void FlashStore::write_vector(size_t offset, const float* vec, size_t d) {
    int fd = fileno(fp_);
    ssize_t nwritten = ::pwrite(fd, vec, d * sizeof(float),
                                static_cast<off_t>(offset));
    if (nwritten != static_cast<ssize_t>(d * sizeof(float))) {
        throw std::runtime_error("FlashStore: pwrite failed in write_vector");
    }
}

void FlashStore::write_leaf_region(size_t region_start, size_t /*region_bytes*/,
                                    const float* vectors, size_t n_vectors, size_t d) {
    int fd = fileno(fp_);
    size_t total = n_vectors * d;
    ssize_t nwritten = ::pwrite(fd, vectors, total * sizeof(float),
                                static_cast<off_t>(region_start));
    if (nwritten != static_cast<ssize_t>(total * sizeof(float))) {
        throw std::runtime_error("FlashStore: pwrite failed in write_leaf_region");
    }
    // No fflush needed: pwrite bypasses userspace buffering
}

void FlashStore::write_all_leaves(const std::vector<std::vector<float>>& leaf_vectors,
                                   size_t leaf_region_bytes, size_t d) {
    size_t offset = 0;
    for (const auto& leaf : leaf_vectors) {
        if (!leaf.empty()) {
            write_leaf_region(offset, leaf_region_bytes, leaf.data(),
                              leaf.size() / d, d);
        }
        offset += leaf_region_bytes;
    }
}

void FlashStore::read_vector(size_t offset, size_t d, float* out) const {
    int fd = fileno(fp_);
    ssize_t nread = ::pread(fd, out, d * sizeof(float),
                            static_cast<off_t>(offset));
    if (nread != static_cast<ssize_t>(d * sizeof(float))) {
        throw std::runtime_error("FlashStore: pread failed in read_vector");
    }
}

void FlashStore::read_batch(const std::vector<size_t>& offsets, size_t d,
                             std::vector<float>& out) const {
    out.resize(offsets.size() * d);

    // Sort requests by offset and merge adjacent reads
    std::vector<std::pair<size_t, size_t>> sorted; // (offset, original_index)
    sorted.reserve(offsets.size());
    for (size_t i = 0; i < offsets.size(); i++) {
        sorted.emplace_back(offsets[i], i);
    }
    std::sort(sorted.begin(), sorted.end());

    constexpr size_t MERGE_THRESHOLD = 65536; // 64KB merge threshold

    int fd = fileno(fp_);

    for (size_t i = 0; i < sorted.size(); ) {
        // Build a merge range
        size_t j = i + 1;
        size_t range_end = sorted[i].first + d * sizeof(float);
        while (j < sorted.size() &&
               sorted[j].first <= range_end &&
               (sorted[j].first - sorted[i].first) < MERGE_THRESHOLD) {
            range_end = std::max(range_end, sorted[j].first + d * sizeof(float));
            j++;
        }

        if (j == i + 1) {
            // Single read — use pread directly
            ssize_t nread = ::pread(fd, out.data() + sorted[i].second * d,
                                    d * sizeof(float),
                                    static_cast<off_t>(sorted[i].first));
            if (nread != static_cast<ssize_t>(d * sizeof(float))) {
                throw std::runtime_error("FlashStore: pread failed in read_batch");
            }
        } else {
            // Merged read
            size_t merge_start = sorted[i].first;
            size_t merge_bytes = sorted[j-1].first - merge_start + d * sizeof(float);
            std::vector<float> buf(merge_bytes / sizeof(float));
            ssize_t nread = ::pread(fd, buf.data(), merge_bytes,
                                    static_cast<off_t>(merge_start));
            if (nread != static_cast<ssize_t>(merge_bytes)) {
                throw std::runtime_error("FlashStore: pread failed in read_batch merge");
            }
            for (size_t k = i; k < j; k++) {
                size_t buf_off = (sorted[k].first - merge_start) / sizeof(float);
                std::memcpy(out.data() + sorted[k].second * d,
                            buf.data() + buf_off, d * sizeof(float));
            }
        }
        i = j;
    }
}

void FlashStore::truncate(size_t total_size) {
    if (fp_ == nullptr) return;
    int fd = fileno(fp_);
    if (ftruncate(fd, static_cast<off_t>(total_size)) != 0) {
        throw std::runtime_error("FlashStore: ftruncate failed");
    }
}

} // namespace curator
