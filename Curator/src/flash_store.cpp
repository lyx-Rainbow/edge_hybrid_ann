// flash_store.cpp — Flash disk storage implementation
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
    if (std::fseek(fp_, static_cast<long>(offset), SEEK_SET) != 0) {
        throw std::runtime_error("FlashStore: fseek failed in write_vector");
    }
    size_t written = std::fwrite(vec, sizeof(float), d, fp_);
    if (written != d) {
        throw std::runtime_error("FlashStore: fwrite failed in write_vector");
    }
}

void FlashStore::write_leaf_region(size_t region_start, size_t region_bytes,
                                    const float* vectors, size_t n_vectors, size_t d) {
    if (std::fseek(fp_, static_cast<long>(region_start), SEEK_SET) != 0) {
        throw std::runtime_error("FlashStore: fseek failed");
    }
    size_t total = n_vectors * d;
    size_t written = std::fwrite(vectors, sizeof(float), total, fp_);
    if (written != total) {
        throw std::runtime_error("FlashStore: fwrite failed");
    }
    std::fflush(fp_);
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
    if (std::fseek(fp_, static_cast<long>(offset), SEEK_SET) != 0) {
        throw std::runtime_error("FlashStore: fseek failed in read_vector");
    }
    size_t nread = std::fread(out, sizeof(float), d, fp_);
    if (nread != d) {
        throw std::runtime_error("FlashStore: fread failed in read_vector");
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
            // Single read
            read_vector(sorted[i].first, d, out.data() + sorted[i].second * d);
        } else {
            // Merged read
            size_t merge_bytes = sorted[j-1].first - sorted[i].first + d * sizeof(float);
            std::vector<float> buf(merge_bytes / sizeof(float));
            if (std::fseek(fp_, static_cast<long>(sorted[i].first), SEEK_SET) != 0) {
                throw std::runtime_error("FlashStore: fseek failed in read_batch");
            }
            std::fread(buf.data(), sizeof(float), buf.size(), fp_);
            for (size_t k = i; k < j; k++) {
                size_t buf_off = (sorted[k].first - sorted[i].first) / sizeof(float);
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
