// flash_store.h — Flash disk storage for full-precision vectors (RAII file handle)
#pragma once

#include <cstdio>
#include <cstddef>
#include <string>
#include <vector>

#include "common.h"

namespace curator {

class FlashStore {
public:
    FlashStore() = default;

    // Open the flash file for reading + writing. Creates parent directories if needed.
    bool open(const std::string& path);

    // Close file handle
    void close();

    // Destructor auto-closes
    ~FlashStore() { close(); }

    bool is_open() const { return fp_ != nullptr; }

    // Write a vector at a specific byte offset (fseek + fwrite)
    void write_vector(size_t offset, const float* vec, size_t d);

    // Write a leaf region: vectors_offset[i] + region_size for each leaf
    void write_leaf_region(size_t region_start, size_t region_bytes,
                           const float* vectors, size_t n_vectors, size_t d);

    // Batch write all leaf regions
    void write_all_leaves(const std::vector<std::vector<float>>& leaf_vectors,
                          size_t leaf_region_bytes, size_t d);

    // Read a single vector at offset into pre-allocated out buffer
    void read_vector(size_t offset, size_t d, float* out) const;

    // Batch read: sort requests by offset, merge adjacent reads
    void read_batch(const std::vector<size_t>& offsets, size_t d,
                    std::vector<float>& out) const;

    // Pre-allocate file size (ftruncate)
    void truncate(size_t total_size);

private:
    FILE* fp_ = nullptr;
};

} // namespace curator
