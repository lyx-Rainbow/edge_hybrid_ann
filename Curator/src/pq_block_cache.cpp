// pq_block_cache.cpp — PQBlockCache implementation
#include "pq_block_cache.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

#include "common.h"

namespace curator {

// Magic number "PQCD" — same as legacy for file compatibility.
// NOTE: The header layout differs from legacy (uint64 fields vs uint32).
//       This is intentional; files written by this code can only be read
//       by this code.  See PQ_EXTERNAL_STORAGE_PLAN.md §11.
static constexpr uint32_t PQ_DISK_MAGIC = 0x50514344;

// ============================================================================
// Destructor
// ============================================================================
PQBlockCache::~PQBlockCache() {
    close(false);  // close fd, don't delete file
}

// ============================================================================
// open
// ============================================================================
bool PQBlockCache::open(const std::string& path, size_t M, size_t nbits,
                         size_t block_size, size_t max_blocks) {
    if (fd_ >= 0) close(false);

    cfg_.block_size = block_size;
    cfg_.max_blocks = max_blocks;

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        fprintf(stderr, "PQBlockCache::open: cannot open '%s': %s\n",
                path.c_str(), strerror(errno));
        return false;
    }

    // Read and validate 32-byte header
    uint8_t header[HEADER_SIZE];
    ssize_t nread = ::pread(fd_, header, HEADER_SIZE, 0);
    if (nread != HEADER_SIZE) {
        fprintf(stderr, "PQBlockCache::open: short header read (%zd bytes)\n", nread);
        ::close(fd_); fd_ = -1;
        return false;
    }

    uint32_t magic;
    uint64_t m_val, nb_val, n_codes;
    std::memcpy(&magic,   header,       sizeof(magic));
    std::memcpy(&m_val,   header + 4,   sizeof(m_val));
    std::memcpy(&nb_val,  header + 12,  sizeof(nb_val));
    std::memcpy(&n_codes, header + 20,  sizeof(n_codes));

    if (magic != PQ_DISK_MAGIC) {
        fprintf(stderr, "PQBlockCache::open: bad magic 0x%08X, expected 0x%08X\n",
                magic, PQ_DISK_MAGIC);
        ::close(fd_); fd_ = -1;
        return false;
    }
    if (m_val != M) {
        fprintf(stderr, "PQBlockCache::open: M mismatch (file=%zu, expected=%zu)\n",
                static_cast<size_t>(m_val), M);
        ::close(fd_); fd_ = -1;
        return false;
    }
    if (nb_val != nbits) {
        fprintf(stderr, "PQBlockCache::open: nbits mismatch (file=%zu, expected=%zu)\n",
                static_cast<size_t>(nb_val), nbits);
        ::close(fd_); fd_ = -1;
        return false;
    }

    M_       = M;
    n_codes_ = static_cast<size_t>(n_codes);
    return true;
}

// ============================================================================
// close
//
// File deletion is managed by PQCodec (which owns pq_file_path_ and the
// persist_pq_codes_ flag), NOT by PQBlockCache.  The delete_file parameter
// is accepted for API symmetry but currently unused — PQBlockCache does not
// store the file path and cannot unlink independently.
// ============================================================================
void PQBlockCache::close(bool delete_file) {
    std::lock_guard<std::mutex> lock(mutex_);
    (void)delete_file;  // file lifecycle managed by PQCodec, not this class
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    cache_.clear();
    lru_list_.clear();
    n_codes_ = 0;
    M_ = 0;
}

// ============================================================================
// get_code — single code, thread-local-friendly (copies out)
// ============================================================================
bool PQBlockCache::get_code(size_t seq_idx, uint8_t* out) {
    if (seq_idx >= n_codes_) return false;

    size_t block_id = get_block_id(seq_idx, cfg_.block_size);

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(block_id);
    if (it == cache_.end()) {
        // Cache miss — load the block
        stats_.cache_misses++;
        while (cache_.size() >= cfg_.max_blocks) {
            evict_one_locked();
        }
        load_block_locked(block_id);
        it = cache_.find(block_id);
        if (it == cache_.end()) return false;  // shouldn't happen
    } else {
        stats_.cache_hits++;
    }

    // Update LRU: move to front
    lru_list_.erase(it->second.second);
    lru_list_.push_front(block_id);
    it->second.second = lru_list_.begin();

    // Copy out
    size_t block_offset = seq_idx - block_id * cfg_.block_size;
    std::memcpy(out, it->second.first.data.data() + block_offset * M_, M_);
    return true;
}

// ============================================================================
// prefetch_and_get_codes — batch fetch with ONE lock acquisition
//
// v5 optimisations:
//   • Loads missing blocks (de-duplicated), then fills pointers.
//   • LRU update de-duplicated: same block_id → only one erase+push_front.
//   • Stats counted per seq_idx (hit-rate denominator = total accesses).
// ============================================================================
void PQBlockCache::prefetch_and_get_codes(
        const std::vector<size_t>& seq_indices,
        std::vector<const uint8_t*>& out_codes) {

    out_codes.resize(seq_indices.size(), nullptr);

    std::lock_guard<std::mutex> lock(mutex_);

    // ── Pass 1: collect needed block_ids, load missing ones (de-duplicated) ──
    for (size_t seq_idx : seq_indices) {
        if (seq_idx >= n_codes_) continue;
        size_t block_id = get_block_id(seq_idx, cfg_.block_size);
        if (cache_.find(block_id) == cache_.end()) {
            while (cache_.size() >= cfg_.max_blocks) {
                evict_one_locked();
            }
            load_block_locked(block_id);
        }
    }

    // ── Pass 2: fill pointers + update LRU + stats (per seq_idx) ──
    size_t last_block_id = SIZE_MAX;
    for (size_t i = 0; i < seq_indices.size(); i++) {
        size_t seq_idx = seq_indices[i];
        if (seq_idx >= n_codes_) continue;   // out_codes[i] stays nullptr

        size_t block_id = get_block_id(seq_idx, cfg_.block_size);
        auto it = cache_.find(block_id);
        if (it == cache_.end()) {
            // Shouldn't happen after pass 1, but be defensive
            stats_.cache_misses++;
            continue;
        }

        // ★ v5: LRU update only when block_id changes (avoid 128× redundant ops)
        if (block_id != last_block_id) {
            lru_list_.erase(it->second.second);
            lru_list_.push_front(block_id);
            it->second.second = lru_list_.begin();
            last_block_id = block_id;
        }

        // ★ v5: count per seq_idx so hit_rate() denominator = total accesses
        stats_.cache_hits++;

        // Full-block allocation: offset = (seq_idx % block_size) * M_
        size_t block_offset = seq_idx - block_id * cfg_.block_size;
        out_codes[i] = it->second.first.data.data() + block_offset * M_;
    }
    // lock released here — callers MUST consume pointers before next cache mutation
}

// ============================================================================
// prefetch_blocks — warm cache with specific block_ids (standalone, NOT chained)
// ============================================================================
void PQBlockCache::prefetch_blocks(const std::unordered_set<size_t>& block_ids) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t block_id : block_ids) {
        if (cache_.find(block_id) != cache_.end()) continue;
        while (cache_.size() >= cfg_.max_blocks) {
            evict_one_locked();
        }
        load_block_locked(block_id);
    }
}

// ============================================================================
// cache_bytes
// ============================================================================
size_t PQBlockCache::cache_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& kv : cache_) {
        total += kv.second.first.data.size();
    }
    return total;
}

// ============================================================================
// stats (thread-safe copy)
// ============================================================================
PQBlockCache::Stats PQBlockCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void PQBlockCache::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.reset();
}

// ============================================================================
// load_block_locked (private — caller MUST hold mutex_)
//
// Full-block allocation strategy (v5 §17):
//   CacheEntry::data is always block_size * M_ bytes.
//   For the last partial block, ::pread returns 0 for bytes beyond EOF
//   (natural zero-fill).  Callers guard against reading past n_codes_
//   via the seq_idx >= n_codes_ check in pass 2.
// ============================================================================
void PQBlockCache::load_block_locked(size_t block_id) {
    size_t block_start = block_id * cfg_.block_size;
    size_t block_end   = std::min(block_start + cfg_.block_size, n_codes_);
    size_t n_in_block  = block_end - block_start;

    CacheEntry entry;
    entry.block_id = block_id;
    // Full-block allocation — always block_size * M_ bytes
    entry.data.resize(cfg_.block_size * M_, 0);

    size_t offset       = HEADER_SIZE + block_id * cfg_.block_size * M_;
    size_t bytes_to_read = n_in_block * M_;

    auto t0 = std::chrono::high_resolution_clock::now();
    ssize_t nread = ::pread(fd_, entry.data.data(), bytes_to_read,
                            static_cast<off_t>(offset));
    auto t1 = std::chrono::high_resolution_clock::now();

    if (nread != static_cast<ssize_t>(bytes_to_read)) {
        fprintf(stderr, "PQBlockCache::load_block_locked: I/O error on block %zu: "
                "expected %zd, got %zd (errno=%s)\n",
                block_id, bytes_to_read, nread, strerror(errno));
        // Entry is still inserted (partial data) — callers will see the
        // seq_idx >= n_codes_ guard in pass 2 and skip invalid entries.
    }

    stats_.bytes_read += bytes_to_read;
    stats_.io_count++;
    stats_.io_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Insert at LRU head
    lru_list_.push_front(block_id);
    cache_[block_id] = {std::move(entry), lru_list_.begin()};
}

// ============================================================================
// evict_one_locked (private — caller MUST hold mutex_)
// ============================================================================
void PQBlockCache::evict_one_locked() {
    if (lru_list_.empty()) return;
    size_t victim = lru_list_.back();
    lru_list_.pop_back();
    cache_.erase(victim);
}

} // namespace curator
