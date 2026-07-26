// pq_block_cache.h — LRU block cache for PQ codes stored on disk
//
// Memory model: two access patterns with different pointer lifetime guarantees.
//
//   Interface                     | Returns              | Lifetime
//   ------------------------------|----------------------|------------------------------
//   get_code(seq_idx, out)        | copies to caller buf | N/A (data is copied)
//   prefetch_and_get_codes(...)   | pointers into cache  | until next cache mutation(*)
//
//   (*) Any thread calling get_code / prefetch_and_get_codes / prefetch_blocks may
//       trigger LRU eviction and invalidate pointers.  Callers MUST finish reading
//       before touching the cache again.  compute_pq_distances satisfies this by
//       doing ADC accumulation immediately after prefetch_and_get_codes returns,
//       without any intervening cache calls.
//
//   When the working set fits in cache (max_blocks * block_size >= dataset size),
//   eviction never happens and pointers are effectively stable.  Default sizing
//   (max_blocks=256, block_size=4096) covers ~1M codes — enough for typical
//   multi-tenant workloads.  For extreme concurrency + tiny cache scenarios,
//   future work could add reference counting (see PQ_EXTERNAL_STORAGE_PLAN.md §14).
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace curator {

class PQBlockCache {
public:
    struct Config {
        size_t block_size = 4096;    // number of PQ codes per block
        size_t max_blocks = 256;     // maximum cached blocks
    };

    PQBlockCache() = default;
    ~PQBlockCache();

    // Non-copyable
    PQBlockCache(const PQBlockCache&) = delete;
    PQBlockCache& operator=(const PQBlockCache&) = delete;

    // ── Lifecycle ──

    // Open the PQ codes disk file (read-only), validate header.
    // Precondition: M and nbits must match the values used at write time.
    // block_size and max_blocks configure the cache behaviour.
    bool open(const std::string& path, size_t M, size_t nbits,
              size_t block_size, size_t max_blocks);

    // Close file and clear cache.  If delete_file=true, unlink the disk file.
    void close(bool delete_file = false);

    bool is_open() const { return fd_ >= 0; }

    // ── Core access (thread-safe) ──

    // Get a single PQ code for seq_idx, copied into out (must hold ≥ M_ bytes).
    // Returns false if seq_idx is out of range.
    bool get_code(size_t seq_idx, uint8_t* out);

    // ★ Batch fetch: load missing blocks + fill pointers + update LRU in ONE lock.
    // seq_indices may contain SIZE_MAX sentinels (→ corresponding out_codes[i]=nullptr).
    // Pointers point into cache internals — see file-header memory-model contract.
    void prefetch_and_get_codes(
            const std::vector<size_t>& seq_indices,
            std::vector<const uint8_t*>& out_codes);

    // Prefetch specific block-ids (e.g. for cache warming).  Loads missing blocks.
    // Do NOT chain with prefetch_and_get_codes — the latter already does its own
    // block loading; chaining causes double-lock + redundant hash-table lookups.
    void prefetch_blocks(const std::unordered_set<size_t>& block_ids);

    // ── Query state ──
    size_t n_codes()  const { return n_codes_; }
    size_t M()        const { return M_; }
    size_t cache_size() const { return cache_.size(); }   // approximate (no lock)
    size_t max_blocks() const { return cfg_.max_blocks; }
    size_t block_size() const { return cfg_.block_size; }
    size_t cache_bytes() const;                            // total cached bytes

    // ── Statistics ──
    struct Stats {
        uint64_t cache_hits   = 0;
        uint64_t cache_misses = 0;
        uint64_t bytes_read   = 0;     // total bytes read from disk
        uint64_t io_count     = 0;     // number of disk I/O operations
        double   io_time_ms   = 0;     // total time spent in disk I/O
        void reset() { *this = Stats{}; }
        double hit_rate() const {
            uint64_t total = cache_hits + cache_misses;
            return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
        }
    };
    Stats stats() const;                 // thread-safe copy
    void reset_stats();                  // thread-safe reset

    // ── Static utility ──
    static size_t get_block_id(size_t seq_idx, size_t block_size) {
        return seq_idx / block_size;
    }

private:
    Config cfg_;
    int fd_ = -1;
    size_t n_codes_ = 0;
    size_t M_ = 0;                       // PQ code bytes (= pq_M, since nbits=8)
    static constexpr size_t HEADER_SIZE = 32;

    // Cache entry — data is ALWAYS block_size * M_ bytes (full-block allocation).
    // For the last partial block the tail bytes are naturally zero-filled by ::pread
    // reading past EOF.  This simplifies offset arithmetic.
    struct CacheEntry {
        size_t block_id;
        std::vector<uint8_t> data;       // [block_size * M_] bytes
    };

    // LRU: list head = most-recently-used, tail = least-recently-used
    std::list<size_t> lru_list_;         // block_id list
    std::unordered_map<size_t,           // block_id → (entry, lru_iterator)
        std::pair<CacheEntry, std::list<size_t>::iterator>> cache_;

    mutable std::mutex mutex_;           // protects all cache state
    Stats stats_;                        // guarded by mutex_

    // ── Internal helpers (caller MUST hold mutex_) ──
    void load_block_locked(size_t block_id);
    void evict_one_locked();
};

} // namespace curator
