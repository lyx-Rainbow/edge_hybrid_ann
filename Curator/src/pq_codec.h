// pq_codec.h — Product Quantization codec (train, encode, ADC distance, block-cache I/O)
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "pq_block_cache.h"

namespace curator {

class PQCodec {
public:
    PQCodec() = default;
    ~PQCodec();

    // ── Training & encoding (unchanged) ──

    // Train PQ codebook from vectors [n × d]. nbits is HARD-CODED to 8.
    void train(size_t n, const float* x, size_t d, size_t M, size_t nbits);

    // Encode a single vector into uint8_t[M] code
    void encode(const float* x, std::vector<uint8_t>& code) const;

    // Encode all vectors
    void encode_all(size_t n, const float* x,
                    std::vector<std::vector<uint8_t>>& codes) const;

    // Build distance lookup table for a query.
    // Precondition: is_trained() must be true.
    void build_distance_table(const float* query,
                              std::vector<float>& d_table) const;

    // ── ★ Core distance computation (v5) ──

    // Compute ADC (Asymmetric Distance Computation) PQ distances for a list of vids.
    // d_table is [M × ksub], pre-built by build_distance_table (once per query).
    // Internally does vid→seq_idx lookup and batch-fetches PQ codes via block cache.
    // Thread-safe: block cache uses internal mutex; d_table is caller-owned.
    void compute_pq_distances(
            const std::vector<float>& d_table,
            const std::vector<int_vid_t>& vids,
            std::vector<std::pair<float, int_vid_t>>& output) const;

    // Compute exact L2 distances for ADC rerank (uses provided raw vectors)
    void compute_exact_batch(
            const float* query,
            const std::vector<int_vid_t>& vids,
            const float* vectors, // [n_vids × d]
            size_t d,
            std::vector<std::pair<float, int_vid_t>>& output) const;

    // ── Disk persistence ──

    // Write encoded PQ codes to disk with 32-byte header.
    void write_to_disk(const std::string& path,
                       const std::vector<std::vector<uint8_t>>& codes,
                       size_t M, size_t nbits);

    // ── Block cache management ──

    // Open the PQ codes file for on-demand block-cached access (after flush).
    bool open_cache(const std::string& path, size_t block_size, size_t max_blocks);

    // Close cache + fd. Does NOT delete the disk file.
    void close_cache();

    bool has_cache() const { return block_cache_ != nullptr && block_cache_->is_open(); }

    // ── Query state ──
    bool is_trained() const { return !pq_codebook_.empty(); }

    // ── Accessors ──
    size_t M()     const { return M_; }
    size_t nbits() const { return nbits_; }
    size_t ksub()  const { return ksub_; }
    size_t dsub()  const { return dsub_; }
    const std::vector<float>& codebook() const { return pq_codebook_; }

    // ── Statistics ──
    PQBlockCache::Stats cache_stats() const;
    void reset_cache_stats();
    size_t cache_bytes() const;

    // ── File path management ──
    const std::string& pq_file_path() const { return pq_file_path_; }
    void set_pq_file_path(const std::string& path) { pq_file_path_ = path; }
    void set_persist(bool persist) { persist_pq_codes_ = persist; }

    // ── Sequence mapping (pointers owned by CuratorIndex) ──
    void set_seq_maps(const std::vector<int_vid_t>* seq_to_vid,
                      const std::unordered_map<int_vid_t, size_t>* vid_to_seq) {
        seq_to_vid_ = seq_to_vid;
        vid_to_seq_ = vid_to_seq;
    }

    // ── Internal: single-code access (uses thread_local buffer) ──
    const uint8_t* get_code(size_t seq_idx) const;

private:
    size_t M_ = 0;
    size_t nbits_ = 8;
    size_t ksub_ = 0;
    size_t dsub_ = 0;
    size_t d_ = 0;

    std::vector<float> pq_codebook_;                     // [M × ksub × dsub]

    // ★ Block cache (mutable: LRU state updated during const search)
    mutable std::unique_ptr<PQBlockCache> block_cache_;

    // ★ Thread-local buffer for get_code()
    static thread_local std::vector<uint8_t> tl_code_buf_;

    // File path & persistence (for destructor cleanup)
    std::string pq_file_path_;
    bool persist_pq_codes_ = false;

    // Pointers to CuratorIndex's sequence mapping (non-owning)
    const std::vector<int_vid_t>* seq_to_vid_ = nullptr;
    const std::unordered_map<int_vid_t, size_t>* vid_to_seq_ = nullptr;
};

} // namespace curator
