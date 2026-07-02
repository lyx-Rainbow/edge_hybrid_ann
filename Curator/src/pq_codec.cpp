// pq_codec.cpp — Product Quantization codec implementation (block-cache based)
#include "pq_codec.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

#include "common.h"
#include "distance.h"
#include "kmeans.h"

namespace curator {

// Magic number "PQCD" — same magic as legacy for file-level compatibility.
// NOTE: header layout differs from legacy (uint64 vs uint32 fields).
//       Files written by this code can only be read by this code.
//       See PQ_EXTERNAL_STORAGE_PLAN.md §11.
static constexpr uint32_t PQ_DISK_MAGIC = 0x50514344;

// Thread-local buffer for get_code() — one per thread, zero contention.
thread_local std::vector<uint8_t> PQCodec::tl_code_buf_;

// ============================================================================
// Destructor
// ============================================================================
PQCodec::~PQCodec() {
    if (block_cache_) {
        block_cache_->close(false);
        block_cache_.reset();
    }
    if (!persist_pq_codes_ && !pq_file_path_.empty()) {
        ::unlink(pq_file_path_.c_str());
    }
}

// ============================================================================
// train (unchanged)
// ============================================================================
void PQCodec::train(size_t n, const float* x, size_t d, size_t M, size_t nbits) {
    if (nbits != 8) {
        CURATOR_THROW_FMT("PQCodec: only nbits=8 is supported, got %zu", nbits);
    }

    M_ = M;
    nbits_ = nbits;
    ksub_ = 1 << nbits;  // 256 for nbits=8
    dsub_ = d / M;
    d_ = d;

    if (d % M != 0) {
        CURATOR_THROW_FMT("PQCodec: d=%zu must be divisible by M=%zu", d, M);
    }

    pq_codebook_.resize(M * ksub_ * dsub_);

    KMeansConfig kcfg;
    kcfg.niter = 20;
    kcfg.seed = 1234;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t m = 0; m < M; m++) {
        auto result = kmeans(static_cast<int>(dsub_), static_cast<int>(n), x + m * dsub_,
                              static_cast<int>(ksub_), kcfg, static_cast<int>(d));

        std::memcpy(pq_codebook_.data() + m * ksub_ * dsub_,
                    result.centroids.data(),
                    ksub_ * dsub_ * sizeof(float));
    }
}

// ============================================================================
// encode / encode_all (unchanged)
// ============================================================================
void PQCodec::encode(const float* x, std::vector<uint8_t>& code) const {
    code.resize(M_);
    for (size_t m = 0; m < M_; m++) {
        const float* sub = x + m * dsub_;
        const float* cb = pq_codebook_.data() + m * ksub_ * dsub_;
        uint8_t best = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (size_t k = 0; k < ksub_; k++) {
            float dist = l2_sqr(sub, cb + k * dsub_, dsub_);
            if (dist < best_dist) {
                best_dist = dist;
                best = static_cast<uint8_t>(k);
            }
        }
        code[m] = best;
    }
}

void PQCodec::encode_all(size_t n, const float* x,
                          std::vector<std::vector<uint8_t>>& codes) const {
    codes.resize(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; i++) {
        encode(x + i * d_, codes[i]);
    }
}

// ============================================================================
// build_distance_table
// v5 §18: added is_trained() precondition check
// ============================================================================
void PQCodec::build_distance_table(const float* query,
                                    std::vector<float>& d_table) const {
    if (!is_trained()) return;

    d_table.resize(M_ * ksub_);
    for (size_t m = 0; m < M_; m++) {
        const float* q_sub = query + m * dsub_;
        const float* cb = pq_codebook_.data() + m * ksub_ * dsub_;
        for (size_t k = 0; k < ksub_; k++) {
            d_table[m * ksub_ + k] = l2_sqr(q_sub, cb + k * dsub_, dsub_);
        }
    }
}

// ============================================================================
// compute_pq_distances — ★ v5 core: single-lock batch fetch + ADC
//
// Design (v5 plan §4.2 / Appendix A):
//   1. vid → seq_idx conversion (with SIZE_MAX sentinels)
//   2. prefetch_and_get_codes — ONE lock: load missing blocks + fill pointers + LRU
//   3. ADC accumulation — lock-free, pointers consumed immediately
//
//   NO separate prefetch_blocks call (v5 §13 — avoids double locking).
//   d_table size is asserted (v5 §23).
// ============================================================================
void PQCodec::compute_pq_distances(
        const std::vector<float>& d_table,
        const std::vector<int_vid_t>& vids,
        std::vector<std::pair<float, int_vid_t>>& output) const {

    if (!block_cache_ || !vid_to_seq_) return;
    assert(d_table.size() == M_ * ksub_ && "d_table size mismatch");

    // 1. vid → seq_idx conversion (with SIZE_MAX sentinels)
    std::vector<size_t> seq_indices;
    seq_indices.reserve(vids.size());
    for (int_vid_t vid : vids) {
        auto it = vid_to_seq_->find(vid);
        if (it != vid_to_seq_->end()) {
            seq_indices.push_back(it->second);
        } else {
            seq_indices.push_back(SIZE_MAX);  // sentinel: vid not found
        }
    }

    // 2. Batch fetch all codes — ONE lock acquisition.
    //    prefetch_and_get_codes handles missing-block loading internally;
    //    no separate prefetch_blocks call needed.
    std::vector<const uint8_t*> codes;
    block_cache_->prefetch_and_get_codes(seq_indices, codes);

    // 3. ADC distance computation — LOCK-FREE.
    //    codes[i] points into cache internals; must be consumed before
    //    any subsequent cache mutation.  Since this loop does no cache
    //    calls, the contract is satisfied.  See pq_block_cache.h memory model.
    for (size_t i = 0; i < vids.size(); i++) {
        const uint8_t* code = codes[i];
        if (!code) continue;  // sentinel or out-of-range

        float dist = 0.0f;
        for (size_t m = 0; m < M_; m++) {
            dist += d_table[m * ksub_ + code[m]];
        }
        output.emplace_back(dist, vids[i]);
    }
}

// ============================================================================
// compute_exact_batch (unchanged)
// ============================================================================
void PQCodec::compute_exact_batch(
        const float* query,
        const std::vector<int_vid_t>& vids,
        const float* vectors,
        size_t d,
        std::vector<std::pair<float, int_vid_t>>& output) const {
    for (size_t i = 0; i < vids.size(); i++) {
        float dist = l2_sqr(query, vectors + i * d, d);
        output.emplace_back(dist, vids[i]);
    }
}

// ============================================================================
// write_to_disk (unchanged header format — 32 bytes)
// ============================================================================
void PQCodec::write_to_disk(const std::string& path,
                             const std::vector<std::vector<uint8_t>>& codes,
                             size_t M, size_t nbits) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        CURATOR_THROW_FMT("PQCodec: cannot open file for writing: %s", path.c_str());
    }

    // Header: magic(4) + M(8) + nbits(8) + n_codes(8) + reserved(4) = 32 bytes
    uint32_t magic   = PQ_DISK_MAGIC;
    uint64_t m_val   = M;
    uint64_t nb_val  = nbits;
    uint64_t n_codes = codes.size();
    uint32_t reserved = 0;

    std::fwrite(&magic,   sizeof(magic),   1, fp);
    std::fwrite(&m_val,   sizeof(m_val),   1, fp);
    std::fwrite(&nb_val,  sizeof(nb_val),  1, fp);
    std::fwrite(&n_codes, sizeof(n_codes), 1, fp);
    std::fwrite(&reserved,sizeof(reserved),1, fp);

    // Write codes: each code is M bytes (for nbits=8)
    size_t code_bytes = M;
    for (const auto& code : codes) {
        std::fwrite(code.data(), 1, code_bytes, fp);
    }

    std::fclose(fp);
}

// ============================================================================
// open_cache / close_cache
// ============================================================================
bool PQCodec::open_cache(const std::string& path, size_t block_size, size_t max_blocks) {
    if (!block_cache_) {
        block_cache_ = std::make_unique<PQBlockCache>();
    }
    return block_cache_->open(path, M_, nbits_, block_size, max_blocks);
}

void PQCodec::close_cache() {
    if (block_cache_) {
        block_cache_->close(false);
        block_cache_.reset();
    }
}

// ============================================================================
// get_code — single code via block cache, uses thread_local buffer
// ============================================================================
const uint8_t* PQCodec::get_code(size_t seq_idx) const {
    if (!block_cache_) return nullptr;

    tl_code_buf_.resize(M_);
    bool ok = block_cache_->get_code(seq_idx, tl_code_buf_.data());
    return ok ? tl_code_buf_.data() : nullptr;
}

// ============================================================================
// Statistics
// ============================================================================
PQBlockCache::Stats PQCodec::cache_stats() const {
    if (!block_cache_) return {};
    return block_cache_->stats();
}

void PQCodec::reset_cache_stats() {
    if (block_cache_) block_cache_->reset_stats();
}

size_t PQCodec::cache_bytes() const {
    if (!block_cache_) return 0;
    return block_cache_->cache_bytes();
}

} // namespace curator
