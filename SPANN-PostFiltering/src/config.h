// config.h — Common infrastructure + SPANN PostFiltering configuration
#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

// ============================================================================
// Common infrastructure macros (inline; no shared/ dependency)
// ============================================================================
#define PREFETCH(ptr) __builtin_prefetch((ptr), 0, 3)

#define THROW_MSG(MSG) throw std::runtime_error(MSG)

#define THROW_IF_NOT(COND, MSG) \
    do { if (!(COND)) THROW_MSG(MSG); } while (0)

#define THROW_FMT(MSG, ...)                                      \
    do {                                                          \
        char _buf[1024];                                          \
        std::snprintf(_buf, sizeof(_buf), MSG, ##__VA_ARGS__);   \
        throw std::runtime_error(std::string(_buf));              \
    } while (0)

#define THROW_IF_NOT_FMT(COND, MSG, ...) \
    do { if (!(COND)) THROW_FMT(MSG, ##__VA_ARGS__); } while (0)

// ============================================================================
// SPANN PostFiltering configuration
// ============================================================================
// Note: d is auto-detected from .npy shape by main.cpp (0 = auto-detect).
struct SPANNConfig {
    // ── SPTAG native parameters ──
    size_t d = 0;                  // 0 = auto-detect from data
    std::string dist_method = "L2";   // L2 / Cosine
    size_t num_threads = 1;           // SPTAG internal search threads
                                       // (set to 1 to avoid nested parallelism
                                       //  with batch_query's OpenMP)
    size_t max_check = 8192;          // head index max check count
    size_t hash_exp = 8;              // BKTree hash table exponent
    size_t bkt_kmeans_k = 0;          // SelectHead K-means cluster count (0 = SPTAG default 32)
    size_t search_internal_result_num = 0;  // SSD search internal result cap
                                             // (0 = SPTAG default 64)

    // ── PostFiltering parameters ──
    size_t k = 10;
    size_t num_warmup = 20;
    size_t overfetch_factor = 50;     // fixed overfetch: k' = k * factor
    bool overfetch_adaptive = true;   // adaptive overfetch based on selectivity
    bool batch_query = false;

    // ── Data paths ──
    std::string index_dir;            // SPTAG index storage directory
    std::string metadata_path;        // label metadata file path (preprocessed)
};
