// config.h — Common infrastructure + PreFiltering configuration
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
// PreFiltering configuration
// ============================================================================
// Note: d is auto-detected from .npy shape by main.cpp (0 = auto-detect).
//       n_labels is auto-detected from access_pairs during build (0 = auto).
struct PreFilteringConfig {
    size_t d = 0;              // 0 = auto-detect from data
    size_t k = 10;
    size_t n_labels = 0;       // 0 = auto-detect from access_pairs (max tid + 1)
    size_t num_warmup = 20;
    bool batch_query = false;
    bool external_scan = false;         // chunked disk scan mode (lower memory, slower)
    size_t scan_chunk_vectors = 4096;   // vectors read per disk I/O in external mode
    std::string vector_file_path;       // binary float32 vector file for external mode

    // Deliberate chunked multi-load slowdown for the exact PreFilter baseline.
    // When enabled, each query processes its candidate list in small chunks and
    // pays a simulated I/O latency per chunk, without changing the search result.
    bool simulate_chunked_io = false;
    size_t io_chunk_vectors = 4096;     // candidate vectors per simulated load
    size_t io_chunk_delay_us = 1000;    // simulated I/O delay per load (microseconds)
    size_t io_base_delay_us = 0;        // fixed per-query simulated I/O overhead (microseconds)
};
