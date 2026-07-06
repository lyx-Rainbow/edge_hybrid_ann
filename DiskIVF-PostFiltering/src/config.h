// config.h — Common infrastructure + DiskIVF configuration
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
// DiskIVF configuration
// ============================================================================
// Note: d is auto-detected from .npy shape by main.cpp (0 = auto-detect).
struct DiskIVFConfig {
    size_t d = 0;              // 0 = auto-detect from data
    size_t nlist = 64;         // number of K-means clusters
    size_t nprobe = 16;        // number of clusters to probe per query
    size_t clus_niter = 20;    // K-means iterations
    size_t k = 10;
    size_t num_warmup = 20;
    bool batch_query = false;
    std::string disk_dir;      // cluster file storage directory
};
