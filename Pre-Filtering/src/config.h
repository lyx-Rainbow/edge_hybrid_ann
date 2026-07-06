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
};
