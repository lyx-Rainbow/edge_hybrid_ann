// distance.h — Distance functions for Curator index (header-only)
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

#include "common.h"

namespace curator {

// ============================================================================
// l2_sqr — scalar L2 squared distance (replaces FAISS fvec_L2sqr)
// ============================================================================
inline float l2_sqr(const float* x, const float* y, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; i++) {
        float diff = x[i] - y[i];
        sum += diff * diff;
    }
    return sum;
}

// ============================================================================
// l2_sqr_4way — true 4-way unrolled L2: 4 independent accumulators
// ============================================================================
inline void l2_sqr_4way(
        const float* x,
        const float* y0, const float* y1, const float* y2, const float* y3,
        size_t d,
        float& d0, float& d1, float& d2, float& d3) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    size_t i = 0;
    for (; i + 3 < d; i += 4) {
        float dx0 = x[i] - y0[i]; s0 += dx0 * dx0;
        float dx1 = x[i] - y1[i]; s1 += dx1 * dx1;
        float dx2 = x[i] - y2[i]; s2 += dx2 * dx2;
        float dx3 = x[i] - y3[i]; s3 += dx3 * dx3;
        dx0 = x[i+1] - y0[i+1]; s0 += dx0 * dx0;
        dx1 = x[i+1] - y1[i+1]; s1 += dx1 * dx1;
        dx2 = x[i+1] - y2[i+1]; s2 += dx2 * dx2;
        dx3 = x[i+1] - y3[i+1]; s3 += dx3 * dx3;
        dx0 = x[i+2] - y0[i+2]; s0 += dx0 * dx0;
        dx1 = x[i+2] - y1[i+2]; s1 += dx1 * dx1;
        dx2 = x[i+2] - y2[i+2]; s2 += dx2 * dx2;
        dx3 = x[i+2] - y3[i+2]; s3 += dx3 * dx3;
        dx0 = x[i+3] - y0[i+3]; s0 += dx0 * dx0;
        dx1 = x[i+3] - y1[i+3]; s1 += dx1 * dx1;
        dx2 = x[i+3] - y2[i+3]; s2 += dx2 * dx2;
        dx3 = x[i+3] - y3[i+3]; s3 += dx3 * dx3;
    }
    for (; i < d; i++) {
        float dx0 = x[i] - y0[i]; s0 += dx0 * dx0;
        float dx1 = x[i] - y1[i]; s1 += dx1 * dx1;
        float dx2 = x[i] - y2[i]; s2 += dx2 * dx2;
        float dx3 = x[i] - y3[i]; s3 += dx3 * dx3;
    }
    d0 = s0; d1 = s1; d2 = s2; d3 = s3;
}

// ============================================================================
// compute_batch_dists — parameterized batch distance computation
// Takes a lambda get_vector_ptr(vid) → const float* for flash/buffer abstraction
// ============================================================================
template <typename GetVectorFn>
inline void compute_batch_dists(
        const float* x,
        const int_vid_t* vids, size_t n_vids,
        GetVectorFn get_vector_ptr,
        size_t d,
        std::vector<std::pair<float, int_vid_t>>& output) {
    if (n_vids == 0) return;

    // For small batches, simple scalar computation
    if (n_vids < 4) {
        for (size_t i = 0; i < n_vids; i++) {
            const float* y = get_vector_ptr(vids[i]);
            output.emplace_back(l2_sqr(x, y, d), vids[i]);
        }
        return;
    }

    // Prefetch first batch
    for (size_t j = 0; j < std::min(n_vids, size_t(8)); j++) {
        PREFETCH(get_vector_ptr(vids[j]));
    }

    // Process in batches of 4 with prefetch
    size_t i = 0;
    for (; i + 3 < n_vids; i += 4) {
        if (i + 8 < n_vids) {
            PREFETCH(get_vector_ptr(vids[i + 4]));
            PREFETCH(get_vector_ptr(vids[i + 5]));
            PREFETCH(get_vector_ptr(vids[i + 6]));
            PREFETCH(get_vector_ptr(vids[i + 7]));
        }

        const float* y0 = get_vector_ptr(vids[i]);
        const float* y1 = get_vector_ptr(vids[i + 1]);
        const float* y2 = get_vector_ptr(vids[i + 2]);
        const float* y3 = get_vector_ptr(vids[i + 3]);

        float d0, d1, d2, d3;
        l2_sqr_4way(x, y0, y1, y2, y3, d, d0, d1, d2, d3);

        output.emplace_back(d0, vids[i]);
        output.emplace_back(d1, vids[i + 1]);
        output.emplace_back(d2, vids[i + 2]);
        output.emplace_back(d3, vids[i + 3]);
    }

    // Handle remaining
    for (; i < n_vids; i++) {
        const float* y = get_vector_ptr(vids[i]);
        output.emplace_back(l2_sqr(x, y, d), vids[i]);
    }
}

} // namespace curator
