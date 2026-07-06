// distance.h — L2 squared distance functions (adapted from Curator)
// Stripped of Curator-specific dependencies: no common.h, no namespace curator.
#pragma once
#include <cmath>
#include <cstddef>

namespace distance {

// ── Scalar L2 squared distance ──
inline float l2_sqr(const float* x, const float* y, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; i++) {
        float diff = x[i] - y[i];
        sum += diff * diff;
    }
    return sum;
}

// ── 4-way unrolled L2: 4 independent accumulators for ILP ──
inline void l2_sqr_4way(
        const float* x,
        const float* y0, const float* y1,
        const float* y2, const float* y3,
        size_t d,
        float& d0, float& d1, float& d2, float& d3) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    size_t i = 0;
    for (; i + 3 < d; i += 4) {
        float dx0 = x[i] - y0[i];     s0 += dx0 * dx0;
        float dx1 = x[i] - y1[i];     s1 += dx1 * dx1;
        float dx2 = x[i] - y2[i];     s2 += dx2 * dx2;
        float dx3 = x[i] - y3[i];     s3 += dx3 * dx3;
        dx0 = x[i+1] - y0[i+1];       s0 += dx0 * dx0;
        dx1 = x[i+1] - y1[i+1];       s1 += dx1 * dx1;
        dx2 = x[i+1] - y2[i+1];       s2 += dx2 * dx2;
        dx3 = x[i+1] - y3[i+1];       s3 += dx3 * dx3;
        dx0 = x[i+2] - y0[i+2];       s0 += dx0 * dx0;
        dx1 = x[i+2] - y1[i+2];       s1 += dx1 * dx1;
        dx2 = x[i+2] - y2[i+2];       s2 += dx2 * dx2;
        dx3 = x[i+2] - y3[i+2];       s3 += dx3 * dx3;
        dx0 = x[i+3] - y0[i+3];       s0 += dx0 * dx0;
        dx1 = x[i+3] - y1[i+3];       s1 += dx1 * dx1;
        dx2 = x[i+3] - y2[i+3];       s2 += dx2 * dx2;
        dx3 = x[i+3] - y3[i+3];       s3 += dx3 * dx3;
    }
    for (; i < d; i++) {
        float dx0 = x[i] - y0[i];     s0 += dx0 * dx0;
        float dx1 = x[i] - y1[i];     s1 += dx1 * dx1;
        float dx2 = x[i] - y2[i];     s2 += dx2 * dx2;
        float dx3 = x[i] - y3[i];     s3 += dx3 * dx3;
    }
    d0 = s0; d1 = s1; d2 = s2; d3 = s3;
}

} // namespace distance
