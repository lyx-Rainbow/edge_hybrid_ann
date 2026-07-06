// kmeans.cpp — Lloyd K-means implementation
// Adapted from Curator/src/kmeans.cpp: replaced CURATOR_THROW_MSG with THROW_MSG,
// removed namespace curator, included config.h instead of common.h.
#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#include "config.h"

#ifdef _OPENMP
#include <omp.h>
#endif

KMeansResult kmeans(
        int d, int n, const float* x,
        int n_clusters, const KMeansConfig& cfg,
        int stride) {
    if (n <= 0) {
        THROW_MSG("kmeans: n must be positive");
    }
    if (n_clusters <= 0) {
        THROW_MSG("kmeans: n_clusters must be positive");
    }

    int effective_stride = (stride > 0) ? stride : d;
    int n_threads = cfg.n_threads;
    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = 1;
#endif
    }

    KMeansResult result;
    result.centroids.resize(n_clusters * d);
    result.assignments.resize(n);

    // ── Random initialization: sample n_clusters vectors without replacement ──
    std::mt19937 rng(cfg.seed);
    std::vector<int> indices(n);
    for (int i = 0; i < n; i++) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), rng);
    if (n_clusters > n) n_clusters = n;
    for (int c = 0; c < n_clusters; c++) {
        int idx = indices[c];
        const float* src = x + static_cast<size_t>(idx) * effective_stride;
        std::memcpy(result.centroids.data() + c * d, src, d * sizeof(float));
    }

    // ── Per-thread accumulation buffers ──
    std::vector<float> local_sum(n_threads * n_clusters * d, 0.0f);
    std::vector<int> local_count(n_threads * n_clusters, 0);

    // ── Iterate ──
    for (int iter = 0; iter < cfg.niter; iter++) {
        // Reset accumulators
        std::fill(local_sum.begin(), local_sum.end(), 0.0f);
        std::fill(local_count.begin(), local_count.end(), 0);

        // Assignment step (parallel)
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
        for (int i = 0; i < n; i++) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            const float* xi = x + static_cast<size_t>(i) * effective_stride;
            int best_c = 0;
            float best_dist = std::numeric_limits<float>::max();
            for (int c = 0; c < n_clusters; c++) {
                const float* cent = result.centroids.data() + c * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = xi[j] - cent[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            result.assignments[i] = best_c;

            // Accumulate
            float* sum_ptr = local_sum.data() + (tid * n_clusters + best_c) * d;
            for (int j = 0; j < d; j++) {
                sum_ptr[j] += xi[j];
            }
            local_count[tid * n_clusters + best_c]++;
        }

        // Reduce per-thread accumulators
        std::vector<float> global_sum(n_clusters * d, 0.0f);
        std::vector<int> global_count(n_clusters, 0);
        for (int t = 0; t < n_threads; t++) {
            for (int c = 0; c < n_clusters; c++) {
                float* gs = global_sum.data() + c * d;
                const float* ls = local_sum.data() + (t * n_clusters + c) * d;
                for (int j = 0; j < d; j++) {
                    gs[j] += ls[j];
                }
                global_count[c] += local_count[t * n_clusters + c];
            }
        }

        // Update centroids
        float global_mean[d];
        for (int j = 0; j < d; j++) global_mean[j] = 0.0f;
        int total_n = 0;
        for (int c = 0; c < n_clusters; c++) {
            if (global_count[c] > 0) {
                float* cent = result.centroids.data() + c * d;
                const float* gs = global_sum.data() + c * d;
                for (int j = 0; j < d; j++) {
                    cent[j] = gs[j] / global_count[c];
                    global_mean[j] += gs[j];
                }
                total_n += global_count[c];
            }
        }
        // Fix global_mean
        if (total_n > 0) {
            for (int j = 0; j < d; j++) global_mean[j] /= total_n;
        }

        // Handle empty clusters: set to global mean + small perturbation
        std::uniform_real_distribution<float> dist(-1e-6f, 1e-6f);
        for (int c = 0; c < n_clusters; c++) {
            if (global_count[c] == 0) {
                float* cent = result.centroids.data() + c * d;
                std::memcpy(cent, global_mean, d * sizeof(float));
                for (int j = 0; j < d; j++) {
                    cent[j] += dist(rng);
                }
            }
        }
    }

    return result;
}
