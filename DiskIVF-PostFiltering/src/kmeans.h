// kmeans.h — Lloyd K-means clustering (L2 distance, OpenMP parallelized)
// Adapted from Curator/src/kmeans.h: removed namespace curator, no common.h dependency.
#pragma once

#include <cstddef>
#include <vector>

struct KMeansConfig {
    int niter = 20;
    int min_points_per_centroid = 1;
    int seed = 42;              // ★ Match FAISS default (seed=42)
    int n_threads = 0;          // 0 = use all available cores
};

struct KMeansResult {
    std::vector<float> centroids;    // n_clusters × d, row-major
    std::vector<int> assignments;    // n vectors → cluster ID
};

// Standard Lloyd K-means. stride=0 means contiguous (d==stride);
// stride>0 used by PQ sub-space access (vectors interleaved at stride=d).
KMeansResult kmeans(
        int d, int n, const float* x,
        int n_clusters, const KMeansConfig& cfg,
        int stride = 0);
