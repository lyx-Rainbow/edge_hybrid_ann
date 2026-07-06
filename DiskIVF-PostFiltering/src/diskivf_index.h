// diskivf_index.h — Disk-backed IVF index with post-filtering
//
// Architecture:
//   - Build: K-means clustering → partition vectors into nlist clusters →
//            save each cluster as binary files on disk (vecs, vids, labels)
//   - Search (single-label): find nprobe nearest clusters by centroid distance →
//            load cluster data from disk → filter by label → L2 on qualified →
//            global top-k via min-heap
//   - Search (complex-predicate): same flow but filter by boolean expression
//   - Search (unfiltered): no label filtering, load + L2 on all cluster vectors
//
// Memory: only centroids + cluster_sizes kept in memory (minimal).
// Vectors are loaded from disk on-demand per query.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include "config.h"

class DiskIVFIndex {
public:
    explicit DiskIVFIndex(const DiskIVFConfig& cfg);

    // ── Build ──
    // vectors:       [n × d] float32, row-major contiguous
    // access_pairs:  [n_pairs × 2] int32, each row = (vid, tid)
    void build(size_t n, const float* vectors,
               const int32_t* access_pairs, size_t n_pairs);

    // ── Single-label search ──
    void search(const float* query, size_t k, int32_t tenant_id,
                float* distances, int32_t* labels) const;

    // ── Complex-predicate search ──
    void search_with_predicate(const float* query, size_t k,
                               const std::string& predicate,
                               float* distances, int32_t* labels) const;

    // ── Unfiltered search ──
    void search_unfiltered(const float* query, size_t k,
                           float* distances, int32_t* labels) const;

    // ── Info ──
    size_t memory_bytes() const;
    size_t disk_bytes() const;
    size_t ntotal() const { return ntotal_; }
    size_t dim() const { return d_; }
    size_t nlist() const { return nlist_; }

private:
    DiskIVFConfig cfg_;
    size_t d_ = 0;
    size_t ntotal_ = 0;
    size_t nlist_ = 0;

    std::vector<float> centroids_;          // [nlist × d]
    std::vector<int32_t> cluster_sizes_;    // [nlist]
    std::string disk_dir_;                  // cluster file directory

    // ── Cluster data structure (loaded on-demand) ──
    struct ClusterData {
        std::vector<float> vecs;                        // [n_vecs × d]
        std::vector<int32_t> vids;                      // [n_vecs] global vector IDs
        std::vector<std::vector<int32_t>> labels;       // [n_vecs] label lists per vector
    };

    // ── Cluster I/O ──
    void save_cluster(int32_t cid, const ClusterData& data) const;
    ClusterData load_cluster(int32_t cid) const;

    // ── Query helpers ──
    // Return IDs of the nprobe nearest clusters to query vector
    std::vector<int32_t> get_nearest_clusters(const float* query) const;

    // Filter cluster data by label/predicate, compute L2 distances,
    // merge into global top-k heap.  Uses distance::l2_sqr internally.
    void process_cluster(const float* query, size_t k, int32_t tenant_id,
                         const ClusterData& cluster,
                         std::vector<std::pair<float, int32_t>>& heap) const;

    void process_cluster_predicate(const float* query, size_t k,
                                   const std::vector<std::string>& tokens,
                                   const ClusterData& cluster,
                                   std::vector<std::pair<float, int32_t>>& heap) const;

    void process_cluster_unfiltered(const float* query, size_t k,
                                    const ClusterData& cluster,
                                    std::vector<std::pair<float, int32_t>>& heap) const;

    // Maintain top-k via sorted insert (k is small, O(k) is fine)
    static void heap_insert(std::vector<std::pair<float, int32_t>>& heap,
                            size_t k, float dist, int32_t vid);
};
