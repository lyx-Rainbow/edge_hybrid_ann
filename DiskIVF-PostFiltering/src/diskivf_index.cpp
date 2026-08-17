// diskivf_index.cpp — DiskIVFIndex implementation
#include "diskivf_index.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sys/stat.h>
#include <sys/types.h>

#include "distance.h"
#include "kmeans.h"
#include "predicate.h"

// ============================================================================
// Constructor
// ============================================================================
DiskIVFIndex::DiskIVFIndex(const DiskIVFConfig& cfg) : cfg_(cfg) {}

// ============================================================================
// Build
// ============================================================================
void DiskIVFIndex::build(size_t n, const float* vectors,
                          const int32_t* access_pairs, size_t n_pairs) {
    d_ = cfg_.d;
    ntotal_ = n;
    nlist_ = cfg_.nlist;

    // Set disk directory
    disk_dir_ = cfg_.disk_dir;
    if (disk_dir_.empty()) {
        disk_dir_ = "diskivf_data";
    }
    mkdir(disk_dir_.c_str(), 0755);
    // (ignore EEXIST)

    // ── 1. Build vid_to_labels from access pairs ──
    std::vector<std::vector<int32_t>> vid_to_labels(n);
    for (size_t i = 0; i < n_pairs; i++) {
        int32_t vid = access_pairs[i * 2];
        int32_t tid = access_pairs[i * 2 + 1];
        THROW_IF_NOT_FMT(vid >= 0 && static_cast<size_t>(vid) < n,
                         "vid %d out of range [0, %zu)", vid, n);
        vid_to_labels[vid].push_back(tid);
    }
    // Sort labels per vector (enables binary_search during query)
    for (size_t i = 0; i < n; i++) {
        std::sort(vid_to_labels[i].begin(), vid_to_labels[i].end());
    }

    // ── 2. K-means clustering ──
    printf("  Running K-means (nlist=%zu, d=%zu, niter=%zu)...\n",
           nlist_, d_, cfg_.clus_niter);
    KMeansConfig km_cfg;
    km_cfg.niter = static_cast<int>(cfg_.clus_niter);
    km_cfg.seed = 42;  // Match FAISS default

    KMeansResult km_result = kmeans(
        static_cast<int>(d_), static_cast<int>(n), vectors,
        static_cast<int>(nlist_), km_cfg);

    // Copy centroids
    centroids_ = std::move(km_result.centroids);

    // ── 3. Group vectors by cluster ──
    cluster_sizes_.assign(nlist_, 0);
    std::vector<std::vector<int32_t>> cluster_vids(nlist_);

    for (size_t i = 0; i < n; i++) {
        int cid = km_result.assignments[i];
        THROW_IF_NOT_FMT(cid >= 0 && static_cast<size_t>(cid) < nlist_,
                         "K-means assignment %d out of range [0, %zu)", cid, nlist_);
        cluster_sizes_[cid]++;
        cluster_vids[cid].push_back(static_cast<int32_t>(i));
    }

    // ── 4. Save each cluster to disk ──
    printf("  Saving %zu clusters to disk (%s)...\n", nlist_, disk_dir_.c_str());
    for (int32_t cid = 0; cid < static_cast<int32_t>(nlist_); cid++) {
        size_t nc = cluster_vids[cid].size();

        ClusterData data;
        data.vecs.resize(nc * d_);
        data.vids.reserve(nc);
        data.labels.reserve(nc);

        for (size_t j = 0; j < nc; j++) {
            int32_t vid = cluster_vids[cid][j];
            data.vids.push_back(vid);
            // Copy vector
            std::memcpy(data.vecs.data() + j * d_,
                       vectors + static_cast<size_t>(vid) * d_,
                       d_ * sizeof(float));
            // Copy labels (already sorted)
            data.labels.push_back(vid_to_labels[vid]);
        }

        save_cluster(cid, data);

        if ((cid + 1) % 100 == 0 || cid + 1 == static_cast<int32_t>(nlist_)) {
            printf("    %d/%zu clusters saved\r", cid + 1, nlist_);
            fflush(stdout);
        }
    }
    printf("\n");

    // ── 5. Save centroids + cluster_sizes + metadata ──
    {
        std::string path = disk_dir_ + "/centroids.bin";
        FILE* f = fopen(path.c_str(), "wb");
        THROW_IF_NOT_FMT(f, "Cannot open %s for writing", path.c_str());
        fwrite(centroids_.data(), sizeof(float), centroids_.size(), f);
        fclose(f);
    }
    {
        std::string path = disk_dir_ + "/cluster_sizes.bin";
        FILE* f = fopen(path.c_str(), "wb");
        THROW_IF_NOT_FMT(f, "Cannot open %s for writing", path.c_str());
        fwrite(cluster_sizes_.data(), sizeof(int32_t), cluster_sizes_.size(), f);
        fclose(f);
    }
    {
        // metadata.bin: 3 int32_t values for search-mode reload
        std::string path = disk_dir_ + "/metadata.bin";
        FILE* f = fopen(path.c_str(), "wb");
        THROW_IF_NOT_FMT(f, "Cannot open %s for writing", path.c_str());
        int32_t meta[3] = {static_cast<int32_t>(d_), static_cast<int32_t>(ntotal_),
                           static_cast<int32_t>(nlist_)};
        fwrite(meta, sizeof(int32_t), 3, f);
        fclose(f);
    }

    printf("  Build complete: %zu vectors, %zu clusters, %.2f MB centroids\n",
           ntotal_, nlist_, centroids_.size() * sizeof(float) / (1024.0 * 1024.0));
}

// ============================================================================
// Load metadata from pre-built index on disk
// ============================================================================
void DiskIVFIndex::load_metadata(const std::string& disk_dir) {
    disk_dir_ = disk_dir;

    // ── Read metadata.bin ──
    {
        std::string path = disk_dir_ + "/metadata.bin";
        FILE* f = fopen(path.c_str(), "rb");
        THROW_IF_NOT_FMT(f, "Cannot open metadata.bin for reading: %s", path.c_str());
        int32_t meta[3] = {0, 0, 0};
        size_t nread = fread(meta, sizeof(int32_t), 3, f);
        fclose(f);
        THROW_IF_NOT_FMT(nread == 3,
            "metadata.bin truncated: expected 3 int32_t, got %zu", nread);
        d_ = static_cast<size_t>(meta[0]);
        ntotal_ = static_cast<size_t>(meta[1]);
        nlist_ = static_cast<size_t>(meta[2]);
    }

    // ── Read centroids ──
    {
        std::string path = disk_dir_ + "/centroids.bin";
        FILE* f = fopen(path.c_str(), "rb");
        THROW_IF_NOT_FMT(f, "Cannot open centroids.bin for reading: %s", path.c_str());
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        size_t n_floats = static_cast<size_t>(fsize) / sizeof(float);
        centroids_.resize(n_floats);
        size_t nread = fread(centroids_.data(), sizeof(float), n_floats, f);
        fclose(f);
        THROW_IF_NOT_FMT(nread == n_floats,
            "centroids.bin truncated: expected %zu floats, got %zu", n_floats, nread);
    }

    // ── Read cluster_sizes ──
    {
        std::string path = disk_dir_ + "/cluster_sizes.bin";
        FILE* f = fopen(path.c_str(), "rb");
        THROW_IF_NOT_FMT(f, "Cannot open cluster_sizes.bin for reading: %s", path.c_str());
        cluster_sizes_.resize(nlist_);
        size_t nread = fread(cluster_sizes_.data(), sizeof(int32_t), nlist_, f);
        fclose(f);
        THROW_IF_NOT_FMT(nread == nlist_,
            "cluster_sizes.bin truncated: expected %zu int32_t, got %zu", nlist_, nread);
    }

    printf("  Metadata loaded: d=%zu, ntotal=%zu, nlist=%zu\n", d_, ntotal_, nlist_);
}

// ============================================================================
// Cluster I/O (binary format)
// ============================================================================
void DiskIVFIndex::save_cluster(int32_t cid, const ClusterData& data) const {
    char path[512];

    // vecs.bin: [n_vecs * d] float32
    std::snprintf(path, sizeof(path), "%s/c%d_vecs.bin", disk_dir_.c_str(), cid);
    FILE* f = fopen(path, "wb");
    THROW_IF_NOT_FMT(f, "Cannot open %s for writing", path);
    fwrite(data.vecs.data(), sizeof(float), data.vecs.size(), f);
    fclose(f);

    // vids.bin: [n_vecs] int32
    std::snprintf(path, sizeof(path), "%s/c%d_vids.bin", disk_dir_.c_str(), cid);
    f = fopen(path, "wb");
    THROW_IF_NOT_FMT(f, "Cannot open %s for writing", path);
    fwrite(data.vids.data(), sizeof(int32_t), data.vids.size(), f);
    fclose(f);

    // mds.bin: per-vector: [1 int32 n_labels] + [n_labels int32 labels]
    std::snprintf(path, sizeof(path), "%s/c%d_mds.bin", disk_dir_.c_str(), cid);
    f = fopen(path, "wb");
    THROW_IF_NOT_FMT(f, "Cannot open %s for writing", path);
    for (size_t i = 0; i < data.labels.size(); i++) {
        int32_t n_labels = static_cast<int32_t>(data.labels[i].size());
        fwrite(&n_labels, sizeof(int32_t), 1, f);
        if (n_labels > 0) {
            fwrite(data.labels[i].data(), sizeof(int32_t), n_labels, f);
        }
    }
    fclose(f);
}

DiskIVFIndex::ClusterData DiskIVFIndex::load_cluster(int32_t cid) const {
    ClusterData data;
    char path[512];

    // vecs.bin
    std::snprintf(path, sizeof(path), "%s/c%d_vecs.bin", disk_dir_.c_str(), cid);
    FILE* f = fopen(path, "rb");
    THROW_IF_NOT_FMT(f, "Cannot open %s for reading", path);
    fseek(f, 0, SEEK_END);
    long vecs_bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n_floats = static_cast<size_t>(vecs_bytes) / sizeof(float);
    data.vecs.resize(n_floats);
    fread(data.vecs.data(), sizeof(float), n_floats, f);
    fclose(f);

    size_t n_vecs = n_floats / d_;

    // vids.bin
    std::snprintf(path, sizeof(path), "%s/c%d_vids.bin", disk_dir_.c_str(), cid);
    f = fopen(path, "rb");
    THROW_IF_NOT_FMT(f, "Cannot open %s for reading", path);
    data.vids.resize(n_vecs);
    fread(data.vids.data(), sizeof(int32_t), n_vecs, f);
    fclose(f);

    // mds.bin
    std::snprintf(path, sizeof(path), "%s/c%d_mds.bin", disk_dir_.c_str(), cid);
    f = fopen(path, "rb");
    THROW_IF_NOT_FMT(f, "Cannot open %s for reading", path);
    data.labels.resize(n_vecs);
    for (size_t i = 0; i < n_vecs; i++) {
        int32_t n_labels = 0;
        fread(&n_labels, sizeof(int32_t), 1, f);
        if (n_labels > 0) {
            data.labels[i].resize(n_labels);
            fread(data.labels[i].data(), sizeof(int32_t), n_labels, f);
        }
    }
    fclose(f);

    return data;
}

// ============================================================================
// get_nearest_clusters — centroid L2 distance, return top-nprobe
// ============================================================================
std::vector<int32_t> DiskIVFIndex::get_nearest_clusters(const float* query) const {
    size_t nprobe = std::min(cfg_.nprobe, nlist_);

    // Compute L2 distances to all centroids
    std::vector<std::pair<float, int32_t>> dists;
    dists.reserve(nlist_);
    for (size_t c = 0; c < nlist_; c++) {
        const float* cent = centroids_.data() + c * d_;
        float dist = distance::l2_sqr(query, cent, d_);
        dists.emplace_back(dist, static_cast<int32_t>(c));
    }

    // Partial sort: top-nprobe by distance
    std::partial_sort(dists.begin(), dists.begin() + static_cast<long>(nprobe),
                      dists.end());

    std::vector<int32_t> result;
    result.reserve(nprobe);
    for (size_t i = 0; i < nprobe; i++) {
        result.push_back(dists[i].second);
    }
    return result;
}

// ============================================================================
// heap_insert — maintain a sorted list of top-k (distance, vid) pairs
// ============================================================================
void DiskIVFIndex::heap_insert(std::vector<std::pair<float, int32_t>>& heap,
                                size_t k, float dist, int32_t vid) {
    if (heap.size() < k) {
        // Find insertion position (ascending order)
        auto it = std::lower_bound(heap.begin(), heap.end(),
                                   std::make_pair(dist, vid));
        heap.insert(it, std::make_pair(dist, vid));
    } else if (dist < heap.back().first) {
        // Replace the worst (largest distance) if better
        auto it = std::lower_bound(heap.begin(), heap.end(),
                                   std::make_pair(dist, vid));
        heap.insert(it, std::make_pair(dist, vid));
        heap.pop_back();  // remove the now-excess largest element
    }
}

// ============================================================================
// process_cluster — single-label filter + L2 → global heap
// ============================================================================
void DiskIVFIndex::process_cluster(const float* query, size_t k, int32_t tenant_id,
                                    const ClusterData& cluster,
                                    std::vector<std::pair<float, int32_t>>& heap) const {
    for (size_t i = 0; i < cluster.vids.size(); i++) {
        // Filter: check if tenant_id is in this vector's label list
        const auto& labels = cluster.labels[i];
        if (!std::binary_search(labels.begin(), labels.end(), tenant_id)) {
            continue;
        }
        // Compute L2 distance
        const float* y = cluster.vecs.data() + i * d_;
        float dist = distance::l2_sqr(query, y, d_);
        heap_insert(heap, k, dist, cluster.vids[i]);
    }
}

// ============================================================================
// process_cluster_predicate — complex-predicate filter + L2 → global heap
// ============================================================================
void DiskIVFIndex::process_cluster_predicate(
        const float* query, size_t k,
        const std::vector<std::string>& tokens,
        const ClusterData& cluster,
        std::vector<std::pair<float, int32_t>>& heap) const {
    for (size_t i = 0; i < cluster.vids.size(); i++) {
        // Filter: evaluate predicate against this vector's label list
        const auto& labels = cluster.labels[i];
        if (!predicate::evaluate(tokens,
                                 labels.data(),
                                 labels.data() + labels.size())) {
            continue;
        }
        // Compute L2 distance
        const float* y = cluster.vecs.data() + i * d_;
        float dist = distance::l2_sqr(query, y, d_);
        heap_insert(heap, k, dist, cluster.vids[i]);
    }
}

// ============================================================================
// process_cluster_unfiltered — no filter, all vectors → global heap
// ============================================================================
void DiskIVFIndex::process_cluster_unfiltered(
        const float* query, size_t k,
        const ClusterData& cluster,
        std::vector<std::pair<float, int32_t>>& heap) const {
    for (size_t i = 0; i < cluster.vids.size(); i++) {
        const float* y = cluster.vecs.data() + i * d_;
        float dist = distance::l2_sqr(query, y, d_);
        heap_insert(heap, k, dist, cluster.vids[i]);
    }
}

// ============================================================================
// Search — single-label
// ============================================================================
void DiskIVFIndex::search(const float* query, size_t k, int32_t tenant_id,
                           float* distances, int32_t* labels) const {
    // Out-of-range tenant_id → empty result
    if (tenant_id < 0) {
        search_unfiltered(query, k, distances, labels);
        return;
    }

    auto nearest = get_nearest_clusters(query);
    std::vector<std::pair<float, int32_t>> heap;

    for (int32_t cid : nearest) {
        if (cluster_sizes_[cid] == 0) continue;
        ClusterData cluster = load_cluster(cid);
        process_cluster(query, k, tenant_id, cluster, heap);
    }

    // Output top-k
    size_t found = heap.size();
    for (size_t i = 0; i < k; i++) {
        if (i < found) {
            labels[i] = heap[i].second;
            distances[i] = heap[i].first;
        } else {
            labels[i] = -1;
            distances[i] = std::numeric_limits<float>::max();
        }
    }
}

// ============================================================================
// Search — complex-predicate
// ============================================================================
void DiskIVFIndex::search_with_predicate(const float* query, size_t k,
                                          const std::string& predicate_str,
                                          float* distances, int32_t* labels) const {
    // Tokenize once
    auto tokens = predicate::tokenize(predicate_str);

    auto nearest = get_nearest_clusters(query);
    std::vector<std::pair<float, int32_t>> heap;

    for (int32_t cid : nearest) {
        if (cluster_sizes_[cid] == 0) continue;
        ClusterData cluster = load_cluster(cid);
        process_cluster_predicate(query, k, tokens, cluster, heap);
    }

    // Output top-k
    size_t found = heap.size();
    for (size_t i = 0; i < k; i++) {
        if (i < found) {
            labels[i] = heap[i].second;
            distances[i] = heap[i].first;
        } else {
            labels[i] = -1;
            distances[i] = std::numeric_limits<float>::max();
        }
    }
}

// ============================================================================
// Search — unfiltered (no label filtering)
// ============================================================================
void DiskIVFIndex::search_unfiltered(const float* query, size_t k,
                                      float* distances, int32_t* labels) const {
    auto nearest = get_nearest_clusters(query);
    std::vector<std::pair<float, int32_t>> heap;

    for (int32_t cid : nearest) {
        if (cluster_sizes_[cid] == 0) continue;
        ClusterData cluster = load_cluster(cid);
        process_cluster_unfiltered(query, k, cluster, heap);
    }

    // Output top-k
    size_t found = heap.size();
    for (size_t i = 0; i < k; i++) {
        if (i < found) {
            labels[i] = heap[i].second;
            distances[i] = heap[i].first;
        } else {
            labels[i] = -1;
            distances[i] = std::numeric_limits<float>::max();
        }
    }
}

// ============================================================================
// Memory accounting
// ============================================================================
size_t DiskIVFIndex::memory_bytes() const {
    size_t mem = 0;
    mem += centroids_.capacity() * sizeof(float);
    mem += cluster_sizes_.capacity() * sizeof(int32_t);
    return mem;
}

size_t DiskIVFIndex::disk_bytes() const {
    // Walk disk_dir_ and sum file sizes
    size_t total = 0;
    // Use a simple approach: iterate over expected cluster files
    for (size_t cid = 0; cid < nlist_; cid++) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/c%zu_vecs.bin", disk_dir_.c_str(), cid);
        FILE* f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            total += static_cast<size_t>(ftell(f));
            fclose(f);
        }
        std::snprintf(path, sizeof(path), "%s/c%zu_vids.bin", disk_dir_.c_str(), cid);
        f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            total += static_cast<size_t>(ftell(f));
            fclose(f);
        }
        std::snprintf(path, sizeof(path), "%s/c%zu_mds.bin", disk_dir_.c_str(), cid);
        f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            total += static_cast<size_t>(ftell(f));
            fclose(f);
        }
    }
    // Add metadata files
    {
        std::string path = disk_dir_ + "/centroids.bin";
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            total += static_cast<size_t>(ftell(f));
            fclose(f);
        }
    }
    {
        std::string path = disk_dir_ + "/cluster_sizes.bin";
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            total += static_cast<size_t>(ftell(f));
            fclose(f);
        }
    }
    return total;
}
