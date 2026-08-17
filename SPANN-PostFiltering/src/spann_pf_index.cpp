// spann_pf_index.cpp — SPTAG-based SPANN index with PostFiltering wrapper
//
// Wraps SPTAG SPANN::Index<float> with label-based post-filtering.
// SPTAG provides approximate nearest-neighbor search; we overfetch results
// and post-filter by label/predicate in this wrapper layer.
#include "spann_pf_index.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>

// ── SPTAG headers (AnnService/ is the include root) ──
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SearchQuery.h"
#include "inc/Core/SearchResult.h"
#include "inc/Core/Common.h"
#include "inc/Core/VectorSet.h"

#include "predicate.h"

namespace fs = std::filesystem;

// ============================================================================
// Construction / Destruction
// ============================================================================

SPANNPostFilterIndex::SPANNPostFilterIndex(const SPANNConfig& cfg)
    : cfg_(cfg) {}

SPANNPostFilterIndex::~SPANNPostFilterIndex() {
    // SPTAG index is managed by shared_ptr — auto cleanup
}

// ============================================================================
// Build
// ============================================================================

void SPANNPostFilterIndex::build(size_t n, const float* vectors,
                                  const int32_t* access_pairs,
                                  size_t n_pairs) {
    ntotal_ = n;
    d_ = cfg_.d;
    if (d_ == 0) {
        THROW_MSG("config.d must be set (auto-detected from .npy by main.cpp)");
    }

    // ── Step 1: Build label metadata ──
    // Scan access_pairs to determine n_labels and populate vid_to_labels_
    size_t n_labels = 0;
    if (n_pairs > 0 && access_pairs != nullptr) {
        // First pass: find max label ID
        for (size_t i = 0; i < n_pairs; i++) {
            int32_t tid = access_pairs[i * 2 + 1];
            if (tid < 0) continue;
            if (static_cast<size_t>(tid) >= n_labels) {
                n_labels = static_cast<size_t>(tid) + 1;
            }
        }
    }

    // Allocate label data structures
    vid_to_labels_.resize(n);
    label_counts_.resize(n_labels, 0);

    // Second pass: populate
    for (size_t i = 0; i < n_pairs; i++) {
        int32_t vid = access_pairs[i * 2];
        int32_t tid = access_pairs[i * 2 + 1];
        if (vid < 0 || tid < 0 || static_cast<size_t>(vid) >= n) continue;
        vid_to_labels_[vid].push_back(tid);
        label_counts_[tid]++;
        total_label_assignments_++;
    }

    // Sort each vector's label list (enables binary_search for PostFilter)
    for (auto& labels : vid_to_labels_) {
        if (!labels.empty()) {
            std::sort(labels.begin(), labels.end());
        }
    }

    printf("  Label metadata: %zu labels, %zu total assignments\n",
           n_labels, total_label_assignments_);

    // ── Step 2: Build SPTAG index ──
    printf("  Building SPTAG SPANN index (n=%zu, d=%zu)...\n", n, d_);

    // Create SPANN<float> index directly (bypass factory for reliability)
    spann_index_ = std::make_shared<SPTAG::SPANN::Index<float>>();

    if (!spann_index_) {
        THROW_MSG("Failed to create SPTAG SPANN index instance");
    }

    // Configure SPTAG parameters
    configure_spann_parameters();

    // Build the index using BasicVectorSet (SPTAG standard approach)
    auto t0 = std::chrono::high_resolution_clock::now();

    // Wrap vectors in SPTAG BasicVectorSet
    SPTAG::ByteArray vec_arr = SPTAG::ByteArray::Alloc(sizeof(float) * n * d_);
    std::memcpy(vec_arr.Data(), vectors, sizeof(float) * n * d_);
    auto vector_set = std::make_shared<SPTAG::BasicVectorSet>(
        vec_arr,
        SPTAG::VectorValueType::Float,
        static_cast<SPTAG::DimensionType>(d_),
        static_cast<SPTAG::SizeType>(n)
    );

    SPTAG::ErrorCode ec = spann_index_->BuildIndex(
        vector_set, nullptr, false, false, false);

    auto t1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();

    if (ec != SPTAG::ErrorCode::Success) {
        THROW_FMT("SPTAG BuildIndex failed with error code %d",
                  static_cast<int>(ec));
    }
    printf("  SPTAG build complete: %.2f s, %zu samples\n",
           build_s, spann_index_->GetNumSamples());

    // ── Step 3: Save SPTAG index + metadata ──
    if (!cfg_.index_dir.empty()) {
        printf("  Saving SPTAG index to %s ...\n", cfg_.index_dir.c_str());
        fs::create_directories(cfg_.index_dir);

        ec = spann_index_->SaveIndex(cfg_.index_dir);
        if (ec != SPTAG::ErrorCode::Success) {
            THROW_FMT("SPTAG SaveIndex failed with error code %d",
                      static_cast<int>(ec));
        }

        // Save label metadata alongside SPTAG index
        std::string meta_path = cfg_.index_dir + "/metadata.bin";
        save_metadata(meta_path);
        printf("  Index saved: %s\n", cfg_.index_dir.c_str());
    }
}

// ============================================================================
// Load (pre-built index)
// ============================================================================

void SPANNPostFilterIndex::load(const std::string& index_dir,
                                 const std::string& metadata_path) {
    cfg_.index_dir = index_dir;
    cfg_.metadata_path = metadata_path;

    printf("  Loading SPTAG index from %s ...\n", index_dir.c_str());

    // Create SPANN<float> instance directly
    spann_index_ = std::make_shared<SPTAG::SPANN::Index<float>>();

    if (!spann_index_) {
        THROW_MSG("Failed to create SPTAG SPANN index instance for loading");
    }

    SPTAG::ErrorCode ec = SPTAG::VectorIndex::LoadIndex(
        index_dir, spann_index_);
    if (ec != SPTAG::ErrorCode::Success) {
        THROW_FMT("SPTAG LoadIndex failed with error code %d",
                  static_cast<int>(ec));
    }

    // Extract metadata from loaded index
    ntotal_ = spann_index_->GetNumSamples();
    d_ = spann_index_->GetFeatureDim();
    printf("  Loaded: %zu vectors, d=%zu\n", ntotal_, d_);

    // Load label metadata
    std::string meta_path = metadata_path.empty()
        ? (index_dir + "/metadata.bin") : metadata_path;
    if (fs::exists(meta_path)) {
        load_metadata(meta_path);
    } else {
        printf("  Warning: metadata file not found at %s, PostFilter disabled\n",
               meta_path.c_str());
    }

    // Re-configure SPTAG search parameters (thread count, max_check).
    // Use set_search_parameters to avoid re-triggering build operations.
    set_search_parameters(cfg_.max_check, cfg_.overfetch_factor);
}

// ============================================================================
// configure_spann_parameters
// ============================================================================

void SPANNPostFilterIndex::configure_spann_parameters() {
    if (!spann_index_) return;

    size_t threads = cfg_.batch_query ? 1 : cfg_.num_threads;
    std::string dist = cfg_.dist_method.empty() ? "L2" : cfg_.dist_method;

    // ── Base parameters (section "Base") ──
    spann_index_->SetParameter("DistCalcMethod", dist.c_str(), "Base");
    spann_index_->SetParameter("NumberOfThreads",
                               std::to_string(threads).c_str(), "Base");
    if (!cfg_.index_dir.empty()) {
        spann_index_->SetParameter("IndexDirectory",
                                   cfg_.index_dir.c_str(), "Base");
    }

    // ── Head selection (section "SelectHead") ──
    // CRITICAL: m_selectHead defaults to false — must explicitly enable.
    spann_index_->SetParameter("isExecute", "true", "SelectHead");
    if (cfg_.bkt_kmeans_k > 0) {
        // Number of K-means clusters used to select head vectors
        // (SPTAG default is 32; larger => finer partitioning => higher recall).
        spann_index_->SetParameter("BKTKmeansK",
                                   std::to_string(cfg_.bkt_kmeans_k).c_str(),
                                   "SelectHead");
    }

    // ── Head building (section "BuildHead") ──
    // CRITICAL: m_buildHead defaults to false — must explicitly enable.
    spann_index_->SetParameter("isExecute", "true", "BuildHead");
    // Head index search parameters (routed to BKT/KDT via SetParameter)
    spann_index_->SetParameter("MaxCheck",
                               std::to_string(cfg_.max_check).c_str(), "BuildHead");
    spann_index_->SetParameter("HashTableExponent",
                               std::to_string(cfg_.hash_exp).c_str(), "BuildHead");

    // ── SSD index building + search (section "BuildSSDIndex") ──
    // CRITICAL: m_buildSsdIndex defaults to false — must explicitly enable.
    spann_index_->SetParameter("BuildSsdIndex", "true", "BuildSSDIndex");
    // CRITICAL: m_enableSSD defaults to false — must explicitly enable for search.
    spann_index_->SetParameter("isExecute", "true", "BuildSSDIndex");
    spann_index_->SetParameter("NumberOfThreads",
                               std::to_string(threads).c_str(), "BuildSSDIndex");
    spann_index_->SetParameter("MaxCheck",
                               std::to_string(cfg_.max_check).c_str(), "BuildSSDIndex");
    if (cfg_.search_internal_result_num > 0) {
        // SPTAG caps SSD search candidates at SearchInternalResultNum
        // (default 64). Raising it lets post-filtering see more candidates,
        // which directly raises filtered recall.
        spann_index_->SetParameter("SearchInternalResultNum",
                                   std::to_string(cfg_.search_internal_result_num).c_str(),
                                   "BuildSSDIndex");
    }
}

// ============================================================================
// set_search_parameters — safe for post-load reconfiguration
// ============================================================================
void SPANNPostFilterIndex::set_search_parameters(size_t max_check,
                                                   size_t overfetch_factor) {
    if (!spann_index_) return;

    cfg_.max_check = max_check;
    cfg_.overfetch_factor = overfetch_factor;

    // Only set search-time knobs on the head index sections.
    // Do NOT set isExecute — that would re-trigger build operations
    // and crash on a pre-built loaded index.
    spann_index_->SetParameter("MaxCheck",
                               std::to_string(max_check).c_str(), "BuildHead");
    spann_index_->SetParameter("MaxCheck",
                               std::to_string(max_check).c_str(), "BuildSSDIndex");

    size_t threads = cfg_.batch_query ? 1 : cfg_.num_threads;
    spann_index_->SetParameter("NumberOfThreads",
                               std::to_string(threads).c_str(), "BuildSSDIndex");
}

// ============================================================================
// compute_overfetch_k — adaptive overfetch factor
// ============================================================================

size_t SPANNPostFilterIndex::compute_overfetch_k(size_t k,
                                                  int32_t tenant_id) const {
    if (!cfg_.overfetch_adaptive) {
        return std::min(ntotal_, k * cfg_.overfetch_factor);
    }

    // Adaptive mode: adjust overfetch based on label selectivity
    // selectivity = count(label) / N
    // overfetch = k * clamp(k / selectivity, 10, 200)
    //            = k * clamp(k * N / count, 10, 200)

    if (tenant_id < 0 ||
        static_cast<size_t>(tenant_id) >= label_counts_.size()) {
        // Unknown tenant → fallback to fixed factor
        return std::min(ntotal_, k * cfg_.overfetch_factor);
    }

    size_t count = label_counts_[tenant_id];
    if (count == 0) {
        // Zero-count label → no candidates expected, but still try fixed overfetch
        return std::min(ntotal_, k * cfg_.overfetch_factor);
    }

    if (ntotal_ == 0) return k;

    // selectivity = count / ntotal_
    double selectivity = static_cast<double>(count) / ntotal_;

    // multiplier = clamp(k / selectivity, 10, 200)
    double multiplier = static_cast<double>(k) / selectivity;
    multiplier = std::max(10.0, std::min(200.0, multiplier));

    size_t overfetch = static_cast<size_t>(k * multiplier);

    // Cap at total vectors (can't fetch more than exist)
    overfetch = std::min(ntotal_, overfetch);

    // Floor at k (must return at least k)
    overfetch = std::max(k, overfetch);

    return overfetch;
}

// ============================================================================
// search — single-label PostFilter
// ============================================================================

void SPANNPostFilterIndex::search(const float* query, size_t k,
                                   int32_t tenant_id,
                                   float* distances,
                                   int32_t* labels) const {
    if (!spann_index_) {
        THROW_MSG("Index not built or loaded");
    }

    // Step 1: Compute overfetch k'
    size_t overfetch_k = compute_overfetch_k(k, tenant_id);

    // Step 2: Call SPTAG SearchIndex
    SPTAG::QueryResult query_result(
        static_cast<const void*>(query),
        static_cast<int>(overfetch_k),
        false  // no metadata
    );

    SPTAG::ErrorCode ec = spann_index_->SearchIndex(query_result, false);
    if (ec != SPTAG::ErrorCode::Success) {
        THROW_FMT("SPTAG SearchIndex failed with error code %d",
                  static_cast<int>(ec));
    }

    // Step 3: PostFilter — collect all qualifying results
    // (Do NOT assume SPTAG returns distance-sorted results;
    //  we collect first, then sort by distance ourselves.)
    std::vector<std::pair<float, int32_t>> qualified;
    qualified.reserve(overfetch_k);

    for (int i = 0; i < query_result.GetResultNum(); i++) {
        auto* res = query_result.GetResult(i);
        if (res->VID < 0) break;

        int32_t vid = static_cast<int32_t>(res->VID);
        if (static_cast<size_t>(vid) >= vid_to_labels_.size()) continue;

        // Check if vector has the required label
        const auto& lbls = vid_to_labels_[vid];
        if (std::binary_search(lbls.begin(), lbls.end(), tenant_id)) {
            qualified.emplace_back(res->Dist, vid);
        }
    }

    // Step 4: Sort qualifying results by distance (ascending)
    std::sort(qualified.begin(), qualified.end());

    // Step 5: Fill output with top-k
    size_t found = 0;
    for (; found < k && found < qualified.size(); found++) {
        labels[found] = qualified[found].second;
        distances[found] = qualified[found].first;
    }

    // Step 6: Padding for insufficient results
    for (; found < k; found++) {
        labels[found] = -1;
        distances[found] = std::numeric_limits<float>::max();
    }
}

// ============================================================================
// search_with_predicate — complex-predicate PostFilter
// ============================================================================

void SPANNPostFilterIndex::search_with_predicate(
        const float* query, size_t k,
        const std::string& predicate_str,
        float* distances, int32_t* labels) const {
    if (!spann_index_) {
        THROW_MSG("Index not built or loaded");
    }

    // Tokenize predicate once (reused for all candidate checks)
    auto tokens = predicate::tokenize(predicate_str);

    // Fixed overfetch for complex predicates (no selectivity info)
    size_t overfetch_k = std::min(ntotal_, k * cfg_.overfetch_factor);

    // Call SPTAG SearchIndex
    SPTAG::QueryResult query_result(
        static_cast<const void*>(query),
        static_cast<int>(overfetch_k),
        false
    );

    SPTAG::ErrorCode ec = spann_index_->SearchIndex(query_result, false);
    if (ec != SPTAG::ErrorCode::Success) {
        THROW_FMT("SPTAG SearchIndex failed with error code %d",
                  static_cast<int>(ec));
    }

    // Collect all qualifying results
    std::vector<std::pair<float, int32_t>> qualified;
    qualified.reserve(overfetch_k);

    for (int i = 0; i < query_result.GetResultNum(); i++) {
        auto* res = query_result.GetResult(i);
        if (res->VID < 0) break;

        int32_t vid = static_cast<int32_t>(res->VID);
        if (static_cast<size_t>(vid) >= vid_to_labels_.size()) continue;

        // Evaluate predicate against vector's label set
        const auto& lbls = vid_to_labels_[vid];
        if (predicate::evaluate(tokens, lbls.data(),
                                lbls.data() + lbls.size())) {
            qualified.emplace_back(res->Dist, vid);
        }
    }

    // Sort and take top-k
    std::sort(qualified.begin(), qualified.end());

    size_t found = 0;
    for (; found < k && found < qualified.size(); found++) {
        labels[found] = qualified[found].second;
        distances[found] = qualified[found].first;
    }

    for (; found < k; found++) {
        labels[found] = -1;
        distances[found] = std::numeric_limits<float>::max();
    }
}

// ============================================================================
// search_unfiltered — direct SPTAG search (no PostFilter)
// ============================================================================

void SPANNPostFilterIndex::search_unfiltered(
        const float* query, size_t k,
        float* distances, int32_t* labels) const {
    if (!spann_index_) {
        THROW_MSG("Index not built or loaded");
    }

    // Direct SPTAG search — request exactly k results
    SPTAG::QueryResult query_result(
        static_cast<const void*>(query),
        static_cast<int>(k),
        false
    );

    SPTAG::ErrorCode ec = spann_index_->SearchIndex(query_result, false);
    if (ec != SPTAG::ErrorCode::Success) {
        THROW_FMT("SPTAG SearchIndex failed with error code %d",
                  static_cast<int>(ec));
    }

    // Copy results directly (SPTAG returns sorted results for unfiltered search)
    for (size_t i = 0; i < k; i++) {
        auto* res = query_result.GetResult(static_cast<int>(i));
        if (res && res->VID >= 0) {
            labels[i] = static_cast<int32_t>(res->VID);
            distances[i] = res->Dist;
        } else {
            labels[i] = -1;
            distances[i] = std::numeric_limits<float>::max();
        }
    }
}

// ============================================================================
// Info
// ============================================================================

size_t SPANNPostFilterIndex::memory_bytes() const {
    size_t mem = 0;

    // Label metadata
    mem += vid_to_labels_.capacity() * sizeof(std::vector<int32_t>);
    for (const auto& v : vid_to_labels_) {
        mem += v.capacity() * sizeof(int32_t);
    }
    mem += label_counts_.capacity() * sizeof(size_t);

    // SPTAG memory is internal — we report a rough estimate
    if (spann_index_) {
        // SPTAG doesn't expose memory directly, but BufferSize gives hints
        auto bufs = spann_index_->BufferSize();
        for (auto sz : *bufs) {
            mem += static_cast<size_t>(sz);
        }
    }

    return mem;
}

size_t SPANNPostFilterIndex::disk_bytes() const {
    size_t total = 0;
    if (cfg_.index_dir.empty()) return 0;

    try {
        for (const auto& entry : fs::recursive_directory_iterator(cfg_.index_dir)) {
            if (entry.is_regular_file()) {
                total += static_cast<size_t>(entry.file_size());
            }
        }
    } catch (...) {
        // Directory might not exist or be inaccessible
    }

    return total;
}

// ============================================================================
// Metadata I/O
// ============================================================================

void SPANNPostFilterIndex::save_metadata(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    THROW_IF_NOT_FMT(out.is_open(), "Cannot open metadata file for writing: %s",
                     path.c_str());

    // Header: n_vectors (uint32)
    uint32_t n = static_cast<uint32_t>(vid_to_labels_.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));

    // For each vector: n_labels (uint32) + labels (int32 array, sorted)
    for (const auto& labels : vid_to_labels_) {
        uint32_t nl = static_cast<uint32_t>(labels.size());
        out.write(reinterpret_cast<const char*>(&nl), sizeof(nl));
        if (nl > 0) {
            out.write(reinterpret_cast<const char*>(labels.data()),
                      nl * sizeof(int32_t));
        }
    }

    out.close();
    printf("  Metadata saved: %s (%u vectors)\n", path.c_str(), n);
}

void SPANNPostFilterIndex::load_metadata(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    THROW_IF_NOT_FMT(in.is_open(), "Cannot open metadata file for reading: %s",
                     path.c_str());

    // Header: n_vectors (uint32)
    uint32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    THROW_IF_NOT_FMT(n == ntotal_,
                     "Metadata vector count mismatch: %u vs %zu", n, ntotal_);

    vid_to_labels_.resize(n);

    // Determine n_labels from max label ID seen
    size_t max_label = 0;

    // Read per-vector label lists
    for (uint32_t i = 0; i < n; i++) {
        uint32_t nl = 0;
        in.read(reinterpret_cast<char*>(&nl), sizeof(nl));
        if (nl > 0) {
            vid_to_labels_[i].resize(nl);
            in.read(reinterpret_cast<char*>(vid_to_labels_[i].data()),
                    nl * sizeof(int32_t));
            // Assume labels are already sorted (as written by save_metadata)
            for (int32_t lbl : vid_to_labels_[i]) {
                if (lbl >= 0 && static_cast<size_t>(lbl) > max_label) {
                    max_label = static_cast<size_t>(lbl);
                }
            }
        }
    }

    // Rebuild label_counts_
    label_counts_.resize(max_label + 1, 0);
    total_label_assignments_ = 0;
    for (const auto& labels : vid_to_labels_) {
        for (int32_t lbl : labels) {
            if (lbl >= 0) {
                label_counts_[lbl]++;
                total_label_assignments_++;
            }
        }
    }

    in.close();
    printf("  Metadata loaded: %s (%u vectors, %zu labels, %zu assignments)\n",
           path.c_str(), n, label_counts_.size(), total_label_assignments_);
}
