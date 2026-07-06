// main.cpp — Curator CLI entry point (bench mode: build + search in one process)
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cnpy.h"
#include "common.h"
#include "config.h"
#include "curator_index.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace curator;

namespace {

// Minimal JSON output helper (avoids nlohmann/json dependency)
void write_json_results(
        const std::string& path,
        const CuratorConfig& cfg,
        double build_time_s,
        size_t memory_bytes,
        const std::vector<std::vector<ext_vid_t>>& all_labels,
        const std::vector<std::vector<float>>& all_dists,
        const std::vector<int32_t>& query_labels) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"config\": {\n";
    out << "    \"d\": " << cfg.d << ",\n";
    out << "    \"nlist\": " << cfg.n_clusters << ",\n";
    out << "    \"pq_M\": " << cfg.pq_M << ",\n";
    out << "    \"pq_cache_block_size\": " << cfg.pq_cache_block_size << ",\n";
    out << "    \"pq_cache_max_blocks\": " << cfg.pq_cache_max_blocks << ",\n";
    out << "    \"search_ef\": " << cfg.search_ef << "\n";
    out << "  },\n";
    out << "  \"build_time_s\": " << build_time_s << ",\n";
    out << "  \"memory_bytes\": " << memory_bytes << ",\n";
    out << "  \"queries\": [\n";

    for (size_t q = 0; q < all_labels.size(); q++) {
        out << "    {\n";
        out << "      \"idx\": " << q << ",\n";
        int32_t tid = (q < query_labels.size()) ? query_labels[q] : -1;
        out << "      \"tenant_id\": " << tid << ",\n";
        out << "      \"labels\": [";
        for (size_t i = 0; i < all_labels[q].size(); i++) {
            if (i > 0) out << ", ";
            out << all_labels[q][i];
        }
        out << "],\n";
        out << "      \"distances\": [";
        for (size_t i = 0; i < all_dists[q].size(); i++) {
            if (i > 0) out << ", ";
            out << all_dists[q][i];
        }
        out << "]\n";
        out << "    }";
        if (q + 1 < all_labels.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    out.close();
}

CuratorConfig load_config_from_json(const std::string& path) {
    CuratorConfig cfg;
    std::ifstream in(path);
    if (!in.is_open()) {
        fprintf(stderr, "Warning: cannot open config file '%s', using defaults\n", path.c_str());
        return cfg;
    }
    // Simple JSON value parser (handles top-level key:value pairs)
    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());

    auto find_val = [&](const std::string& key) -> std::string {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = content.find(':', pos);
        if (pos == std::string::npos) return "";
        pos++;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n'))
            pos++;
        size_t end = pos;
        while (end < content.size() && content[end] != ',' && content[end] != '\n' && content[end] != '}')
            end++;
        while (end > pos && (content[end-1] == ' ' || content[end-1] == '\t'))
            end--;
        return content.substr(pos, end - pos);
    };

    auto get_int = [&](const std::string& key, size_t& out) {
        auto v = find_val(key);
        if (!v.empty()) out = std::stoull(v);
    };
    auto get_float = [&](const std::string& key, float& out) {
        auto v = find_val(key);
        if (!v.empty()) out = std::stof(v);
    };
    auto get_bool = [&](const std::string& key, bool& out) {
        auto v = find_val(key);
        if (v == "true") out = true;
        else if (v == "false") out = false;
    };
    auto get_str = [&](const std::string& key, std::string& out) {
        auto v = find_val(key);
        if (!v.empty() && v.front() == '"') {
            v = v.substr(1, v.size() - 2);
        }
        if (!v.empty()) out = v;
    };

    get_int("d", cfg.d);
    get_int("n_clusters", cfg.n_clusters);
    get_int("bf_capacity", cfg.bf_capacity);
    get_float("bf_false_pos", cfg.bf_false_pos);
    get_int("max_sl_size", cfg.max_sl_size);
    get_int("clus_niter", cfg.clus_niter);
    get_int("max_leaf_size", cfg.max_leaf_size);
    get_int("pq_M", cfg.pq_M);
    get_int("pq_nbits", cfg.pq_nbits);
    get_bool("pq_enabled", cfg.pq_enabled);
    get_bool("pq_use_adc_rerank", cfg.pq_use_adc_rerank);
    get_int("pq_rerank_topk_factor", cfg.pq_rerank_topk_factor);
    get_bool("persist_pq_codes", cfg.persist_pq_codes);
    get_str("pq_codes_path", cfg.pq_codes_path);
    get_int("pq_cache_block_size", cfg.pq_cache_block_size);
    get_int("pq_cache_max_blocks", cfg.pq_cache_max_blocks);
    get_bool("use_flash_storage", cfg.use_flash_storage);
    get_str("flash_path", cfg.flash_path);
    get_int("nprobe", cfg.nprobe);
    get_float("prune_thres", cfg.prune_thres);
    get_float("variance_boost", cfg.variance_boost);
    get_int("search_ef", cfg.search_ef);
    get_int("beam_size", cfg.beam_size);
    get_bool("use_temp_index_caching", cfg.use_temp_index_caching);
    get_bool("batch_query", cfg.batch_query);

    return cfg;
}

void print_usage() {
    printf("Usage: curator bench [options]\n");
    printf("Options:\n");
    printf("  --train_vecs PATH      Training vectors .npy file [N, d] float32\n");
    printf("  --train_access PATH    Training access pairs .npy file [M, 2] int32\n");
    printf("  --queries PATH         Query vectors .npy file [Q, d] float32\n");
    printf("  --query_labels PATH    Query tenant labels .npy file [Q] int32 (-1=unfiltered)\n");
    printf("  --filter EXPR          Complex predicate filter (e.g. \"1 AND 2\")\n");
    printf("  --config PATH          JSON config file (optional, defaults used)\n");
    printf("  --k K                  Number of results per query (default: 10)\n");
    printf("  --batch-query          Enable inter-query OpenMP parallelism\n");
    printf("  --output PATH          Output JSON results file (default: results.json)\n");
    printf("  --profile              Enable profiling output\n");
    printf("  --help                 Show this help\n");
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::string train_vecs_path, train_access_path, queries_path, query_labels_path;
    std::string config_path, output_path = "results.json";
    std::string filter_expr;  // complex predicate (AND/OR/NOT), empty = simple query
    size_t k = 10;
    bool batch_query = false;
    bool profile = false;

    // Parse CLI args
    int start_idx = 1;
    if (argc > 1 && std::string(argv[1]) == "bench") {
        start_idx = 2;
    }
    for (int i = start_idx; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") { print_usage(); return 0; }
        else if (arg == "--train_vecs" && i + 1 < argc) train_vecs_path = argv[++i];
        else if (arg == "--train_access" && i + 1 < argc) train_access_path = argv[++i];
        else if (arg == "--queries" && i + 1 < argc) queries_path = argv[++i];
        else if (arg == "--query_labels" && i + 1 < argc) query_labels_path = argv[++i];
        else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--k" && i + 1 < argc) k = std::stoull(argv[++i]);
        else if (arg == "--batch-query") batch_query = true;
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--filter" && i + 1 < argc) filter_expr = argv[++i];
        else if (arg == "--profile") profile = true;
        else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage();
            return 1;
        }
    }

    if (train_vecs_path.empty() || queries_path.empty()) {
        fprintf(stderr, "Error: --train_vecs and --queries are required\n");
        print_usage();
        return 1;
    }

    printf("=== Curator bench mode ===\n");

    // Load config
    CuratorConfig cfg;
    if (!config_path.empty()) {
        cfg = load_config_from_json(config_path);
    }
    if (batch_query) cfg.batch_query = true;

    // Load training vectors
    printf("Loading training vectors from %s ...\n", train_vecs_path.c_str());
    std::vector<size_t> tv_shape;
    auto train_vecs = cnpy::npy_load<float>(train_vecs_path, tv_shape);
    size_t ntrain = tv_shape.empty() ? 0 : tv_shape[0];
    if (tv_shape.size() >= 2) cfg.d = tv_shape[1];
    printf("  Loaded %zu vectors, d=%zu\n", ntrain, cfg.d);

    // Load training access pairs
    std::vector<int32_t> access_pairs;
    if (!train_access_path.empty()) {
        printf("Loading access pairs from %s ...\n", train_access_path.c_str());
        std::vector<size_t> ap_shape;
        access_pairs = cnpy::npy_load<int32_t>(train_access_path, ap_shape);
        printf("  Loaded %zu access pairs\n", access_pairs.size() / 2);
    }

    // Load query vectors
    printf("Loading queries from %s ...\n", queries_path.c_str());
    std::vector<size_t> qv_shape;
    auto query_vecs = cnpy::npy_load<float>(queries_path, qv_shape);
    size_t n_queries = qv_shape.empty() ? 0 : qv_shape[0];
    printf("  Loaded %zu queries\n", n_queries);

    // Load query labels
    std::vector<int32_t> query_labels;
    if (!query_labels_path.empty()) {
        std::vector<size_t> ql_shape;
        query_labels = cnpy::npy_load<int32_t>(query_labels_path, ql_shape);
        printf("  Loaded %zu query labels\n", query_labels.size());
    }

    // ── Build index ──
    printf("\nBuilding index...\n");
    auto t_build_start = std::chrono::high_resolution_clock::now();

    CuratorIndex index(cfg);
    index.train(ntrain, train_vecs.data());

    for (size_t i = 0; i < ntrain; i++) {
        index.add_vector(train_vecs.data() + i * cfg.d,
                          static_cast<ext_vid_t>(i));
    }

    // Grant access from pairs
    for (size_t i = 0; i + 1 < access_pairs.size(); i += 2) {
        ext_vid_t vid = static_cast<ext_vid_t>(access_pairs[i]);
        ext_lid_t tid = static_cast<ext_lid_t>(access_pairs[i + 1]);
        index.grant_access(vid, tid);
    }

    index.flush();

    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_time_s = std::chrono::duration<double>(t_build_end - t_build_start).count();
    size_t mem_bytes = index.memory_bytes();
    printf("Build complete: %.2f s, %.2f MB memory\n",
           build_time_s, mem_bytes / (1024.0 * 1024.0));

    // ── Search ──
    printf("\nSearching %zu queries (k=%zu)...\n", n_queries, k);
    index.enable_profiling(profile);

    std::vector<std::vector<ext_vid_t>> all_labels(n_queries);
    std::vector<std::vector<float>> all_dists(n_queries);

    // If a complex filter predicate is specified, build a filter index once
    // and use it for all queries (replaces per-query tenant_id filtering).
    ext_lid_t filter_label = -1;
    if (!filter_expr.empty()) {
        printf("Evaluating complex filter: %s\n", filter_expr.c_str());
        auto qualified = index.find_all_qualified_vecs(filter_expr);
        printf("  Qualified vectors: %zu\n", qualified.size());
        if (!qualified.empty()) {
            filter_label = index.build_filter_index(
                filter_expr, qualified.data(), qualified.size());
            printf("  Filter index built, label=%d\n", static_cast<int>(filter_label));
        } else {
            fprintf(stderr, "Warning: no vectors match filter, all results will be empty\n");
        }
    }

    auto t_search_start = std::chrono::high_resolution_clock::now();

    if (cfg.batch_query) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (size_t q = 0; q < n_queries; q++) {
            all_labels[q].resize(k);
            all_dists[q].resize(k);
            if (filter_label >= 0) {
                // Complex predicate: use the pre-built filter index
                index.search(query_vecs.data() + q * cfg.d, k, filter_label,
                              all_dists[q].data(), all_labels[q].data());
            } else {
                ext_lid_t tid = (q < query_labels.size()) ?
                    static_cast<ext_lid_t>(query_labels[q]) : -1;
                index.search(query_vecs.data() + q * cfg.d, k, tid,
                              all_dists[q].data(), all_labels[q].data());
            }
        }
    } else {
        for (size_t q = 0; q < n_queries; q++) {
            all_labels[q].resize(k);
            all_dists[q].resize(k);
            if (filter_label >= 0) {
                index.search(query_vecs.data() + q * cfg.d, k, filter_label,
                              all_dists[q].data(), all_labels[q].data());
            } else {
                ext_lid_t tid = (q < query_labels.size()) ?
                    static_cast<ext_lid_t>(query_labels[q]) : -1;
                index.search(query_vecs.data() + q * cfg.d, k, tid,
                              all_dists[q].data(), all_labels[q].data());
            }
        }
    }

    auto t_search_end = std::chrono::high_resolution_clock::now();
    double search_time_s = std::chrono::duration<double>(t_search_end - t_search_start).count();
    if (filter_label >= 0) {
        printf("Search complete (filter mode): %.2f s total, %.2f ms/query\n",
               search_time_s, search_time_s / n_queries * 1000.0);
    } else {
        printf("Search complete: %.2f s total, %.2f ms/query\n",
               search_time_s, search_time_s / n_queries * 1000.0);
    }

    // ── Output ──
    printf("\nWriting results to %s ...\n", output_path.c_str());
    write_json_results(output_path, cfg, build_time_s, mem_bytes,
                       all_labels, all_dists, query_labels);

    if (profile) {
        const auto& prof = index.last_profile();
        printf("\nProfiling (last query):\n");
        printf("  query_type: %s\n", prof.query_type);
        printf("  beam_search: %.3f ms\n", prof.beam_search_ms);
        printf("  frontier_search: %.3f ms (nodes_popped=%d, shortlists=%d, expanded=%d)\n",
               prof.frontier_search_ms, prof.frontier_nodes_popped,
               prof.frontier_shortlists_scanned, prof.frontier_children_expanded);
        if (prof.pq_table_build_ms > 0)
            printf("  pq_table_build: %.3f ms\n", prof.pq_table_build_ms);
        if (prof.pq_distance_compute_ms > 0)
            printf("  pq_distance_compute: %.3f ms\n", prof.pq_distance_compute_ms);
        printf("  candidate_merge: %.3f ms\n", prof.candidate_merge_ms);
        printf("  rerank: %.3f ms (count=%d)\n", prof.rerank_ms, prof.rerank_count);
        printf("  total: %.3f ms\n", prof.total_search_ms);
    }

    index.print_tree_info();
    printf("\nDone.\n");
    return 0;
}
