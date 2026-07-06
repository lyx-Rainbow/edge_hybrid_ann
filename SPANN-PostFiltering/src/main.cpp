// main.cpp — SPANN-PostFiltering CLI entry point (bench mode: build + search)
//
// Usage: ./spann_pf bench --train_vecs <path> --queries <path> [options]
// Options: --train_access <path> --query_labels <path> --config <path>
//          --k <int> --output <path> --filter <expr>
//          --index_dir <path> --profile --batch-query
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cnpy.h"
#include "config.h"
#include "spann_pf_index.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// ============================================================================
// Minimal JSON output helper
// ============================================================================
void write_json_results(
        const std::string& path,
        const SPANNConfig& cfg,
        double build_time_s,
        size_t memory_bytes,
        size_t disk_bytes_val,
        const std::vector<std::vector<int32_t>>& all_labels,
        const std::vector<std::vector<float>>& all_dists,
        const std::vector<int32_t>& query_labels) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"index\": \"SPANN-PostFiltering\",\n";
    out << "  \"config\": {\n";
    out << "    \"d\": " << cfg.d << ",\n";
    out << "    \"dist_method\": \"" << cfg.dist_method << "\",\n";
    out << "    \"num_threads\": " << cfg.num_threads << ",\n";
    out << "    \"max_check\": " << cfg.max_check << ",\n";
    out << "    \"hash_exp\": " << cfg.hash_exp << ",\n";
    out << "    \"k\": " << cfg.k << ",\n";
    out << "    \"overfetch_factor\": " << cfg.overfetch_factor << ",\n";
    out << "    \"overfetch_adaptive\": " << (cfg.overfetch_adaptive ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"build_time_s\": " << build_time_s << ",\n";
    out << "  \"memory_bytes\": " << memory_bytes << ",\n";
    out << "  \"disk_bytes\": " << disk_bytes_val << ",\n";
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

// ============================================================================
// Simple JSON config loader (string-based)
// ============================================================================
SPANNConfig load_config_from_json(const std::string& path) {
    SPANNConfig cfg;
    std::ifstream in(path);
    if (!in.is_open()) {
        fprintf(stderr, "Warning: cannot open config file '%s', using defaults\n",
                path.c_str());
        return cfg;
    }
    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());

    auto find_val = [&](const std::string& key) -> std::string {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = content.find(':', pos);
        if (pos == std::string::npos) return "";
        pos++;
        while (pos < content.size() &&
               (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n'))
            pos++;
        size_t end = pos;
        while (end < content.size() &&
               content[end] != ',' && content[end] != '\n' && content[end] != '}')
            end++;
        while (end > pos && (content[end-1] == ' ' || content[end-1] == '\t'))
            end--;
        return content.substr(pos, end - pos);
    };

    auto get_str = [&](const std::string& key, std::string& out) {
        auto v = find_val(key);
        if (!v.empty()) {
            // Strip quotes
            if (v.front() == '"' && v.back() == '"') {
                out = v.substr(1, v.size() - 2);
            } else {
                out = v;
            }
        }
    };
    auto get_int = [&](const std::string& key, size_t& out) {
        auto v = find_val(key);
        if (!v.empty()) out = std::stoull(v);
    };
    auto get_bool = [&](const std::string& key, bool& out) {
        auto v = find_val(key);
        if (v == "true") out = true;
        else if (v == "false") out = false;
    };

    get_int("d", cfg.d);
    get_str("dist_method", cfg.dist_method);
    get_int("num_threads", cfg.num_threads);
    get_int("max_check", cfg.max_check);
    get_int("hash_exp", cfg.hash_exp);
    get_int("k", cfg.k);
    get_int("num_warmup", cfg.num_warmup);
    get_int("overfetch_factor", cfg.overfetch_factor);
    get_bool("overfetch_adaptive", cfg.overfetch_adaptive);
    get_bool("batch_query", cfg.batch_query);
    get_str("index_dir", cfg.index_dir);
    get_str("metadata_path", cfg.metadata_path);

    return cfg;
}

void print_usage() {
    printf("Usage: spann_pf bench [options]\n");
    printf("Options:\n");
    printf("  --train_vecs PATH      Training vectors .npy [N, d] float32\n");
    printf("  --train_access PATH    Access pairs .npy [M, 2] int32 (vid, tid)\n");
    printf("  --queries PATH         Query vectors .npy [Q, d] float32\n");
    printf("  --query_labels PATH    Query labels .npy [Q] int32 (-1=unfiltered)\n");
    printf("  --config PATH          JSON config file (optional)\n");
    printf("  --k K                  Results per query (default: 10)\n");
    printf("  --filter EXPR          Complex predicate filter (e.g. \"AND 0 NOT 1\")\n");
    printf("  --index_dir PATH       SPTAG index storage directory\n");
    printf("  --batch-query          Enable inter-query OpenMP parallelism\n");
    printf("  --output PATH          Output JSON (default: results.json)\n");
    printf("  --profile              Print last-query timing breakdown\n");
    printf("  --help                 Show this help\n");
}

} // anonymous namespace

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    std::string train_vecs_path, train_access_path, queries_path, query_labels_path;
    std::string config_path, output_path = "results.json";
    std::string filter_expr;
    std::string index_dir;
    size_t k = 10;
    bool batch_query = false;
    bool profile = false;

    // ── Parse CLI args ──
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
        else if (arg == "--index_dir" && i + 1 < argc) index_dir = argv[++i];
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

    printf("=== SPANN-PostFiltering bench mode ===\n");

    // ── Load config ──
    SPANNConfig cfg;
    if (!config_path.empty()) {
        cfg = load_config_from_json(config_path);
    }
    cfg.k = k;
    if (batch_query) {
        cfg.batch_query = true;
        cfg.num_threads = 1;  // avoid nested parallelism with SPTAG
    }
    if (!index_dir.empty()) {
        cfg.index_dir = index_dir;
    } else {
        cfg.index_dir = "4_Results/SPANN-PostFiltering/spann_index_" +
                        std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    }

    // ── Load training vectors ──
    printf("Loading training vectors from %s ...\n", train_vecs_path.c_str());
    std::vector<size_t> tv_shape;
    auto train_vecs = cnpy::npy_load<float>(train_vecs_path, tv_shape);
    size_t ntrain = tv_shape.empty() ? 0 : tv_shape[0];
    if (tv_shape.size() >= 2) cfg.d = tv_shape[1];
    printf("  Loaded %zu vectors, d=%zu\n", ntrain, cfg.d);

    // ── Load access pairs ──
    std::vector<int32_t> access_pairs;
    size_t n_pairs = 0;
    if (!train_access_path.empty()) {
        printf("Loading access pairs from %s ...\n", train_access_path.c_str());
        std::vector<size_t> ap_shape;
        access_pairs = cnpy::npy_load<int32_t>(train_access_path, ap_shape);
        n_pairs = access_pairs.size() / 2;
        printf("  Loaded %zu access pairs\n", n_pairs);
    }

    // ── Load query vectors ──
    printf("Loading queries from %s ...\n", queries_path.c_str());
    std::vector<size_t> qv_shape;
    auto query_vecs = cnpy::npy_load<float>(queries_path, qv_shape);
    size_t n_queries = qv_shape.empty() ? 0 : qv_shape[0];
    printf("  Loaded %zu queries\n", n_queries);

    // ── Load query labels ──
    std::vector<int32_t> query_labels;
    if (!query_labels_path.empty()) {
        std::vector<size_t> ql_shape;
        query_labels = cnpy::npy_load<int32_t>(query_labels_path, ql_shape);
        printf("  Loaded %zu query labels\n", query_labels.size());
    }

    // ── Build index ──
    printf("\nBuilding SPANN-PostFiltering index...\n");
    printf("  Config: dist=%s, threads=%zu, max_check=%zu, hash_exp=%zu\n",
           cfg.dist_method.c_str(), cfg.num_threads, cfg.max_check, cfg.hash_exp);
    printf("  Index dir: %s\n", cfg.index_dir.c_str());

    auto t_build_start = std::chrono::high_resolution_clock::now();

    SPANNPostFilterIndex index(cfg);
    index.build(ntrain, train_vecs.data(), access_pairs.data(), n_pairs);

    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_time_s = std::chrono::duration<double>(
        t_build_end - t_build_start).count();
    size_t mem_bytes = index.memory_bytes();
    size_t disk_bytes_val = index.disk_bytes();
    printf("Build complete: %.2f s, %.2f MB memory, %.2f MB disk\n",
           build_time_s, mem_bytes / (1024.0 * 1024.0),
           disk_bytes_val / (1024.0 * 1024.0));

    // ── Warmup queries ──
    size_t num_warmup = cfg.num_warmup;
    if (n_queries > 0 && num_warmup > 0) {
        num_warmup = std::min(num_warmup, n_queries);
        printf("Warming up %zu queries...\n", num_warmup);
        std::vector<int32_t> wlabels(cfg.k);
        std::vector<float> wdists(cfg.k);
        for (size_t q = 0; q < num_warmup; q++) {
            if (!filter_expr.empty()) {
                index.search_with_predicate(query_vecs.data() + q * cfg.d, cfg.k,
                                            filter_expr,
                                            wdists.data(), wlabels.data());
            } else {
                int32_t tid = (q < query_labels.size()) ? query_labels[q] : -1;
                if (tid >= 0) {
                    index.search(query_vecs.data() + q * cfg.d, cfg.k, tid,
                                 wdists.data(), wlabels.data());
                } else {
                    index.search_unfiltered(query_vecs.data() + q * cfg.d, cfg.k,
                                            wdists.data(), wlabels.data());
                }
            }
        }
    }

    // ── Search ──
    const char* query_type = filter_expr.empty() ? "single-label" : "complex-predicate";
    printf("\nSearching %zu queries (k=%zu, type=%s)...\n", n_queries, cfg.k, query_type);
    if (!filter_expr.empty()) {
        printf("  Filter: %s\n", filter_expr.c_str());
    }
    if (cfg.batch_query) {
        printf("  batch_query ON (OpenMP parallel, SPTAG internal threads=1)\n");
    }

    std::vector<std::vector<int32_t>> all_labels(n_queries);
    std::vector<std::vector<float>> all_dists(n_queries);

    auto t_search_start = std::chrono::high_resolution_clock::now();

    if (cfg.batch_query) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (size_t q = 0; q < n_queries; q++) {
            all_labels[q].resize(cfg.k);
            all_dists[q].resize(cfg.k);
            if (!filter_expr.empty()) {
                index.search_with_predicate(query_vecs.data() + q * cfg.d, cfg.k,
                                            filter_expr,
                                            all_dists[q].data(), all_labels[q].data());
            } else {
                int32_t tid = (q < query_labels.size()) ? query_labels[q] : -1;
                if (tid >= 0) {
                    index.search(query_vecs.data() + q * cfg.d, cfg.k, tid,
                                 all_dists[q].data(), all_labels[q].data());
                } else {
                    index.search_unfiltered(query_vecs.data() + q * cfg.d, cfg.k,
                                            all_dists[q].data(), all_labels[q].data());
                }
            }
        }
    } else {
        for (size_t q = 0; q < n_queries; q++) {
            all_labels[q].resize(cfg.k);
            all_dists[q].resize(cfg.k);
            if (!filter_expr.empty()) {
                index.search_with_predicate(query_vecs.data() + q * cfg.d, cfg.k,
                                            filter_expr,
                                            all_dists[q].data(), all_labels[q].data());
            } else {
                int32_t tid = (q < query_labels.size()) ? query_labels[q] : -1;
                if (tid >= 0) {
                    index.search(query_vecs.data() + q * cfg.d, cfg.k, tid,
                                 all_dists[q].data(), all_labels[q].data());
                } else {
                    index.search_unfiltered(query_vecs.data() + q * cfg.d, cfg.k,
                                            all_dists[q].data(), all_labels[q].data());
                }
            }
        }
    }

    auto t_search_end = std::chrono::high_resolution_clock::now();
    double search_time_s = std::chrono::duration<double>(
        t_search_end - t_search_start).count();
    printf("Search complete: %.2f s total, %.2f ms/query\n",
           search_time_s, search_time_s / n_queries * 1000.0);

    // ── Profile last query (if requested) ──
    if (profile && n_queries > 0) {
        size_t last_q = n_queries - 1;
        std::vector<int32_t> plabels(cfg.k);
        std::vector<float> pdists(cfg.k);

        auto t0 = std::chrono::high_resolution_clock::now();
        if (!filter_expr.empty()) {
            index.search_with_predicate(query_vecs.data() + last_q * cfg.d, cfg.k,
                                        filter_expr,
                                        pdists.data(), plabels.data());
        } else {
            int32_t tid = (last_q < query_labels.size()) ? query_labels[last_q] : -1;
            if (tid >= 0) {
                index.search(query_vecs.data() + last_q * cfg.d, cfg.k, tid,
                             pdists.data(), plabels.data());
            } else {
                index.search_unfiltered(query_vecs.data() + last_q * cfg.d, cfg.k,
                                        pdists.data(), plabels.data());
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("\nProfiling (last query):\n");
        printf("  query_type: %s\n",
               filter_expr.empty() ? "single-label" : "complex-predicate");
        if (!filter_expr.empty()) printf("  filter: %s\n", filter_expr.c_str());
        printf("  overfetch_factor: %zu\n", cfg.overfetch_factor);
        printf("  total: %.3f ms\n", ms);
    }

    // ── Output ──
    printf("\nWriting results to %s ...\n", output_path.c_str());
    write_json_results(output_path, cfg, build_time_s, mem_bytes,
                       disk_bytes_val, all_labels, all_dists, query_labels);

    printf("Done.\n");
    return 0;
}
