// main.cpp — PreFiltering CLI entry point (bench mode: build + search in one process)
//
// Usage: ./prefiltering bench --train_vecs <path> --queries <path> [options]
// Options: --train_access <path> --query_labels <path> --config <path>
//          --k <int> --output <path> --profile --batch-query
//          --filter EXPR --filters_file <path>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

#include "cnpy.h"
#include "config.h"
#include "prefiltering_index.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// ============================================================================
// RSS sampling helper (Linux /proc/self/statm)
// ============================================================================
long get_rss_bytes() {
    std::ifstream statm("/proc/self/statm");
    if (!statm.is_open()) return 0;
    long size_pages = 0, rss_pages = 0;
    statm >> size_pages >> rss_pages;
    long page_size = sysconf(_SC_PAGESIZE);
    return rss_pages * page_size;
}

// ============================================================================
// Minimal JSON output helper (no external JSON library dependency)
// ============================================================================
void write_json_results(
        const std::string& path,
        const PreFilteringConfig& cfg,
        double build_time_s,
        size_t memory_bytes,
        const std::vector<std::vector<int32_t>>& all_labels,
        const std::vector<std::vector<float>>& all_dists,
        const std::vector<int32_t>& query_labels,
        const std::vector<double>& search_times_us,
        double rss_peak_query_mb,
        const std::vector<std::string>& cp_filters = {},
        const std::vector<std::vector<std::vector<int32_t>>>& cp_all_labels = {},
        const std::vector<std::vector<std::vector<float>>>& cp_all_dists = {},
        const std::vector<std::vector<double>>& cp_search_times_us = {}) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"index\": \"PreFiltering\",\n";
    out << "  \"config\": {\n";
    out << "    \"d\": " << cfg.d << ",\n";
    out << "    \"k\": " << cfg.k << ",\n";
    out << "    \"n_labels\": " << cfg.n_labels << "\n";
    out << "  },\n";
    out << "  \"build_time_s\": " << build_time_s << ",\n";
    out << "  \"memory_bytes\": " << memory_bytes << ",\n";
    out << "  \"disk_bytes\": 0,\n";
    out << "  \"rss_peak_query_mb\": " << rss_peak_query_mb << ",\n";

    bool is_batch = !cp_filters.empty();

    if (is_batch) {
        // ── Batch CP mode: filters_results ──
        out << "  \"queries\": [],\n";
        out << "  \"filters_results\": [\n";
        for (size_t fi = 0; fi < cp_filters.size(); fi++) {
            out << "    {\n";
            out << "      \"filter\": \"" << cp_filters[fi] << "\",\n";
            out << "      \"queries\": [\n";
            const auto& labels_vec = cp_all_labels[fi];
            const auto& dists_vec  = cp_all_dists[fi];
            const auto& times_vec  = cp_search_times_us[fi];
            for (size_t q = 0; q < labels_vec.size(); q++) {
                out << "        {\n";
                out << "          \"idx\": " << q << ",\n";
                out << "          \"labels\": [";
                for (size_t i = 0; i < labels_vec[q].size(); i++) {
                    if (i > 0) out << ", ";
                    out << labels_vec[q][i];
                }
                out << "],\n";
                out << "          \"distances\": [";
                for (size_t i = 0; i < dists_vec[q].size(); i++) {
                    if (i > 0) out << ", ";
                    out << dists_vec[q][i];
                }
                out << "],\n";
                out << "          \"search_time_us\": " << times_vec[q] << "\n";
                out << "        }";
                if (q + 1 < labels_vec.size()) out << ",";
                out << "\n";
            }
            out << "      ]\n";
            out << "    }";
            if (fi + 1 < cp_filters.size()) out << ",";
            out << "\n";
        }
        out << "  ]\n";
    } else {
        // ── Single filter / SL mode: queries ──
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
            out << "],\n";
            out << "      \"search_time_us\": " << search_times_us[q] << "\n";
            out << "    }";
            if (q + 1 < all_labels.size()) out << ",";
            out << "\n";
        }
        out << "  ]\n";
    }
    out << "}\n";
    out.close();
}

// ============================================================================
// Simple JSON config loader (string-based, matches Curator's approach)
// ============================================================================
PreFilteringConfig load_config_from_json(const std::string& path) {
    PreFilteringConfig cfg;
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
    get_int("k", cfg.k);
    get_int("n_labels", cfg.n_labels);
    get_int("num_warmup", cfg.num_warmup);
    get_bool("batch_query", cfg.batch_query);

    return cfg;
}

void print_usage() {
    printf("Usage: prefiltering <bench|search> [options]\n");
    printf("\nbench options:\n");
    printf("  --train_vecs PATH      Training vectors .npy [N, d] float32\n");
    printf("  --train_access PATH    Access pairs .npy [M, 2] int32 (vid, tid)\n");
    printf("  --queries PATH         Query vectors .npy [Q, d] float32\n");
    printf("  --query_labels PATH    Query labels .npy [Q] int32 (-1=unfiltered)\n");
    printf("  --config PATH          JSON config file (optional)\n");
    printf("  --k K                  Results per query (default: 10)\n");
    printf("  --batch-query          Enable inter-query OpenMP parallelism\n");
    printf("  --output PATH          Output JSON (default: results.json)\n");
    printf("  --filter EXPR          Complex predicate filter (e.g. \"AND 0 NOT 1\")\n");
    printf("  --filters_file PATH    Batch CP: file with one filter per line (mutually exclusive with --filter)\n");
    printf("  --profile              Print last-query timing breakdown\n");
    printf("\nsearch options (re-load data, skip allocation messages):\n");
    printf("  --train_vecs PATH      Training vectors .npy [N, d] float32 (required)\n");
    printf("  --train_access PATH    Access pairs .npy [M, 2] int32 (vid, tid) (required)\n");
    printf("  --queries PATH         Query vectors .npy [Q, d] float32 (required)\n");
    printf("  --query_labels PATH    Query labels .npy [Q] int32 (-1=unfiltered)\n");
    printf("  --k K                  Results per query (default: 10)\n");
    printf("  --batch-query          Enable inter-query OpenMP parallelism\n");
    printf("  --output PATH          Output JSON (default: results.json)\n");
    printf("  --filter EXPR          Complex predicate filter\n");
    printf("  --filters_file PATH    Batch CP: file with one filter per line (mutually exclusive with --filter)\n");
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
    std::string filter_expr;  // complex predicate (AND/OR/NOT), empty = simple query
    std::string filters_file_path;
    size_t k = 10;
    bool batch_query = false;
    bool profile = false;

    // ── Parse CLI args ──
    bool is_search = false;
    int start_idx = 1;
    if (argc > 1) {
        std::string cmd = argv[1];
        if (cmd == "bench") {
            start_idx = 2;
        } else if (cmd == "search") {
            is_search = true;
            start_idx = 2;
        }
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
        else if (arg == "--filters_file" && i + 1 < argc) filters_file_path = argv[++i];
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

    // ── Read batch filters ──
    std::vector<std::string> cp_filters;
    if (!filters_file_path.empty()) {
        std::ifstream ff(filters_file_path);
        if (!ff.is_open()) {
            fprintf(stderr, "Error: cannot open filters_file '%s'\n", filters_file_path.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(ff, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
                line.pop_back();
            if (!line.empty()) cp_filters.push_back(line);
        }
        ff.close();
        if (!filter_expr.empty()) {
            printf("Note: --filters_file provided, ignoring --filter\n");
            filter_expr.clear();
        }
    }

    printf("=== PreFiltering %s mode ===\n", is_search ? "search" : "bench");

    // ── Load config ──
    PreFilteringConfig cfg;
    if (!config_path.empty()) {
        cfg = load_config_from_json(config_path);
    }
    cfg.k = k;
    if (batch_query) cfg.batch_query = true;

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
    printf("\nBuilding index...\n");
    auto t_build_start = std::chrono::high_resolution_clock::now();

    PreFilteringIndex index(cfg);
    index.build(ntrain, train_vecs.data(), access_pairs.data(), n_pairs);

    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_time_s = std::chrono::duration<double>(
        t_build_end - t_build_start).count();
    size_t mem_bytes = index.memory_bytes();
    printf("Build complete: %.2f s, %.2f MB memory, %zu labels\n",
           build_time_s, mem_bytes / (1024.0 * 1024.0), index.n_labels());

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
            } else if (!cp_filters.empty()) {
                index.search_with_predicate(query_vecs.data() + q * cfg.d, cfg.k,
                                            cp_filters[0],
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
    std::vector<std::vector<int32_t>> all_labels(n_queries);
    std::vector<std::vector<float>> all_dists(n_queries);
    std::vector<double> search_times_us(n_queries, 0.0);
    double rss_peak_query_mb = 0.0;

    // Batch CP containers
    std::vector<std::vector<std::vector<int32_t>>> cp_all_labels;
    std::vector<std::vector<std::vector<float>>> cp_all_dists;
    std::vector<std::vector<double>> cp_search_times_us;

    if (!cp_filters.empty()) {
        // ═══════════════════════════════════════════════════════════
        // Batch CP mode: one build, N filters
        // ═══════════════════════════════════════════════════════════
        printf("\nBatch CP search: %zu queries (k=%zu), %zu filters\n",
               n_queries, cfg.k, cp_filters.size());
        if (cfg.batch_query) {
            printf("  batch_query ON (OpenMP parallel)\n");
        }

        cp_all_labels.resize(cp_filters.size());
        cp_all_dists.resize(cp_filters.size());
        cp_search_times_us.resize(cp_filters.size());

        auto t_search_start = std::chrono::high_resolution_clock::now();
        long rss_peak = 0;

        for (size_t fi = 0; fi < cp_filters.size(); fi++) {
            const std::string& filter = cp_filters[fi];
            printf("  [%zu/%zu] Filter: %s ...", fi + 1, cp_filters.size(), filter.c_str());
            fflush(stdout);

            auto& labels_vec = cp_all_labels[fi];
            auto& dists_vec  = cp_all_dists[fi];
            auto& times_vec  = cp_search_times_us[fi];
            labels_vec.resize(n_queries);
            dists_vec.resize(n_queries);
            times_vec.resize(n_queries, 0.0);

            auto t_filter_start = std::chrono::high_resolution_clock::now();

            if (cfg.batch_query) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                for (size_t q = 0; q < n_queries; q++) {
                    labels_vec[q].resize(cfg.k);
                    dists_vec[q].resize(cfg.k);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    index.search_with_predicate(query_vecs.data() + q * cfg.d, cfg.k,
                                                filter,
                                                dists_vec[q].data(), labels_vec[q].data());
                    auto t1 = std::chrono::high_resolution_clock::now();
                    times_vec[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                }
            } else {
                for (size_t q = 0; q < n_queries; q++) {
                    labels_vec[q].resize(cfg.k);
                    dists_vec[q].resize(cfg.k);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    index.search_with_predicate(query_vecs.data() + q * cfg.d, cfg.k,
                                                filter,
                                                dists_vec[q].data(), labels_vec[q].data());
                    auto t1 = std::chrono::high_resolution_clock::now();
                    times_vec[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();

                    long rss = get_rss_bytes();
                    if (rss > rss_peak) rss_peak = rss;
                }
            }

            auto t_filter_end = std::chrono::high_resolution_clock::now();
            double filter_s = std::chrono::duration<double>(t_filter_end - t_filter_start).count();
            printf(" %.1f s\n", filter_s);
        }

        auto t_search_end = std::chrono::high_resolution_clock::now();
        double search_time_s = std::chrono::duration<double>(
            t_search_end - t_search_start).count();
        printf("Batch search complete: %.2f s total, %.2f ms/query/filter\n",
               search_time_s, search_time_s / (n_queries * cp_filters.size()) * 1000.0);

        rss_peak_query_mb = rss_peak / (1024.0 * 1024.0);

    } else {
        // ═══════════════════════════════════════════════════════════
        // Single filter / SL mode (existing logic + per-query timing)
        // ═══════════════════════════════════════════════════════════
        const char* query_type = filter_expr.empty() ? "single-label" : "complex-predicate";
        printf("\nSearching %zu queries (k=%zu, type=%s)...\n", n_queries, cfg.k, query_type);
        if (!filter_expr.empty()) {
            printf("  Filter: %s\n", filter_expr.c_str());
        }

        long rss_peak = 0;
        auto t_search_start = std::chrono::high_resolution_clock::now();

        if (cfg.batch_query) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (size_t q = 0; q < n_queries; q++) {
                all_labels[q].resize(cfg.k);
                all_dists[q].resize(cfg.k);
                auto t0 = std::chrono::high_resolution_clock::now();
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
                auto t1 = std::chrono::high_resolution_clock::now();
                search_times_us[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            }
        } else {
            for (size_t q = 0; q < n_queries; q++) {
                all_labels[q].resize(cfg.k);
                all_dists[q].resize(cfg.k);
                auto t0 = std::chrono::high_resolution_clock::now();
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
                auto t1 = std::chrono::high_resolution_clock::now();
                search_times_us[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();

                long rss = get_rss_bytes();
                if (rss > rss_peak) rss_peak = rss;
            }
        }

        auto t_search_end = std::chrono::high_resolution_clock::now();
        double search_time_s = std::chrono::duration<double>(
            t_search_end - t_search_start).count();
        printf("Search complete: %.2f s total, %.2f ms/query\n",
               search_time_s, search_time_s / n_queries * 1000.0);

        rss_peak_query_mb = rss_peak / (1024.0 * 1024.0);

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
            printf("  total: %.3f ms\n", ms);
        }
    }

    // ── Output ──
    printf("\nWriting results to %s ...\n", output_path.c_str());
    write_json_results(output_path, cfg, build_time_s, mem_bytes,
                       all_labels, all_dists, query_labels,
                       search_times_us, rss_peak_query_mb,
                       cp_filters, cp_all_labels, cp_all_dists, cp_search_times_us);

    printf("Done.\n");
    return 0;
}
