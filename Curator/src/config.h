// config.h — CuratorConfig structure (no heavy dependencies)
#pragma once

#include <cstddef>
#include <string>

namespace curator {

//TODO:实验过程中真的使用该文件中的默认值了吗？
struct CuratorConfig {
    // Data dimensions
    size_t d = 128;
    // Tree structure
    size_t n_clusters = 64;
    size_t bf_capacity = 1000;
    float  bf_false_pos = 0.01f;
    size_t max_sl_size = 128;
    size_t clus_niter = 20;
    size_t max_leaf_size = 128;
    // PQ (Product Quantization)
    size_t pq_M = 16;
    size_t pq_nbits = 8;
    bool   pq_enabled = true;
    bool   pq_use_adc_rerank = false;
    size_t pq_rerank_topk_factor = 4;
    bool   persist_pq_codes = false;
    std::string pq_codes_path;
    // PQ block cache (for on-disk PQ codes with on-demand loading)
    size_t pq_cache_block_size = 4096;   // codes per block
    size_t pq_cache_max_blocks = 256;    // max cached blocks (M=16→16MB, M=128→128MB)
    // Flash storage
    bool   use_flash_storage = true;
    std::string flash_path;
    // Search
    size_t nprobe = 3000;
    float  prune_thres = 1.6f;
    float  variance_boost = 0.4f;
    size_t search_ef = 128;
    size_t beam_size = 2;
    bool   use_temp_index_caching = true;
    bool   batch_query = false;
};

} // namespace curator
