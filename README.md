# Curator 向量检索基准测试项目

基于层次化聚类树索引（Curator）的过滤向量检索基准测试平台，对比 DiskIVF-PostFiltering、Pre-Filtering 和 SPANN-PostFiltering 三种基线方法。

Curator 是**独立 C++ 可执行文件**（零 FAISS 依赖），通过 CMake + OpenMP 构建，WSL-Ubuntu 端运行。

---

## 目录

- [环境信息](#环境信息)
- [快速开始](#快速开始)
- [编译 Curator](#编译-curator)
- [运行实验](#运行实验)
- [评估结果](#评估结果)
- [项目结构](#项目结构)
- [配置参数说明](#配置参数说明)

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11 + WSL Ubuntu 24.04 |
| C++ 编译器 | GCC 13.3.0（需 C++17） |
| CMake | ≥ 3.10 |
| Conda 环境 | `edge_ann`（Python 3.10, numpy, faiss-cpu 1.7.4） |
| WSL 路径 | `/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines` |

> **注**: FAISS 仅用于其他基线方法（DiskIVF/SPANN）。Curator 本身**不依赖 FAISS**。

---

## 快速开始

### 一键运行（推荐）

```bash
# 在 WSL 中激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/Curator

# arxiv_small（50K×384，~20秒构建）
python python/run_experiment.py --dataset arxiv_small

# yfcc100m_small（50K×192，~10秒构建）
python python/run_experiment.py --dataset yfcc100m_small

# 使用自定义配置
python python/run_experiment.py --dataset arxiv_small \
    --config /tmp/arxiv_config.json --k 10 --profile
```

### 手动运行 C++ 二进制

```bash
# 编译后直接调用
./build/curator bench \
    --train_vecs     $DATA/train_vecs.npy \
    --train_access   $DATA/train_access.npy \
    --queries        $DATA/query_vecs.npy \
    --query_labels   $DATA/query_labels.npy \
    --config         /tmp/config.json \
    --k 10 --profile --output results.json
```

---

## 编译 Curator

仅在首次搭建或修改 C++ 源码后需要。

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/Curator
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

编译产物：`build/curator`（独立可执行文件，仅依赖 OpenMP）。

### 系统依赖

```bash
sudo apt install libomp-dev   # OpenMP（K-means 并行 + batch query）
```

---

## 运行实验

### 可用数据集

| 数据集 | 向量数 | 维度 | 标签数 | 类型 |
|--------|:-----:|:----:|:-----:|------|
| sift1m_small | 5,000 | 128 | 50 | 快速验证 |
| gist1m_small | 3,000 | 960 | 50 | 快速验证 |
| arxiv_small | 50,000 | 384 | 99 | 中等规模 |
| yfcc100m_small | 50,000 | 192 | 1,000 | 中等规模 |
| sift1m | 1,000,000 | 128 | 100 | 完整规模 |
| gist1m | 1,000,000 | 960 | 100 | 完整规模 |
| arxiv | 1,600,000 | 384 | 99 | 完整规模 |
| yfcc100m | 800,000 | 192 | 1,000 | 完整规模 |

数据位于 `1_Data/ground_truth/<dataset>/`，需包含：
`train_vecs.npy`, `train_mds.pkl`, `query_vecs.npy`, `query_labels.npy`, `ground_truth.npy`。

### 常用配置模板

#### arxiv（d=384, M=128）

```bash
cat > /tmp/arxiv_config.json << 'JSON'
{
  "d": 384, "n_clusters": 32,
  "max_sl_size": 256, "max_leaf_size": 128,
  "pq_M": 128, "pq_nbits": 8,
  "pq_enabled": true, "pq_use_adc_rerank": true, "pq_rerank_topk_factor": 4,
  "pq_cache_block_size": 4096, "pq_cache_max_blocks": 256,
  "use_flash_storage": true,
  "variance_boost": 0.4, "search_ef": 1024, "beam_size": 4
}
JSON
```

#### yfcc100m（d=192, M=64）

```bash
cat > /tmp/yfcc_config.json << 'JSON'
{
  "d": 192, "n_clusters": 32,
  "max_sl_size": 256, "max_leaf_size": 128,
  "pq_M": 64, "pq_nbits": 8,
  "pq_enabled": true, "pq_use_adc_rerank": true, "pq_rerank_topk_factor": 4,
  "pq_cache_block_size": 4096, "pq_cache_max_blocks": 256,
  "use_flash_storage": true,
  "variance_boost": 0.2, "search_ef": 1024, "beam_size": 4
}
JSON
```

### 参数扫描

```bash
# 搜索参数网格扫描（构建一次，复用多次）
for VAR_BOOST in 0.0 0.2 0.4 0.6; do
for SEARCH_EF in 256 512 1024 2048; do
    # 生成临时配置 → 运行 bench
done
done
```

### 消融实验

| 实验 | 配置变更 |
|------|---------|
| 关闭 PQ（纯精确距离） | `"pq_enabled": false` |
| 关闭 ADC Rerank（纯 PQ 近似） | `"pq_use_adc_rerank": false` |
| 关闭 Flash（纯内存模式） | `"use_flash_storage": false` |
| 多线程查询 | `"batch_query": true` |

---

## 评估结果

### 计算 Recall@k

```bash
python3 << 'EOF'
import json, numpy as np
gt = np.load("1_Data/ground_truth/arxiv_small/ground_truth.npy")
with open("results.json") as f:
    results = json.load(f)
k = 10
recalls = []
for q, qr in enumerate(results["queries"]):
    gt_set = set(gt[q, :k])
    res_set = set(qr["labels"][:k])
    recalls.append(len(gt_set & res_set) / k)
recalls = np.array(recalls)
print(f"Recall@{k}: mean={np.mean(recalls):.4f}, median={np.median(recalls):.4f}")
EOF
```

### Profiling 输出解读

```
Profiling (last query):
  query_type: standard
  beam_search: 0.014 ms        ← Phase 1: Beam Search
  frontier_search: 0.403 ms    ← Phase 2: Frontier 优先队列搜索
    (nodes_popped=59, shortlists=56, expanded=2)
  pq_table_build: 0.063 ms     ← PQ 距离查表构建（每 query 仅一次）
  pq_distance_compute: 0.199 ms ← PQ ADC 距离计算（含按需 I/O）
  candidate_merge: 0.089 ms    ← 候选集合并
  rerank: 0.074 ms (count=40)  ← Phase 3: ADC 精确重排
  total: 0.490 ms              ← 总耗时
```

关键诊断：
- `pq_table_build` 有值 → PQ 距离计算已启用
- `pq_distance_compute` > 0 → PQ 码从磁盘按需加载
- `rerank_count=40` (= k×factor) → ADC rerank 正常
- `nodes_popped` 很大 → 调大 `beam_size` 或调整 `variance_boost`

---

## 项目结构

```
new-Baselines/
├── README.md                           # 本文件
│
├── Curator/                            # ★ Curator 独立 C++ 索引
│   ├── CMakeLists.txt                  #   CMake 构建（仅依赖 OpenMP）
│   ├── src/                            #   C++ 源码（14 模块）
│   │   ├── main.cpp                    #     CLI 入口（bench 命令）
│   │   ├── curator_index.h/.cpp        #     主编排类（构建/搜索/管理）
│   │   ├── common.h                    #     公共基础设施（类型/工具类/异常宏）
│   │   ├── config.h                    #     配置参数结构体
│   │   ├── profiling.h                 #     性能分析/内存分解
│   │   ├── distance.h                  #     距离函数（header-only）
│   │   ├── tree_node.h                 #     树节点数据结构
│   │   ├── bloom_filter.h              #     Bloom Filter（header-only）
│   │   ├── cluster_tree.h/.cpp         #     聚类树构建与遍历
│   │   ├── shortlist.h/.cpp            #     短列表分裂/合并
│   │   ├── kmeans.h/.cpp               #     Lloyd K-means 聚类
│   │   ├── pq_codec.h/.cpp             #     PQ 编解码 + 块缓存集成
│   │   ├── pq_block_cache.h/.cpp       #     ★ PQ 码外存 LRU 块缓存（v2 新增）
│   │   ├── flash_store.h/.cpp          #     全精度向量磁盘 I/O
│   │   ├── temp_index.h/.cpp           #     Bitmap Filter 临时索引
│   │   ├── complex_predicate.h/.cpp    #     复杂谓词解析/求值
│   │   └── cnpy.h/.cpp                 #     .npy 文件读写
│   ├── python/                         #   Python 实验脚本
│   │   ├── run_experiment.py           #     实验编排（预处理 → C++ bench → recall）
│   │   ├── preprocess_train.py         #     .pkl → .npy 转换
│   │   └── prepare_queries.py          #     查询数据准备
│   ├── legacy/                         #   旧版 FAISS-SWIG 代码（归档，不再使用）
│   ├── build/                          #   编译产物
│   ├── EXPERIMENT_GUIDE.md             #   实验操作手册（详细）
│   ├── LEARNING_GUIDE.md               #   代码 100% 掌握学习路线
│   ├── REFACTOR_PLAN.md                #   重构设计文档
│   └── PQ_EXTERNAL_STORAGE_PLAN.md     #   PQ 外存改造执行计划（v5）
│
├── DiskIVF-PostFiltering/              # DiskIVF 基线
├── Pre-Filtering/                      # Pre-Filtering 基线
├── SPANN-PostFiltering/                # SPANN 基线
│
├── 1_Data/                             # 数据目录
│   ├── ground_truth/                   #   预处理后的数据集
│   │   ├── arxiv/  arxiv_small/        #     arxiv (1.6M / 50K)
│   │   ├── yfcc100m/  yfcc100m_small/  #     yfcc100m (800K / 50K)
│   │   ├── sift1m/  sift1m_small/      #     sift1m (1M / 5K)
│   │   └── gist1m/  gist1m_small/      #     gist1m (1M / 3K)
│   └── download_ann_datasets.py        #   数据下载+预处理工具
│
├── 3_Config/                           # 实验配置文件
├── 4_Results/                          # 实验结果输出
├── 5_Plot/                             # 可视化脚本
├── analysis/                           # 设计文档与编译指南
└── tests/                              # 诊断脚本
```

---

## 配置参数说明

配置文件为 JSON 格式，所有字段可选（未指定的使用默认值）。通过 `--config` 传入。

### 构建参数

| 参数 | 默认值 | 说明 |
|------|:-----:|------|
| `d` | 128 | 向量维度（自动从数据推断） |
| `n_clusters` | 64 | K-means 聚类分支数（≤ 64） |
| `bf_capacity` | 1000 | Bloom Filter 预期元素数 |
| `bf_false_pos` | 0.01 | Bloom Filter 假阳性率 |
| `max_sl_size` | 128 | 短列表最大长度（超限触发分裂） |
| `clus_niter` | 20 | K-means 迭代次数 |
| `max_leaf_size` | 128 | 叶子节点最大向量数 |

### PQ（乘积量化）参数

| 参数 | 默认值 | 说明 |
|------|:-----:|------|
| `pq_M` | 16 | 子空间数量（d 必须能被 M 整除） |
| `pq_nbits` | 8 | 每子空间量化位数（**仅支持 8**） |
| `pq_enabled` | true | 是否启用 PQ 距离计算 |
| `pq_use_adc_rerank` | false | 是否对 top-K×factor 候选做精确重排 |
| `pq_rerank_topk_factor` | 4 | ADC 重排因子 |
| `persist_pq_codes` | false | 是否在析构后保留 PQ 码磁盘文件 |
| `pq_codes_path` | — | 自定义 PQ 码文件路径（为空则自动生成临时文件） |
| `pq_cache_block_size` | 4096 | ★ PQ 块缓存的每块码数 |
| `pq_cache_max_blocks` | 256 | ★ PQ 块缓存最大块数（M=16→~16MB, M=128→~128MB 上限） |

> ★ **PQ 码外存架构（v2）**：编码后写入磁盘，搜索时通过 LRU 块缓存按需加载。仅实际用到的块进入内存，大幅减少内存占用。`pq_cache_block_size` 和 `pq_cache_max_blocks` 控制缓存行为，默认值可覆盖典型工作集。

### Flash 存储参数

| 参数 | 默认值 | 说明 |
|------|:-----:|------|
| `use_flash_storage` | true | 是否将全精度向量写入磁盘 |
| `flash_path` | — | Flash 文件路径（为空则自动生成临时文件） |

### 搜索参数

| 参数 | 默认值 | 说明 |
|------|:-----:|------|
| `nprobe` | 3000 | 无过滤搜索的探测桶数 |
| `prune_thres` | 1.6 | 无过滤搜索的剪枝阈值倍数 |
| `variance_boost` | 0.4 | 方差提升系数（降低高方差节点优先级） |
| `search_ef` | 128 | Frontier 搜索候选集大小 |
| `beam_size` | 2 | Beam search 宽度（0 = 关闭） |
| `use_temp_index_caching` | true | 是否缓存 bitmap filter 的临时索引 |
| `batch_query` | false | 是否启用查询间 OpenMP 并行化 |

---

## 核心设计概要

Curator 是一棵**层次化 K-means 聚类树**：

```
                 [Root]
               /   |   \  (n_clusters=64 路分支)
            [L1]  [L1]  ...  每个节点含：
            / \    / \        • Bloom Filter（汇总子树租户）
          ...    ...  ...     • Shortlists（租户→vid 有序列表）
          /       \    /      • Centroid + Variance
       [Leaf]   [Leaf]       叶子额外含 vector_indices
```

**搜索流程**（单租户）：

1. **Beam Search** — 沿树下降，每层保留 top-`beam_size` 个节点
2. **Frontier Search** — 优先队列遍历，命中短列表后：
   - **PQ 路径**（默认）：`build_distance_table`（一次）→ `compute_pq_distances`（ADC 近似距离，PQ 码按需 I/O）
   - **精确路径**（回退）：`compute_vector_distance`（FlashStore 全精度读取）
3. **ADC Rerank**（可选）— 对 top-(k×factor) 候选用全精度 L2 重排

**关键不变量**：
- 所有非叶子节点恰好有 `n_clusters` 个子节点
- 短列表大小 ≤ `max_sl_size`（超限触发向下分裂）
- PQ nbits=8 硬约束；PQ 码始终写入磁盘（外存化）
- int_vid_t 为 64 位路径编码（每层 6bit + 最后 10bit leaf local index）
