# 过滤向量检索基准测试项目

基于层次化聚类树索引（Curator）的过滤向量检索基准测试平台，对比 **DiskIVF-PostFiltering**、**Pre-Filtering** 和 **SPANN-PostFiltering** 三种基线方法。

Curator 是**独立 C++ 可执行文件**（零 FAISS 依赖），通过 CMake + OpenMP 构建，WSL-Ubuntu 端运行。三种基线方法也为独立 C++ 实现，各自通过 CMake 构建。

---

## 目录

- [环境信息](#环境信息)
- [快速开始](#快速开始)
- [统一运行脚本](#统一运行脚本)
- [编译](#编译)
- [可用数据集](#可用数据集)
- [CLI 参数速查](#cli-参数速查)
- [复杂谓词查询](#复杂谓词查询)
- [配置参数说明](#配置参数说明)
- [评估结果](#评估结果)
- [项目结构](#项目结构)
- [核心设计概要](#核心设计概要)

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11 + WSL Ubuntu 24.04 |
| C++ 编译器 | GCC 13.3.0（需 C++17） |
| CMake | ≥ 3.10 |
| Conda 环境 | `edge_ann`（Python 3.10, numpy, faiss-cpu 1.7.4） |
| WSL 路径 | `/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines` |

> **注**: FAISS 仅用于 DiskIVF / SPANN 基线方法。Curator 本身**不依赖 FAISS**，Pre-Filtering 基线基于暴力搜索也不依赖 FAISS。

---

## 快速开始

### 环境激活

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
```

### 一键运行（推荐）

```bash
# 使用 run.sh 统一脚本
bash run.sh curator arxiv_small            # Curator 单次 benchmark
bash run.sh diskivf yfcc100m_small         # DiskIVF 单次 benchmark
bash run.sh pre-filter arxiv_small --sweep # Pre-Filtering 参数扫描
bash run.sh spann yfcc100m_small --sweep   # SPANN 参数扫描
```

### 100K benchmark 统一流程（当前主流程）

```bash
# 全方法 SL+CP sweep（结果写入 4_Results/{Curator,DiskIVF,SPANN,Pre-Filtering}/sweep_{dataset}.json）
python3 run_100k_sweep.py --dataset sift1m_100k --method all --run_sl --run_cp

# 增量补跑：只跑清单内的新参数 combo 并合并进现有 sweep JSON（已存在 combo 自动跳过）
python3 run_incremental.py --method DiskIVF
python3 run_incremental.py --method Curator
python3 run_incremental.py --method SPANN

# 折线验收报告（每条 Pareto frontier 的点数/左右端/是否 ≥0.95）
python3 tests/report_frontier.py
```

### 手动运行 C++ 二进制

```bash
# 编译后直接调用 Curator
./Curator/build/curator bench \
    --train_vecs     $DATA/train_vecs.npy \
    --train_access   $DATA/train_access.npy \
    --queries        $DATA/query_vecs.npy \
    --query_labels   $DATA/query_labels.npy \
    --config         /tmp/config.json \
    --k 10 --profile --output results.json
```

---

## 统一运行脚本

`run.sh` 支持四种索引 × 多种数据集的统一入口：

```bash
bash run.sh <index> <dataset> [--sweep]
```

| 参数 | 可选值 |
|------|--------|
| `<index>` | `curator` \| `pre-filter` \| `diskivf` \| `spann` |
| `<dataset>` | `arxiv` \| `arxiv_small` \| `yfcc100m` \| `yfcc100m_small` |
| `--sweep` | 可选，运行参数网格扫描 |

配置文件位于 `3_Config/<Index>/<index>_<dataset>.json`，扫描配置位于 `3_Config/<Index>/sweep.json`。

---

## 编译

### Curator（独立 C++，仅依赖 OpenMP）

```bash
cd Curator
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

编译产物：`Curator/build/curator`。

系统依赖：
```bash
sudo apt install libomp-dev   # OpenMP（K-means 并行 + batch query）
```

### 基线方法

各基线方法（DiskIVF-PostFiltering、Pre-Filtering、SPANN-PostFiltering）也有独立的 CMake 构建系统，编译方式相同：

```bash
cd <BaselineDir>
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

> SPANN-PostFiltering 依赖 `SPTAG-main/` 开源代码（SPTAG ANN 库），首次编译需要先构建 SPTAG。

---

## 可用数据集

| 数据集 | 向量数 | 维度 | 标签数 | 查询数 | 类型 |
|--------|:-----:|:----:|:-----:|:-----:|------|
| sift1m_small | 5,000 | 128 | 100 | 500 | 快速验证 |
| gist1m_small | 3,000 | 960 | 100 | 500 | 快速验证 |
| arxiv_small | 50,000 | 384 | 99 | 1,000 | 中等规模 |
| yfcc100m_small | 50,000 | 192 | 1,000 | 1,000 | 中等规模 |
| wit_small | 50,000 | 384 | 1,000 | 500 | 中等规模（WIT 文本嵌入） |
| sift1m | 1,000,000 | 128 | 100 | 500 | 完整规模 |
| gist1m | 1,000,000 | 960 | 100 | 500 | 完整规模 |
| arxiv | 1,600,000 | 384 | 99 | 500 | 完整规模 |
| yfcc100m | 800,000 | 192 | 1,000 | 500 | 完整规模 |
| wit | ~3,000,000 | 384 | 1,000 | 1,000 | 完整规模（WIT 文本嵌入） |

数据位于 `1_Data/ground_truth/<dataset>/`。

### 100K 统一规格数据集（当前 benchmark 主力）

| 数据集 | 向量数 | 维度 | 标签数 | 标签方式 | 查询数 |
|--------|:-----:|:----:|:-----:|---------|:-----:|
| sift1m_100k | 100,000 | 128 | 100 | random（层次 Bernoulli） | 1,000 |
| yfcc100m_100k | 100,000 | 192 | 1,000 | natural（Flickr tag） | 1,000 |
| arxiv_100k | 100,000 | 384 | 99 | natural（学科分类） | 1,000 |
| gist1m_100k | 100,000 | 960 | 100 | random（层次 Bernoulli） | 500 |
| wit_100k | 100,000 | 384 | 1,000 | random（层次 Bernoulli） | 1,000 |

- 每个 100k 数据集均含单标签（SL）与复杂谓词（CP）两套 ground truth；
- 目录：`1_Data/ground_truth/<dataset>_100k/`，规格详见 `1_Data/ground_truth/DATASETS_100K.md`；
- 100k 的 sweep 配置：`3_Config/*/sweep_100k.json`；实验结果：`4_Results/{Curator,DiskIVF,SPANN,Pre-Filtering}/sweep_*_100k.json`（不入版本库）。

### 各数据集必需文件

```
1_Data/ground_truth/<dataset>/
├── train_vecs.npy         # 训练向量 [N, d] float32
├── train_mds.pkl          # 训练访问列表 (Python pickle: list[list[int]])
├── train_access.npy       # 访问对 [M, 2] int32（由 train_mds.pkl 预处理生成）
├── query_vecs.npy         # 查询向量 [Q, d] float32
├── query_labels.npy       # 每条查询的 tenant_id [Q] int32（-1 = 无过滤）
├── ground_truth.npy       # Ground Truth 标签 [Q, k] int32
├── metadata.json          # 数据集元信息（维度、标签数、查询数等）
├── all_labels.json        # 所有标签 ID 的集合
├── query_info.json        # 查询详细信息（标签、选择性等）
└── complex_predicate/     # （部分数据集）复杂谓词语料
    ├── filters.json        #   预定义的 RPN 过滤器列表
    ├── query_vecs.npy      #   复杂谓词专用查询向量
    ├── query_indices.npy   #   查询索引映射
    └── gt_*.npy            #   每个过滤器的 Ground Truth
```

### 数据集说明

- **arxiv / arxiv_small**: arXiv 学术论文摘要经 `all-MiniLM-L6-v2` 嵌入（384 维），标签为 arXiv 学科分类（99 类，多标签）
- **yfcc100m / yfcc100m_small**: Yahoo Flickr Creative Commons 100M 子采样，向量为图像特征（192 维），标签为用户 tag（1,000 类，多标签）
- **sift1m / sift1m_small**: SIFT 图像描述符（128 维），标签通过 K-means 聚类合成（100 类）
- **gist1m / gist1m_small**: GIST 图像描述符（960 维），标签通过 K-means 聚类合成（100 类）
- **wit / wit_small**: Wikipedia-based Image Text 数据集，文本经 `all-MiniLM-L6-v2` 嵌入（384 维），标签通过 K-means 聚类合成（1,000 类）

> **标签语义**: `train_mds[i] = [1, 5, 23]` 表示向量 `i` 可被租户 1、5、23 访问。过滤检索时，查询限定租户，仅搜索该租户有权访问的向量子集。

---

## CLI 参数速查

### Curator bench 命令

```
curator bench [options]
```

| 参数 | 必需 | 说明 |
|------|:--:|------|
| `--train_vecs PATH` | ✅ | 训练向量 `.npy` [N, d] float32 |
| `--train_access PATH` | — | 访问对 `.npy` [M, 2] int32（无标签过滤时可省略） |
| `--queries PATH` | ✅ | 查询向量 `.npy` [Q, d] float32 |
| `--query_labels PATH` | — | 查询标签 `.npy` [Q] int32（-1=无过滤，省略则全部无过滤） |
| `--filter EXPR` | — | 全局复杂谓词表达式（RPN 后缀表达式），应用于全部查询 |
| `--query_filters PATH` | — | Per-query 复杂谓词文件（每行一个 RPN 表达式，空行=回退到 `--filter`） |
| `--config PATH` | — | JSON 配置文件（省略则使用 C++ 默认值） |
| `--k K` | — | 返回结果数（默认 10） |
| `--batch-query` | — | 启用查询间 OpenMP 并行化 |
| `--output PATH` | — | 结果 JSON 路径（默认 `results.json`） |
| `--profile` | — | 打印最后一个查询的详细计时分解 |
| `--help` | — | 显示帮助信息 |

### 优先级规则

当同时指定多种过滤方式时，优先级为：
1. **per-query filter**（`--query_filters` 中非空行）
2. **全局 filter**（`--filter`）
3. **query_labels**（`--query_labels` 中的单租户 ID）
4. **无过滤搜索**（全部向量参与检索）

---

## 复杂谓词查询

Curator 支持 AND / OR / NOT 布尔组合的复杂谓词过滤。谓词表达式使用 **Reverse Polish Notation (RPN / 后缀表达式)**。

### RPN 语法速查

| 含义 | ❌ 中缀（不可用） | ✅ RPN（正确） |
|------|----------------|---------------|
| A AND B | `1 AND 2` | `1 2 AND` |
| A OR B | `3 OR 4` | `3 4 OR` |
| (A AND B) OR C | `(1 AND 2) OR 3` | `1 2 AND 3 OR` |
| A AND NOT B | `7 AND NOT 2` | `7 2 NOT AND` |
| NOT A | `NOT 1` | `1 NOT` |
| 单租户 | `1` | `1` |

> **注意**: 谓词中的数字引用**内部租户 ID**（int_lid_t），而非外部标签（ext_lid_t）。对于从 0 开始连续编号的标签（arxiv 0..98, yfcc100m 0..999, sift 0..99），两者恒等。

### 使用方式一：全局谓词（所有查询共用）

```bash
# 查询同时属于租户 1 AND 2 的向量
./Curator/build/curator bench \
    --train_vecs   $DATA/train_vecs.npy \
    --train_access $DATA/train_access.npy \
    --queries      $DATA/query_vecs.npy \
    --query_labels $DATA/query_labels.npy \
    --filter       '1 2 AND' \
    --config       /tmp/arxiv_config.json \
    --k 10 --output $DATA/results.json
```

### 使用方式二：Per-Query 谓词（不同查询使用不同过滤条件）

创建 `query_filters.txt`（每行一个 RPN 表达式，空行=回退到 `--filter` 或 `--query_labels`）：

```
1 2 AND
3 4 AND 5 OR

7 2 NOT AND
1
```

```bash
./Curator/build/curator bench \
    --train_vecs     $DATA/train_vecs.npy \
    --train_access   $DATA/train_access.npy \
    --queries        $DATA/query_vecs.npy \
    --query_labels   $DATA/query_labels.npy \
    --query_filters  query_filters.txt \
    --config         /tmp/arxiv_config.json \
    --k 10 --output $DATA/results.json
```

### 实现机制

复杂谓词查询分两阶段执行：

1. **Phase 1（串行预构建）**: 遍历所有查询的谓词表达式，对每个唯一谓词调用 `find_all_qualified_vecs()`（全量扫描 + RPN 栈式求值），然后调用 `build_filter_index()` 构建临时索引。利用 `filter_to_label_` 映射自动去重，相同谓词只构建一次。
2. **Phase 2（并行搜索）**: 每条查询使用预构建的 filter_label 调用 `index.search()`。此阶段仅执行 const 方法，对 OpenMP `batch_query` 模式安全。

### 复杂谓词 Ground Truth 数据

部分数据集（arxiv_small、yfcc100m_small、sift1m_small、yfcc100m）预置了复杂谓词 Ground Truth，位于 `complex_predicate/` 子目录。每个 `gt_*.npy` 文件对应一个 RPN 过滤器，命名遵循 `gt_{OP}_{ID1}_{OP}_{ID2}...` 规则（如 `gt_AND_10_61.npy`、`gt_NOT_31.npy`、`gt_OR_13_OR_56_81.npy`）。

---

## 配置参数说明

配置文件为 JSON 格式，所有字段可选（未指定的使用默认值）。通过 `--config` 传入。

### 构建参数（影响索引结构）

| 参数 | 默认值 | 说明 |
|------|:-----:|------|
| `d` | 128 | 向量维度（自动从数据推断） |
| `n_clusters` | 64 | K-means 聚类分支数（≤ 64） |
| `bf_capacity` | 1000 | Bloom Filter 预期元素数 |
| `bf_false_pos` | 0.01 | Bloom Filter 假阳性率 |
| `max_sl_size` | 128 | 短列表最大长度（超限触发分裂） |
| `clus_niter` | 20 | K-means 迭代次数 |
| `max_leaf_size` | 128 | 叶子节点最大向量数（超过则继续分裂） |

### PQ（乘积量化）参数

| 参数 | 默认值 | 说明 |
|------|:-----:|------|
| `pq_M` | 16 | 子空间数量（d 必须能被 M 整除） |
| `pq_nbits` | 8 | 每子空间量化位数（**当前仅支持 8**） |
| `pq_enabled` | true | 是否启用 PQ 距离计算 |
| `pq_use_adc_rerank` | false | 是否对 top-(k×factor) 候选做精确距离重排 |
| `pq_rerank_topk_factor` | 4 | ADC 重排因子 |
| `persist_pq_codes` | false | 是否在析构后保留 PQ 码磁盘文件 |
| `pq_codes_path` | — | 自定义 PQ 码磁盘文件路径（为空则自动生成临时文件） |
| `pq_cache_block_size` | 4096 | PQ 块缓存的每块码数（块 = 连续 block_size 条向量的 PQ 码） |
| `pq_cache_max_blocks` | 256 | PQ 块缓存最大块数（M=16→~16MB, M=128→~128MB 内存上限） |

> **PQ 码外存架构（v2）**: 编码后写入磁盘，搜索时通过 LRU 块缓存按需加载。仅实际用到的块进入内存，大幅减少内存占用。`pq_cache_block_size` 和 `pq_cache_max_blocks` 控制缓存行为，默认值可覆盖典型工作集。

### Flash 存储参数

| 参数 | 默认值 | 说明 |
|------|:-----:|------|
| `use_flash_storage` | true | 是否将全精度向量写入磁盘 |
| `flash_path` | — | Flash 文件路径（为空则自动生成临时文件） |

### 搜索参数

| 参数 | 默认值 | 说明 |
|------|:-----:|------|
| `nprobe` | 3000 | 无过滤搜索的探测桶数（仅 `search_unfiltered`） |
| `prune_thres` | 1.6 | 无过滤搜索的剪枝阈值倍数 |
| `variance_boost` | 0.4 | 方差提升系数（降低高方差节点优先级） |
| `search_ef` | 128 | Frontier 搜索候选集大小 |
| `beam_size` | 2 | Beam search 宽度（0 = 关闭 beam search） |
| `use_temp_index_caching` | true | 是否缓存 bitmap filter 的临时索引 |
| `batch_query` | false | 是否启用查询间 OpenMP 并行化 |

### 常用配置模板

#### arxiv / arxiv_small（d=384, M=128）

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

#### yfcc100m / yfcc100m_small（d=192, M=64）

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

#### sift1m / sift1m_small（d=128, M=32）

```bash
cat > /tmp/sift_config.json << 'JSON'
{
  "d": 128, "n_clusters": 32,
  "max_sl_size": 256, "max_leaf_size": 128,
  "pq_M": 32, "pq_nbits": 8,
  "pq_enabled": true, "pq_use_adc_rerank": true, "pq_rerank_topk_factor": 4,
  "pq_cache_block_size": 4096, "pq_cache_max_blocks": 256,
  "use_flash_storage": true,
  "variance_boost": 0.4, "search_ef": 1024, "beam_size": 4
}
JSON
```

---

## 评估结果

### 输出 JSON 格式

```json
{
  "config": { "d": 384, "nlist": 32, "pq_M": 128, "search_ef": 1024 },
  "build_time_s": 18.38,
  "memory_bytes": 11470000,
  "memory_breakdown": {
    "num_tree_nodes": 156,
    "tree_node_attrs_bytes": 49920,
    "centroids_bytes": 498624,
    "bloom_filter_bytes": 52416,
    "shortlists_overhead_bytes": 280000,
    "shortlists_payload_bytes": 1048576,
    "vector_indices_bytes": 204800,
    "id_allocator_bytes": 262144,
    "tenant_id_allocator_bytes": 4096,
    "pq_codebook_bytes": 1572864,
    "pq_cache_bytes": 262144,
    "flash_index_bytes": 2800000,
    "raw_vectors_buffer_bytes": 0,
    "temp_index_cache_bytes": 0,
    "temp_qualified_vecs_bytes": 0,
    "total_bytes": 8023584
  },
  "queries": [
    {
      "idx": 0,
      "tenant_id": 48,
      "labels": [37965, 34908, 47118, 28570, 14872, 18150, 2343, 3369, 17616, 5198],
      "distances": [12.34, 15.67, 18.90, ...]
    }
  ]
}
```

### 关键指标

| 指标 | 含义 | 来源 |
|------|------|------|
| `build_time_s` | 索引构建总耗时（训练 + 插入 + 授权 + flush） | JSON |
| `memory_bytes` | 索引内存占用（全部组件合计） | JSON |
| `memory_breakdown` | 各组件内存分解（树节点、Bloom Filter、短列表、PQ 缓存等） | JSON |
| `avg_latency_ms` | 平均查询延迟 | 从 `total_search_time / n_queries` 推算 |
| `Recall@k` | 前 k 个结果与 Ground Truth 的交集比例 | Python 计算 |
| `profiling` 输出 | beam / frontier / rerank 各阶段耗时分解 | stdout（需 `--profile`） |

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
print(f"Recall@{k}: mean={np.mean(recalls):.4f}, median={np.median(recalls):.4f}, "
      f"P90={np.percentile(recalls, 90):.4f}, Min={np.min(recalls):.4f}")
EOF
```

### Profiling 输出解读

```
Profiling (last query):
  query_type: standard
  beam_search: 0.015 ms        ← Phase 1: Beam Search
  frontier_search: 1.295 ms    ← Phase 2: Frontier 优先队列搜索
    (nodes_popped=59, shortlists=56, expanded=2)
  pq_table_build: 0.063 ms     ← PQ 距离查表构建（每 query 仅一次）
  pq_distance_compute: 0.199 ms ← PQ ADC 距离计算（含按需 I/O）
  candidate_merge: 0.075 ms    ← 候选集合并
  rerank: 0.040 ms (count=40)  ← Phase 3: ADC 精确重排
  total: 1.350 ms              ← 总耗时
```

关键诊断：
- `pq_table_build` 有值 → PQ 距离计算已启用
- `pq_distance_compute` > 0 → PQ 码从磁盘按需加载（首次查询可能略慢，冷启动 I/O）
- `rerank_count=40` (= k×factor) → ADC rerank 正常工作
- `nodes_popped` 很大 → 调大 `beam_size` 或调整 `variance_boost`
- `shortlists=0` → BF 过于保守，尝试提高 `bf_false_pos`
- `query_type: temp_index` → 命中复杂谓词临时索引缓存（快速路径）

### 消融实验

| 实验 | 配置变更 |
|------|---------|
| 关闭 PQ（纯精确距离） | `"pq_enabled": false` |
| 关闭 ADC Rerank（纯 PQ 近似） | `"pq_use_adc_rerank": false` |
| 关闭 Flash（纯内存模式） | `"use_flash_storage": false` |
| 多线程查询 | `"batch_query": true`（或 CLI `--batch-query`） |

---

## 项目结构

```
new-Baselines/
├── README.md                              # 本文件
├── run.sh                                 # 旧版全量数据集实验入口（4 种索引 × 4 种数据集）
├── run_100k_sweep.py                      # ★ 100K 统一 sweep 执行器（SL+CP，全方法）
├── run_incremental.py                     # ★ 增量补跑器（只跑清单内新 combo，合并进 sweep JSON）
├── run_night_queue.sh                     # 夜间队列脚本（Curator→SPANN→验收→出图→清理）
├── setup_curator.sh                       # [已弃用] 旧版 FAISS-SWIG 构建脚本
│
├── Curator/                               # ★ Curator 独立 C++ 索引
│   ├── CMakeLists.txt                     #   CMake 构建（仅依赖 OpenMP）
│   ├── src/                               #   C++ 源码（16 模块）
│   │   ├── main.cpp                       #     CLI 入口（bench 命令；构建后释放原始向量/堆紧缩）
│   │   ├── curator_index.h/.cpp           #     主编排类（含 compact_memory() 查询期内存紧缩）
│   │   ├── common.h                       #     公共基础设施（类型/工具类/异常宏）
│   │   ├── config.h                       #     配置参数结构体
│   │   ├── profiling.h                    #     性能分析/内存分解
│   │   ├── distance.h                     #     距离函数（header-only）
│   │   ├── tree_node.h                    #     树节点（Bloom Filter + Shortlists）
│   │   ├── bloom_filter.h                 #     Bloom Filter（header-only）
│   │   ├── cluster_tree.h/.cpp            #     聚类树构建与遍历
│   │   ├── shortlist.h/.cpp               #     短列表分裂/合并
│   │   ├── kmeans.h/.cpp                  #     Lloyd K-means 聚类
│   │   ├── pq_codec.h/.cpp                #     PQ 编解码 + 块缓存集成
│   │   ├── pq_block_cache.h/.cpp          #     ★ PQ 码外存 LRU 块缓存（v2）
│   │   ├── flash_store.h/.cpp             #     全精度向量磁盘 I/O
│   │   ├── temp_index.h/.cpp              #     Bitmap Filter 临时索引
│   │   ├── complex_predicate.h/.cpp       #     ★ 复杂谓词解析/求值（RPN）
│   │   └── cnpy.h/.cpp                    #     .npy 文件读写
│   ├── python/                            #   Python 实验脚本
│   ├── legacy/                            #   旧版 FAISS-SWIG 代码（归档，不再使用）
│   ├── build/                             #   编译产物
│   └── EXPERIMENT_GUIDE.md                #   实验操作手册（详细）
│
├── DiskIVF-PostFiltering/                 # DiskIVF 基线（C++ 实现）
│   ├── CMakeLists.txt
│   ├── src/  python/  build/  legacy/
│   └── memory/                            #   内存分析数据
│
├── Pre-Filtering/                         # Pre-Filtering 基线（C++ 暴力搜索实现）
│   ├── CMakeLists.txt
│   ├── src/  python/  build/  legacy/
│
├── SPANN-PostFiltering/                   # SPANN 基线（C++ 实现，依赖 SPTAG）
│   ├── CMakeLists.txt
│   ├── SPTAG-main/                        #   SPTAG ANN 库（开源代码）
│   └── src/  python/  build/  legacy/
│       └── src/config.h                   #   额外支持 bkt_kmeans_k / search_internal_result_num
│
├── 1_Data/                                # 数据目录
│   ├── arxiv/  sift1m/  gist1m/           #   原始数据
│   ├── yfcc100m/  wit/                    #   原始数据
│   ├── ground_truth/                      #   预处理后的数据集（含 *_100k）
│   ├── description.md                     #   数据集详细说明文档
│   ├── ground_truth/description.md        #   Ground Truth 计算流程文档
│   ├── ground_truth/DATASETS_100K.md      #   100K 统一规格说明
│   └── prepare_*.py / compute_cp_gt.py / gt_computing.py  # 数据生成脚本
│
├── 2_Utils/                               # Python 工具库
│   ├── predicate.py                       #   复杂谓词评估（Python 版）
│   ├── memory_profiler.py                 #   内存分析工具
│   ├── memory_utils.py                    #   内存工具函数
│   ├── query_profiler.py                  #   查询性能分析工具
│   └── utils.h                            #   C++ 工具头文件
│
├── 3_Config/                              # 实验配置文件
│   ├── Curator/                           #   Curator 配置（8 数据集 + sweep + sweep_100k）
│   ├── DiskIVF-PostFiltering/             #   DiskIVF 配置（+ sweep_100k）
│   ├── Pre-Filtering/                     #   Pre-Filtering 配置（+ sweep_100k）
│   └── SPANN-PostFiltering/               #   SPANN 配置（+ sweep_100k / sweep_100k_reduced）
│
├── 4_Results/                             # 实验结果输出（除 fig_100k 外不入版本库）
│   ├── Curator/  DiskIVF-PostFiltering/   #   各方法结果与中间产物
│   ├── Pre-Filtering/  SPANN-PostFiltering/
│   ├── fig_100k/                          #   ★ 100K 可视化输出（PNG + SVG，入库）
│   │   ├── fig_sl_qps_recall_100k.*       #     SL QPS-Recall 折线（Pareto frontier）
│   │   ├── fig_cp_qps_recall_100k.*       #     CP QPS-Recall 折线
│   │   ├── fig_sl_qps_recall_by_bucket.*  #     选择率分面（5×3）
│   │   ├── fig_memory_100k.*              #     Memory 柱状图（查询期峰值 RSS）
│   │   └── fig_memory_100k_v2.*           #     Memory v2（Curator 内存优化配置）
│   ├── memory_tuned/                      #   Curator 内存优化版测量数据
│   └── RESULTS_ANALYSIS.md                #   结果分析文档
│
├── 5_Plot/                                # 可视化脚本
│   ├── fig_sl_qps_recall_100k.py          #   SL QPS-Recall（自适应横轴 + 单调 frontier）
│   ├── fig_cp_qps_recall_100k.py          #   CP QPS-Recall
│   ├── fig_sl_qps_recall_by_bucket.py     #   选择率分面
│   ├── fig_memory_100k.py                 #   Memory（统一 RSS 口径）
│   ├── fig_memory_100k_v2.py              #   Memory v2（Curator 内存优化配置）
│   ├── fig1_sl_latency_recall.py          #   [旧] FIG1: 全量数据集 SL
│   ├── fig2_cp_latency_recall.py          #   [旧] FIG2: 全量数据集 CP
│   ├── fig3_memory.py                     #   [旧] FIG3: 内存占用对比
│   ├── fig4_build_time.py                 #   [旧] FIG4: 构建时间对比
│   └── utils.py                           #   绘图工具函数（INDEX_META 等）
│
└── tests/                                 # 诊断与验收脚本
    ├── report_frontier.py                 #   折线验收报告（点数/左右端/≥0.95）
    ├── quick_check.py                     #   单个 C++ 输出 JSON 的 recall/QPS 速查
    ├── run_memory_tuned.py                #   Curator 内存优化版测量
    ├── test_sir.py                        #   SPTAG SearchInternalResultNum 假设验证
    └── check_recall.py / prune_empty_entries.py
```

---

## 核心设计概要

Curator 是一棵**层次化 K-means 聚类树**：

```
                 [Root]
               /   |   \  (n_clusters=64 路分支)
            [L1]  [L1]  ...  每个节点含：
            / \    / \        • Bloom Filter（汇总子树租户）
          ...    ...  ...     • Shortlists（租户→vid 有序列表）
          /       \    /      • Centroid + Running Variance
       [Leaf]   [Leaf]       叶子额外含 vector_indices
```

### 构建流程

```
train()                     # K-means 递归构建聚类树
  └─ add_vector() × N       # 逐向量分配叶节点 + 路径编码 vid
       └─ grant_access() × M # 授权：vid→tid 沿树向下推送
            └─ flush()       # PQ 训练→编码→写入磁盘 + Flash 存储固化
                             #   释放 raw_buffer，后续按需从磁盘加载
```

### 搜索流程（单租户过滤检索）

1. **Beam Search** — 沿树下降，每层保留 top-`beam_size` 个节点
2. **Frontier Search** — 优先队列遍历，命中短列表后：
   - **PQ 路径**（默认）：`build_distance_table`（每查询一次）→ `compute_pq_distances`（ADC 近似距离，PQ 码通过 LRU 块缓存按需 I/O 加载）
   - **精确路径**（回退）：`compute_vector_distance`（FlashStore 全精度读取）
3. **ADC Rerank**（可选，`pq_use_adc_rerank=true`）— 对 top-(k×factor) 候选通过 FlashStore 批量读取全精度向量，计算精确 L2 距离重排

### 复杂谓词搜索流程

```
find_all_qualified_vecs(filter)    # 全量扫描 + RPN 栈式求值，收集满足谓词的 vid
  └─ build_filter_index(filter)    # 构建 TempIndexNode 树（bitmap 临时索引）+ 缓存
       └─ search(query, k, filter_label)
            └─ search_one()        # 命中 temp_index 缓存，走快速路径
                 └─ search_temp_index()  # 在临时索引上执行 beam + frontier 搜索
```

### 搜索路径对比

| 查询类型 | query_type | 搜索方法 | 说明 |
|---------|-----------|---------|------|
| 单租户（有过滤） | `standard` | `search_one()` | 主集群树 beam + frontier + PQ/精确回退 |
| 单租户（无过滤） | — | `search_unfiltered()` | 按质心距离探测叶子桶 |
| 复杂谓词 | `temp_index` | `search_temp_index()` | 从缓存读取 TempIndexNode 树进行搜索 |
| Bitmap 直接 | `bitmap_filter` | `search_with_bitmap()` | [保留接口，当前未使用] |

### 关键不变量

- 所有非叶子节点恰好有 `n_clusters` 个子节点
- 短列表大小 ≤ `max_sl_size`（超限触发向下分裂）
- PQ nbits=8 硬约束；PQ 码始终写入磁盘（外存化），搜索时通过 LRU 块缓存按需加载
- `int_vid_t` 为 64 位路径编码（每层 6bit + 最后 10bit leaf local index）
- `int_lid_t` 为 16 位内部标签 ID（通过 `TenantIdAllocator` 的 hole-reuse 策略保持紧凑）

### 数据流概览

```
原始数据 (1_Data/{arxiv,yfcc100m,sift1m,gist1m,wit}/)
    │
    ├─ prepare_*.py  ──→  ground_truth/<dataset>/
    │                        ├── train_vecs.npy + train_mds.pkl
    │                        ├── query_vecs.npy + query_labels.npy
    │                        └── ground_truth.npy
    │
    ├─ preprocess_train.py  ──→  train_access.npy  (C++ 消费)
    │
    └─ run_experiment.py  ──→  curator bench
          │                       ├── 加载 .npy → 构建索引 → 搜索
          │                       └── 输出 results.json
          │
          └─ 计算 Recall@k (vs ground_truth.npy)
```
