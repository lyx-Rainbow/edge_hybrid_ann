# Curator 查询阶段分析

> 最后更新: 2026-07-09  
> 分析范围: Curator 索引的查询阶段 Profiling 分解与内存开销  
> 测试配置: n_clusters=32, search_ef=1024, beam_size=4, k=10

---

## 一、查询阶段 Profiling 分解

### 1.1 sift1m_small (50K×128, M=32, 100 labels)

| 指标 | 值 |
|------|:---:|
| Recall@10 (mean/median/min) | 0.9974 / 1.0000 / 0.8000 |
| 完美查询 (Recall=1.0) | 489/500 |
| 构建时间 | 5.0 s |
| 索引内存 | 9.0 MB |
| 平均查询延迟 | 0.23 ms |

**Profiling** (nodes_popped=31, shortlists=31):

| 步骤 | 耗时 (ms) | 占比 |
|------|:---:|:---:|
| Beam Search | 0.006 | 2.3% |
| **Frontier Search** | **0.238** | **90.2%** |
|  PQ Table Build | 0.012 | 4.5% |
|  PQ Distance Compute | 0.117 | 44.3% |
|  Candidate Merge | 0.056 | 21.2% |
| ADC Rerank (count=40) | 0.020 | 7.6% |
| **总计** | **0.264** | **100%** |

```
PQ Distance Compute  ██████████████████████████████████████████ 44.3%
Candidate Merge      ██████████████████ 21.2%
ADC Rerank           ███████ 7.6%
PQ Table Build       ████ 4.5%
Beam Search          ██ 2.3%
Frontier Overhead    ████████████████ 20.1%
```

### 1.2 gist1m_small (50K×960, M=320, 100 labels)

| 指标 | 值 |
|------|:---:|
| Recall@10 (mean/median/min) | 0.9582 / 1.0000 / 0.4000 |
| 完美查询 (Recall=1.0) | 398/500 |
| 构建时间 | 57.7 s |
| 索引内存 | 27.5 MB |
| 平均查询延迟 | 0.82 ms |

**Profiling** (nodes_popped=49, shortlists=45):

| 步骤 | 耗时 (ms) | 占比 |
|------|:---:|:---:|
| Beam Search | 0.079 | 8.7% |
| **Frontier Search** | **0.697** | **76.8%** |
|  PQ Table Build | 0.169 | 18.6% |
|  PQ Distance Compute | 0.421 | 46.4% |
|  Candidate Merge | 0.062 | 6.8% |
| ADC Rerank (count=40) | 0.130 | 14.3% |
| **总计** | **0.908** | **100%** |

```
PQ Distance Compute  ██████████████████████████████████████████████ 46.4%
PQ Table Build       ██████████████████ 18.6%
ADC Rerank           ██████████████ 14.3%
Beam Search          ████████ 8.7%
Candidate Merge      ██████ 6.8%
Frontier Overhead    █████ 5.0%
```

### 1.3 wit_small (48K×384, M=128, 1000 labels)

| 指标 | 值 |
|------|:---:|
| Recall@10 (mean/median/min) | 0.9680 / 1.0000 / 0.0000 |
| 完美查询 (Recall=1.0) | 450/500 |
| 构建时间 | 24.4 s |
| 索引内存 | 23.2 MB |
| 平均查询延迟 | 0.32 ms |

**Profiling** (nodes_popped=1, shortlists=1):

| 步骤 | 耗时 (ms) | 占比 |
|------|:---:|:---:|
| Beam Search | 0.001 | 1.0% |
| Frontier Search | 0.068 | 68.0% |
|  PQ Table Build | 0.057 | 57.0% |
|  PQ Distance Compute | 0.009 | 9.0% |
|  Candidate Merge | 0.000 | 0.0% |
| ADC Rerank (count=40) | 0.031 | 31.0% |
| **总计** | **0.100** | **100%** |

### 1.4 arxiv_small (50K×384, M=128, 99 labels)

| 指标 | 值 |
|------|:---:|
| Recall@10 (mean/median/min) | 0.9970 / 1.0000 / 0.9000 |
| 完美查询 (Recall=1.0) | 485/500 |
| 构建时间 | 22.8 s |
| 索引内存 | 10.9 MB |
| 平均查询延迟 | 0.42 ms |

**Profiling — 硬查询** (nodes_popped=59, shortlists=56):

| 步骤 | 耗时 (ms) | 占比 |
|------|:---:|:---:|
| Beam Search | 0.014 | 3.2% |
| **Frontier Search** | — | — |
|  PQ Table Build | 0.064 | 14.8% |
|  **PQ Distance Compute** | **0.196** | **45.4%** |
|  Candidate Merge | 0.070 | 16.2% |
| ADC Rerank (count=40) | 0.040 | 9.3% |
| **总计** | **0.432** | **100%** |

**Profiling — 简单查询** (nodes_popped=1, shortlists=1):

| 步骤 | 耗时 (ms) | 占比 |
|------|:---:|:---:|
| Beam Search | 0.001 | 1.3% |
| PQ Table Build | 0.032 | 40.0% |
| PQ Distance Compute | 0.010 | 12.5% |
| ADC Rerank (count=40) | 0.034 | 42.5% |
| **总计** | **0.080** | **100%** |

### 1.5 yfcc100m_small (50K×192, M=64, 1000 labels)

| 指标 | 值 |
|------|:---:|
| Recall@10 (mean/median/min) | 0.9970 / 1.0000 / 0.7000 |
| 完美查询 (Recall=1.0) | 488/500 |
| 构建时间 | 11.4 s |
| 索引内存 | 13.4 MB |
| 平均查询延迟 | 0.34 ms |

**Profiling** (nodes_popped≈30, shortlists≈30):

| 步骤 | 耗时 (ms) | 占比 |
|------|:---:|:---:|
| Beam Search | 0.005 | 1.5% |
| PQ Table Build | 0.032 | 9.4% |
| **PQ Distance Compute** | **0.150** | **44.1%** |
| Candidate Merge | 0.045 | 13.2% |
| Frontier Overhead | 0.075 | 22.1% |
| ADC Rerank (count=40) | 0.033 | 9.7% |
| **总计** | **0.340** | **100%** |

---

## 二、查询阶段组件内存开销统计

### 2.0 术语说明

Curator 中有两套独立的聚类中心，名称容易混淆：

| 名称 | 存储位置 | 数量 | 用途 |
|------|------|:---:|------|
| **质心 (Centroids)** | 每个树节点存储 1 个 d 维向量 | `n_nodes` 个 | Beam Search 导航——判断查询向量从哪个分支下降 |
| **PQ 码本 (PQ Codebook)** | PQCodec 内部，M 个子空间 | `M × 256` 个 | PQ 近似距离计算——从压缩码重建子向量算 ADC 距离 |

> **"码本"指的是 PQ 码本**。质心是树的导航结构（每个节点到其父节点的聚类中心），不是通常意义上的"码本"。两者服务于搜索流程的不同阶段：质心用于 **Phase 1 (Beam Search)** 的树导航；PQ 码本用于 **Phase 2 (Frontier Search)** 的近似距离计算。

**理论大小公式**：

| 组件 | 公式 | 说明 |
|------|------|------|
| 质心 | `n_nodes × d × 4` | 每节点 1 个 d 维 float32 向量 |
| PQ 码本 | `256 × d × 4` ⚠ | **与 M 无关**（M × 256 × d/M × 4 = 256 × d × 4） |
| PQ 块缓存 | `pq_cache_max_blocks × pq_cache_block_size × M` | LRU 缓存，运行时懒加载 |
| PQ Table (每查询) | `M × 256 × 4` | 瞬时分配，查询结束即释放 |

### 2.1 各组件内存分解 (bytes)

| 组件 | sift1m_small | gist1m_small | wit_small | 理论公式 |
|------|:---:|:---:|:---:|------|
| | d=128, M=32 | d=960, M=320 | d=384, M=128 | |
| | 1,409 nodes | 3,745 nodes | 5,089 nodes | |
| **Bloom Filters** | 2.42 MB | 6.42 MB | 8.73 MB | `Σ bf.size()` per node |
| **质心 (Centroids)** | 0.69 MB | 13.72 MB | 7.46 MB | `n_nodes × d × 4` ✓ |
| **PQ 码本 (PQ Codebook)** | 0.13 MB | 0.94 MB | 0.38 MB | `256 × d × 4` ✓ |
| **PQ 块缓存** | 1.63 MB | 16.25 MB | 6.00 MB | `max_blocks × block_size × M` |
| **PQ Table (每查询瞬时)** | 0.03 MB | 0.31 MB | 0.13 MB | `M × 256 × 4` |
| Flash 索引映射 | 2.32 MB | 2.37 MB | 2.31 MB | `n × sizeof(FlashEntry)` |
| 短列表 Overhead | 0.18 MB | 0.18 MB | 0.16 MB | hash table 开销 |
| 短列表 Payload | 1.14 MB | 1.14 MB | 1.10 MB | `n × avg_labels × 8` bytes |
| 树节点属性 | 0.39 MB | 1.03 MB | 1.40 MB | `n_nodes × sizeof(TreeNode)` |
| 向量索引 | 0.38 MB | 0.38 MB | 0.37 MB | `n_leaves × avg_leaf_size × 8` |
| ID 映射表 | 1.34 MB | 1.34 MB | 1.28 MB | `n × 2 × 8` (vid_map + tid_map) |
| Tenant ID 映射 | 0.00 MB | 0.00 MB | 0.02 MB | `n_tenants × 4` |
| **总计 (搜索后, 热缓存)** | **10.61 MB** | **43.76 MB** | **29.19 MB** | |

> ⚠ **冷/热缓存差异**: bench 输出的 `memory_bytes`（sift: 8.98 MB / gist: 27.51 MB / wit: 23.19 MB）是在搜索前采样的**冷缓存**值（PQ 块缓存为空）。上表中的总计来自搜索后采样的 `memory_breakdown()`，包含了已填充的 PQ 块缓存。差值精确等于 `pq_cache_bytes`。搜索阶段实际运行态应使用**热缓存**数据。

### 2.2 内存组件分类

按搜索阶段的角色分类：

**常驻内存（构建后即分配，搜索期间不变）**：

| 类别 | 组件 | sift | gist | wit |
|------|------|:---:|:---:|:---:|
| 树结构 | 质心 + 节点属性 | 1.08 MB (10.2%) | 14.74 MB (33.7%) | 8.86 MB (30.4%) |
| 过滤结构 | Bloom Filters | 2.42 MB (22.8%) | 6.42 MB (14.7%) | 8.73 MB (29.9%) |
| 索引映射 | Flash + 向量索引 + ID | 4.04 MB (38.1%) | 4.09 MB (9.3%) | 3.96 MB (13.6%) |
| 短列表 | Overhead + Payload | 1.33 MB (12.5%) | 1.32 MB (3.0%) | 1.26 MB (4.3%) |
| PQ 静态 | 码本 | 0.13 MB (1.2%) | 0.94 MB (2.1%) | 0.38 MB (1.3%) |

**懒加载内存（搜索时按需填充）**：

| 组件 | sift | gist | wit | 说明 |
|------|:---:|:---:|:---:|------|
| PQ 块缓存 | 1.63 MB | 16.25 MB | 6.00 MB | 搜索时 LRU 填充，冷启动为空 |

**瞬时内存（每查询分配，用完释放）**：

| 组件 | sift | gist | wit | 说明 |
|------|:---:|:---:|:---:|------|
| PQ 距离表 | 0.03 MB | 0.31 MB | 0.13 MB | `M × 256 × 4` bytes |

### 2.3 维度对内存影响的量化分析

| 项目 | sift (d=128) | gist (d=960) | wit (d=384) | 缩放关系 |
|------|:---:|:---:|:---:|------|
| 质心 | 0.69 MB | 13.72 MB | 7.46 MB | ∝ `n_nodes × d` |
| PQ 码本 | 0.13 MB | 0.94 MB | 0.38 MB | ∝ `d`（与 M 无关） |
| PQ 块缓存 | 1.63 MB | 16.25 MB | 6.00 MB | ∝ `M`（`M ≈ d/3`） |
| Bloom Filters | 2.42 MB | 6.42 MB | 8.73 MB | ∝ `n_nodes`（树规模，间接相关 d） |
| 其他 | 5.76 MB | 7.63 MB | 5.19 MB | ≈ const（与 n 相关，与 d 弱相关） |

> **结论**: 高维数据集（gist d=960）的常驻内存主要由质心（33.7%）主导，懒加载内存中 PQ 缓存（16.25 MB）也成为显著开销。低维数据集（sift d=128）内存集中于过滤结构和索引映射。PQ 码本本身开销极小（1-2%，`256×d×4`），真正的 PQ 内存开销来自**块缓存**而非码本。

---

*分析日期: 2026-07-09 | 数据来源: Curator bench --profile + memory_breakdown*
