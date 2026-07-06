# 基线索引 C++ 迁移执行方案（修订版）

> 将 Pre-Filtering、DiskIVF-PostFiltering、SPANN-PostFiltering 三个基线索引从纯 Python 实现改为 C++ 实现 + Python 离线预处理与实验编排的分工模式。

**创建日期**: 2026-07-05  
**修订日期**: 2026-07-06（v2：取消 shared/ 公共基础设施，各索引文件夹自包含）  
**参考模型**: Curator 索引的 C++/Python 分工模式

---

## 目录

- [0. 背景与分工模式](#0-背景与分工模式)
- [1. 总体设计原则：文件夹自包含](#1-总体设计原则文件夹自包含)
- [2. 任务一：Pre-Filtering 索引 C++ 化](#2-任务一pre-filtering-索引-c-化)
- [3. 任务二：DiskIVF-PostFiltering 索引 C++ 化](#3-任务二diskivf-postfiltering-索引-c-化)
- [4. 任务三：SPANN-PostFiltering 索引 C++ 化（基于 SPTAG-main）](#4-任务三spann-postfiltering-索引-c-化基于-sptag-main)
- [5. 实施顺序与里程碑](#5-实施顺序与里程碑)
- [6. 跨 Phase 一致性要求](#6-跨-phase-一致性要求)
- [7. 验证方案](#7-验证方案)
- [8. 风险与注意事项](#8-风险与注意事项)
- [附录 A：文件变更总览](#附录-a文件变更总览)
- [附录 B：关键接口对照表](#附录-b关键接口对照表)

---

## 0. 背景与分工模式

### 0.1 当前状态

| 索引 | 实现语言 | 核心依赖 | 构建方式 | 检索方式 |
|------|:---:|------|------|------|
| Curator | C++ | OpenMP（零 FAISS） | CMake → 独立可执行文件 | CLI `curator bench` |
| Pre-Filtering | Python | numpy | — | `python run_prefiltering.py` |
| DiskIVF-PostFiltering | Python | numpy, faiss-cpu | — | `python run_diskivf.py` |
| SPANN-PostFiltering | Python | numpy, faiss-cpu | — | `python run_spann.py` |

### 0.2 目标分工模式（参照 Curator）

```
┌─────────────────────────────────────────────────────────┐
│ Python（离线预处理 + 实验编排）                           │
│  • 数据格式转换（.pkl → .npy / .bin）                    │
│  • 生成配置文件（JSON）                                  │
│  • 调用 C++ 可执行文件（subprocess）                      │
│  • 计算 Recall 指标                                      │
│  • 参数扫描 / 结果汇总                                   │
└─────────────────────────────────────────────────────────┘
                          │ subprocess
                          ▼
┌─────────────────────────────────────────────────────────┐
│ C++（索引构建 + 检索，独立可执行文件）                     │
│  • 读取 .npy / .bin 数据                                 │
│  • 构建索引（训练 + 插入）                                │
│  • 执行检索（Single-label + Complex-predicate）           │
│  • 输出 JSON 结果                                        │
│  • Profiling 输出                                        │
└─────────────────────────────────────────────────────────┘
```

### 0.3 数据流约定

所有索引使用统一的数据格式：

```
1_Data/ground_truth/<dataset>/
├── train_vecs.npy       # [N, d] float32  训练向量
├── train_access.npy     # [M, 2] int32    (vid, tid) 访问对
├── train_mds.pkl        # Python pickle   标签列表（仅预处理阶段使用）
├── query_vecs.npy       # [Q, d] float32  查询向量
├── query_labels.npy     # [Q] int32       查询标签（-1 = 无过滤）
├── query_info.json      # 查询元信息（bucket 分布等）
└── ground_truth.npy     # [Q, 100] int32  Ground Truth
```

**Python 预处理职责**（对所有索引统一）：
1. 读取 `train_mds.pkl`，转换为 `train_access.npy`（`[vid, tid]` pairs）或索引专用的 metadata 二进制文件
2. 复杂谓词查询数据的预处理（生成 `query_vecs.npy` + `filters.json` + `gt_*.npy`）

---

## 1. 总体设计原则：文件夹自包含

### 1.1 核心原则

**每个索引的 C++ 实现完全局限在其自己的文件夹内，不提取共享基础设施。**

| 对比维度 | 原方案（Phase 1） | 修订方案 |
|------|:---:|------|
| 共享代码组织 | `shared/` 目录，所有索引引用 | 各索引自行复制所需文件 |
| 编译方式 | 各索引 CMake 引用 shared/ 路径 | 各索引独立 CMake，完全自包含 |
| Curator 源码 | 零变更 | 零变更（保持不变） |
| 代码复用 | 通过路径引用 | 通过文件复制（允许独立演进） |
| 耦合度 | shared/ 修改影响所有索引 | 各索引完全解耦 |

### 1.2 每个索引文件夹的 C++ 部分结构

```
<Index-Name>/
├── CMakeLists.txt                    # 独立 CMake 构建
├── src/
│   ├── main.cpp                      # CLI 入口（bench 命令）
│   ├── config.h                      # 配置结构体
│   ├── <index>_index.h/.cpp          # 核心索引类
│   ├── cnpy.h / cnpy.cpp             # .npy 读写（从 Curator 复制）
│   ├── distance.h                    # L2 距离函数（从 Curator 精简复制）
│   └── predicate.h                   # 前缀记法谓词求值器（全新 header-only）
├── python/
│   ├── run_experiment.py             # 实验编排脚本
│   └── preprocess.py                 # 数据预处理
├── build/                            # 编译产物
└── run_<index>.py                    # 保留原 Python 实现为兼容性 wrapper
```

### 1.3 从 Curator 复制的文件清单（每个索引自行复制）

| 文件 | 来源 | 复制方式 | 修改 |
|------|------|------|:---:|
| `cnpy.h` | `Curator/src/cnpy.h` | 直接复制 | 无 |
| `cnpy.cpp` | `Curator/src/cnpy.cpp` | 直接复制 | 无 |
| `distance.h` | `Curator/src/distance.h`（精简） | 提取复制 | 删除 `compute_batch_dists`（依赖 Curator 特有类型），删除 `#include "common.h"` 和 `namespace curator`，仅保留 `l2_sqr` + `l2_sqr_4way` |
| `kmeans.h` | `Curator/src/kmeans.h` | 直接复制 | 仅 DiskIVF 需要。需同时复制 `kmeans.cpp` 并修改其 `#include "common.h"` → 内联异常宏 |
| `kmeans.cpp` | `Curator/src/kmeans.cpp` | 复制+微调 | 仅 DiskIVF 需要。将 `CURATOR_THROW_*` 宏替换为本地定义的等价宏 |

### 1.4 每个索引自行新增的文件

| 文件 | 说明 |
|------|------|
| `common_base.h` | 最小公共基础设施（PREFETCH 宏 + 异常宏 + 类型别名），~25 行，内联在 `src/` 中或合并到 `config.h` |
| `predicate.h` | 前缀记法（Polish Notation）谓词求值器，~60 行，header-only |

### 1.5 与 Curator 的关系

**Curator 源码零变更。** 各索引从 Curator 复制的文件成为独立副本，后续可独立演进。

### 1.6 统一 CLI 接口

所有 C++ 可执行文件使用一致的 CLI 参数风格，与 Curator 保持一致：

```
./<index_name> bench \
    --train_vecs     <path>    # 必需：[N, d] float32 训练向量
    --train_access   <path>    # 可选：[M, 2] int32 (vid, tid) 访问对
    --queries        <path>    # 必需：[Q, d] float32 查询向量
    --query_labels   <path>    # 可选：[Q] int32 查询标签（-1=无过滤）
    --config         <path>    # 可选：JSON 配置文件
    --k              <int>     # 默认 10
    --output         <path>    # 默认 results.json
    --profile                  # 可选 flag：打印最后一个查询的时间分解
    --batch-query              # 可选 flag：启用查询间 OpenMP 并行
```

### 1.7 统一 JSON 输出格式

所有 C++ 索引输出统一格式，便于 Python 端解析：

```json
{
  "index": "PreFiltering",
  "dataset": "arxiv_small",
  "config": { "d": 384, "n_labels": 100 },
  "build_time_s": 0.5,
  "memory_bytes": 150000000,
  "disk_bytes": 0,
  "queries": [
    {
      "idx": 0,
      "tenant_id": 48,
      "labels": [37965, 34908, 47118, 28570, 14872, 18150, 2343, 3369, 17616, 5198],
      "distances": [12.34, 15.67, 18.90, 21.12, 23.45, 25.67, 27.89, 30.12, 32.34, 34.56]
    }
  ]
}
```

---

## 2. 任务一：Pre-Filtering 索引 C++ 化

### 2.1 当前 Python 实现分析

**文件**: [Pre-Filtering/run_prefiltering.py](Pre-Filtering/run_prefiltering.py)  
**代码量**: ~330 行（含 runner 逻辑）

**核心类**: `PreFilteringIndex`
- **Build**: 存储 `train_vecs` (np.ndarray) + `train_mds` (list[list[int]])，构建 `label_to_indices` 倒排索引（dict[int, np.ndarray]）
- **Query (single-label)**: `query(x, k, tenant_id)` → 查倒排索引获取候选集 → 候选集向量与查询向量做 L2 距离 → argpartition 取 top-k
- **Query (complex-predicate)**: `query_with_complex_predicate(x, k, predicate)` → 调用 `compute_qualified_indices()` 获取候选集 → L2 距离 → top-k
- **关键依赖**: [2_Utils/predicate.py](2_Utils/predicate.py)（倒排索引构建 + 前缀记法布尔表达式求值）

**性能特征**:
- 构建：极快（仅构建倒排索引，无聚类/训练）
- 查询：候选集小时极快，候选集大时需暴力计算全部候选向量 L2
- 内存：需全量训练向量 + 倒排索引（与训练数据总量等大）

### 2.2 C++ 实现设计

#### 2.2.1 目录结构

```
Pre-Filtering/
├── CMakeLists.txt
├── src/
│   ├── main.cpp                      # CLI 入口
│   ├── config.h                      # 配置结构体
│   ├── prefiltering_index.h          # PreFilteringIndex 核心类声明
│   ├── prefiltering_index.cpp        # PreFilteringIndex 核心类实现
│   ├── predicate.h                   # ★ 新增：前缀记法谓词求值器（header-only）
│   ├── distance.h                    # 从 Curator 精简复制：l2_sqr + l2_sqr_4way
│   ├── cnpy.h                        # 从 Curator 复制
│   └── cnpy.cpp                      # 从 Curator 复制
├── python/
│   ├── run_experiment.py             # 实验编排脚本
│   └── preprocess.py                 # 数据预处理
├── build/                            # 编译产物
└── run_prefiltering.py               # 保留原 Python 实现作为兼容性入口
```

#### 2.2.2 模块设计

**`config.h`** — 配置结构体 + 公共基础设施：
```cpp
// Pre-Filtering/src/config.h
#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

// ── 公共基础设施（内联，不设 shared/） ──
#define PREFETCH(ptr) __builtin_prefetch((ptr), 0, 3)

#define THROW_MSG(MSG) throw std::runtime_error(MSG)
#define THROW_IF_NOT(COND, MSG) do { if (!(COND)) THROW_MSG(MSG); } while (0)
#define THROW_FMT(MSG, ...) do { \
    char _buf[1024]; \
    std::snprintf(_buf, sizeof(_buf), MSG, ##__VA_ARGS__); \
    throw std::runtime_error(std::string(_buf)); \
} while (0)
#define THROW_IF_NOT_FMT(COND, MSG, ...) \
    do { if (!(COND)) THROW_FMT(MSG, ##__VA_ARGS__); } while (0)

// ── 索引配置 ──
// 注: d 由 main.cpp 从 .npy shape 自动检测并覆盖；
//      n_labels 由 build() 从 access_pairs 自动检测（max tid + 1）。
//      配置文件中的值作为可选覆盖（通常不需要）。
struct PreFilteringConfig {
    size_t d = 0;              // 0 = 自动从数据检测
    size_t k = 10;
    size_t n_labels = 0;       // 0 = 自动从 access_pairs 检测
    size_t num_warmup = 20;
    bool batch_query = false;
};
```

**`prefiltering_index.h`** — 核心索引类：
```cpp
// Pre-Filtering/src/prefiltering_index.h
#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "config.h"

class PreFilteringIndex {
public:
    explicit PreFilteringIndex(const PreFilteringConfig& cfg);

    // Build index from raw data.
    // vectors: [n × d] float32, row-major contiguous
    // access_pairs: [n_pairs × 2] int32, each row = (vid, tid)
    // d is taken from cfg.d (auto-detected from .npy shape by main.cpp).
    // n_labels is auto-detected from max(tid) in access_pairs.
    void build(size_t n, const float* vectors,
               const int32_t* access_pairs, size_t n_pairs);

    // Single-label query: top-k among vectors that have `tenant_id` label
    void search(const float* query, size_t k, int32_t tenant_id,
                float* distances, int32_t* labels) const;

    // Complex-predicate query (Polish notation: "AND 0 NOT 1")
    // Iterates ALL vectors, evaluates predicate per-vector via vid_to_labels_.
    void search_with_predicate(const float* query, size_t k,
                               const std::string& predicate,
                               float* distances, int32_t* labels) const;

    // Unfiltered brute-force search (all vectors are candidates)
    void search_unfiltered(const float* query, size_t k,
                           float* distances, int32_t* labels) const;

    size_t memory_bytes() const;
    size_t ntotal() const { return ntotal_; }
    size_t dim() const { return d_; }
    size_t n_labels() const { return n_labels_; }

private:
    PreFilteringConfig cfg_;
    size_t d_ = 0;
    size_t ntotal_ = 0;
    size_t n_labels_ = 0;

    // ★ 双向索引：
    std::vector<float> train_vecs_;                       // [N × d] 全量训练向量
    std::vector<std::vector<int32_t>> label_to_vids_;     // 倒排: label → vid列表 (单标签查询)
    std::vector<std::vector<int32_t>> vid_to_labels_;     // 正向: vid → label列表 (谓词求值)

    // 内部方法：在候选集中暴力 L2 搜索
    void search_candidates(const float* query, size_t k,
                           const std::vector<int32_t>& candidates,
                           float* distances, int32_t* labels) const;
};
```

**关键数据结构**:
- `train_vecs_`: `std::vector<float>` — 全量训练向量，连续存储 [N × d]
- `label_to_vids_`: `std::vector<std::vector<int32_t>>` — 倒排索引，`label_to_vids_[label]` = 拥有该 label 的所有 vid 列表。供 `search()` 使用，O(1) 查候选集。
- `vid_to_labels_`: `std::vector<std::vector<int32_t>>` — 正向索引，`vid_to_labels_[vid]` = 该向量的所有 label 列表（已排序）。供 `search_with_predicate()` 使用，逐向量求值谓词表达式。

**build() 流程**:
```
build(n, vectors, access_pairs, n_pairs):
  1. 复制 vectors → train_vecs_（连续存储 [N × d]）
  2. 扫描 access_pairs，找到 max(tid) → n_labels_
  3. 分配 label_to_vids_.resize(n_labels_) 和 vid_to_labels_.resize(n)
  4. 遍历 access_pairs:
     for each (vid, tid):
       label_to_vids_[tid].push_back(vid)
       vid_to_labels_[vid].push_back(tid)
  5. 对每个内部向量排序（保证 binary_search 可用）:
     for each vid: std::sort(vid_to_labels_[vid])
     for each label: std::sort(label_to_vids_[label])
```

**查询流程**:
```
search(x, k, tenant_id):
  1. if tenant_id 越界: return all [-1, ...]
  2. candidates = label_to_vids_[tenant_id]
  3. if candidates empty: return all [-1, ...]
  4. for each vid in candidates:
       dist = l2_sqr(x, train_vecs_[vid * d_], d_)
  5. partial_sort top-k → output
  6. 不足 k 个填充 -1

search_with_predicate(x, k, "AND 0 NOT 1"):
  1. tokens = predicate::tokenize(formula)
  2. for vid in 0..ntotal_-1:
       if predicate::evaluate(tokens, vid_to_labels_[vid]):
         candidates.push_back(vid)
  3. search_candidates(x, k, candidates) → output

search_unfiltered(x, k):
  1. 所有 vid 均为候选 → 全量暴力 L2 → top-k
```

#### 2.2.3 `predicate.h` 设计（header-only，内置于 Pre-Filtering/src/）

```cpp
// Pre-Filtering/src/predicate.h
#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace predicate {

// Tokenize a formula string (space-delimited)
inline std::vector<std::string> tokenize(const std::string& formula) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : formula) {
        if (c == ' ' || c == '\t') {
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
        } else {
            current += c;
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

// Evaluate prefix-notation formula against a label set (as [begin, end) iterators).
// Uses right-to-left stack evaluation — matches Python predicate.py algorithm.
//
// labels must be sorted for binary_search, or use std::find for unsorted.
inline bool evaluate(const std::vector<std::string>& tokens,
                     const int32_t* labels_begin,
                     const int32_t* labels_end) {
    std::vector<bool> stack;
    for (auto it = tokens.rbegin(); it != tokens.rend(); ++it) {
        const std::string& token = *it;
        if (token == "AND") {
            if (stack.size() < 2) return false;
            bool a = stack.back(); stack.pop_back();
            bool b = stack.back(); stack.pop_back();
            stack.push_back(a && b);
        } else if (token == "OR") {
            if (stack.size() < 2) return false;
            bool a = stack.back(); stack.pop_back();
            bool b = stack.back(); stack.pop_back();
            stack.push_back(a || b);
        } else if (token == "NOT") {
            if (stack.empty()) return false;
            bool a = stack.back(); stack.pop_back();
            stack.push_back(!a);
        } else {
            int32_t label_id = std::stoi(token);
            // Linear search — label lists are short (< 50 labels per vector)
            stack.push_back(std::find(labels_begin, labels_end, label_id)
                            != labels_end);
        }
    }
    return stack.size() == 1 && stack.back();
}

// Convenience: tokenize + evaluate
inline bool evaluate(const std::string& formula,
                     const int32_t* labels_begin,
                     const int32_t* labels_end) {
    return evaluate(tokenize(formula), labels_begin, labels_end);
}

} // namespace predicate
```

**算法正确性验证**（与 Python `predicate.py` 对比）：

| 输入 | Python (`reversed` 遍历) | C++ (`rbegin→rend` 遍历) | 结果 |
|------|------|------|:---:|
| `"AND 0 1"`, labels=[0,1] | push(1∈L)=T, push(0∈L)=T, AND TT=T | push(1∈L)=T, push(0∈L)=T, AND TT=T | ✅ 一致 |
| `"AND 0 NOT 1"`, labels=[0] | push(1∈L)=F, NOT F=T, push(0∈L)=T, AND TT=T | push(1∈L)=F, NOT F=T, push(0∈L)=T, AND TT=T | ✅ 一致 |
| `"OR 0 1"`, labels=[2] | push(1∈L)=F, push(0∈L)=F, OR FF=F | push(1∈L)=F, push(0∈L)=F, OR FF=F | ✅ 一致 |
| `"NOT 0"`, labels=[] | push(0∈L)=F, NOT F=T | push(0∈L)=F, NOT F=T | ✅ 一致 |

#### 2.2.4 `distance.h` 设计（从 Curator 精简复制）

从 `Curator/src/distance.h` 中提取 `l2_sqr()` 和 `l2_sqr_4way()` 两个函数，删除依赖 Curator 特有类型的 `compute_batch_dists`，移除 `#include "common.h"` 和 `namespace curator` 包装。

```cpp
// Pre-Filtering/src/distance.h
#pragma once
#include <cmath>
#include <cstddef>

namespace distance {

inline float l2_sqr(const float* x, const float* y, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; i++) {
        float diff = x[i] - y[i];
        sum += diff * diff;
    }
    return sum;
}

inline void l2_sqr_4way(
        const float* x,
        const float* y0, const float* y1, const float* y2, const float* y3,
        size_t d,
        float& d0, float& d1, float& d2, float& d3) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    size_t i = 0;
    for (; i + 3 < d; i += 4) {
        float dx0 = x[i] - y0[i];     s0 += dx0 * dx0;
        float dx1 = x[i] - y1[i];     s1 += dx1 * dx1;
        float dx2 = x[i] - y2[i];     s2 += dx2 * dx2;
        float dx3 = x[i] - y3[i];     s3 += dx3 * dx3;
        dx0 = x[i+1] - y0[i+1];       s0 += dx0 * dx0;
        dx1 = x[i+1] - y1[i+1];       s1 += dx1 * dx1;
        dx2 = x[i+1] - y2[i+1];       s2 += dx2 * dx2;
        dx3 = x[i+1] - y3[i+1];       s3 += dx3 * dx3;
        dx0 = x[i+2] - y0[i+2];       s0 += dx0 * dx0;
        dx1 = x[i+2] - y1[i+2];       s1 += dx1 * dx1;
        dx2 = x[i+2] - y2[i+2];       s2 += dx2 * dx2;
        dx3 = x[i+2] - y3[i+2];       s3 += dx3 * dx3;
        dx0 = x[i+3] - y0[i+3];       s0 += dx0 * dx0;
        dx1 = x[i+3] - y1[i+3];       s1 += dx1 * dx1;
        dx2 = x[i+3] - y2[i+3];       s2 += dx2 * dx2;
        dx3 = x[i+3] - y3[i+3];       s3 += dx3 * dx3;
    }
    for (; i < d; i++) {
        float dx0 = x[i] - y0[i];     s0 += dx0 * dx0;
        float dx1 = x[i] - y1[i];     s1 += dx1 * dx1;
        float dx2 = x[i] - y2[i];     s2 += dx2 * dx2;
        float dx3 = x[i] - y3[i];     s3 += dx3 * dx3;
    }
    d0 = s0; d1 = s1; d2 = s2; d3 = s3;
}

} // namespace distance
```

#### 2.2.5 CLI 设计

```bash
./prefiltering bench \
    --train_vecs     $DATA/train_vecs.npy \
    --train_access   $DATA/train_access.npy \
    --queries        $DATA/query_vecs.npy \
    --query_labels   $DATA/query_labels.npy \
    --config         /tmp/prefilter_config.json \
    --k 10 \
    --output         results.json \
    --profile
```

#### 2.2.6 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(prefiltering LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

set(SOURCES
    src/cnpy.cpp
    src/prefiltering_index.cpp
    src/main.cpp
)

add_executable(prefiltering ${SOURCES})
target_include_directories(prefiltering PRIVATE src)
target_link_libraries(prefiltering PRIVATE OpenMP::OpenMP_CXX)
```

#### 2.2.7 Python 实验编排脚本

```python
# Pre-Filtering/python/run_experiment.py
# 职责：
#   1. 预处理（.pkl → train_access.npy）
#   2. 生成 JSON 配置文件
#   3. subprocess 调用 C++ 二进制（./build/prefiltering bench ...）
#   4. 读取 C++ 输出的 results.json，计算 Recall@k
#   5. 输出汇总结果到 4_Results/Pre-Filtering/
```

### 2.3 实现步骤

| 步骤 | 内容 | 预估工作量 |
|:---:|------|:---:|
| 2.1 | 创建目录结构（src/, python/, build/） | 小 |
| 2.2 | 从 Curator 复制 cnpy.h/.cpp（零修改） | 小 |
| 2.3 | 从 Curator 精简复制 distance.h（仅 l2_sqr + l2_sqr_4way） | 小 |
| 2.4 | 编写 predicate.h（前缀记法谓词求值器，header-only） | 小 |
| 2.5 | 编写 config.h（公共基础设施 + 索引配置） | 小 |
| 2.6 | 实现 `PreFilteringIndex::build()`（倒排索引构建） | 中 |
| 2.7 | 实现 `PreFilteringIndex::search()`（单标签查询 + L2 暴力搜索） | 中 |
| 2.8 | 实现 `search_with_predicate()`（复杂谓词查询） | 中 |
| 2.9 | 编写 `main.cpp`（CLI + JSON 输出 + 计时） | 中 |
| 2.10 | 编写 CMakeLists.txt | 小 |
| 2.11 | 编写 Python 编排脚本 `run_experiment.py` | 小 |
| 2.12 | WSL 编译验证 + 与 Python 原版结果对比（Recall@k 差异 < 0.1%） | 中 |

---

## 3. 任务二：DiskIVF-PostFiltering 索引 C++ 化

### 3.1 当前 Python 实现分析

**文件**: [DiskIVF-PostFiltering/run_diskivf.py](DiskIVF-PostFiltering/run_diskivf.py)  
**代码量**: ~475 行（含 runner 逻辑）

**核心类**: `DiskIVFIndex`
- **Build**: FAISS K-means → 分配向量到簇 → 每个簇保存为磁盘文件（`c{cid}_vecs.npy`, `c{cid}_indices.npy`, `c{cid}_mds.pkl`）
- **Query (single-label)**: 计算查询向量到所有质心的距离 → 选最近 nprobe 个簇 → 依次加载簇文件 → 过滤标签 → L2 距离 → top-k
- **Query (complex-predicate)**: 同上，但过滤条件为布尔表达式
- **关键依赖**: FAISS K-means（`faiss.Kmeans`），numpy 磁盘 I/O，pickle

**性能特征**:
- 构建：K-means 聚类（O(N·nlist·d·niter)）+ 磁盘写入（多文件、.npy 格式）
- 查询：质心距离计算（O(nlist·d)）+ 磁盘读取 nprobe 个簇文件 + 过滤 + L2
- 内存：仅需质心 + 簇大小统计（极小），向量均在磁盘

### 3.2 C++ 实现设计

#### 3.2.1 目录结构

```
DiskIVF-PostFiltering/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── config.h                      # 配置 + 公共基础设施
│   ├── diskivf_index.h               # DiskIVFIndex 核心类声明
│   ├── diskivf_index.cpp             # DiskIVFIndex 核心类实现
│   ├── kmeans.h                      # 从 Curator 复制
│   ├── kmeans.cpp                    # 从 Curator 复制 + 微调（去除 Curator 宏依赖）
│   ├── predicate.h                   # ★ 新增：前缀记法谓词求值器（与 Pre-Filtering 相同）
│   ├── distance.h                    # 从 Curator 精简复制
│   ├── cnpy.h                        # 从 Curator 复制
│   └── cnpy.cpp                      # 从 Curator 复制
├── python/
│   ├── run_experiment.py
│   └── preprocess.py
├── build/
└── run_diskivf.py                    # 保留原 Python 实现
```

#### 3.2.2 模块设计

**`config.h`**：
```cpp
// DiskIVF-PostFiltering/src/config.h
#pragma once
// ... 与 Pre-Filtering 相同的公共基础设施宏（PREFETCH, THROW_*）...

struct DiskIVFConfig {
    size_t d = 128;
    size_t nlist = 64;         // 聚类簇数（可 > 64，不受 Curator MAX_BRANCH_FACTOR 限制）
    size_t nprobe = 16;        // 探测簇数
    size_t clus_niter = 20;    // K-means 迭代次数
    size_t k = 10;
    bool batch_query = false;
    std::string disk_dir;      // 簇文件存储目录
};
```

**`diskivf_index.h`** — 核心索引类：
```cpp
// DiskIVF-PostFiltering/src/diskivf_index.h
#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "config.h"

class DiskIVFIndex {
public:
    explicit DiskIVFIndex(const DiskIVFConfig& cfg);

    // vectors: [N × d] float32, access_pairs: [M × 2] int32 (vid, tid)
    void build(size_t n, const float* vectors,
               const int32_t* access_pairs, size_t n_pairs);

    void search(const float* query, size_t k, int32_t tenant_id,
                float* distances, int32_t* labels) const;

    void search_with_predicate(const float* query, size_t k,
                               const std::string& predicate,
                               float* distances, int32_t* labels) const;

    void search_unfiltered(const float* query, size_t k,
                           float* distances, int32_t* labels) const;

    size_t memory_bytes() const;
    size_t disk_bytes() const;
    size_t ntotal() const { return ntotal_; }

private:
    DiskIVFConfig cfg_;
    size_t d_ = 0;
    size_t ntotal_ = 0;

    std::vector<float> centroids_;         // [nlist × d]
    std::vector<int32_t> cluster_sizes_;   // [nlist]
    std::string disk_dir_;

    // 查询辅助
    std::vector<int32_t> get_nearest_clusters(const float* query) const;

    // 加载单个簇：返回 (vecs, vids, mds)
    struct ClusterData {
        std::vector<float> vecs;
        std::vector<int32_t> vids;
        std::vector<std::vector<int32_t>> mds;  // 每条向量的标签列表
    };
    ClusterData load_cluster(int32_t cid) const;

    // 筛选 + L2 距离计算
    void filter_and_compute_dists(const float* query, const ClusterData& cluster,
                                  int32_t tenant_id, size_t k,
                                  std::vector<std::pair<float, int32_t>>& heap) const;
};
```

#### 3.2.3 磁盘文件格式

**采用原始二进制格式**（`.bin`）替代 Python 版的多文件 `.npy` + `.pkl`：

```
<disk_dir>/
├── centroids.bin       # [nlist × d] float32
├── cluster_sizes.bin   # [nlist] int32
├── c0/
│   ├── vecs.bin        # [n0 × d] float32（原始向量）
│   ├── vids.bin        # [n0] int32（向量全局 ID）
│   └── mds.bin         # flat int32: [n_labels_i, label_1, label_2, ...] 重复
├── c1/
│   └── ...
└── ...
```

> **设计选择**: 二进制格式比 .npy + .pkl 组合更快（无 Python dict header 解析、无 pickle 反序列化）。簇文件数量多（nlist 可达 256），每个文件小，二进制直读（fread）最快。
>
> **备选方案**: 继续使用 .npy 格式以保持兼容性和可调试性。首次实现推荐使用 .npy（cnpy 已可用），后续可优化为 .bin。

#### 3.2.4 查询流程

```
search(x, k, tenant_id):
  1. 计算查询向量到所有质心的 L2 距离 → dists[nlist]
  2. partial_sort 选取距离最小的 nprobe 个簇 ID
  3. for each 选出簇:
     a. 从磁盘加载 vecs + vids + mds
     b. 过滤: 保留 mds[i] 包含 tenant_id 的向量
     c. 计算过滤后向量与查询向量的 L2 距离
     d. 合并到全局 top-k 堆（std::priority_queue）
  4. 返回 top-k (vid, dist)
```

**关键优化点**:
- 使用 min-heap 或固定大小的排序数组维护全局 top-k
- 4-way unrolled L2 distance（复用 `distance.h` 的 `l2_sqr_4way`）
- 可选的批量簇文件预取（OpenMP 并行加载多个簇）

#### 3.2.5 K-means 实现

从 Curator 复制 `kmeans.h` + `kmeans.cpp`，进行以下微调：
1. 将 `#include "common.h"` 替换为 `#include "config.h"`（使用本地的 `THROW_MSG` 等宏）
2. 移除 `namespace curator`，改用 `namespace diskivf` 或匿名命名空间
3. K-means 算法本身无 64 簇上限（`n_clusters` 是自由参数），DiskIVF 的 nlist=256 可直接使用

#### 3.2.6 CLI 设计

```bash
./diskivf bench \
    --train_vecs     $DATA/train_vecs.npy \
    --train_access   $DATA/train_access.npy \
    --queries        $DATA/query_vecs.npy \
    --query_labels   $DATA/query_labels.npy \
    --config         /tmp/diskivf_config.json \
    --k 10 --output results.json --profile
```

#### 3.2.7 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(diskivf LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

set(SOURCES
    src/cnpy.cpp
    src/kmeans.cpp
    src/diskivf_index.cpp
    src/main.cpp
)

add_executable(diskivf ${SOURCES})
target_include_directories(diskivf PRIVATE src)
target_link_libraries(diskivf PRIVATE OpenMP::OpenMP_CXX)
```

#### 3.2.8 复杂谓词处理

与 Pre-Filtering 相同：C++ 端使用 `predicate.h` 实现波兰表达式解析器，在加载的簇数据上逐向量求值。

### 3.3 实现步骤

| 步骤 | 内容 | 预估工作量 |
|:---:|------|:---:|
| 3.1 | 创建目录结构 | 小 |
| 3.2 | 从 Curator 复制 cnpy.h/.cpp（零修改） | 小 |
| 3.3 | 从 Curator 精简复制 distance.h | 小 |
| 3.4 | 编写 predicate.h（与 Pre-Filtering 相同，直接复制） | 小 |
| 3.5 | 编写 config.h（公共基础设施 + 索引配置） | 小 |
| 3.6 | 从 Curator 复制 kmeans.h/.cpp 并微调（去除 Curator 宏/命名空间依赖） | 中 |
| 3.7 | 实现 `DiskIVFIndex::build()`（K-means + 簇文件写入） | 大 |
| 3.8 | 实现簇文件 I/O（写入/读取 vecs, vids, mds，使用 cnpy 或二进制） | 中 |
| 3.9 | 实现 `DiskIVFIndex::search()`（单标签 + 磁盘读取 + 过滤 + L2） | 中 |
| 3.10 | 实现 `search_with_predicate()`（复杂谓词） | 中 |
| 3.11 | 编写 `main.cpp`（CLI + JSON 输出） | 中 |
| 3.12 | 编写 CMakeLists.txt | 小 |
| 3.13 | 编写 Python 编排脚本 | 小 |
| 3.14 | WSL 编译验证 + 与 Python 原版结果对比 | 中 |

---

## 4. 任务三：SPANN-PostFiltering 索引 C++ 化（基于 SPTAG-main）

### 4.1 当前实现分析

**原 Python 实现** ([SPANN-PostFiltering/run_spann.py](SPANN-PostFiltering/run_spann.py)):
- 代码结构与 DiskIVF 几乎完全相同，仅默认参数不同（nlist=256 vs 64, nprobe=64 vs 16）
- 本质上是"细粒度 DiskIVF"：更细的聚类 → 每簇更少的向量 → 加载更快

**SPTAG-main 开源代码** ([SPANN-PostFiltering/SPTAG-main/](SPANN-PostFiltering/SPTAG-main/)):
- Microsoft 开源的 SPANN（Space-Partitioned ANN）实现
- 两阶段架构：
  - **Head Index**（内存）：BKT/KDT 树索引质心
  - **SSD Index**（磁盘）：posting lists（每个质心对应一组向量 ID 和距离）
- 核心类：`SPTAG::SPANN::Index<T>`（定义在 `AnnService/inc/Core/SPANN/Index.h`）
  - `BuildIndex(p_data, p_vectorNum, p_dimension)`: 构建 head index + 生成 posting lists 到磁盘
  - `SearchIndex(p_query, p_searchDeleted)`: head index 搜索 → 加载对应 posting lists → 精确距离重排
  - `SearchIndexWithFilter(p_query, filterFunc, maxCheck, p_searchDeleted)`: **已有 Filter 支持**
  - `GetIterator(p_target, p_searchDeleted, p_filterFunc, p_maxCheck)`: 迭代式搜索
- 依赖：Boost (system, thread, serialization, filesystem), OpenMP, zstd, TBB
- **不直接支持按 label 过滤**，但提供了 `std::function<bool(const ByteArray&)>` 过滤器接口

### 4.2 PostFiltering 集成策略

#### 核心策略：Overfetch + PostFilter

SPTAG 的 `SearchIndex` 接口返回结果时不支持自定义 filter。但 `GetIterator` 和 `SearchIndexWithFilter` 支持 `std::function<bool(const ByteArray&)>` 过滤器。

**方案 A（推荐）**：使用 SPTAG 的 `SearchIndex` 检索足够多的结果（overfetch），然后在 C++ wrapper 层进行 PostFilter。这是最简单且侵入性最小的方案。

```
SPTAG SearchIndex (k' = k × overfetch_factor)
    │
    ▼  返回 top-k' 个未过滤结果 (VID + distance)
    │
PostFilter (遍历 k' 个结果，筛选符合标签条件的)
    │
    ▼  返回 top-k 过滤结果
```

**方案 B**：使用 SPTAG 的 `SearchIndexWithFilter`，在 metadata 中编码 label 信息，利用 SPTAG 内置的 filter callback。这需要将 label 数据存入 SPTAG metadata 系统，较复杂。

**推荐采用方案 A**，理由：
1. 不侵入 SPTAG 源码
2. 标签元数据独立管理（wrapper 层的 `vid_to_labels_`）
3. 实现简单，逻辑清晰

#### Overfetch Factor 选择逻辑：

- 根据标签选择性自适应调整
- `overfetch_factor = max(10, min(100, int(1.0 / selectivity)))`
- 即：选择性 1% → 检索 k×100 个；选择性 10% → 检索 k×10 个
- 默认固定值 `overfetch_factor = 50` 也可满足大多数场景

#### 4.2.1 目录结构

```
SPANN-PostFiltering/
├── SPTAG-main/                     # ★ 原始 SPTAG 开源代码（只读，作为编译依赖）
├── CMakeLists.txt                  # ★ 新增：独立 CMake（编译 spann_pf 可执行文件）
├── src/                            # ★ 新增：PostFiltering 封装层
│   ├── main.cpp                    #   CLI 入口（bench 模式）
│   ├── config.h                    #   配置结构体 + 公共基础设施
│   ├── spann_pf_index.h            #   PostFiltering 封装类声明
│   ├── spann_pf_index.cpp          #   PostFiltering 封装类实现
│   ├── predicate.h                 #   前缀记法谓词求值器（与其他索引相同）
│   ├── distance.h                  #   L2 距离函数（从 Curator 精简复制）
│   ├── cnpy.h                      #   从 Curator 复制
│   └── cnpy.cpp                    #   从 Curator 复制
├── python/
│   ├── run_experiment.py           #   实验编排
│   └── preprocess.py               #   预处理（.pkl → 元数据二进制文件）
├── legacy/
│   ├── run_spann.py                #   保留原 Python 实现为兼容性入口
│   └── run_spann_sweep.py
├── build/                          #   编译产物
└── run_spann.py                    #   保留原 Python 实现（重定向到 legacy/）
```

#### 4.2.2 模块设计

**`config.h`**：
```cpp
// SPANN-PostFiltering/src/config.h
#pragma once
// ... 公共基础设施宏（同 Pre-Filtering）...
#include <string>
#include <cstdint>

struct SPANNConfig {
    // SPANN 原参数
    size_t d = 128;
    std::string dist_method = "L2";    // L2 / Cosine
    size_t num_threads = 32;           // SPTAG 内部搜索线程数（设为 1 避免与 batch_query 嵌套并行）
    size_t max_check = 8192;           // head index 最大检查数
    size_t hash_exp = 8;               // BKTree hash table exponent

    // PostFiltering 参数
    size_t k = 10;
    size_t num_warmup = 20;
    size_t overfetch_factor = 50;      // k × factor = k' 检索量（固定模式）
    bool overfetch_adaptive = true;    // 是否根据选择性自适应调整（推荐）
    bool batch_query = false;

    // 数据路径
    std::string index_dir;             // SPTAG 索引存储目录
    std::string metadata_path;         // 标签元数据文件路径（预处理生成）
};
```

> **线程安全注意事项**：当 `batch_query = true` 时，main.cpp 使用 OpenMP 并行化查询，此时必须将 `num_threads` 设为 1（通过 `SetParameter("NumberOfThreads", "1", "SSD")`），避免 SPTAG 内部线程池与 OpenMP 嵌套并行导致性能退化。单查询模式下（`batch_query = false`），`num_threads` 可设为 > 1 以加速 SPTAG 内部搜索。

**`spann_pf_index.h`** — PostFiltering 封装层：
```cpp
// SPANN-PostFiltering/src/spann_pf_index.h
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "config.h"

// Forward declare SPTAG types (avoid pulling in full SPTAG headers in public header)
namespace SPTAG {
    class VectorIndex;
}

class SPANNPostFilterIndex {
public:
    explicit SPANNPostFilterIndex(const SPANNConfig& cfg);
    ~SPANNPostFilterIndex();

    // ── Build ──
    // vectors: [N × d] float32
    // access_pairs: [M × 2] int32 (vid, tid)
    void build(size_t n, const float* vectors,
               const int32_t* access_pairs, size_t n_pairs);

    // ── Search ──
    void search(const float* query, size_t k, int32_t tenant_id,
                float* distances, int32_t* labels) const;

    void search_with_predicate(const float* query, size_t k,
                               const std::string& predicate,
                               float* distances, int32_t* labels) const;

    void search_unfiltered(const float* query, size_t k,
                           float* distances, int32_t* labels) const;

    // ── Info ──
    size_t memory_bytes() const;
    size_t disk_bytes() const;
    size_t ntotal() const;

private:
    SPANNConfig cfg_;
    size_t ntotal_ = 0;
    size_t d_ = 0;

    // SPTAG 索引（模板实例化 uint8_t 或 float，通过 void* 擦除类型）
    std::shared_ptr<SPTAG::VectorIndex> spann_index_;

    // 元数据（标签信息，用于 PostFiltering）
    // vid_to_labels_[vid] = 该向量的标签列表（已排序）
    std::vector<std::vector<int32_t>> vid_to_labels_;

    // 标签频率统计（用于自适应 overfetch_factor 计算）
    std::vector<size_t> label_counts_;

    // 内部方法
    size_t compute_overfetch_k(size_t k, int32_t tenant_id) const;

    // PostFilter: 遍历 SPTAG 结果，筛选符合 filter_fn 的向量
    void postfilter_results(
        const float* query,
        size_t k,
        const std::function<bool(int32_t vid)>& filter_fn,
        float* distances,
        int32_t* labels
    ) const;
};
```

#### 4.2.3 PostFiltering 查询流程

```
search(x, k, tenant_id):
  1. 计算 overfetch_k:
     - adaptive: 若 tenant_id 有效且 count>0:
         selectivity = count(tenant_id) / N
         overfetch_k = min(N, k × clamp(k / selectivity, 10, 200))
     - fixed: overfetch_k = k × overfetch_factor
     - 安全上限: overfetch_k = min(overfetch_k, N)  // 不超过总向量数
     - 零值保护: 若 count(tenant_id)==0, 回退到 fixed 模式
  2. 构造 SPTAG::QueryResult(query, overfetch_k, false)
  3. 调用 spann_index_->SearchIndex(query_result, false)
     → 返回 top-overfetch_k 结果 (VID, Dist)
  4. PostFilter（收集→排序→取top-k，不依赖 SPTAG 内部排序）:
     a. 遍历 overfetch_k 个结果
     b. 对每个结果检查 vid_to_labels_[VID] 是否包含 tenant_id
     c. 将合格的 (Dist, VID) 收集到临时数组 qualified
     d. 对 qualified 按 Dist 升序排序
     e. 取前 k 个作为最终结果
     f. 不足 k 个填充 -1

search_with_predicate(x, k, "AND 0 NOT 1"):
  1. overfetch_k = k × overfetch_factor（复杂谓词缺乏统计信息，使用固定因子）
  2. 调用 spann_index_->SearchIndex(query_result, false)
  3. PostFilter: 同上collect→sort→topk流程
     - 用 predicate::evaluate() 对每个结果判断
     - 检查 vid_to_labels_[VID] 是否满足谓词表达式
  4. 返回前 k 个

search_unfiltered(x, k):
  1. 直接调用 spann_index_->SearchIndex(query_result, false)，k' = k
  2. 无需 PostFilter，直接返回 SPTAG 结果
  3. 这是纯 SPTAG 搜索，性能最高
```

#### 4.2.4 SPTAG 搜索接口集成细节

SPTAG 的关键 API（来自 `VectorIndex.h` 和 `SPANN/Index.h`）：

```cpp
// SearchIndex — 基础检索
virtual ErrorCode SearchIndex(QueryResult& p_results, bool p_searchDeleted = false) const = 0;

// SearchIndexWithFilter — 带 filter 的检索（方案 B 使用）
virtual ErrorCode SearchIndexWithFilter(QueryResult& p_query,
    std::function<bool(const ByteArray&)> filterFunc,
    int maxCheck = 0, bool p_searchDeleted = false) const = 0;

// GetIterator — 迭代式检索
virtual std::shared_ptr<ResultIterator> GetIterator(const void* p_target,
    bool p_searchDeleted = false,
    std::function<bool(const ByteArray&)> p_filterFunc = nullptr,
    int p_maxCheck = 0) const = 0;
```

**方案 A**（推荐）的包装调用实现：

```cpp
void SPANNPostFilterIndex::search(const float* query, size_t k, int32_t tenant_id,
                                   float* distances, int32_t* labels) const {
    // Step 1: 计算 overfetch_k（含除零保护）
    size_t overfetch_k = compute_overfetch_k(k, tenant_id);

    // Step 2: 调用 SPTAG 检索
    SPTAG::QueryResult query_result(query, (int)overfetch_k, false);
    spann_index_->SearchIndex(query_result, false);

    // Step 3: PostFilter — 收集所有合格结果（不依赖 SPTAG 内部排序）
    std::vector<std::pair<float, int32_t>> qualified;
    for (int i = 0; i < query_result.GetResultNum(); i++) {
        auto* res = query_result.GetResult(i);
        if (res->VID < 0) break;  // 无更多结果

        int32_t vid = static_cast<int32_t>(res->VID);
        const auto& labels_vec = vid_to_labels_[vid];
        // 检查标签（vid_to_labels_[vid] 已排序，使用 binary_search）
        if (std::binary_search(labels_vec.begin(), labels_vec.end(), tenant_id)) {
            qualified.emplace_back(res->Dist, vid);
        }
    }

    // Step 4: 排序合格结果并取 top-k
    std::sort(qualified.begin(), qualified.end());
    size_t found = 0;
    for (; found < k && found < qualified.size(); found++) {
        labels[found] = qualified[found].second;
        distances[found] = qualified[found].first;
    }

    // Step 5: 不足 k 个填充 -1
    for (; found < k; found++) {
        labels[found] = -1;
        distances[found] = std::numeric_limits<float>::max();
    }
}
```

**关键设计决策**：

1. **元数据存储**: `vid_to_labels_` 作为 `std::vector<std::vector<int32_t>>`，内存占用约 `N × avg_labels × 4 bytes`。
   - arxiv (1.6M, avg 2.8 labels): ~18 MB
   - yfcc100m (800K, avg 31 labels): ~99 MB
   - 可以接受。

2. **SPTAG 链接方式**: 将 SPTAG 编译为静态库（`libSPTAGLibStatic.a`），PostFiltering wrapper 链接它。
   - SPTAG 编译命令（最小配置）:
     ```bash
     cd SPANN-PostFiltering/SPTAG-main
     mkdir -p Release && cd Release
     cmake .. -DCMAKE_BUILD_TYPE=Release \
              -DLIBRARYONLY=ON \
              -DROCKSDB=OFF \
              -DSPDK=OFF \
              -DURING=OFF \
              -DTBB=ON
     make -j$(nproc)
     ```

3. **SPTAG 依赖处理**:
   - 必须: Boost (system, thread, serialization, filesystem), OpenMP, zstd, TBB
   - WSL 安装: `sudo apt install libboost-all-dev libtbb-dev libzstd-dev`

4. **SPTAG 源码修改最小化**:
   - **不修改** SPTAG 源码（除编译所需的 CMake 参数调整）
   - PostFiltering 逻辑完全在外层 wrapper 中实现
   - 标签数据不进入 SPTAG metadata 系统

#### 4.2.5 Python 预处理

```python
# SPANN-PostFiltering/python/preprocess.py
# 职责:
#   1. train_mds.pkl → train_access.npy（与其他索引相同）
#   2. train_mds.pkl → metadata.bin（vid_to_labels 二进制格式，C++ 直接读取）
#   3. 生成 JSON 配置文件
```

**metadata.bin 格式**：
```
[4 bytes] n_vectors (uint32)
For each vector i:
  [4 bytes] n_labels_i (uint32)
  [n_labels_i × 4 bytes] labels (int32 array, sorted)
```

#### 4.2.6 CMakeLists.txt（SPANN-PostFiltering/ 根目录，与 Pre-Filtering/DiskIVF 一致）

```cmake
cmake_minimum_required(VERSION 3.12)
project(spann_pf LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP for batch_query
find_package(OpenMP)

# SPTAG 依赖
find_package(Boost 1.66 COMPONENTS system thread serialization filesystem REQUIRED)
find_package(TBB REQUIRED)

# 编译选项（与 SPTAG 保持一致）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

# ★ SPTAG 库路径（需先编译 SPTAG-main）
set(SPTAG_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/SPTAG-main)
set(SPTAG_BUILD_DIR ${SPTAG_ROOT}/Release)

# 包含路径
include_directories(${SPTAG_ROOT}/AnnService)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})

# 源文件
set(SPANN_SOURCES
    src/cnpy.cpp
    src/spann_pf_index.cpp
    src/main.cpp
)

# 头文件（IDE 集成，对编译非必需）
set(SPANN_HEADERS
    src/cnpy.h
    src/config.h
    src/distance.h
    src/predicate.h
    src/spann_pf_index.h
)

# 可执行文件
add_executable(spann_pf ${SPANN_SOURCES} ${SPANN_HEADERS})
target_include_directories(spann_pf PRIVATE src)

# 链接 SPTAG 静态库及其所有依赖
target_link_libraries(spann_pf
    ${SPTAG_BUILD_DIR}/libSPTAGLibStatic.a
    ${SPTAG_BUILD_DIR}/libDistanceUtils.a
    ${Boost_LIBRARIES}
    tbb
    zstd
    -lrt
    -lpthread
)

# 可选：OpenMP
if(OpenMP_FOUND)
    target_link_libraries(spann_pf PRIVATE OpenMP::OpenMP_CXX)
endif()
```

> **编译顺序**：必须先编译 SPTAG-main（生成 `libSPTAGLibStatic.a` 和 `libDistanceUtils.a`），再编译 spann_pf wrapper。SPTAG 编译命令见 4.3 步骤 4.1。
>
> **注意**：`march=native` 在 WSL 跨文件系统场景下可能有问题。若 SPTAG 开启了 AVX-512 而包装层未开启，链接可能失败。建议在 WSL 原生文件系统（非 `/mnt/d/`）中编译，或统一 `march` 级别。

#### 4.2.7 CLI 设计

```bash
./spann_pf bench \
    --train_vecs     $DATA/train_vecs.npy \
    --train_access   $DATA/train_access.npy \
    --queries        $DATA/query_vecs.npy \
    --query_labels   $DATA/query_labels.npy \
    --config         /tmp/spann_config.json \
    --k 10 \
    --output         results.json \
    --filter         "AND 0 NOT 1" \
    --profile \
    --batch-query
```

**CLI 参数完整列表**：

| 参数 | 必需 | 说明 |
|------|:--:|------|
| `--train_vecs PATH` | ✅ | 训练向量 `.npy` [N, d] float32 |
| `--train_access PATH` | — | 访问对 `.npy` [M, 2] int32 |
| `--queries PATH` | ✅ | 查询向量 `.npy` [Q, d] float32 |
| `--query_labels PATH` | — | 查询标签 `.npy` [Q] int32（-1=无过滤） |
| `--config PATH` | — | JSON 配置文件 |
| `--k K` | — | 返回结果数（默认 10） |
| `--filter EXPR` | — | 复杂谓词表达式（如 "AND 0 NOT 1"） |
| `--batch-query` | — | 启用查询间 OpenMP 并行（此时 SPTAG 内部线程=1） |
| `--output PATH` | — | 结果 JSON 路径（默认 results.json） |
| `--profile` | — | 打印最后一个查询的详细计时分解 |

### 4.3 实现步骤

| 步骤 | 内容 | 预估工作量 |
|:---:|------|:---:|
| 4.1 | 在 WSL 中安装 SPTAG 依赖（Boost, TBB, zstd）并编译 SPTAG-main（验证依赖可满足） | 大 |
| 4.2 | 创建 PostFiltering 封装层目录结构 | 小 |
| 4.3 | 从 Curator 复制 cnpy.h/.cpp | 小 |
| 4.4 | 从 Curator 精简复制 distance.h | 小 |
| 4.5 | 复制 predicate.h（与其他索引相同） | 小 |
| 4.6 | 编写 config.h（公共基础设施 + SPANN 配置） | 小 |
| 4.7 | 实现 `SPANNPostFilterIndex::build()`（调用 SPTAG BuildIndex + 存储标签元数据） | 大 |
| 4.8 | 实现元数据 I/O（metadata.bin 读写） | 中 |
| 4.9 | 实现 `SPANNPostFilterIndex::search()`（overfetch + PostFilter） | 中 |
| 4.10 | 实现 `search_with_predicate()`（复杂谓词 PostFilter） | 中 |
| 4.11 | 编写 `main.cpp`（CLI + JSON 输出） | 中 |
| 4.12 | 编写 CMakeLists.txt | 中 |
| 4.13 | 编写 Python 预处理 + 编排脚本 | 小 |
| 4.14 | WSL 编译验证 + 端到端测试（SPTAG 检索 + PostFilter 正确性） | 大 |

---

## 5. 实施顺序与里程碑

### 5.1 推荐实施顺序

```
Phase 1: Pre-Filtering（~2 天）★ 最简单，先验证 C++/Python 分工模式
├── 复制 cnpy, distance → 编写 predicate.h, config.h
├── 实现 C++ PreFilteringIndex
├── 编写 CLI + CMakeLists.txt
├── Python 编排脚本
└── WSL 编译 + 与 Python 原版 Recall 对比验证

Phase 2: DiskIVF-PostFiltering（~3 天）★ 中等复杂度
├── 复制 cnpy, distance, kmeans（微调）→ 编写 predicate.h, config.h
├── 实现 C++ DiskIVFIndex（K-means + 磁盘 I/O）
├── 编写 CLI + CMakeLists.txt
├── Python 编排脚本
└── WSL 编译 + 与 Python 原版 Recall 对比验证

Phase 3: SPANN-PostFiltering（~4 天）★ 最复杂
├── WSL 中编译 SPTAG-main（依赖安装 + 编译验证）
├── 复制 cnpy, distance, predicate.h → 编写 config.h
├── 实现 PostFiltering wrapper（build + search + overfetch）
├── 编写 CLI + CMakeLists.txt
├── Python 编排脚本
└── WSL 编译 + 端到端验证
```

### 5.2 里程碑

| 里程碑 | 验收标准 |
|------|------|
| M1: Pre-Filtering C++ 版 | Recall@10 与 Python 版一致（误差 < 0.1%），延迟显著降低 |
| M2: DiskIVF C++ 版 | Recall@10 与 Python 版一致（误差 < 0.1%），磁盘 I/O 正常工作 |
| M3: SPANN-PF C++ 版 | SPTAG 编译通过，PostFilter 后 Recall@10 与 Python 版可比 |
| M4: 全系统验证 | 3 个索引在 arxiv_small + yfcc100m_small 上端到端运行通过 |

---

## 6. 跨 Phase 一致性要求

> 由于三个索引各自独立（无 shared/ 目录），以下文件应从 Pre-Filtering（Phase 1 已验证）**直接复制**到 DiskIVF 和 SPANN 的 `src/` 目录，确保行为一致。

### 6.1 必须保持完全相同的文件

| 文件 | 说明 | Phase 1 验证状态 |
|------|------|:---:|
| `cnpy.h` | .npy 文件读写声明 | ✅ 从 Curator 原样复制，零修改 |
| `cnpy.cpp` | .npy 文件读实现（float, int32, int64, uint8） | ✅ 从 Curator 原样复制，零修改 |
| `distance.h` | `l2_sqr()` + `l2_sqr_4way()` | ✅ 从 Curator 精简复制，仅保留两个距离函数 |
| `predicate.h` | 前缀记法谓词求值器（tokenize + evaluate） | ✅ Phase 1 通过 40 个 CP 查询验证，与 Python 结果一致 |

### 6.2 必须保持相同但需微调的文件

| 文件 | 说明 | 差异点 |
|------|------|------|
| `config.h` | 公共基础设施宏 + 索引配置结构体 | **宏部分完全相同**（PREFETCH, THROW_*）；**配置结构体**各索引不同（PreFilteringConfig / DiskIVFConfig / SPANNConfig） |

### 6.3 数据结构设计要求

以下设计模式从 Phase 1 验证得出，DiskIVF 和 SPANN 必须遵循：

#### 6.3.1 双向标签索引（需要支持复杂谓词时）

```
label_to_vids_[label] → vector<vid>   // 倒排索引：单标签查询 O(1) 查候选集
vid_to_labels_[vid]   → vector<label> // 正向索引：谓词求值时查向量的标签列表
```

**DiskIVF 适用性**：DiskIVF 的搜索在加载的簇内进行。单标签查询可以在加载的簇内使用 `label_to_vids_` 全局倒排索引做预过滤（减少加载的向量数）；复杂谓词查询需要对加载的簇内每条向量调 `predicate::evaluate()`，需要对应簇的 `vid_to_labels_`。

**SPANN 适用性**：SPANN 使用 overfetch + PostFilter 模式。SPTAG 返回 top-k' 个未过滤结果后，在 wrapper 层需要 `vid_to_labels_[vid]` 检查每个结果是否满足标签条件。**仅需正向索引，不需要倒排索引**（因为不过滤候选集，而是对结果做后过滤）。

#### 6.3.2 配置自动检测

```
d          ← 从 .npy shape[1] 自动检测，config 文件可覆盖
n_labels   ← 从 access_pairs 自动检测 (max tid + 1)，config 文件可覆盖
```

### 6.4 Phase 1 验证基准

| 指标 | arxiv_small | 说明 |
|------|------|------|
| Single-label Recall@10 | 1.0000 (500/500) | 精确暴力搜索，与 GT 完全一致 |
| CP Recall@10 ("AND 1 NOT 72") | 1.0000 (40/40) | 谓词求值与 Python `predicate.py` 一致 |
| Build time | 0.04s | 50K vectors, 85K access pairs |
| SL latency | 0.25 ms/query | 含 warmup 后测量 |
| CP latency | 2.38 ms/query | 全量扫描 + 谓词求值 |
| Memory | 75.21 MB | 全量向量 + 双向索引 |

---

## 7. 验证方案

### 7.1 正确性验证

#### 6.1.1 单元级验证

- **距离计算**: C++ `l2_sqr()` 与 Python `np.sum((a-b)**2)` 结果对比（误差 < 1e-5）
- **谓词求值**: C++ `predicate::evaluate()` 与 Python `evaluate_predicate()` 在 1000 个随机标签集上的结果一致（交叉验证）
- **倒排索引（Pre-Filtering）**: C++ `label_to_vids_` 与 Python `label_to_indices` 元素一致
- **K-means 聚类（DiskIVF）**: C++ Lloyd 与 FAISS Kmeans 的质心差异（allclose < 1e-3）

#### 6.1.2 检索结果验证

对每条查询，对比 C++ 和原 Python 版的 top-k 结果：
- **Recall@k 差异**: < 0.1%（绝对差异）
- **结果一致性**: top-10 结果中至少 9 个相同（允许浮点舍入引起的顺序交换）
- **距离值**: L2 距离误差 < 1e-5

#### 6.1.3 端到端验证

- 在 `arxiv_small`（50K）和 `yfcc100m_small`（50K）上运行完整实验
- 对比 JSON 输出文件中的关键指标：`build_time_s`, `avg_recall`, `avg_latency_ms`

### 6.2 性能验证

- C++ 版本查询延迟应**不低于** Python 版本（预期显著优于 Python）
- 内存占用应小于 Python 版本（无 Python 解释器 + numpy 开销）
- 构建时间应与 Python 版本可比（K-means 部分 C++ 实现预期更快）

### 6.3 验证脚本

```bash
# 一键验证：运行所有索引的 C++ 版本并与 Python 基线对比
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines

# Step 1: 编译所有索引
#   cd Pre-Filtering && mkdir -p build && cd build && cmake .. && make -j
#   cd DiskIVF-PostFiltering && mkdir -p build && cd build && cmake .. && make -j
#   cd SPANN-PostFiltering/src && mkdir -p build && cd build && cmake .. && make -j

# Step 2: 运行验证实验
python scripts/verify_all.py --dataset arxiv_small --k 10

# Step 3: 对比结果
python scripts/compare_results.py --dataset arxiv_small
```

---

## 8. 风险与注意事项

### 8.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|------|
| SPTAG 依赖链复杂（Boost + TBB + zstd） | WSL 编译失败 | 预先在 WSL 中完整编译 SPTAG，锁定依赖版本 |
| SPTAG 的 SearchIndex 可能不返回足够结果 | PostFilter 后不足 k 个结果 | 设置足够大的 overfetch_factor（默认 50，上限 10000），支持自适应调整 |
| K-means 结果差异导致 recall 偏差 | C++ 与 FAISS 聚类结果不同 | 接受小幅度 recall 差异（类比 Curator 自实现 vs FAISS），固定 random seed |
| 文件复制导致代码不一致 | 不同索引的 cnpy/distance 版本分歧 | 在修订记录中标注复制来源 commit，便于追溯 |
| 磁盘 I/O 性能问题 | 查询延迟劣化 | 使用原始二进制格式 + 批量读取优化；或先用 cnpy 验证正确性再优化 |
| float32 精度差异 | 检索结果排序微妙不同 | 使用 double 进行关键累加（如距离求和） |

### 8.2 工程风险

| 风险 | 影响 | 缓解措施 |
|------|------|------|
| 代码重复维护成本 | 修改 bug 需同步多处 | 接受这一成本（文件夹自包含原则优先）；核心工具文件（cnpy, predicate.h）变更极少 |
| SPTAG 许可证兼容性 | 发布限制 | SPTAG 使用 MIT License，无问题 |
| SPTAG 源码更新 | 上游改动导致编译失败 | 固定 SPTAG commit 版本，不自动拉取更新 |

### 8.3 注意事项

1. **K-means 无上限限制**：Curator 的 K-means 算法本身 `n_clusters` 是自由参数，64 簇上限仅来自 Curator 树结构的 `MAX_BRANCH_FACTOR` 位编码约束。DiskIVF 复制后独立使用，nlist 可设为 256 等任意值。

2. **数据兼容性**：C++ 版本应与 Python 版本使用相同的预处理数据（相同的 train_vecs.npy 等），确保公平对比。

3. **WSL 路径**：所有路径处理注意 Windows ↔ WSL 映射（`/mnt/d/...`）。

4. **保留原 Python 代码**：原 Python 实现保留作为 baseline reference。新增 `python/` 子目录存放新的编排脚本，原 `run_*.py` 保留为兼容性入口。

5. **SPTAG 最小化编译**：
   ```bash
   cd SPANN-PostFiltering/SPTAG-main && mkdir -p Release && cd Release
   cmake .. -DCMAKE_BUILD_TYPE=Release \
            -DLIBRARYONLY=ON \
            -DROCKSDB=OFF \
            -DSPDK=OFF \
            -DURING=OFF \
            -DTBB=ON
   make -j$(nproc)
   ```

6. **每个索引的可执行文件名**：
   - Pre-Filtering: `build/prefiltering`
   - DiskIVF-PostFiltering: `build/diskivf`
   - SPANN-PostFiltering: `build/spann_pf`

7. **CMake 与 WSL 注意事项**：
   - 编译必须在 WSL 文件系统（非 `/mnt/d/`）下执行，否则 CMake 的 `march=native` 可能出错
   - 推荐方式：源码在 `/mnt/d/`，build 目录也在 `/mnt/d/`
   - 若遇到跨文件系统编译问题，可考虑将 build 目录放在 `/tmp/` 下

---

## 附录 A：文件变更总览

### 新增文件

```
Pre-Filtering/
├── CMakeLists.txt                               # ★ 新增
├── src/
│   ├── main.cpp                                 # ★ 新增
│   ├── config.h                                 # ★ 新增（含公共基础设施）
│   ├── prefiltering_index.h                     # ★ 新增
│   ├── prefiltering_index.cpp                   # ★ 新增
│   ├── predicate.h                              # ★ 新增（前缀记法谓词求值器）
│   ├── distance.h                               # 从 Curator 精简复制
│   ├── cnpy.h                                   # 从 Curator 复制
│   └── cnpy.cpp                                 # 从 Curator 复制
├── python/
│   ├── run_experiment.py                        # ★ 新增
│   └── preprocess.py                            # ★ 新增
└── build/                                       # 编译产物目录

DiskIVF-PostFiltering/
├── CMakeLists.txt                               # ★ 新增
├── src/
│   ├── main.cpp                                 # ★ 新增
│   ├── config.h                                 # ★ 新增（含公共基础设施）
│   ├── diskivf_index.h                          # ★ 新增
│   ├── diskivf_index.cpp                        # ★ 新增
│   ├── kmeans.h                                 # 从 Curator 复制
│   ├── kmeans.cpp                               # 从 Curator 复制 + 微调
│   ├── predicate.h                              # ★ 新增
│   ├── distance.h                               # 从 Curator 精简复制
│   ├── cnpy.h                                   # 从 Curator 复制
│   └── cnpy.cpp                                 # 从 Curator 复制
├── python/
│   ├── run_experiment.py                        # ★ 新增
│   └── preprocess.py                            # ★ 新增
└── build/

SPANN-PostFiltering/
├── SPTAG-main/                                  # 已有（开源代码）
├── CMakeLists.txt                               # ★ 新增（根目录级别）
├── src/
│   ├── main.cpp                                 # ★ 新增
│   ├── config.h                                 # ★ 新增（含公共基础设施）
│   ├── spann_pf_index.h                         # ★ 新增
│   ├── spann_pf_index.cpp                       # ★ 新增
│   ├── predicate.h                              # ★ 新增（与其他索引相同）
│   ├── distance.h                               # 从 Curator 精简复制
│   ├── cnpy.h                                   # 从 Curator 复制
│   └── cnpy.cpp                                 # 从 Curator 复制
├── python/
│   ├── run_experiment.py                        # ★ 新增
│   ├── preprocess.py                            # ★ 新增
│   └── compute_recall.py                        # ★ 新增
├── legacy/
│   ├── run_spann.py                             # 从根目录移入
│   └── run_spann_sweep.py                       # 从根目录移入
└── build/
```

### Curator 源码变更

**零变更。**

### 保留不变的文件

- `Curator/` — 全部保持原样
- `SPANN-PostFiltering/SPTAG-main/` — 开源代码保持原样（仅需 CMake 编译参数调整）
- `SPANN-PostFiltering/run_spann.py` — 保留为兼容性入口
- `SPANN-PostFiltering/run_spann_sweep.py` — 保留
- `Pre-Filtering/run_prefiltering.py` — 保留为兼容性入口
- `DiskIVF-PostFiltering/run_diskivf.py` — 保留为兼容性入口
- `1_Data/` — 数据目录不变
- `2_Utils/` — 工具模块不变
- `3_Config/` — 配置文件目录不变
- `4_Results/` — 结果输出目录不变

---

## 附录 B：关键接口对照表

### C++ ↔ Python 接口对照

| 原 Python 方法 | 新 C++ 方法 | 说明 |
|------|------|------|
| `PreFilteringIndex.build(vecs, mds)` | `PreFilteringIndex::build(n, vecs, access_pairs, n_pairs)` | 输入从 Python 对象变为 raw pointers |
| `PreFilteringIndex.query(x, k, tid)` | `PreFilteringIndex::search(query, k, tid, dists, labels)` | 输出通过指针参数返回 |
| `PreFilteringIndex.query_with_complex_predicate(x, k, p)` | `PreFilteringIndex::search_with_predicate(query, k, predicate, dists, labels)` | C++ 端实现谓词求值 |
| `DiskIVFIndex.build(vecs, mds)` | `DiskIVFIndex::build(n, vecs, access_pairs, n_pairs)` | K-means 改为 C++ 实现 |
| `DiskIVFIndex.query(x, k, tid)` | `DiskIVFIndex::search(query, k, tid, dists, labels)` | 磁盘加载逻辑用 C FILE I/O |
| `DiskIVFIndex._load_cluster(cid)` | `DiskIVFIndex::load_cluster(cid)` | 返回结构体而非 tuple |
| `SPANNIndex` (原 Python) | `SPANNPostFilterIndex` (新 C++) | 完全不同的底层实现（SPTAG） |
| `compute_recall()` | Python 端实现 | 与原 Python 编排脚本相同 |

### 配置文件参数对照

| 原 Python 参数 | 新 JSON 参数 | 说明 |
|------|------|------|
| `n_labels` | `n_labels` | 相同 |
| `k` | `k` | 相同 |
| `nlist` (DiskIVF/SPANN) | `nlist` | 相同 |
| `nprobe` (DiskIVF/SPANN) | `nprobe` | 相同 |
| — | `batch_query` | 新增（OpenMP 并行） |
| — | `num_threads` | 新增（线程数控制） |
| — | `overfetch_factor` (SPANN) | 新增（PostFilter overfetch 因子） |

---

> **文档状态**: v2 修订完成  
> **下一步**: 讨论确认方案细节 → 从 Phase 1（Pre-Filtering）开始实施
