# Curator 索引代码重构方案

> **目标**: 将 Curator 索引从 FAISS SWIG 重型包装中剥离，重组织为精简、独立的 C++ 索引库。C++ 为独立可执行程序（在线构建+检索），Python 仅负责离线数据预处理，两者通过 `.npy` 文件交互。

---

## 一、现状诊断

### 1.1 当前文件清单与规模

**总计: 13,512 行 C++/Python**

| 文件 | 行数 | 类别 | 重构去留 |
|------|------|------|----------|
| `MultiTenantIndexIVFHierarchical.h` | 773 | Core Header | **保留，拆分** |
| `MultiTenantIndexIVFHierarchical.cpp` | 2790 | Core Source | **保留，拆分** |
| `BloomFilter.h` | 734 | Core Utility | **保留** |
| `complex_predicate.h` | 231 | Core Utility | **保留，精简 namespace** |
| `complex_predicate.cpp` | 392 | Core Source | **保留，拆分** |
| `MetricType.h` | 63 | FAISS 原版（定义 `idx_t`、`tid_t`、`vid_t`、`label_t`、`Buffer`、`AccessMap`） | **并入 common.h** |
| `MultiTenantIndex.h` | 270 | 继承基类 | **删除** |
| `MultiTenantIndex.cpp` | 199 | 基类实现 | **删除** |
| `MultiTenantIndexIVF.h` | 379 | IVF 基类 | **删除** |
| `MultiTenantIndexIVF.cpp` | 1089 | IVF 基类 | **删除** |
| `MultiTenantIndexIVFFlat.h` | 100 | IVF 子类 | **删除** |
| `MultiTenantIndexIVFFlat.cpp` | 705 | IVF 子类 | **删除** |
| `MultiTenantIndexIVFFlatBF.h` | 76 | 基线对比 | **删除** |
| `MultiTenantIndexIVFFlatBF.cpp` | 615 | 基线对比 | **删除** |
| `MultiTenantIndexIVFFlatSep.h` | 71 | 基线对比 | **删除** |
| `MultiTenantIndexIVFFlatSep.cpp` | 144 | 基线对比 | **删除** |
| `MultiTenantIndexHNSW.h` | 214 | 基线对比 | **删除** |
| `MultiTenantIndexHNSW.cpp` | 268 | 基线对比 | **删除** |
| `ACORN.h` | 338 | HybridCurator | **删除** |
| `ACORN.cpp` | 1739 | HybridCurator | **删除** |
| `IndexACORN.h` | 147 | HybridCurator | **删除** |
| `IndexACORN.cpp` | 645 | HybridCurator | **删除** |
| `IndexHybridCurator.h` | 97 | HybridCurator | **删除** |
| `IndexHybridCurator.cpp` | 192 | HybridCurator | **删除** |
| `impl/IDSelector.h` | 187 | 标准 FAISS（含 MultiTenantIDSelector 扩展） | **删除** |
| `impl/IDSelector.cpp` | 153 | 标准 FAISS（含 MultiTenantIDSelector 扩展） | **删除** |

**预计精简至 ~5,500 行 C++ + ~700 行 Python (减少 ~55%)**

### 1.2 核心问题

#### (a) 深度绑定 FAISS —— 且完全多余

- 所有代码位于 `namespace faiss` 下
- `MultiTenantIndexIVFHierarchical` 直接继承 `MultiTenantIndex`（**不是** `MultiTenantIndexIVF`），但仅使用了基类的 `d`、`ntotal`、`metric_type` 三个字段，继承链完全可消除
- Curator 的核心数据结构是**层次化聚类树**（`TreeNode`），构建和搜索完全走自己的树遍历逻辑，**根本不使用 IVF 倒排列表**
- `MultiTenantIndexIVF` 类存在但 Curator 从未继承它——IVF 相关的基类（`MultiTenantIndexIVF`、`MultiTenantIndexIVFFlat`）仅用于基线对比方法，不是 Curator 的依赖

#### (b) SWIG 包装——纯开销

- 需要修改 FAISS 的 `swigfaiss.swig`、`CMakeLists.txt`、覆盖 `MetricType.h` 和 `IDSelector.h`
- Python 端需要 `faiss.swig_ptr()`、`faiss.cast_integer_to_idx_t_ptr()` 等不透明的 SWIG 函数
- 大量 `%ignore` 指令来隐藏内部成员

#### (c) 大杂烩文件

- `MultiTenantIndexIVFHierarchical.cpp`（2790行）包含：树构建、树搜索、PQ 编码、Flash 存储、复杂谓词、临时索引、性能分析、内存统计
- `MultiTenantIndexIVFHierarchical.h`（773行）定义了 25+ 个 SWIG 兼容 getter

#### (d) 混杂无关代码

- 4 个基线索引（FlatBF, FlatSep, HNSW）+ ACORN/HybridCurator 总计 ~4400 行
- 这些是论文实验的对比算法，不属于 Curator 本身

---

## 二、6 个关键决策（已确认）

### 决策汇总

| # | 问题 | 决策 | 理由 |
|---|------|------|------|
| Q1 | K-means 依赖 | **自实现 K-means**，提供可选 FAISS Clustering 回退 | 自实现 ~150 行 Lloyd 算法；若质量/性能差异显著，可启用 FAISS Clustering 编译选项 |
| Q2 | 基线索引文件 | **全部删除** | 非 Curator 核心，需要时可从 git 历史找回 |
| Q3 | 向量文件格式 | **保持 `.npy` 格式** | 现有数据无需重新生成；C++ 端使用 `cnpy` 库读取，与 MiniRFANN 一致 |
| Q4 | 复杂谓词 State/VarMap | **完整保留** | 保留 `ExprNode` 树 + `StateNode`/`VarMapNode` 符号执行体系，用于 `find_all_qualified_vecs` |
| Q5 | Profiling 字段 | **全部保留**（~25 个字段） | 维持现有的细粒度性能分析能力 |
| Q6 | Python↔C++ 分工 | **C++ 独立可执行程序 + Python 离线预处理**，通过 `.npy` 文件交互，**无直接绑定** | 完全遵循 REFERENCE_STYLE.md 定义的分工模式（§六） |

### 决策 Q1 补充说明

K-means 在 Curator 中的角色：`train_helper()` 在每个树节点对向量做递归 K-means 聚类，产生子节点质心。

**默认方案（自实现）**:
- 实现标准 Lloyd K-means（初始化 + 分配 + 更新，迭代 `clus_niter` 次）
- 仅支持 L2 距离（Curator 仅使用 METRIC_L2）
- 约 150 行 C++，无外部依赖

**回退方案（可选 FAISS Clustering）**:
- CMake 选项 `CURATOR_USE_FAISS_CLUSTERING=ON/OFF`
- 开启后链接 FAISS 的 `Clustering` 类
- 在 `train_helper()` 中用 `#ifdef` 切换实现

**回退触发条件**:
- 自实现 K-means 最多允许 **3 次调优迭代**（调参：初始化策略、迭代次数、收敛阈值）
- 若 3 次调优后召回率损失仍 >1%（vs FAISS Clustering），则切换到 FAISS

### 补充决策（已确认）

| # | 问题 | 决策 | 理由 |
|---|------|------|------|
| D1 | 索引序列化格式 | **保持当前 Flash region 布局**，不设计新紧凑格式 | 设计新格式增加核心重构任务的复杂度，收益不够明显；可作为后续优化项 |
| D2 | `train_mds.pkl` 数据格式 | Python 预处理脚本将 `.pkl` 转为 `.npy`（COO pairs 格式 `[vid, tid]`），C++ 通过 cnpy 读取 | 转换仅需数秒（arxiv full: 11MB pkl → npy），**不修改任何已有文件**，GT/query 文件完全不受影响 |
| D3 | K-means 回退边界 | 自实现允许 **≤3 次调优**；若召回率损失 >1% 则切换到 FAISS Clustering | 兼顾零依赖理想与质量保障 |
| D4 | `curator.py` 保留 | **不保留**，由 `run_experiment.py` 承担所有编排逻辑（`subprocess` 调用 C++ 可执行文件） | 代码结构更清晰，避免不必要的 Python wrapper 层 |

### D2 数据流举例

```
重构前 (SWIG 直接调用):
  run_curator.py → faiss.MultiTenantIndexIVFHierarchical.grant_access(label, tid)

重构后 (文件系统交互):
  preprocess_train.py:
      train_mds = pickle.load("train_mds.pkl")
      pairs = [(vid, tid) for vid, tids in enumerate(train_mds) for tid in tids]
      np.save("train_access.npy", np.array(pairs, dtype=np.int32))
  
  curator build:
      access = cnpy::npy_load("train_access.npy")  // 读取 [N×2] int32
      for each row: index.grant_access(vid, tid)
```

---

## 三、重构目标架构

### 3.1 Python↔C++ 分工（遵循 REFERENCE_STYLE.md §六）

```
┌──────────────────────────┐          ┌──────────────────────────┐
│   Python 离线预处理        │          │   C++ 在线检索引擎         │
│                          │          │                          │
│  preprocess_train.py     │          │  curator bench        │
│  ├─ K-means 层次聚类      │  .npy    │  ├─ 读取全部 .npy       │
│  ├─ PQ 码本训练 (faiss)   │─────────▶│  ├─ 构建聚类树索引       │
│  ├─ 属性/标签生成         │  文件     │  ├─ 执行过滤向量检索     │
│  └─ 保存预处理结果        │          │  └─ 输出 results.json   │
│                          │          │                          │
│  prepare_queries.py      │          │                          │
│  ├─ 查询向量提取          │─────────▶│                          │
│  ├─ ground truth 计算     │          │                          │
│  └─ 产出查询文件           │          │                          │
│                          │          │                          │
│  run_experiment.py       │subprocess│                          │
│  ├─ .pkl→.npy 格式转换   │─────────▶│                          │
│  ├─ 编排流程              │◀─────────│                          │
│  └─ recall 分析 + 报告    │ stdout   │                          │
└──────────────────────────┘          └──────────────────────────┘
```

**关键特征**（与 MiniRFANN 一致）：
- Python 和 C++ **是两个独立进程**，无 SWIG、pybind11 等绑定层
- 数据交互桥梁是 `.npy` 文件：Python 通过 `numpy.save()` 写入，C++ 通过 `cnpy` 读取
- Python 承担**离线预处理和实验编排**（数据生成、FAISS 训练、流程控制）
- C++ 承担**在线构建和检索**（所有对延迟敏感的操作）

### 3.2 目标目录结构

```
Curator/
├── CMakeLists.txt                  # 独立构建，仅依赖: cnpy + OpenMP + nlohmann/json
├── README.md
├── REFACTOR_PLAN.md                # 本文件
├── src/                            # C++ 源码
│   ├── common.h                    # 类型定义、常量、IdAllocator、SortedList、RunningMean
│   ├── bloom_filter.h              # Bloom Filter（来自原 BloomFilter.h）
│   ├── cnpy.h                      # .npy 文件 I/O（来自 MiniRFANN / github.com/rogersce/cnpy）
│   ├── cnpy.cpp
│   ├── kmeans.h                    # K-means 聚类（自实现 Lloyd 算法）
│   ├── kmeans.cpp
│   ├── tree_node.h                 # TreeNode 数据结构
│   ├── cluster_tree.h              # 聚类树操作（构建、指派、路径查询）
│   ├── cluster_tree.cpp
│   ├── shortlist.h                 # 短列表管理（split / merge）
│   ├── shortlist.cpp
│   ├── pq_codec.h                  # PQ 编解码 + 距离查表 + ADC 重排
│   ├── pq_codec.cpp
│   ├── flash_store.h               # Flash 磁盘存储（fseek/fread）
│   ├── flash_store.cpp
│   ├── temp_index.h                # 临时索引（bitmap filter 搜索加速）
│   ├── temp_index.cpp
│   ├── complex_predicate.h         # 复杂谓词（ExprNode 树 + State/VarMap 完整体系）
│   ├── complex_predicate.cpp
│   ├── distance.h                  # 距离函数（header-only: L2, batch, prefetch）
│   ├── profiling.h                 # 性能分析数据结构（25+ 字段，完整保留）
│   ├── curator_index.h             # CuratorIndex 主类声明
│   ├── curator_index.cpp           # CuratorIndex 主类实现
│   ├── diagnostics.h               # [可选] 诊断工具：sanity_check + memory_breakdown + print_tree_info
│   ├── diagnostics.cpp             # [可选] 诊断工具实现（~200 行，从 curator_index.cpp 提取以控制其体量）
│   └── main.cpp                    # CLI 入口（build / search / bench 模式）
├── python/                         # Python 离线工具 & 实验脚本
│   ├── preprocess_train.py         # 训练数据预处理（格式转换、可选 K-means/PQ 离线训练）
│   ├── prepare_queries.py          # 查询数据准备 + Ground Truth 计算
│   └── run_experiment.py           # 实验编排（.pkl→.npy 转换、subprocess 调用 C++、结果收集分析）
├── examples/
│   └── basic_usage.sh              # Shell 使用示例
└── tests/
    └── test_curator.cpp            # 单元测试 (可选)
```

**总计: 约 24 个 C++ 文件（14 .h + 10 .cpp）+ 3 个 Python 文件，每个 .cpp 文件 150-800 行。若启用可选的 `diagnostics.h/.cpp`，增至 26 个文件（15 .h + 11 .cpp）**

### 3.3 模块依赖关系（星型拓扑）

```
                     ┌──────────────┐
                     │   main.cpp   │  (CLI 入口)
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │curator_index │  (编排层: 唯一耦合点)
                     │  .h / .cpp   │
                     └──┬──┬──┬──┬──┘
            ┌───────────┤  │  │  ├───────────┐
            │        ┌──┘  │  └──┐        ┌─┘
      ┌─────▼────┐ ┌▼──────▼──┐ ┌▼──────┐ ┌▼──────────────┐
      │cluster   │ │ shortlist │ │  pq   │ │  flash_store  │
      │_tree     │ │ .h / .cpp │ │_codec │ │  .h / .cpp    │
      │.h / .cpp │ └───────────┘ │.h/cpp │ └──────────────┘
      └────┬─────┘               └──┬────┘
           │                        │
      ┌────▼────────────────────────▼──────┐
      │           common.h                  │
      │  (类型、常量、IdAllocator、工具类)    │
      └──┬─────┬──────┬────────────────────┘
         │     │      │
  ┌──────▼──┐ ┌▼──────▼──────┐ ┌───────────▼───────────┐
  │ bloom   │ │  distance.h  │ │  diagnostics .h/.cpp   │
  │_filter  │ │(header-only) │ │  [可选] sanity_check,  │
  │  .h     │ └──────┬───────┘ │  memory_breakdown 等   │
  └────┬────┘        │         └───────────────────────┘
       │             │
  ┌────▼─────────────▼──────────┐
  │  complex_predicate .h / .cpp│
  │  temp_index         .h / .cpp│
  └────────────┬────────────────┘
               │
  ┌────────────▼───────────────────┐
  │  kmeans .h / .cpp   (自实现)    │
  └────────────┬───────────────────┘
               │
  ┌────────────▼───────────────────┐
  │  cnpy .h / .cpp (.npy I/O)     │
  └────────────────────────────────┘
```

**耦合特性**（与 REFERENCE_STYLE.md 一致）:
- `common.h` 是唯一的底层依赖，所有上层模块依赖它
- `cluster_tree`、`shortlist`、`pq_codec`、`flash_store` 之间**零相互依赖**
- `curator_index` 是唯一的编排层耦合点
- 这是一种**星型依赖拓扑**，极易测试和替换

### 3.4 FAISS 依赖 → 替换策略（逐项分析"保留 vs 自实现"）

> **核心原则**: 代码简洁性和结构清晰性优先，**不以"零 FAISS"为教条**。若保留某项 FAISS 工具能更简洁，则保留。若自实现更简洁（或 FAISS 工具与 FAISS 内部深度耦合），则自实现。

当前核心文件 `MultiTenantIndexIVFHierarchical.cpp` 对 FAISS 的直接依赖共 7 项，逐项分析如下：

| FAISS 依赖 | 使用次数 | 保留方案 | 自实现方案 | 决策 |
|------------|---------|----------|-----------|:--:|
| `fvec_L2sqr()` | 16 次 | 保留需 `#include <faiss/utils/distances.h>`，需 FAISS 源码树在 include path 中 | `l2_sqr()` ~10 行标量循环 | **自实现**：10 行代码不值得引入 include path 依赖 |
| `prefetch_L1()` | 6 次 | 保留需 `#include <faiss/utils/prefetch.h>` | `PREFETCH(ptr)` 1 行宏 | **自实现**：1 行宏 vs 引入 FAISS header，自实现更简洁 |
| `FAISS_THROW_*` / `FAISS_ASSERT_*` | 31 次 | 保留需 `#include <faiss/impl/FaissException.h>`，引入 FAISS 异常继承体系 | `CURATOR_THROW_*` ~15 行宏（底层用 `std::runtime_error`） | **自实现**：15 行宏可消除整个 FAISS 异常体系依赖 |
| `CMax` / `heap_*` / `maxheap_*` | 5 次 | **(a)** 引入 `<faiss/utils/Heap.h>`（~500 行模板，独立性强）；**(b)** vendor 该文件到 `src/` | `RunningList`（**已存在**于当前代码，tenant search 已验证） | **自实现**：RunningList 是已有代码，零额外成本。若未来需要高性能堆，可 vendor `Heap.h` |
| `Clustering` | 1 次 | 保留需要 `#include <faiss/Clustering.h>`，该头文件连带引入 `Index`、`IndexFlat`、`ClusteringParameters` 等 FAISS 核心基础设施 | `kmeans()` ~250 行 Lloyd + OpenMP | **自实现**：FAISS Clustering **非独立工具**，耦合 FAISS Index 体系。保留意味着整个 FAISS 构建依赖 |
| `IndexFlatL2` | 1 次 | 作为 `Clustering` 的 quantizer，随 Clustering 一起 | K-means 直接返回 `assignments`，无需 quantizer 抽象 | **自实现**：消除不必要的中间层 |
| `ProductQuantizer` | 1 次 | 保留需要 `#include <faiss/impl/ProductQuantizer.h>`，内部依赖 `Clustering` | `PQCodec` ~550 行，复用自实现 `kmeans` 模块 | **自实现**：ProductQuantizer 依赖 Clustering，保留即保留整个 FAISS 训练基础设施 |

**结论: 7 项全部自实现。总新增代码约 850 行（kmeans 250 + pq_codec 550 + 其他 50），换来完全独立的构建系统。**

**补充说明**:

- **`compute_vector_distance`**: 虽然名义上是纯距离计算函数，但它深度访问 `CuratorIndex` 内部状态（`raw_vectors_buffer`、`vid_to_buffer_offset`、`flash_finalized` 等）。重构后作为 `CuratorIndex` 的 **private 方法**（而非 `distance.h` 中的自由函数），通过 lambda 参数化向量获取逻辑（`flash → buffer → 错误` 三级回退）。参见 §4.6a 更新后的归属表。
- **`scratch_` 缓冲区线程安全**: `compute_vector_distance` 使用 `thread_local std::vector<float> scratch_`（而非 `mutable` 成员）避免 per-call 堆分配。`thread_local` 保证多线程并行查询时每个线程有独立缓冲区，无需 `#pragma omp critical` 保护，不引入串行化开销。参见 §4.9 和 §4.14。

- **`evaluate_formula` 模板函数**: `complex_predicate.cpp` 中有 4 个显式模板实例化使用 `tid_t` 类型（`std::unordered_set<tid_t>`、`std::vector<tid_t>`）。重构后 `common.h` 保留 `using tid_t = int16_t` 别名以确保兼容，同时将 `evaluate_formula` 模板定义移入头文件（header-only），消除显式实例化需求。参见 §4.11 迁移说明。

### 3.4a "零 FAISS 依赖"的收益

| 收益 | 说明 |
|------|------|
| **独立构建** | `cmake -B build && cmake --build build`，无需 FAISS 源码树或 `libfaiss.so` |
| **头文件隔离** | 不引入任何 `<faiss/...>` include，不污染命名空间 |
| **可移植性** | 可在任何有 g++/cmake/openmp 的系统上编译（包括当前 WSL 中 FAISS 源码树已不可用的情况） |
| **可调试性** | 所有代码在 `Curator/src/` 下，单步调试不跳入 FAISS 框架代码 |
| **对齐 REFERENCE_STYLE.md** | MiniRFANN 同样仅依赖 cnpy + OpenMP，零 FAISS 依赖 |

### 3.4b 若后续发现自实现不够用的回退路径

| 场景 | 回退方案 |
|------|----------|
| K-means 聚类质量不达标 | CMake option `CURATOR_USE_FAISS_CLUSTERING=ON` 切换回 FAISS Clustering |
| 需要 AVX/SIMD 加速距离计算 | 将 `l2_sqr` 替换为 SIMD 内联汇编（~50 行），无需引入 FAISS |
| 需要更复杂的 PQ 变体（OPQ、PQ+IVF） | 在 `pq_codec` 基础上扩展；极端情况下 vendor FAISS `ProductQuantizer.h/.cpp` 到 `src/` |

---

## 四、模块职责设计

### 4.1 `common.h` — 公共基础设施

**职责**: 所有模块共享的类型别名、编译期常量、小工具类模板、异常宏

| 内容 | 来源 |
|------|------|
| 类型别名: `ext_vid_t`, `int_vid_t`, `ext_lid_t`, `int_lid_t` | 原 `MetricType.h` + `MultiTenantIndexIVFHierarchical.h` |
| **保留别名**: `tid_t = int16_t`（供 `complex_predicate` 模板实例化兼容） | 原 `MetricType.h` |
| 类型别名: `Buffer` = `std::vector<int_vid_t>` | 原 `MetricType.h`（`complex_predicate` 依赖此类型进行 buffer 集合操作） |
| 常量: `MAX_BRANCH_FACTOR`, `MAX_LEAF_SIZE`, `MAX_TREE_DEPTH` | 原头文件 |
| `PREFETCH(ptr)` 宏: `__builtin_prefetch((ptr), 0, 3)` | 替代 FAISS `prefetch_L1`。参数 `0`=读预取, `3`=高时间局部性(L1 cache)。语义与 FAISS `prefetch_L1` 的 `__builtin_prefetch(ptr, 0, 3)` 完全等价 |
| `CURATOR_THROW_*` 宏族: `THROW_MSG`, `THROW_IF_NOT`, `THROW_IF_NOT_MSG`, `THROW_FMT`, `ASSERT_MSG`, `ASSERT_FMT` | 替代 FAISS `FAISS_THROW_*` / `FAISS_ASSERT_*` |
| `IdAllocator<Ext,Int>`: 连续 ID 分配器 | 原 `TenantIdAllocator` |
| `IdMapping<Ext,Int>`: 双向 ID 映射 | 原 `VectorIdAllocator` |
| `SortedList<T>`: 有序列表 | 原头文件 |
| `RunningList`: top-K 有序候选集（O(k) 插入，被 `curator_index` 和 `temp_index` 共享） | 原 .cpp 匿名 namespace（[第 568-662 行](Curator/src/MultiTenantIndexIVFHierarchical.cpp#L568-L662)） |
| `RunningMean`: 动态均值追踪 | 原头文件 |

**`CURATOR_THROW_FMT` 实现要点**（C++17，无 `std::format`）:

```cpp
// 底层使用 snprintf 实现格式化消息
#define CURATOR_THROW_FMT(MSG, ...)                                    \
    do {                                                               \
        char _buf[1024];                                               \
        std::snprintf(_buf, sizeof(_buf), MSG, ##__VA_ARGS__);        \
        throw std::runtime_error(std::string(_buf));                   \
    } while (0)

// 其他宏类似构造
#define CURATOR_THROW_MSG(MSG) \
    throw std::runtime_error(MSG)

#define CURATOR_THROW_IF_NOT(COND, MSG) \
    do { if (!(COND)) CURATOR_THROW_MSG(MSG); } while (0)

#define CURATOR_THROW_IF_NOT_MSG CURATOR_THROW_IF_NOT  // 语义别名

#define CURATOR_THROW_IF_NOT_FMT(COND, MSG, ...) \
    do { if (!(COND)) CURATOR_THROW_FMT(MSG, ##__VA_ARGS__); } while (0)

#define CURATOR_ASSERT_MSG(COND, MSG) \
    do { if (!(COND)) CURATOR_THROW_MSG(MSG); } while (0)

#define CURATOR_ASSERT_FMT(COND, MSG, ...) \
    do { if (!(COND)) CURATOR_THROW_FMT(MSG, ##__VA_ARGS__); } while (0)
```

> **注意**: `complex_predicate.cpp` 中有 3 处 `FAISS_THROW_MSG`（`make_state` 函数，第 60/69/77 行），Phase 3 迁移时必须替换为 `CURATOR_THROW_MSG`。

### 4.2 `bloom_filter.h` — Bloom Filter

**来源**: 原 `BloomFilter.h`，header-only  
**保留**: `bloom_parameters` + `bloom_filter` 两个类  
**去除**: `compressible_bloom_filter`（Curator 未使用）  
**迁移**: 所有符号移入 `namespace curator`

### 4.3 `cnpy.h/.cpp` — .npy 文件 I/O

**来源**: 外部引入轻量库，与 MiniRFANN 同款。可直接从 MiniRFANN 的 `main/` 目录复制 `cnpy.h` 和 `cnpy.cpp`（位于 `d:/23235/Documents/Aftergraduate/experiments/new-Baselines/MiniRFANN/main/`），或从 [rogersce/cnpy](https://github.com/rogersce/cnpy) 获取。MiniRFANN 的 `bptree_common.h` 中也包含 `readNpy<T>()` 模板封装可供参考。
**职责**: 读取 NumPy `.npy` 格式文件为 `std::vector<T>`，支持任意维度和 dtype

### 4.4 `kmeans.h/.cpp` — K-means 聚类

**职责**: 标准 Lloyd K-means 聚类（L2 距离），**必须包含 OpenMP 并行化**

> **关键**: 当前 FAISS `Clustering` 内部使用 OpenMP 并行化分配步骤。核心 Curator 文件仅在查询批量并行化处使用 OpenMP（第 527 行），**不包含**训练阶段的并行化。自实现 K-means 的分配步骤（每个向量 → 最近质心）必须用 `#pragma omp parallel for` 并行化，否则训练时间将比 FAISS 慢 4-8 倍。M 个 PQ 子空间的 K-means 也可在子空间级别并行（各子空间独立）。

```cpp
struct KMeansConfig {
    int niter = 20;
    int min_points_per_centroid = 1;
    int seed = 1234;
    int n_threads = 0;  // 0 = 使用全部可用核心
};

struct KMeansResult {
    std::vector<float> centroids;    // n_clusters × d，连续存储
    std::vector<int> assignments;    // n 个向量的聚类 ID
};

KMeansResult kmeans(int d, int n, const float* x,
                    int n_clusters, const KMeansConfig& cfg,
                    int stride = 0);  // stride=0 表示连续存储（d==stride）；stride>0 用于 PQ 子空间访问
```

**实现细节**:

1. **随机初始化**: 使用 `std::mt19937` 生成器（种子 `cfg.seed`），从 `[0, n)` 中无放回随机抽样 `n_clusters` 个索引作为初始质心。不实现 K-means++（Curator 的宽分支因子稀释了初始化的影响）。

2. **迭代**: 固定 `niter` 次迭代，**不检查收敛**（与 FAISS `Clustering` 行为一致）。每轮迭代：
   - **分配步骤** (`#pragma omp parallel for`): 每个向量 `x[i*d : (i+1)*d]` 与所有 `n_clusters` 个质心计算 `l2_sqr`，选择距离最小的聚类
   - **累积步骤** (`#pragma omp parallel for` over clusters): 各聚类原子累加归属于它的向量和（`float[d]`）及计数
   - **更新步骤** (串行): `centroid[j][dim] = sum[j][dim] / count[j]`

3. **空聚类处理**: 若某聚类分配了 0 个向量（`count[j] == 0`），将其质心设为全局均值加小扰动（`global_mean + epsilon * random_unit_vector`），其中 `epsilon = 1e-6`。这确保所有子节点质心有效，满足 `cluster_tree::build_tree` 依赖的 **"所有子节点质心有效"** 不变式。

4. **最小点数**: `min_points_per_centroid = 1`（与 FAISS 默认一致），不强制最小点数约束。

5. **线程安全**: 累积步骤使用 `#pragma omp atomic` 或 per-thread 局部累加数组（推荐后者——分配 `n_threads × n_clusters × d` 局部空间，最后 reduce），避免原子操作竞争开销。

**回退方案**: 通过 CMake option `CURATOR_USE_FAISS_CLUSTERING` 切换到 FAISS 实现

**预估**: ~250 行（含 OpenMP 并行化，比初版 +50 行）

### 4.5 `tree_node.h` — 树节点

**职责**: `TreeNode` 数据结构定义 + 构造函数 + BF 初始化/重算

```cpp
struct TreeNode {
    size_t level, sibling_id;
    TreeNode* parent;
    std::vector<TreeNode*> children;
    int_vid_t node_id;
    std::vector<float> centroid;     // d floats（替代 aligned_alloc+free）
    RunningMean variance;
    bloom_filter bf;
    std::unordered_map<int_lid_t, SortedList<int_vid_t>> shortlists;
    SortedList<int_vid_t> vector_indices;  // leaf only

    TreeNode(size_t level, size_t sibling_id, TreeNode* parent,
             const float* centroid, size_t d, size_t bf_cap, float bf_fp);
    // 析构: children 递归 delete（同原逻辑）
    bloom_filter make_bf() const;
    bloom_filter recompute_bf() const;
};
```

### 4.6 `cluster_tree.h/.cpp` — 聚类树构建

**职责**: 层次化 K-means 训练的递归逻辑

| 函数 | 说明 |
|------|------|
| `build_tree(root, n, x, cfg)` | 递归 K-means 聚类建树（替代原 `train_helper`） |
| `assign_to_leaf(root, x, d)` | 按最近质心将向量指派到叶子节点 |
| `get_vector_path(root, vid)` | 查询向量在树中的路径 |
| `find_assigned_leaf(root, vid)` | 从 vid 解码路径并定位叶子节点 |

**来源**: 从 `MultiTenantIndexIVFHierarchical::train_helper` / `assign_vec_to_leaf` / `get_vector_path` / `find_assigned_leaf` 提取

> **关键实现细节**:
>
> **根节点质心初始化**: 原 `train_helper` 对 `centroid == nullptr` 的节点计算全局均值（`new float[d]` 后填均值，[第 303-317 行](Curator/src/MultiTenantIndexIVFHierarchical.cpp#L303-L317)）。重构后 `TreeNode::centroid` 为 `std::vector<float>`，`build_tree` 需先调用 `root->centroid.resize(d)` 并填入全局均值，再进行聚类。子树节点的质心由 `new TreeNode(…, clus.centroids.data() + clus_id * d, …)` 在构造时拷贝。
>
> **重构消除了原代码的不一致**: 原代码中根节点用 `new float[d]`（非对齐），子节点用 `aligned_alloc(64, ...)`（64 字节对齐）。统一为 `std::vector<float>` 后两者一致。标量 `l2_sqr` 不需要对齐——若后续引入 SIMD，可对 `std::vector<float>` 使用 `alignas(64)` 或恢复 `aligned_alloc`。
>
> **子节点创建不变式**: `train_helper` 递归处理**所有** `n_clusters` 个子节点，包括分配了 0 个向量的子节点。此时子节点的质心从 `clus.centroids` 复制（有效），但 `variance` 从未更新、`vector_indices` 为空。重构后必须保持此行为——空子节点不参与搜索但保留在树结构中。所有非叶子节点**恰好有 `n_clusters` 个子节点**。此不变式被 `build_temp_index_for_filter` 依赖（通过位前缀二分定位，直接访问 `children[child_idx]`）。重构后在 `cluster_tree.cpp` 中添加 `assert(node->children.empty() || node->children.size() == cfg.n_clusters)`。

### 4.6a 匿名 namespace 函数的归属

当前 .cpp 匿名 namespace 中的搜索辅助函数重构后归属如下：

| 原函数 | 新归属 | 理由 |
|--------|--------|------|
| `RunningList` | `common.h`（模板类） | 被 `curator_index` 和 `temp_index` 两个模块共享 |
| `node_score()` | `distance.h`（内联） | 纯距离计算，无状态。**保留 `var_boost==0.0` 的短路路径**，因为 `temp_index` 搜索设置 `variance_boost=0` |
| `beam_search()` | `curator_index.cpp`（私有 static） | 搜索算法核心，依赖 `TreeNode`、`CuratorConfig`。**注意**: `beam_search` 当前通过 `index.variance_boost` 隐式获取方差提升系数；重构后作为静态函数，需显式传入 `variance_boost` 参数，tenant search 传非零值，temp index search 传 0 |
| `compute_child_scores_with_prefetch()` | `curator_index.cpp`（私有 static） | 含 prefetch 逻辑，且需访问 `index.d` 和 `index.variance_boost` |
| `compute_dists_with_prefetch()` | `curator_index.cpp`（**私有 static**） | 深度访问 `CuratorIndex` 内部状态（`raw_vectors_buffer_`、`vid_to_buffer_offset_`、`flash_finalized_`）。**不放入 `distance.h`**，而是参数化：接收 `get_vector_ptr(vid)→const float*` lambda，由 `CuratorIndex` 根据 flash→buffer 状态构造 lambda 后传入 |
| `compute_pq_distances()` | `pq_codec.h/.cpp` | PQ 距离计算核心，仅依赖 `PQCodec` 和 `vid_to_seq`（通过参数传入） |

> **注意**: `TreeNode` 的 `centroid` 从原始 `float*`(aligned_alloc) 改为 `std::vector<float>`。当前代码使用 64 字节对齐的 `aligned_alloc`（用于 SIMD `fvec_L2sqr`），重构后自实现 `l2_sqr` 为标量实现，对齐不再关键。若后续引入 SIMD 加速，可用 `alignas(64)` 或恢复 `aligned_alloc`。

### 4.7 `shortlist.h/.cpp` — 短列表管理

**职责**: 树节点的短列表分裂和合并，保证 `max_sl_size` 约束

| 函数 | 说明 |
|------|------|
| `split_shortlist(node, tid, max_size)` | 将超限短列表按 vid 前缀下推到子节点 |
| `try_merge_shortlists(node, tid, max_size)` | 尝试将子节点的同租户短列表合并到父节点 |

**来源**: 从原 `split_short_list` / `merge_short_list` 提取

### 4.8 `pq_codec.h/.cpp` — PQ 编解码

**职责**: Product Quantization 码本训练、编码、距离查表、ADC 重排，以及 PQ 编码的磁盘持久化（mmap 零拷贝访问）

> **实现要点**: PQ 训练 = M 个子空间各自做 K-means 聚类（每个子空间 `ksub = 1<<nbits` 个聚类中心）。**复用 `kmeans` 模块**进行子空间聚类。
>
> **子空间数据访问（stride 模式）**: 子空间 `m` 的向量数据从 `x + m*dsub` 开始，步进 `d` 个 float（即 `x[m*dsub + i*d + k]` 访问第 `i` 个向量的第 `m` 个子空间的第 `k` 维）。**不复制数据**——`kmeans` 模块接收 stride 参数（`stride = d`），在距离计算时按 stride 跳转。这避免了分配 M 个 `[N×dsub]` 临时矩阵，内存峰值与当前 FAISS 实现持平。

| 方法 | 说明 |
|------|------|
| `train(ntotal, vectors, d, M, nbits)` | 将向量拆为 M 段，每段独立 K-means → 拼成 `pq_codebook[M * ksub * dsub]` |
| `encode(vector, out_code)` | 逐子空间找最近质心 → 编码为 `uint8_t[M]` |
| `encode_all(vectors, out_codes)` | 全量编码 |
| `compute_distances(query, vids, codes, out)` | 构建 M×ksub 距离查表 → 逐向量查表累加 |
| `compute_exact_batch(query, vids, vectors, out)` | 精确 L2 距离（prefetch + 4 路展开） |
| `write_to_disk(path, codes, M, nbits)` | 将 PQ 编码写入磁盘文件（32 字节头 + 编码体） |
| `load_from_disk(path)` | mmap 磁盘文件以实现零拷贝只读访问 |
| `free_in_memory_codes()` | 释放内存中编码，回退到 mmap 磁盘访问 |
| `get_code(seq_idx)` | 透明地从内存或 mmap 获取编码（统一访问接口） |

**来源**: 从原 `compute_pq_distances`、`train_pq_codebook`、`ProductQuantizer`、`write_pq_codes_to_disk`、`load_pq_codes_from_disk`、`free_pq_codes`、`get_pq_code_by_seq` 中提取并重写  
**预估**: ~550 行（含子空间 K-means 训练 + 磁盘持久化，比初版多 ~200 行）

> **关键约束与设计决策**:
>
> **`nbits=8` 硬约束**: 当前所有配置默认 `nbits=8`，PQ 编码结构使用 `uint8_t[M]`，磁盘格式中 `code_bytes = M`（即 `pq_M * pq_nbits / 8`）。**重构后添加 `static_assert(nbits == 8, "Only nbits=8 is currently supported")`**，防止未来误用 `nbits≠8` 导致静默数据损坏。若后续需要支持 `nbits=4` 或 `nbits=16`，需引入位压缩/扩展层。
>
> **磁盘格式 Magic Number**: 原代码使用 `0x50514344`（"PQCD"）。重构后可选保留此 magic 或改为 `0x43555251`（"CURQ"）标识独立格式。若改动 magic，需同步更新 `load_from_disk` 中的验证逻辑。**默认建议**: 保持不变，降低迁移风险。
>
> **Profiling 的 PQ table build / distance compute 拆分**: 原代码用硬编码 20/80 比例拆分第一次 `compute_pq_distances` 调用的耗时（[第 1032-1037 行](Curator/src/MultiTenantIndexIVFHierarchical.cpp#L1032-L1037)）。重构后在 `pq_codec` 中分别暴露 `build_distance_table(query) → d_table` 和 `compute_distances(d_table, vids, codes) → (dist,vid) pairs` 两个独立方法，让 `curator_index` 分别计时，消除 20/80 启发式 hack。

### 4.9 `flash_store.h/.cpp` — Flash 磁盘存储

**职责**: 全精度向量的文件级 I/O 管理（打开/关闭/读写），**不持有**树结构映射

> **耦合说明**: `FlashStore` 只提供原始文件操作。`vid → flash_offset` 的映射查找（`vid_to_leaf_id_`、`leaf_node_id_to_seq_`、`vid_to_local_idx_`）由 `CuratorIndex` 持有并负责解析，解析后调用 `FlashStore` 的底层 read/write。`finalize_flash_storage()` 的**组织逻辑**（DFS 遍历叶子节点分配序号 → 按叶子分组向量 → 按 local_index 排序）在 `CuratorIndex` 中执行，组装完成后调用 `FlashStore` 的批量写入。

| 方法 | 说明 |
|------|------|
| `open(path)` / `close()` | 打开/关闭文件（`fopen`/`fclose`），**遵循 RAII**：构造时打开，析构时自动关闭 |
| `write_region(leaf_seq, region_size, vectors)` | 写入一个 leaf region（fseek + 连续 fwrite） |
| `write_all_leaves(leaf_vectors, leaf_region_size)` | 批量写入所有叶子：对每个叶子 fseek 到 region_start 后连续 fwrite。**保持与原代码相同的写入顺序**（按 leaf_seq 遍历，每个叶子内按 `local_index` 升序） |
| `read_vector(offset, d, out)` | 单向量 fseek + fread。**调用方负责提供 out 缓冲区**（避免 callee 堆分配） |
| `read_batch(requests, d, out)` | 按磁盘偏移排序后批量合并读取（合并阈值 65536 字节） |
| `truncate(total_size)` | 预分配文件大小（`ftruncate` 或 fseek+fwrite） |
| `is_open()` | 文件是否已打开 |

**来源**: 从原 `finalize_flash_storage` / `load_vector_from_flash` / `load_vectors_from_flash_batched` 提取（纯文件 I/O 部分）  
**预估**: ~250 行（纯文件 I/O 层）

> **系统依赖**: `FlashStore` 需要以下 POSIX 头文件（Linux/WSL 标准可用，非 Windows 可移植）:
> - `<cstdio>` — `fopen`, `fclose`, `fseek`, `fread`, `fwrite`, `fflush`
> - `<unistd.h>` — `ftruncate`, `fileno`, `getpid`（`getpid` 仅用于生成默认 flash 文件名）
> - `<sys/mman.h>` — `mmap`, `munmap`（仅 `pq_codec` 的磁盘持久化路径需要，FlashStore 自身不使用）
> - `<sys/stat.h>`, `<fcntl.h>` — `open`, `fstat`（仅 `pq_codec` 的 mmap 路径需要）
> - `<filesystem>` — `std::filesystem::create_directories`（创建父目录）
>
> 若未来需要 Windows/MSVC 兼容，可将 `ftruncate`/`mmap` 替换为 Win32 API 等效调用（`SetFilePointerEx`+`SetEndOfFile`、`CreateFileMapping`+`MapViewOfFile`）。重构第一阶段仅支持 Linux/WSL 平台。

> **单向量读取的堆分配优化**: 原 `compute_vector_distance` 在 flash 路径中每次调用执行 `std::vector<float> x(d)` 堆分配（[第 1201 行](Curator/src/MultiTenantIndexIVFHierarchical.cpp#L1201)）。ADC rerank 对 `k × rerank_topk_factor` 个候选逐一调用此函数（如 `k=10, factor=4 → 40 次分配`）。重构后在 `CuratorIndex` 中添加 **`static thread_local std::vector<float> scratch_`**（`resize(d)` 一次），`compute_vector_distance` 复用此 scratch buffer，消除 per-call 堆分配。
>
> **选择 `thread_local` 而非 `mutable` 成员的理由**:
> - `mutable` 成员在 `#pragma omp parallel for` 并行查询中引发竞态条件（多线程写同一缓冲区）
> - `#pragma omp critical` 保护会串行化 flash 读取，抵消并行查询的收益
> - `thread_local` 每个线程独立拥有缓冲区，零同步开销，且线程数受 OpenMP 限制（通常 ≤ CPU 核心数），内存开销可忽略（每线程 `d × 4` 字节 ≈ 数 KB）
>
> **初始化策略**: `scratch_` 在首次使用时 resize（延迟初始化），因为 `d` 在 `CuratorIndex` 构造时已知但 `thread_local` 变量不能使用构造函数的初始化列表。实现方式：
> ```cpp
> // curator_index.cpp
> static thread_local std::vector<float> tl_scratch;
> 
> float CuratorIndex::compute_vector_distance(const float* query, int_vid_t vid) const {
>     if (tl_scratch.size() != cfg_.d) {
>         tl_scratch.resize(cfg_.d);
>     }
>     // ... use tl_scratch.data() as the read buffer
> }
> ```

### 4.10 `temp_index.h/.cpp` — 临时索引

**职责**: 为 bitmap filter 搜索构建轻量级临时树，基于排序后的 vid 列表快速构建

| 函数 | 说明 |
|------|------|
| `build_temp_index(root, sorted_vids, n_clusters, out_nodes)` | 构建 `TempIndexNode` 列表 |
| `search_temp_index(nodes, vids, query, cfg, out)` | 在临时索引上执行 beam + frontier 搜索 |

**来源**: 从原 `complex_predicate::build_temp_index_for_filter` / `search_temp_index` 提取

> **关键设计细节**:
>
> **分支因子获取**: 原 `build_temp_index_for_filter` 通过 `index->n_clusters` 获取分支因子并用于两层循环（[第 71 行](Curator/src/MultiTenantIndexIVFHierarchical.cpp#L71) 和 [第 96 行](Curator/src/MultiTenantIndexIVFHierarchical.cpp#L96)），同时假设 `curr_node->children[child_idx]` 对 `0 ≤ child_idx < n_clusters` 均有效。重构后 `build_temp_index` 显式接收 `n_clusters` 参数（来自 `CuratorConfig`），调用方负责保证 `n_clusters ≤ node->children.size()`。此不变式由 `cluster_tree::build_tree` 保证（所有非叶子节点恰好有 `n_clusters` 个子节点，参见 §4.6）。
>
> **`TempIndexNode::centroid` 非拥有指针**: `centroid` 指向主聚类树中 `TreeNode::centroid`（`std::vector<float>`）。由于 `TreeNode::centroid` 在构造时赋值后不再 reallocate，指针在索引生命周期内有效。`temp_index.h` 中需显式注释此生命周期约束："Non-owning pointer to TreeNode::centroid — valid for the lifetime of the main cluster tree."
>
> **`build_temp_index` 对子节点的访问模式**: 即使视频率分配了 0 个向量的子节点（空 `TreeNode`），位前缀二分仍扫描全部分支（0→n_clusters-1），仅当 `range.first != range.second` 时才递归进入。这与原行为一致。

### 4.11 `complex_predicate.h/.cpp` — 复杂谓词

**职责**: AND/OR/NOT 表达式的解析和求值（Polish Notation）

**完整保留**:
- `tokenize_formula()`: 字符串分词
- `parse_formula()`: Polish Notation → `ExprNode` 抽象语法树
- `evaluate_formula()`: 直接在 access_list 上布尔求值（模板函数，**改为 header-only**）
- `StateNode` / `VarMapNode` / `State` / `VarMap` 符号执行体系
- `ExprNode` 继承体系: `VariableNode`, `NotNode`, `AndNode`, `OrNode`
- `buffer_intersect` / `buffer_union` / `buffer_difference`: Buffer 集合操作

**迁移**: 所有符号从 `faiss::complex_predicate` 移入 `curator::predicate`

**迁移注意事项**:
1. **模板实例化**: 原 `complex_predicate.cpp` 中有 4 个 `evaluate_formula` 的显式模板实例化（第 376-390 行），使用 `std::unordered_set<tid_t>` 和 `std::vector<tid_t>`。重构后 `evaluate_formula` 模板定义移入 `.h` 文件（header-only），**消除显式实例化需求**。调用方 include 头文件后自动获得正确的模板实例化。`tid_t` 别名保留在 `common.h` 中以保证兼容。
2. **异常宏替换**: `complex_predicate.cpp` 中 `make_state` 函数有 3 处 `FAISS_THROW_MSG`（第 60/69/77 行），Phase 3 迁移时必须替换为 `CURATOR_THROW_MSG`。
3. **Buffer 类型**: `Buffer = std::vector<vid_t>`（`vid_t = uint64_t = int_vid_t`）定义在 `common.h` 中，`complex_predicate.h` 通过 include `common.h` 获取。

### 4.12 `distance.h` — 距离函数

**职责**: 各种距离计算函数（header-only，内联）

| 函数 | 说明 |
|------|------|
| `l2_sqr(x, y, d)` | 标量 L2 平方距离（直接替换 `fvec_L2sqr`，相同算法） |
| `l2_sqr_4way(x, y0,y1,y2,y3, d)` | **真正的** 4 路展开 L2（4 个独立累加器，减少依赖链，比原代码的"伪展开"——连续 4 次独立 `fvec_L2sqr` 调用——更高效） |
| `node_score(node, x, d, var_boost)` | 树节点评分 = dist - boost×var。**保留 `var_boost==0.0` 的短路路径**（直接返回 dist，不调用 `variance.get_mean()`），因为 temp_index 搜索全部使用 `var_boost=0` |
| `compute_batch_dists(x, vids, get_vec_fn, out)` | 参数化的批量距离计算，接收 `get_vector_ptr(vid)→const float*` lambda（由 `CuratorIndex` 根据 flash→buffer 状态构造），含 prefetch + 4 路计算 |

> **`RunningList` 类**: 当前定义于 .cpp 匿名 namespace（[第 568-662 行](Curator/src/MultiTenantIndexIVFHierarchical.cpp#L568-L662)），同时被 `curator_index.cpp`（tenant/unfiltered search 路径）和 `temp_index.cpp`（bitmap filter search 路径）使用。重构后移入 `common.h` 作为模板类或公共工具类，供两个模块共享。

> **RunningList vs HeapForL2 性能说明**: `RunningList` 使用 O(k) 有序插入，FAISS `HeapForL2` 使用 O(log k) 堆操作。对于 k≤100 的典型场景，O(k) ≈ 100 次移位操作（微秒级），差异可忽略。Unfiltered search 路径扫描大量向量（每个 `maxheap_replace_top` 调用原为 O(log k)），替换为 `RunningList` 后 P99 延迟可能增加 1-3 微秒，**在 ±2% 整体查询延迟范围内**。若后续需要极致性能，可 vendor FAISS 的 `Heap.h`（独立性强，~500 行模板）到 `src/` 目录。

### 4.13 `profiling.h` — 性能分析

**职责**: 搜索各阶段的计时和数据收集（header-only 结构体）

> **线程安全说明**: `SearchProfile` 和 `profiling_on_` 是 `mutable` 成员。单线程查询（`search_one` / `search_with_bitmap_filter` 逐个调用）下安全。**并行批量查询模式**（`batch_query=true`，`#pragma omp parallel for`）下多线程写同一 `SearchProfile` 会导致数据损坏。此问题在原始代码中已存在（[第 527 行](Curator/src/MultiTenantIndexIVFHierarchical.cpp#L527)），重构后**不修复**——并行查询时 profiling 数据仅供粗略参考，不应依赖其精度。若未来需要线程安全的 profiling，可在 `search()` 中使用 `#pragma omp critical` 保护写入或使用 thread-local 累加器。

```cpp
struct SearchProfile {
    char query_type[32];

    // 标准单租户路径
    double beam_search_ms = 0;
    int    beam_layers_visited = 0;
    int    beam_nodes_scored = 0;
    double frontier_search_ms = 0;
    int    frontier_nodes_popped = 0;
    int    frontier_shortlists_scanned = 0;
    int    frontier_children_expanded = 0;
    double pq_table_build_ms = 0;
    double pq_distance_compute_ms = 0;
    double exact_distance_compute_ms = 0;
    double candidate_merge_ms = 0;
    double rerank_ms = 0;
    int    rerank_count = 0;
    double total_search_ms = 0;

    // Bitmap filter 路径
    double preproc_ms = 0;
    double sort_ms = 0;
    double build_temp_index_ms = 0;
    double search_ms = 0;
    size_t qualified_labels_count = 0;
    size_t temp_nodes_count = 0;
};

// 组件级内存分解（Python get_memory_breakdown() 的返回类型）
struct MemoryBreakdown {
    size_t num_tree_nodes = 0;
    size_t tree_node_attrs_bytes = 0;
    size_t centroids_bytes = 0;
    size_t bloom_filter_bytes = 0;
    size_t shortlists_overhead_bytes = 0;
    size_t shortlists_payload_bytes = 0;
    size_t vector_indices_bytes = 0;
    size_t id_allocator_bytes = 0;
    size_t tenant_id_allocator_bytes = 0;
    size_t pq_codebook_bytes = 0;
    size_t pq_codes_bytes = 0;
    size_t flash_index_bytes = 0;
    size_t raw_vectors_buffer_bytes = 0;
    size_t temp_index_cache_bytes = 0;
    size_t temp_qualified_vecs_bytes = 0;
    size_t total_bytes = 0;
};
```

### 4.14 `curator_index.h/.cpp` — 主类

**职责**: Curator 索引的唯一入口类，编排所有子模块

```cpp
namespace curator {

struct CuratorConfig {
    size_t d;
    size_t n_clusters;
    size_t bf_capacity = 1000;
    float  bf_false_pos = 0.01f;
    size_t max_sl_size = 128;
    size_t clus_niter = 20;
    size_t max_leaf_size = 128;
    // PQ
    size_t pq_M = 16;
    size_t pq_nbits = 8;
    bool   pq_enabled = true;
    bool   pq_use_adc_rerank = false;
    size_t pq_rerank_topk_factor = 4;
    bool   persist_pq_codes = false;
    std::string pq_codes_path;
    // Flash
    bool   use_flash_storage = true;
    std::string flash_path;
    // Search
    size_t nprobe = 3000;
    float  prune_thres = 1.6f;
    float  variance_boost = 0.4f;
    size_t search_ef = 128;
    size_t beam_size = 2;
    bool   use_temp_index_caching = true;
    bool   batch_query = false;  // 替代原 getenv("BATCH_QUERY")
};

class CuratorIndex {
public:
    explicit CuratorIndex(const CuratorConfig& cfg);
    ~CuratorIndex();

    // ── 构建 ──
    void train(size_t n, const float* x);
    void add_vector(const float* x, ext_vid_t label);
    void grant_access(ext_vid_t label, ext_lid_t tenant);
    void flush();  // 训练 PQ + 写入 Flash

    // ── 查询（均为单查询接口；批量查询由 CLI/main.cpp 外循环驱动）──
    void search(const float* x, size_t k, ext_lid_t tenant,
                float* distances, ext_vid_t* labels) const;
    void search_unfiltered(const float* x, size_t k,
                           float* distances, ext_vid_t* labels) const;
    void search_with_bitmap(const float* x, size_t k,
                            const ext_vid_t* qualified, size_t n_qualified,
                            float* distances, ext_vid_t* labels) const;
    // 优化版本：跳过 ext→int ID 转换和排序阶段，直接传入已排序的内部 vid
    void search_with_bitmap(const float* x, size_t k,
                            const int_vid_t* sorted_qualified, size_t n_qualified,
                            float* distances, ext_vid_t* labels) const;

    // > **API 变更说明**: 当前 C++ 的 `search()`/`search_with_bitmap_filter()` 接受
    // > `idx_t n` 参数支持批量查询。重构后改为**单查询接口**（每次处理一个 query
    // > vector），批量查询由 CLI/main.cpp 的外层循环（配合 `#pragma omp parallel for`）
    // > 驱动。理由：(1) 单查询方法更易测试和调优；(2) 并行策略（inter-query vs
    // > intra-query）由调用方根据场景选择；(3) 与 MiniRFANN 接口风格一致。

    // ── 管理 ──
    bool revoke_access(ext_vid_t label, ext_lid_t tenant);
    bool remove_vector(ext_vid_t label);  // 保留 stub（当前为 NOT_SUPPORTED），供 API 兼容
    ext_lid_t build_filter_index(const std::string& predicate,
                                  const ext_vid_t* qualified, size_t n);
    ext_lid_t get_filter_label(const std::string& predicate) const;

    // ── 运行时参数调优 ──
    void set_rerank_params(bool enabled, size_t topk_factor);
    bool get_rerank_enabled() const;
    size_t get_rerank_topk_factor() const;

    // ── ID 映射导出（Python/调试用） ──
    void get_label_to_vid_mapping(const ext_vid_t* labels, size_t n,
                                   int_vid_t* vids_out) const;

    // ── 内存 ──
    size_t memory_bytes() const;
    MemoryBreakdown memory_breakdown() const;
    void print_tree_info() const;

    // ── 性能 ──
    void enable_profiling(bool on);
    const SearchProfile& last_profile() const;

    // ── 缓存 ──
    size_t cached_temp_index_memory() const;
    void clear_temp_index_cache();

    // ── 存取器 ──
    const CuratorConfig& config() const { return cfg_; }
    size_t ntotal() const { return ntotal_; }

private:
    CuratorConfig cfg_;
    TreeNode* root_ = nullptr;
    size_t ntotal_ = 0;

    // ID 映射
    IdMapping<ext_vid_t, int_vid_t> vid_map_;
    IdAllocator<ext_lid_t, int_lid_t> tid_map_;

    // 原始向量缓冲（flush 前）
    std::vector<float> raw_buffer_;
    std::unordered_map<int_vid_t, size_t> vid_to_buf_offset_;
    std::unordered_map<int_vid_t, int_vid_t> vid_to_leaf_id_;
    std::unordered_map<int_vid_t, size_t> vid_to_local_idx_;
    std::vector<int_vid_t> seq_to_vid_;
    std::unordered_map<int_vid_t, size_t> vid_to_seq_;

    // Flash 存储元数据（finalize_flash_storage 时填充）
    std::unordered_map<int_vid_t, size_t> leaf_node_id_to_seq_;
    size_t num_leaves_ = 0;
    size_t leaf_region_size_ = 0;

    // 子模块
    PQCodec pq_;
    FlashStore flash_;
    bool flash_finalized_ = false;

    // 复杂谓词辅助
    std::unordered_map<std::string, ext_lid_t> filter_to_label_;

    // 临时索引缓存
    std::unordered_map<int_lid_t, std::vector<TempIndexNode>> temp_indexes_;
    std::unordered_map<int_lid_t, std::vector<int_vid_t>> qualified_cache_;

    // Profiling
    mutable SearchProfile profile_;
    mutable bool profiling_on_ = false;

    // ── 内部辅助 ──
    void grant_access_impl(TreeNode* node, int_vid_t vid, int_lid_t tid);
    void batch_grant_access(const std::vector<int_vid_t>& vids, int_lid_t tid);
    float compute_vector_distance(const float* query, int_vid_t vid) const;
    // 实现逻辑: flash → raw_buffer_ 两级回退（去除原 storage 兼容路径）。
    // 使用 static thread_local std::vector<float> 避免 per-call 堆分配
    // 且保证多线程并行安全（每线程独立缓冲区）。
    std::vector<int_vid_t> find_all_qualified_vecs(const std::string& filter) const;
    std::string convert_complex_predicate(const std::string& filter) const;
    bool get_cached_temp_index_data(int_lid_t tid,
                                     std::vector<TempIndexNode>*& nodes,
                                     std::vector<int_vid_t>*& vids) const;
    // 注意：当前代码中此方法接收 ext_lid_t 参数，但缓存 map 的 key 是 int_lid_t
    // （两者底层均为 int16_t，碰巧兼容）。重构后统一为 int_lid_t，消除类型不一致。
};

} // namespace curator
```

### 4.15 `main.cpp` — CLI 入口

> **重要**: 当前代码不支持索引序列化（`save_to_file`/`load_from_file` 均为 `NotImplementedError`），因此重构第一阶段**仅支持 `bench` 模式**（单次进程完成构建+检索）。独立的 `build`/`search` 模式留待后续实现（需设计全索引序列化格式）。

```bash
# bench 模式: 构建 + 查询 + 统计（一站式，单进程）
./curator bench \
    --train_vecs     data/train_vecs.npy \
    --train_access   data/train_access.npy \
    --queries        data/query_vecs.npy \
    --query_labels   data/query_labels.npy \
    --config         config.json \
    --k 10 \
    --batch-query \
    --output         results.json
```

**bench 模式流程**:
1. 读取所有 `.npy` 数据（cnpy）
2. `CuratorIndex::train()` → `add_vector()` × N → `grant_access()` × M（从 `train_access.npy` 读取）→ `flush()`
3. 遍历查询向量，根据 `query_labels.npy` 中的 tenant_id 执行 `search()` 或 `search_unfiltered()`（tid=-1），收集 top-k 结果
4. 将搜索结果写入 `results.json`：
   ```json
   {
     "config": { "d": 384, "nlist": 64, ... },
     "build_time_s": 12.3,
     "memory_bytes": 123456789,
     "queries": [
       {
         "idx": 0,
         "tenant_id": 5,
         "labels": [100, 200, 300, ...],
         "distances": [0.12, 0.34, 0.56, ...]
       },
       ...
     ]
   }
   ```
5. stdout 输出汇总统计（构建时间、查询延迟分布、内存占用）

> **`config.json` 格式**:
> ```json
> {
>   "d": 384,
>   "n_clusters": 64,
>   "bf_capacity": 1000,
>   "bf_false_pos": 0.01,
>   "max_sl_size": 128,
>   "clus_niter": 20,
>   "max_leaf_size": 128,
>   "pq_M": 24,
>   "pq_nbits": 8,
>   "pq_enabled": true,
>   "pq_use_adc_rerank": false,
>   "pq_rerank_topk_factor": 4,
>   "persist_pq_codes": false,
>   "pq_codes_path": "",
>   "use_flash_storage": true,
>   "flash_path": "",
>   "nprobe": 3000,
>   "prune_thres": 1.6,
>   "variance_boost": 0.4,
>   "search_ef": 128,
>   "beam_size": 2,
>   "use_temp_index_caching": true,
>   "batch_query": false
> }
> ```
> 所有字段均为可选，未指定时使用 `CuratorConfig` 的默认值。

> **`.npy` 文件格式约定**:
>
> | 文件 | Shape | Dtype | 说明 |
> |------|-------|-------|------|
> | `train_vecs.npy` | `[N, d]` | float32 | 训练向量矩阵 |
> | `train_access.npy` | `[M, 2]` | int32 | COO pairs: 每行 `[vid, tid]`，表示 vid 被授权给 tid |
> | `query_vecs.npy` | `[Q, d]` | float32 | 查询向量矩阵 |
> | `query_labels.npy` | `[Q]` | int32 | 每个查询的 tenant_id（-1 = 无过滤） |
> | `ground_truth.npy` | `[Q, k]` | int32 | ground truth labels（仅 Python 端加载，用于 recall 计算） |

> **注意**: recall 计算由 Python `run_experiment.py` 负责（读取 `results.json` + `ground_truth.npy` 比对）。C++ 不加载 `ground_truth.npy`，保持输出简洁。

> **查询并行化**: 当前代码通过 `getenv("BATCH_QUERY")` 环境变量控制 OpenMP 查询并行化。重构后改为 CLI 参数 `--batch-query`（或 `CuratorConfig` 中的 `batch_query` 字段），不再使用环境变量。

> **设计决策**: recall 计算保留在 Python 端（`run_experiment.py` 读取 `results.json` + `ground_truth.npy`），C++ 仅输出原始搜索结果。理由：(1) 减少 C++ 代码量；(2) Python 已有成熟的 recall 计算逻辑；(3) 便于后续在 Python 端灵活调整评估指标。

---

## 五、调用链条设计

### 5.1 构建流程

```
main() [bench 模式]
  └─ 读取 .npy 文件 (cnpy::npy_load)
  └─ CuratorIndex::CuratorIndex(cfg)
       ├─ 创建 root TreeNode
       ├─ cfg.n_clusters ≤ MAX_BRANCH_FACTOR 检查
       └─ cfg.max_leaf_size ≤ MAX_LEAF_SIZE 检查
  └─ CuratorIndex::train(n, x)
       └─ cluster_tree::build_tree(root, n, x, cfg)
            ├─ 根节点: centroid.resize(d) → 计算全局均值 → 填入 centroid
            ├─ kmeans() × N 次递归               // 自实现
            ├─ new TreeNode() × n_clusters         // 所有非叶子节点恰好 n_clusters 个子节点
            ├─ assert: node.children.empty() || node.children.size() == cfg.n_clusters
            └─ 递归 build_tree 到子节点
  └─ for each vector:
       ├─ CuratorIndex::add_vector(x, label)
       │    ├─ cluster_tree::assign_to_leaf()    // 逐层最近质心
       │    ├─ 分配 int_vid_t (路径位编码)
       │    ├─ 存入 raw_buffer_
       │    └─ 向上更新 TreeNode::variance
       └─ CuratorIndex::grant_access(label, tid) (by reading train_access.npy)
            └─ grant_access_impl(node, vid, tid)
                 ├─ 叶子节点: 短列表插入
                 ├─ 非叶子: 短列表满 → shortlist::split()
                 └─ 向上传播 bf.insert(tid)
  └─ CuratorIndex::flush()
       ├─ pq_codec::train() + encode_all()       // 训练+编码 PQ
       ├─ flash_store 写入全量向量                  // 写入磁盘（保持现有 region 布局）
       └─ 释放 raw_buffer_
  └─ for each query: CuratorIndex::search()
  └─ 输出 results.json
```

### 5.2 单租户查询流程

```
CuratorIndex::search(x, k, tenant_id)
  ├─ 1. beam_search() — 从根节点 Beam Search
  │    └─ node_score() × N  (distance::)
  │    └─ BF 检查 + shortlist 检查
  ├─ 2. frontier_search() — 优先队列扩展
  │    ├─ 叶子节点: 计算短列表距离
  │    │    ├─ [if PQ] pq_codec::compute_distances()
  │    │    └─ [else]  distance::compute_batch_dists()
  │    ├─ 合并到 RunningList (top-search_ef)
  │    └─ 非叶子: compute_child_scores → 入队
  ├─ 3. [if ADC] pq_codec::rerank(top_k×factor)
  │    └─ flash_store::load_vector() × N
  └─ 4. 输出 top-k: int_vid_t → ext_vid_t (vid_map_)
```

### 5.2a 无过滤查询流程（`search_unfiltered`）

当前 unfiltered search（原 `search_one` 无 tid 参数，第 1135-1196 行）使用**不同于**租户搜索的算法：

```
CuratorIndex::search_unfiltered(x, k, distances, labels)
  ├─ 1. 优先队列遍历树节点（MinHeap<Candidate>，非 beam search）
  │    └─ node_score() = dist - variance_boost × var
  │    └─ 出队 → 展开子节点并入队 → 直到 n_cand_vecs >= nprobe
  │    └─ 收集叶子节点到 buckets
  ├─ 2. 按质心距离排序 buckets
  ├─ 3. 剪枝: prune_thres × min_buck_dist（默认 1.6×）
  ├─ 4. 对每个幸存 bucket:
  │    └─ 遍历所有 vector_indices → compute_vector_distance() → 插入 RunningList
  └─ 5. RunningList 已是升序，直接取前 k 个（无需 heap_reorder）
```

> **关键差异**: unfiltered search 使用 `nprobe` + `prune_thres` 控制剪枝（而非 `search_ef` + `beam_size`），且直接扫描叶子节点的 `vector_indices`（不走 shortlists）。重构后将其作为 `CuratorIndex::search_unfiltered()` 的独立实现路径，使用 `RunningList` 替代 FAISS 堆操作。

### 5.3 Bitmap Filter 查询流程

```
CuratorIndex::search_with_bitmap(x, k, qualified_labels, n_q)  // 单查询接口
  ├─ 1. ext_vid_t → int_vid_t (vid_map_)
  ├─ 2. std::sort(qualified_vids)
  ├─ 3. temp_index::build_temp_index(root, sorted_vids, cfg_.n_clusters)
  │    └─ 按 vid 位前缀二分分组（循环 0→n_clusters-1）→ 递归构建 TempIndexNode
  │    └─ 依赖不变式: 所有非叶子节点 children.size() == n_clusters
  ├─ 4. temp_index::search_temp_index(nodes, vids, query)
  │    ├─ Beam search (variance_boost=0，通过参数传入)
  │    └─ Frontier search (RunningList top-search_ef)
  └─ 5. 输出 top-k (int_vid_t → ext_vid_t via vid_map_)

// 批量查询由 CLI/main.cpp 外循环驱动:
//   for each query: search_with_bitmap(x_i, k, qualified_labels, ...)
//   或 #pragma omp parallel for（batch_query=true 时）
```

### 5.4 过滤器索引构建流程

```
CuratorIndex::build_filter_index(predicate, qualified_labels, n)
  ├─ 1. ext_vid_t → int_vid_t (vid_map_)
  ├─ 2. 分配新的内部 tenant ID (tid_allocator_)
  ├─ 3. [if use_temp_index_caching]
  │    ├─ std::sort(qualified_vids)
  │    ├─ temp_index::build_temp_index(root, sorted_vids)
  │    └─ 缓存 temp_index_nodes + qualified_vids (key = int_lid_t)
  └─ 4. [else]
       └─ batch_grant_access(vids, filter_tid)  // 直接写入主索树短列表
```

### 5.5 TreeNode 节点 ID 位编码规则

当前实现使用 `int_vid_t`（uint64_t）编码向量在树中的路径：
- 根节点 `node_id = 0`
- 第 L 层子节点：`node_id = parent.node_id | (sibling_id << shift)`，其中 `shift = 64 - L * MAX_BRANCH_FACTOR_LOG2`
- 叶子向量 vid：`vid = leaf.node_id | (local_vid << (64 - leaf.level * MAX_BRANCH_FACTOR_LOG2 - MAX_LEAF_SIZE_LOG2))`

重构时**完整保留**此编码规则，不修改。`cluster_tree.cpp` 中的 `get_vector_path()` / `find_assigned_leaf()` 依赖此规则。

### 5.6 调试与诊断工具

以下调试函数保留但非关键路径：
- `sanity_check()`: 递归验证 BF 一致性、短列表约束、合并正确性 → 归入 `curator_index.cpp` 私有方法（或 `diagnostics.cpp` 若拆分）
- `check_bloom_filter(node)`: 递归检查每个节点的 BF 是否等于 `recompute_bloom_filter()` 的结果 → 作为 `sanity_check` 的 private static 辅助函数
- `check_shortlists(node)`: 递归验证短列表的 3 条不变式——(1) 不超过 `max_sl_size`；(2) 合并正确（若所有子节点短列表之和 ≤ max_sl_size 则应已合并到父节点）；(3) 同一租户在任意路径上不重复出现 → 作为 `sanity_check` 的 private static 辅助函数
- `print_tree_info()`: 打印树结构统计（节点数、深度分布、叶子大小直方图） → 保留为 `CuratorIndex` 公开方法
- `locate_vector()`: 按 label 查找并打印路径 → 删除（可被 `get_vector_path()` 替代）

---

## 六、C++ 与 Python 分工总览

### 6.1 C++ 完成的工作

| 模块 | 文件 | 完成的工作 |
|------|------|------------|
| 类型与工具 | `common.h` | `ext_vid_t`, `int_vid_t`, `IdAllocator`, `IdMapping`, `SortedList`, `RunningMean` |
| Bloom Filter | `bloom_filter.h` | `bloom_parameters`（参数优化计算）、`bloom_filter`（插入、查询、交集、并集） |
| .npy I/O | `cnpy.h/.cpp` | `readNpy<T>()` 模板函数，读取 NumPy 文件为 `std::vector<T>` |
| K-means | `kmeans.h/.cpp` | 标准 Lloyd 算法，L2 距离，可选的 FAISS 回退 |
| 树节点 | `tree_node.h` | `TreeNode`（层级、质心、短列表、BF、向量索引）、构造/析构、BF 重算 |
| 聚类树 | `cluster_tree.h/.cpp` | `build_tree()`（递归聚类建树）、`assign_to_leaf()`（指派向量）、`get_vector_path()` |
| 短列表 | `shortlist.h/.cpp` | `split_shortlist()`（下推）、`try_merge_shortlists()`（上合） |
| PQ 编解码 | `pq_codec.h/.cpp` | 码本训练、向量编码、查表距离计算、ADC 重排 |
| Flash 存储 | `flash_store.h/.cpp` | 按 leaf 分 region 写入磁盘、单读、批量合并读 |
| 临时索引 | `temp_index.h/.cpp` | `build_temp_index()`（vid 前缀二分建树）、`search_temp_index()`（beam + frontier 搜索） |
| 复杂谓词 | `complex_predicate.h/.cpp` | 完整表达式树（ExprNode 体系）、符号执行（State/VarMap 体系）、布尔求值 |
| 距离函数 | `distance.h` | `l2_sqr()`、`l2_sqr_4way()`、`node_score()`、`compute_batch_dists()` |
| 性能分析 | `profiling.h` | `SearchProfile` 结构体（~25 字段，完整保留） |
| 主类 | `curator_index.h/.cpp` | `CuratorIndex`：构建、查询、管理、内存统计、性能分析的编排 |
| 诊断工具 | `diagnostics.h/.cpp` | [可选] `sanity_check`（BF一致性+短列表约束验证）、`memory_breakdown`（组件级内存分解）、`print_tree_info`（树结构统计）。当 `curator_index.cpp` 超过 1000 行时从此提取 |
| CLI 入口 | `main.cpp` | 命令行解析、`.npy` 文件 I/O、`bench` 流程编排、JSON 结果输出 |

### 6.2 Python 完成的工作

| 脚本 | 完成的工作 |
|------|------------|
| `preprocess_train.py` | **(1) 格式转换**: `.pkl` → `.npy`（train_mds 转 COO pairs），C++ 可通过 cnpy 直接读取；**(2) 可选预处理**: 使用 FAISS 提前训练 PQ 码本、层次 K-means 聚类，产出 centroids 供 C++ 使用 |
| `prepare_queries.py` | **(1) 查询生成**: 根据 selectivity 配置生成带标签的查询向量；**(2) Ground Truth 计算**: 暴力搜索计算 ground truth 用于 recall 评估 |
| `run_experiment.py` | **(1) 实验编排**: 格式转换（.pkl→.npy）→ `subprocess` 调用 C++ `curator bench` → 读取 `results.json`；**(2) recall 计算**: 对比 `results.json` 与 `ground_truth.npy`（Python 端）；**(3) 报告生成**: 延迟分布、recall@k 统计、per-bucket 分析、JSON 报告 |

> **注意**: `curator.py`（Python Curator wrapper 类）已从重构方案中移除。`run_experiment.py` 直接通过 `subprocess` 与 C++ 可执行文件交互，无需中间 wrapper 层。代码结构更清晰——Python 端仅保留 3 个脚本，职责分明。

### 6.3 Python 是否调用 C++ 代码？

**Python 调用 C++ 可执行文件（通过 `subprocess`），而非调用 C++ 库函数。** 两者是**离线解耦、通过文件系统交互**的独立进程：

```
┌─────────────────────┐          ┌─────────────────────┐
│   Python 离线预处理    │          │   C++ 在线检索引擎     │
│                     │          │                     │
│  preprocess_train   │  .npy    │  curator bench       │
│  ├─ .pkl→.npy 转换   │─────────▶│  ├─ 读取全部 .npy     │
│  ├─ 数据校验          │  文件     │  ├─ 建树+插入+授权    │
│  └─ 保存              │          │  ├─ flush (PQ+Flash) │
│                     │          │  ├─ 批量查询           │
│  prepare_queries    │          │  └─ 输出 results.json │
│  ├─ 查询向量提取      │─────────▶│                     │
│  ├─ 属性/标签生成     │          │  输出:                │
│  └─ GT 计算          │          │  results.json        │
│                     │          │  (每个 query 的        │
│  run_experiment     │subprocess│   top-k labels+dist)  │
│  ├─ 调用 curator     │─────────▶│                     │
│  ├─ 读取 results.json│◀─────────│                     │
│  └─ recall 分析+报告  │          │                     │
└─────────────────────┘          └─────────────────────┘
```

**数据交互的桥梁是 `.npy` 文件**：
- Python 端通过 `numpy.save()` 写入
- C++ 端通过 `cnpy::npy_load()` 读取

**这种设计的优势**：
1. 两种语言完全解耦，可独立开发、调试
2. C++ 无 FAISS 编译依赖（唯一外部依赖是 cnpy）
3. Python 端可利用 FAISS/NumPy 丰富的科学计算生态
4. C++ 端专注于低延迟的在线构建和检索
5. 与 MiniRFANN（REFERENCE_STYLE.md）的架构模式完全一致

---

## 七、重构执行步骤

### Phase 0: 准备工作

| # | 步骤 | 产出 | 预估工作量 |
|---|------|------|------------|
| P0.1 | 创建 `Curator/src/` 下所有新文件骨架 | 24 个空 .h/.cpp 文件（14 .h + 10 .cpp），若启用 diagnostics 则为 26 个 | 0.5h |
| P0.2 | 引入 `cnpy.h/.cpp` 到 `src/` | .npy 文件读取能力 | 0.5h |

### Phase 1: 公共层提取

| # | 步骤 | 来源 | 产出 | 工作量 |
|---|------|------|------|--------|
| P1.1 | `common.h` — 类型别名 + 常量 + `IdAllocator`/`IdMapping`/`SortedList`/`RunningMean` 模板 | 原 `MetricType.h` + 原头文件工具类段 | ~200 行 | 1h |
| P1.2 | `bloom_filter.h` — 移除 `compressible_bloom_filter`，移入 `namespace curator` | 原 `BloomFilter.h` | ~650 行 | 0.5h |
| P1.3 | `distance.h` — `l2_sqr`, `node_score`, `compute_batch_dists` 等 | 原 .cpp 中的匿名函数段 | ~150 行 | 1h |
| P1.4 | `profiling.h` — `SearchProfile` + `MemoryBreakdown` 结构体 | 原头文件 profiling/memory 段 | ~100 行 | 0.5h |

### Phase 2: 独立子模块

| # | 步骤 | 来源 | 产出 | 工作量 |
|---|------|------|------|--------|
| P2.1 | `kmeans.h/.cpp` — 自实现 Lloyd K-means（含 OpenMP 并行化 + 性能对比 FAISS Clustering） | 新写 | ~250 行 | 3.5h |
| P2.2 | `tree_node.h` — `TreeNode` 结构体 + 构造/析构/BF 方法 | 原头文件 TreeNode 段 | ~100 行 | 0.5h |
| P2.3 | `cluster_tree.h/.cpp` — `build_tree`（含根节点质心初始化 + `n_clusters` children 不变式 assert）/`assign_to_leaf`/`get_vector_path`/`find_assigned_leaf` | .cpp 第 292-380, 1214-1272 行 | ~250 行 | 2.5h |
| P2.4 | `shortlist.h/.cpp` — `split_shortlist`/`try_merge_shortlists` | .cpp 第 1250-1319 行 | ~200 行 | 1.5h |
| P2.5 | `pq_codec.h/.cpp` — 码本训练（M 子空间 K-means 复用 kmeans 模块）、编码、距离查表、ADC 重排、磁盘持久化（mmap）。**含 `static_assert(nbits==8)`、`build_distance_table`/`compute_distances` 拆分、子空间数据切片（从 [N×d] 抽 M 个 [N×dsub] 子矩阵）** | .cpp 第 848-897, 2290-2522 行 + 原 FAISS ProductQuantizer | ~550 行 | 5.0h |
| P2.6 | `flash_store.h/.cpp` — Flash 文件级 I/O（open/read/write/close），不持有树映射 | .cpp 第 2341-2725 行（仅文件操作部分） | ~250 行 | 2h |
| P2.7 | `temp_index.h/.cpp` — `build_temp_index`/`search_temp_index` | .cpp 第 34-114, 1958-2112 行 | ~300 行 | 2h |

### Phase 3: 复杂谓词迁移

| # | 步骤 | 来源 | 产出 | 工作量 |
|---|------|------|------|--------|
| P3.1 | `complex_predicate.h/.cpp` — 完整迁移，移入 `namespace curator::predicate`。**将 `evaluate_formula` 模板改为 header-only（消除 `.cpp` 中 4 个显式实例化）；替换 3 处 `FAISS_THROW_MSG` → `CURATOR_THROW_MSG`（`make_state` 函数）** | 原 `complex_predicate.h/.cpp` | ~600 行 | 2.5h |

### Phase 4: 主类组装

| # | 步骤 | 产出 | 工作量 |
|---|------|------|--------|
| P4.1 | `curator_index.h` — `CuratorIndex` 类声明 | ~300 行 | 2h |
| P4.2 | `curator_index.cpp` — `train`/`add`/`grant`/`flush`/`search`/`revoke`/`build_filter`/`memory`/`profile`/`find_all_qualified_vecs`/`sanity_check`/`beam_search` 等实现。**注意**：若文件超过 1000 行，将 `sanity_check` + `memory_breakdown` + `print_tree_info` 提取为独立的 `diagnostics.h/.cpp`（~200 行），保持 `curator_index.cpp` 在 ~800 行以内 | ~1000 行（或 ~800 行 + ~200 行 diagnostics） | 5h |

### Phase 5: CLI 入口与构建系统

| # | 步骤 | 产出 | 工作量 |
|---|------|------|--------|
| P5.1 | `main.cpp` — 命令行解析（`bench` 单一模式）+ `.npy` 文件读取 + JSON 结果输出（nlohmann/json） | ~350 行 | 2.5h |
| P5.2 | `CMakeLists.txt` — 独立构建，依赖 cnpy + OpenMP + nlohmann/json (header-only) | ~60 行 | 0.5h |

### Phase 6: Python 工具脚本 (3 个)

| # | 步骤 | 产出 | 工作量 |
|---|------|------|--------|
| P6.1 | `python/preprocess_train.py` — `.pkl→.npy` 格式转换 + 可选 FAISS 预处理 | ~150 行 | 1.5h |
| P6.2 | `python/prepare_queries.py` — 查询生成 + Ground Truth 计算 | ~150 行 | 1.5h |
| P6.3 | `python/run_experiment.py` — 实验编排（格式转换 + subprocess 调用 C++ + recall 计算 + 报告生成）| ~400 行 | 3h |

### Phase 7: 验证

| # | 步骤 | 说明 | 工作量 |
|---|------|------|--------|
| P7.1 | 编译测试 | 确保 CMake 构建通过 | 0.5h |
| P7.2 | 功能验证 | 在 arxiv_small 上运行完整 build + search 流程，对比重构前后的 recall@10 和延迟 | 2h |
| P7.3 | 内存对比 | 对比重构前后的索引内存占用（`memory_bytes()` 输出） | 0.5h |

**总预估工作量: 约 49 小时**（含 kmeans 调优 3.5h, pq_codec 5.0h, cluster_tree 2.5h, complex_predicate 迁移 2.5h, curator_index 组装 5.0h, 集成验证 4h）

> **说明**: 原估算 39h 偏低。调整后的 49h 考虑了：
> - K-means 自实现 + 与 FAISS Clustering 的 A/B 性能对比（+1h）
> - PQ codec 子空间数据切片（stride 访问） + `build_table`/`compute` 拆分 + mmap 验证（+1.5h）
> - `cluster_tree` 根节点质心初始化 + 不变式 assert 添加（+0.5h）
> - `complex_predicate` 模板 header-only 迁移 + `FAISS_THROW_MSG` 替换（+0.5h）
> - `curator_index.cpp` 实际代码量 > 1000 行，增加 1h；可选 `diagnostics.h/.cpp` 拆分（+1h）
> - 集成测试与 recall/延迟/内存 三方对比预留充足时间（+2h）
> - 其余 Phase 微调（+1.5h）

---

## 八、外部依赖与风险

### 8.1 新引入的第三方依赖

| 依赖 | 类型 | 用途 | 获取方式 |
|------|------|------|----------|
| `cnpy` | .h + .cpp (2 文件) | 读取 `.npy` 文件 | 从 MiniRFANN 复制或 git clone [rogersce/cnpy](https://github.com/rogersce/cnpy) |
| `nlohmann/json` | single-header (.hpp) | JSON 结果输出 | CMake `FetchContent` 或直接放入 `src/json.hpp` |
| OpenMP | 系统库 | 并行 K-means / PQ 训练 / 批量查询 | 系统包管理器 (`apt install libomp-dev`) |
| zlib | 系统库 | cnpy 的 .npz 支持所需（若仅保留 .npy 读取可移除） | 系统自带或 `apt install zlib1g-dev` |

### 8.2 风险与缓解

| 风险 | 严重度 | 缓解措施 |
|------|:------:|--------|
| K-means 自实现精度/性能不如 FAISS | **高** | 必须在分配步骤使用 `#pragma omp parallel for`；保留 CMake option 开关可回退 FAISS；≤3 次调优；验证阶段 A/B 对比训练时间和 recall |
| K-means **单线程**回退（若 OpenMP 未正确实现） | **严重** | 构建时间将膨胀 **4-8 倍**（arxiv: 数分钟→数十分钟）。Phase 2.1 必须通过 OpenMP 并行性测试（多核利用率 >80%） |
| `build_temp_index_for_filter` 依赖 `children.size() == n_clusters` 不变式 | **严重** | 若 `cluster_tree::build_tree` 未严格遵守，temp index 构建时会**越界崩溃**。缓解：在 `cluster_tree.cpp` 中添加 `assert(node.children.empty() \|\| node.children.size() == cfg.n_clusters)` |
| `compute_vector_distance` 单向量堆分配 | **中** | ADC rerank 阶段对 `k × factor` 个候选逐一分配。缓解：使用 `static thread_local std::vector<float>` 复用缓冲区，每个线程独立，零同步开销。参见 §4.9 实现细节 |
| PQ `nbits≠8` 静默数据损坏 | **中** | `code_bytes = M * nbits / 8` 仅在 `nbits=8` 时正确。缓解：在 `pq_codec` 中添加 `static_assert(nbits == 8)` |
| `complex_predicate` 模板实例化类型不兼容 | **中** | `evaluate_formula` 显式实例化使用 `tid_t`。缓解：保留 `tid_t` 别名在 `common.h`，并将模板定义移入 header-only |
| PQ 子空间 K-means 训练时间 | **中** | M 个子空间在子空间级别并行 + 每个子空间内部分配并行；复用同一 kmeans 模块 |
| PQ 码本训练质量偏差 | **低** | 使用与 FAISS 相同的训练流程（逐子空间独立 K-means）；验证阶段逐元素对比 recall |
| 复杂谓词行为差异 | **低** | Phase 3 的 complex_predicate 是**直接迁移**（改 namespace），不改逻辑 |
| Profiling 并行查询竞态 | **低** | 原代码已存在此问题。重构后不修复，文档注明 profiling 不支持并行查询模式 |
| cnpy 不支持某些 dtype | **低** | 仅使用 `float32`、`int32`、`int64`、`uint8`，均在 cnpy 支持范围内 |
| 索引无法持久化（build↔search 分离不可行） | **低** | 重构第一阶段仅支持 `bench` 模式（单进程 build+search）；独立 build/search 留待后续 |
| PQ 编码磁盘持久化（mmap）行为差异 | **低** | pq_codec 将保留完整的 write_to_disk / mmap_load / free_in_memory 功能链，验证阶段逐字节对比 |
| centroid 对齐从 64 字节降级 | **低** | 当前使用标量 `l2_sqr`（无 SIMD），对齐无影响。若后续引入 SIMD 需改回 `aligned_alloc` |

---

## 九、性能回归评估

> **核心发现**: 当前核心 Curator 文件（`MultiTenantIndexIVFHierarchical.cpp`）**仅在第 527 行使用一处 OpenMP**（`BATCH_QUERY` 查询并行化）。K-means 和 PQ 训练的并行性完全来自 FAISS 内部的 OpenMP 实现。自实现必须显式添加 OpenMP，否则训练将回退到单线程。

### 9.1 逐组件评估

| 组件 | FAISS 原实现 | 自实现 | 查询延迟影响 | 构建时间影响 |
|------|-------------|--------|:----------:|:----------:|
| **K-means** | Clustering 类，内部 OpenMP 并行 + 高效批量距离计算 | 自实现 Lloyd + 显式 OpenMP `parallel for` | **≈0%**（聚类树结构相同，仅质心位置可能因初始化不同而微调） | **+30%~+100%**（若未加 OpenMP: **+300%~+500%**） |
| **PQ 训练** | ProductQuantizer，M 个子空间独立 OpenMP K-means | 复用自实现 K-means，M 个子空间 `parallel for` | **≈0%**（查表距离计算算法完全相同） | **+30%~+100%**（若未加 OpenMP: **+500%~+1000%**） |
| **L2 距离** | `fvec_L2sqr()` 标量循环 | `l2_sqr()` 标量循环（相同指令） | **0%** | **0%** |
| **Prefetch** | `prefetch_L1` = `__builtin_prefetch(ptr, 0, 3)` | `PREFETCH(ptr)` = 相同宏 | **0%** | **0%** |
| **堆操作** | `CMax`/`heap_heapify`/`maxheap_replace_top` O(log k) | `RunningList::insert()` O(k) 向量移位 | **<0.5%**（k≤100 时 O(k) 约 100 次移位。Unfiltered search 扫描数万向量时每个都调 `insert` vs `replace_top`，P99 延迟影响 1-3μs，在 ±2% 总延迟范围内。若后续需要极致性能可 vendor `Heap.h`） | N/A |
| **SWIG** | Python↔C++ 每次调用 ~1-5μs | subprocess 启动 ~10-50ms（一次性） | **净提升**（去除了每次查询的 SWIG 开销） | N/A |
| **centroid 对齐** | `aligned_alloc(64, ...)` 用于 SIMD | `std::vector<float>` 默认对齐（标量 `l2_sqr` 不需要 SIMD 对齐） | **0%** | **0%** |

### 9.2 构建时间详细分析（以 arxiv 1.6M×384 为例）

| 阶段 | FAISS 当前耗时（估算） | 自实现预估 | 风险 |
|------|----------------------|-----------|------|
| 层次 K-means（根节点 1.6M→64 类 ×20 迭代） | ~5-10s（内部 OpenMP） | ~8-20s（显式 OpenMP） | 需确保分配步骤并行化正确 |
| 子节点递归 K-means（小批次，总计 ~2-3 层） | ~3-5s | ~5-10s | 小批次并行收益有限，可回退串行 |
| PQ 训练（24 子空间 × 1.6M→256 类 ×20 迭代） | ~10-20s（内部 OpenMP × M 个独立聚类） | ~15-40s（M 个子空间 + 内部分配并行） | 最耗时阶段；M=24 的并行度充足 |
| Flash 写入（fseek/fwrite） | ~2-5s | ~2-5s（不变） | 无变化 |
| **总计** | **~20-40s** | **~30-75s** | **+50%~+100%**（若 OpenMP 实现正确） |

### 9.3 内存占用评估

| 变化 | 影响 |
|------|------|
| 移除 `IndexFlat* storage`（8 字节指针） | -8B |
| 移除 `MultiTenantIndex` 基类 vtable（~8 字节）+ 继承成员对齐 | -16~32B |
| 移除 25+ SWIG getter（无运行时开销，仅代码体积） | 0B 运行时 |
| `float* centroid` → `std::vector<float>`（添加 24 字节 vector 头部） | +24B/节点 × ~5000 节点 ≈ +120KB |
| **净影响** | **基本持平，<1% 差异** |

### 9.4 查询延迟详细分析

查询热路径上的关键操作（按频率排序）：

1. **`node_score()` ≈ 2-10 万次/查询**：`l2_sqr` 替代 `fvec_L2sqr` → **0% 差异**
2. **`bf.contains()` ≈ 1-5 万次/查询**：完全相同 → **0% 差异**
3. **`compute_pq_distances()` ≈ 100-500 次/查询**：相同查表算法 → **0% 差异**
4. **`compute_vector_distance()` ≈ 0-500 次/查询**：精简为 flash→buffer 两级回退（去除 storage 兼容路径）→ **略微加快**
5. **堆操作** ≈ 仅在无过滤搜索中使用 → **<0.1% 差异**
6. **SWIG 开销** ≈ 每次查询 3-5 次 Python↔C++ 调用 → **去除后略微加快**

**综合评估: 查询延迟 ±2% 范围内，基本持平。**

### 9.5 Recall 评估

- K-means 使用随机初始化（非 K-means++）可能导致聚类质量轻微下降
- 但 Curator 采用 **64 路分支**的宽树结构，单层聚类质量对最终搜索路径影响被稀释
- PQ 码本训练算法相同（逐子空间 Lloyd），码本质量理论上等价
- **预估 recall@10 变化: ±0.5% 范围内**

### 9.6 其他注意事项

| 项 | 说明 |
|------|------|
| `search_ef` 默认值 | 原代码默认 `search_ef = 0`（触发异常 "search_ef must be greater than 0"）。重构后默认 `search_ef = 128`，消除陷阱默认值 |
| `set_optimized_search_enabled` | 原头文件（第 745-751 行）声明但 `use_optimized_search` **未被任何搜索路径检查**——死代码。重构后**删除** |
| `memory_usage()` → `memory_bytes()` | 原函数打印到 stdout 且仅用于调试。重构后拆分为 `memory_bytes()`（返回 size_t）和 `memory_breakdown()`（返回结构体），语义更清晰 |
| 批量 `add_vector_with_ids` | 原接口接收批量。重构后 `add_vector()` 单条调用（main.cpp 循环），每条的树遍历开销远大于函数调用开销，性能影响可忽略 |
| Python↔C++ 默认参数不一致 | 当前 Python `curator.py` 自定义默认值（`nprobe=1200, variance_boost=0.2, beam_size=1, bf_error_rate=0.001`）与 C++ 构造函数默认值（`nprobe=3000, variance_boost=0.4, beam_size=2, bf_error_rate=0.01`）不一致。Python 包装器将其值传给 C++ 构造函数，故 C++ 默认值**从未实际生效**。重构后移除 Python 包装层，默认值统一由 `CuratorConfig` 或 JSON 配置文件决定，消除此隐性不一致。**推荐采用 C++ 原始默认值**（与论文实验设置一致），在 `CuratorConfig` 中详注每个参数的语义 |

---

## 十、验证标准

重构成功判据：
1. **构建**: `cmake -B build && cmake --build build` 在仅安装 g++/cmake/openmp 的系统上通过（**无需 FAISS**）
2. **功能**: 在 arxiv_small 数据集上 recall@10 ≥ 重构前水平 -0.5%（由 Python `run_experiment.py` 验证）
3. **查询延迟**: 单租户查询 P50/P99 延迟在重构前的 ±3% 范围内
4. **构建时间**: 总构建时间 ≤ 重构前的 2×（若超过则检查 OpenMP 是否生效；若超 3× 则切换到 FAISS Clustering 回退）
5. **内存**: `memory_bytes()` 输出与重构前差异 <1%
6. **接口**: `./curator bench --config config.json --output results.json` 正常执行并产出正确 JSON
7. **OpenMP**: K-means 训练阶段全部 CPU 核心利用率 >80%（验证并行化正确）

---

## 十一、已知设计决策与约束速查

以下汇总了方案中所有关键的隐式假设、显式约束和不变量，供开发时快速查阅。

### 数据结构不变式

| 不变式 | 说明 | 检查位置 |
|--------|------|----------|
| 所有非叶子节点恰好有 `n_clusters` 个子节点 | 包括空子节点（0 向量）。被 `build_temp_index_for_filter` 依赖 | `cluster_tree.cpp: build_tree` |
| 叶子节点 `children.empty()` | 叶子无子节点，但持有 `vector_indices` | `tree_node.h` |
| `int_vid_t` 位编码路径 | 位布局: `[level_L-1 | ... | level_0 | local_vid]`，各级占 `MAX_BRANCH_FACTOR_LOG2` 位 | `cluster_tree.cpp` |
| `leaf.node_id` 的低位编码路径 | `find_assigned_leaf` 通过 `(vid >> shift) & mask` 解码 | `cluster_tree.cpp` |

### 模块约束

| 约束 | 说明 | 位置 |
|------|------|------|
| `pq_codec` 仅支持 `nbits=8` | `static_assert` 保证，通用化需位压缩层 | `pq_codec.h` |
| `kmeans` 仅支持 L2 距离 | Curator 仅使用 METRIC_L2 | `kmeans.h` |
| `FlashStore` 遵循 RAII | 构造打开/析构关闭文件句柄 | `flash_store.h` |
| `TempIndexNode::centroid` 为 Non-owning pointer | 生命周期绑定到主聚类树的 `TreeNode::centroid` | `temp_index.h` |
| `scratch_` buffer 线程安全 | 使用 `static thread_local std::vector<float>`，每线程独立缓冲区 | `curator_index.cpp: compute_vector_distance` |
| `profiling` 不支持并行查询 | 多线程写 `SearchProfile` 会导致数据损坏（原代码遗留问题） | `profiling.h` |

### 类型别名

| 别名 | 底层类型 | 说明 |
|------|---------|------|
| `ext_vid_t` | `uint32_t` | 外部向量 ID |
| `int_vid_t` | `uint64_t` | 内部向量 ID（含路径编码） |
| `ext_lid_t` | `int16_t` | 外部 tenant/label ID |
| `int_lid_t` | `int16_t` | 内部 tenant ID（连续分配） |
| `tid_t` | `int16_t` | 保留别名（与 `ext_lid_t`/`int_lid_t` 等价，用于 `complex_predicate` 模板兼容） |
| `Buffer` | `std::vector<int_vid_t>` | complex_predicate 集合操作类型 |

### 已知后续优化项

| 优化项 | 说明 | 触发条件 |
|--------|------|----------|
| SIMD 距离计算 | 将 `l2_sqr` 替换为 AVX2 向量化实现 | 查询延迟需要进一步优化 |
| `nbits≠8` 支持 | PQ 编码位压缩/扩展 | 需要更低内存占用（`nbits=4`）或更高精度（`nbits=16`） |
| vendor FAISS `Heap.h` | 替换 `RunningList` 为 O(log k) 堆 | Unfiltered search P99 延迟成为瓶颈 |
| 索引序列化 | `build`→`save`→`load`→`search` 分离 | 大型数据集需要一次构建多次查询 |

---

*文档创建日期: 2026-07-01*
*最后更新: 2026-07-01（第二轮审查：修复 10 个潜在问题——行数估计、thread_local scratch_、API变更文档、kmeans细节、PQ stride访问、key类型修复、POSIX头文件、PREFETCH宏、诊断辅助函数、默认参数一致性）*
*状态: 已批准执行，可开始 Phase 0*
