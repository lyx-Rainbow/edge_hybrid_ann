# Curator 索引代码 100% 掌握学习路线

> **目标**: 在 10-12 小时内达到对 Curator 索引代码 100% 掌握。  
> **前提**: 你已具备 C++/Python 基础，对 Curator 核心设计思想（层次化聚类树 + 短列表 + Bloom Filter 过滤）有较好理解。  
> **策略**: 分层递进 — 先基础工具层，再数据结构层，再算法逻辑层，最后编排层。

---

## 一、代码库全貌速览（5 分钟）

### 1.1 当前代码结构

```
Curator/
├── src/                          # 当前活跃的 C++ 源码（独立版，零 FAISS 依赖）
│   ├── common.h                  # 公共基础设施（类型/常量/工具类/异常宏）
│   ├── config.h                  # 配置参数结构体
│   ├── bloom_filter.h            # Bloom Filter（header-only）
│   ├── distance.h                # 距离函数（header-only）
│   ├── tree_node.h               # 树节点数据结构
│   ├── profiling.h               # 性能分析/Memory 分解结构体
│   ├── cluster_tree.h/.cpp       # 聚类树构建与遍历
│   ├── shortlist.h/.cpp          # 短列表分裂/合并
│   ├── kmeans.h/.cpp             # Lloyd K-means 聚类
│   ├── pq_codec.h/.cpp           # PQ 编解码 + 块缓存集成
│   ├── pq_block_cache.h/.cpp      # ★ PQ 码外存 LRU 块缓存
│   ├── flash_store.h/.cpp        # 全精度向量磁盘 I/O
│   ├── temp_index.h/.cpp         # Bitmap Filter 临时索引
│   ├── complex_predicate.h/.cpp  # 复杂谓词解析/求值
│   ├── curator_index.h/.cpp      # 主编排类
│   ├── cnpy.h/.cpp               # .npy 文件读写
│   └── main.cpp                  # CLI 入口
├── python/
│   ├── run_experiment.py         # 实验编排（preprocess → subprocess C++ → recall）
│   ├── preprocess_train.py       # 训练数据 .pkl → .npy 转换
│   └── prepare_queries.py        # 查询数据准备
├── legacy/                       # 旧版 FAISS-SWIG 集成代码（不再用）
├── build/                        # 编译产物
├── CMakeLists.txt                # 独立 CMake 构建
├── EXPERIMENT_GUIDE.md           # 实验操作手册
└── REFACTOR_PLAN.md              # 重构方案（详细设计文档）
```

### 1.2 核心设计思想（一句话总结）

> Curator 是一棵**层次化 K-means 聚类树**：每个非叶子节点拥有当前层的短列表（shortlist，存储租户→向量ID映射）和 Bloom Filter（汇总子树中的租户）；查询时通过 Beam Search + Frontier Search 沿树下降，利用 BF 跳过无关分支，利用短列表批量计算距离。

---

## 二、学习路线（8 个阶段）

---

### 阶段 0：预备知识确立（预计 15 分钟）

**目标**: 建立正确的全局观，理解"这是什么东西、为什么这样设计"。

**阅读顺序**:

| 步骤 | 文件 | 聚焦内容 | 耗时 |
|------|------|----------|:---:|
| 0.1 | [README.md](README.md) | 项目整体介绍、目录结构、参数说明 | 5min |
| 0.2 | [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) | 编译方式、数据集、CLI 用法、Profiling 解读 | 5min |
| 0.3 | [REFACTOR_PLAN.md](REFACTOR_PLAN.md) §一~§三 | 现状诊断、核心问题、目标架构图、模块依赖星型拓扑 | 5min |

**检验标准**:
- 能说出 Curator 的 5 个核心组件（树构建、短列表、bloom filter、PQ、flash storage）
- 能画出模块间的星型依赖关系图（common.h 是最底层，curator_index 是唯一编排点，其余模块零相互依赖）
- 理解为什么重构去掉了 FAISS（因为 Curator 根本不走 IVF 倒排列表逻辑，树遍历是完全自有的）

---

### 阶段 1：基础工具层 — 类型、常量、宏（预计 40 分钟）

**目标**: 100% 掌握所有类型别名、编译期常量、工具类模板、异常宏的**定义和含义**。这是后续所有代码理解的**先决条件**。

**核心文件**: [common.h](src/common.h)（349 行）

**阅读指导**:

| 行号 | 内容 | 关键理解点 | 必须掌握 |
|------|------|-----------|:--:|
| 17 | `PREFETCH(ptr)` 宏 | `__builtin_prefetch(ptr, 0, 3)` — 0=读预取, 3=高时间局部性(L1 cache) | ✅ |
| 22-43 | `CURATOR_THROW_*` 6 个宏 | 底层都是 `std::runtime_error`，**不需要 FAISS 异常体系** | ✅ |
| 50-55 | 5 个类型别名 | **区分 ext（外部）vs int（内部）**: `ext_vid_t=uint32_t` 是用户给的 label，`int_vid_t=uint64_t` 含路径位编码。`int_lid_t=int16_t` 是内部连续分配的 tenant id | ✅✅✅ |
| 60-65 | 编译期常量 | `MAX_BRANCH_FACTOR=64`, `MAX_LEAF_SIZE=1024`, `MAX_TREE_DEPTH` 由位宽公式算出 | ✅ |
| 70-89 | `RunningMean` | 动态均值追踪（add/remove/get_mean），用于树节点 variance 统计 | ✅ |
| 93-138 | `SortedList<T>` | 有序列表，封装 `std::vector` + 二分插入/删除/合并。**ShortList 就是 `SortedList<int_vid_t>`** | ✅✅ |
| 143-231 | `RunningList` | **top-K 有序候选集**。核心数据结构！O(k) 有序插入。注意 `batch_insert` 是合并两个有序序列。被 `curator_index` 和 `temp_index` 共享 | ✅✅✅ |
| 236-305 | `IdAllocator<Ext,Int>` | 连续 ID 分配器（含 free_list 回收）。**只有 TenantIdAllocator 使用！** | ✅ |
| 311-343 | `IdMapping<Ext,Int>` | 双向 ID 映射（不含分配）。**VectorIdAllocator 使用！** | ✅ |

**重点理解**:
1. **为什么 int_vid_t 是 uint64_t？** → 因为要在 64 位中编码完整树路径（每层 6 位分支 + 最后 10 位局部索引）
2. **RunningList 为什么不用堆而用有序插入？** → k≤100 时 O(k) 移位 ≈ 1-3μs，差异可忽略。且实现简洁，无外部依赖
3. **SortedList 的 merge 是 O(n+m)** → 使用 `std::merge`（两个有序序列归并），非逐个插入

**检验标准**:
- 能不看代码默写出 5 个类型别名及其底层类型和用途
- 能写出 `int_vid_t` 的位编码公式（各层占多少位）
- 能解释 `RunningList::batch_insert` 的归并逻辑

---

### 阶段 2：配置与数据结构层（预计 30 分钟）

**目标**: 理解 CuratorConfig 的每个参数的作用、TreeNode 的字段含义、Bloom Filter 的原理。

**阅读顺序**:

| 步骤 | 文件 | 聚焦内容 | 耗时 |
|------|------|----------|:---:|
| 2.1 | [config.h](src/config.h) | 全部 21 个参数的**含义和默认值** | 10min |
| 2.2 | [tree_node.h](src/tree_node.h) | TreeNode 结构体的全部字段 + `node_score()` 函数 | 15min |
| 2.3 | [bloom_filter.h](src/bloom_filter.h) | `bloom_parameters::compute_optimal_parameters()`（参数优化）和 `bloom_filter::insert/contains`（核心 API） | 5min |

**关键理解点**:

**CuratorConfig 参数分组记忆**:
| 分组 | 参数 | 记忆口诀 |
|------|------|---------|
| 数据结构 | d, n_clusters(≤64), max_leaf_size(≤1024), max_sl_size | "维度64叶128" |
| Bloom | bf_capacity, bf_false_pos | "容量1000误报0.01" |
| K-means | clus_niter | "迭代20次" |
| PQ | pq_M, pq_nbits(仅8), pq_enabled, pq_use_adc_rerank | "M子空间8bit" |
| Flash | use_flash_storage, flash_path | "全精度写磁盘" |
| 搜索 | nprobe, prune_thres(1.6), variance_boost(0.4), search_ef(128), beam_size(2) | "探3000剪1.6方0.4" |

**TreeNode 字段记忆法**:
```
TreeNode
├── 树结构: level, sibling_id, parent, children, node_id（位编码路径）
├── 聚类信息: centroid(float[]), variance(RunningMean)
├── 过滤结构: bf(BloomFilter), shortlists(map: tenant_id → SortedList<vid>)
└── 叶子专用: vector_indices(SortedList<vid>)  ← 仅叶子节点有值
```

**node_score 公式**: `score = L2_dist(query, centroid) - variance_boost × variance_mean`
- variance_boost=0 直接返回距离（temp_index 搜索使用）
- variance_boost>0 时，高方差节点被降权（更易被优先队列弹出）
- 这是 Curator **剪枝的核心机制**！

**Bloom Filter 核心**:
- `bloom_parameters::compute_optimal_parameters()`: 给定元素数和假阳性率，自动计算最优 bit 数和 hash 函数数
- `bloom_filter::insert(key)`: 对每个 salt 计算 hash，对应 bit 置 1
- `bloom_filter::contains(key)`: 所有 hash 位都为 1 则返回 true
- `operator|=(other)`: 位或合并（用于子节点 BF 传播到父节点）

**检验标准**:
- 能否说出 nprobe 和 search_ef 的区别？（前者用于 unfiltered search 的节点探测数，后者用于 tenant search 的候选集大小）
- 能否画出 TreeNode 在内存中的字段布局（哪些是叶子独有的）
- 能否解释 node_score 为什么要减去方差项？

---

### 阶段 3：距离计算与 Profiling（预计 20 分钟）

**目标**: 掌握距离计算函数和性能分析数据结构。

**核心文件**: [distance.h](src/distance.h)（122 行）、[profiling.h](src/profiling.h)（67 行）

**阅读指导**:

| 函数/结构 | 位置 | 关键要点 | 必须掌握 |
|-----------|------|---------|:--:|
| `l2_sqr(x,y,d)` | distance.h:17-24 | 标量 L2 平方距离，~10 行。不平方根因为只需要相对比较 | ✅ |
| `l2_sqr_4way(x,y0,y1,y2,y3,d)` | distance.h:29-61 | **真** 4 路展开：4 个独立累加器，减少 CPU 依赖链。内层循环每 4 维一组 | ✅ |
| `compute_batch_dists(x,vids,get_vec_fn,d,out)` | distance.h:67-119 | 模板函数，接受 `get_vector_ptr(vid)→float*` lambda。含 prefetch + 4 路计算。**n<4 退化到标量** | ✅✅ |
| `SearchProfile` | profiling.h:14-44 | 两套字段: standard 路径(beam/frontier/rerank) + bitmap filter 路径(preproc/sort/build/search)。**线程不安全（已知问题）** | ✅ |
| `MemoryBreakdown` | profiling.h:48-65 | 14 个组件级字节计数，递归遍历树 | ✅ |

**l2_sqr_4way 为什么比连续 4 次 l2_sqr 快？**
```cpp
// 连续 4 次: 依赖链
d0 = l2_sqr(x,y0);  // 需要 d0 完成才能继续
d1 = l2_sqr(x,y1);  // ...

// 4 路展开: 4 个独立累加器
for (i = 0; i < d; i += 4) {
    s0 += (x[i] - y0[i])²;  // 4 条独立指令流
    s1 += (x[i] - y1[i])²;  // CPU 可以并行执行
    s2 += (x[i] - y2[i])²;
    s3 += (x[i] - y3[i])²;
    // 4 维一组，共 d/4 轮
}
```

**检验标准**:
- 能否写出 `compute_batch_dists` 的参数列表（5 个参数 + 模板）？
- 能否说出 SearchProfile 中哪些字段属于 "standard" 路径，哪些属于 "bitmap" 路径？

---

### 阶段 4：子模块层 — 树构建、K-means、短列表（预计 1.5 小时）

**目标**: 理解聚类树的完整构建流程和短列表维护机制。

#### 4.1 K-means 模块（20 分钟）

**核心文件**: [kmeans.h](src/kmeans.h)、[kmeans.cpp](src/kmeans.cpp)（150 行）

**阅读指导**:

| 关注点 | 位置 | 关键要点 |
|--------|------|---------|
| `KMeansConfig` | kmeans.h:9-14 | niter(20), seed(1234), n_threads(0=all) |
| `KMeansResult` | kmeans.h:16-19 | centroids[n_clusters×d] 连续存储, assignments[n] |
| `stride` 参数 | kmeans.h:26 | **PQ 子空间复用的关键设计**: stride=0 表示向量连续（d==stride），stride=d 表示跳步访问子空间 |
| 随机初始化 | kmeans.cpp:44-53 | `std::mt19937` + shuffle 无放回抽样。**不实现 K-means++**（64 路宽分支稀释了初始化影响） |
| 分配步骤 | kmeans.cpp:66-97 | `#pragma omp parallel for` + Per-thread 局部累加数组 |
| 空聚类处理 | kmeans.cpp:135-142 | 设为全局均值 + 随机小扰动(1e-6) |
| Per-thread reduce | kmeans.cpp:99-111 | 串行归并各线程的累加数组 |

**关键设计决策**:
1. **为什么不实现 K-means++？** → Curator 用 64 路宽分支 + 多层递归，单层初始化质量对最终搜索路径影响被稀释
2. **stride 参数的妙用**: PQ 训练需要从 [N×d] 大矩阵中抽 M 个 [N×dsub] 子矩阵。不复制数据，用 `stride=d` 在距离计算时按步跳转访问
3. **空聚类不崩溃**: 确保所有子节点质心有效 → 满足 `build_tree` 的不变式

#### 4.2 cluster_tree 模块（40 分钟）

**核心文件**: [cluster_tree.h](src/cluster_tree.h)、[cluster_tree.cpp](src/cluster_tree.cpp)（140 行）

**4 个核心函数**:

| 函数 | 职责 | 关键算法 |
|------|------|---------|
| `build_tree(node, n, x, cfg)` | 递归 K-means 建树 | 根节点: 计算全局均值 → K-means → 按 assignment 排序 → `new TreeNode()` × n_clusters → **递归所有子节点（含空节点！）** |
| `assign_to_leaf(root, x, d)` | 将向量指派到最近叶子 | 逐层 `l2_sqr` 找最近质心 → 下降 |
| `get_vector_path(root, vid)` | 从 vid 解码全路径 | `(vid >> offset) & mask` 逐层提取 child_id |
| `find_assigned_leaf(root, vid)` | 同 get_vector_path，返回叶子节点指针 | 同上，直接返回 TreeNode* |

**关键不变式**（必须牢记！）:
> **所有非叶子节点恰好有 n_clusters 个子节点。** 包括分配了 0 个向量的空子节点。此不变式被 `temp_index::build_temp_index` 依赖（通过位前缀二分直接访问 `children[child_idx]`）。

**build_tree 流程详解**:
```
build_tree(node, n, x):
  1. 若 centroid 为空 → 计算全局均值（仅根节点）
  2. 若 n ≤ max_leaf_size 或 level ≥ MAX_TREE_DEPTH → 停止（叶子）
  3. K-means(n → n_clusters)
  4. 按 cluster assignment 排序 x → sorted_x
  5. for clus_id = 0..n_clusters-1:
       new TreeNode(level+1, clus_id, node, centroid[clus_id])
       build_tree(child, cluster_size[clus_id], sorted_x[offset])
  6. assert children.size() == n_clusters  ← 不变式检查
```

**位编码细节**:
```
int_vid_t (64位) 布局:
[level_0: 6bit | level_1: 6bit | ... | local_vid: 10bit | unused]

根:     node_id = 0
第L层:  node_id = parent.node_id | (sibling_id << (64 - L*6))
叶子向量: vid = leaf.node_id | (local_vid << (64 - level*6 - 10))
```

#### 4.3 shortlist 模块（30 分钟）

**核心文件**: [shortlist.h](src/shortlist.h)、[shortlist.cpp](src/shortlist.cpp)（85 行）

**2 个核心函数**:

| 函数 | 触发条件 | 操作 | 向上/向下 |
|------|---------|------|:--:|
| `split_shortlist(node, tid, max_sl_size)` | 短列表大小 > max_sl_size | 按 vid 位前缀分配到子节点 → 清空当前节点短列表 → 更新子节点 BF | **向下** |
| `try_merge_shortlists(node, tid, max_sl_size)` | 子节点同租户短列表之和 ≤ max_sl_size | 合并子节点短列表到父节点 → 清空子节点短列表 → 更新父子 BF | **向上** |

**shortlist 的生命周期**:
```
插入: grant_access → grant_access_impl
  ├── 叶子: 直接插入 shortlist
  ├── 非叶子+短列表存在: 插入 → 若超限则 split_shortlist (向下推)
  ├── 非叶子+BF 无此 tenant: 创建新短列表
  └── 非叶子+BF 有此 tenant 但无短列表: 递归到子节点

删除: revoke_access
  ├── 找到含此 vid 的短列表 → erase
  └── 递归 try_merge_shortlists 向上合并
```

**检验标准**:
- 能手动画出 build_tree 在 n_clusters=4 时的递归过程（节点层次、质心来源）
- 能解释为什么空子节点也必须保留在树结构中
- 能说明 split_shortlist 和 try_merge_shortlists 的相互制约关系
- 能手算一个 int_vid_t 的位编码（给定 level=2, sibling_id=3 → node_id）

---

### 阶段 5：PQ 与 Flash 存储（预计 1 小时）

**目标**: 理解 PQ 压缩和 Flash 外存的完整实现。

#### 5.1 PQ Codec（50 分钟）

**核心文件**: [pq_codec.h](src/pq_codec.h)、[pq_codec.cpp](src/pq_codec.cpp)、**[pq_block_cache.h](src/pq_block_cache.h)**、**[pq_block_cache.cpp](src/pq_block_cache.cpp)**

**架构概览（v2 外存化改造后）**:

```
┌──────────────────────────────────────────────────┐
│                   CuratorIndex                    │
│  ┌──────────┐  ┌───────────────────────────────┐ │
│  │ PQCodec  │  │  PQBlockCache (LRU + mutex)   │ │
│  │ codebook │  │  ┌─────┬─────┬─────┬────────┐ │ │
│  │ (内存)   │  │  │Blk 0│Blk 5│Blk 3│  ...   │ │ │
│  │          │  │  └─────┴─────┴─────┴────────┘ │ │
│  │ mutable  │  │  容量可配(默认256块)            │ │
│  └──────────┘  └───────────┬───────────────────┘ │
│                            │ cache miss          │
│                            ▼                     │
│                 ┌─────────────────────┐          │
│                 │ PQ Codes File (磁盘) │          │
│                 │ 32B header+N×M bytes │          │
│                 └─────────────────────┘          │
│                                                  │
│  FlashStore (磁盘): 全精度向量，仅 rerank 时读取  │
└──────────────────────────────────────────────────┘
```

**核心设计思想**: PQ 码不常驻内存，编码后写入磁盘文件，检索时通过 LRU 块缓存按需加载。码本始终在内存（<1MB，每次查询必需）。

**PQCodec 核心 API**:

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `train(n, x, d, M, nbits)` | 原始向量 [N×d] | pq_codebook_ | M 个子空间独立 K-means（#pragma omp parallel for），复用 `kmeans()` 模块 |
| `encode(x, code)` | 单个向量 | uint8_t[M] | 逐子空间找最近质心 |
| `encode_all(n, x, codes)` | [N×d] | vector<uint8_t[M]> | OpenMP 并行编码，**结果写入参数（局部变量），不存入成员** |
| `build_distance_table(query, d_table)` | 查询向量 | [M×256] floats | 预计算每个子空间到 256 个质心的距离。前置条件: `is_trained()` |
| **`compute_pq_distances(d_table, vids, output)`** | 距离表 + vid列表 | (dist,vid) pairs | ★ **核心新增**：内部 vid→seq_idx 转换 → 批量获取 PQ 码（一次加锁）→ ADC 累加（无锁）。参见下方线程安全设计 |
| `compute_exact_batch(query, vids, vectors, d, out)` | 查询 + 全精度向量 | (dist,vid) pairs | ADC 重排使用，标量 l2_sqr |
| `write_to_disk(path, codes, M, nbits)` | 编码 | 磁盘文件 | 32B 头 + 编码体（与 legacy 同 magic `0x50514344`，但字段布局为 uint64） |
| `open_cache(path, block_size, max_blocks)` | 磁盘文件路径 | bool | 打开 PQ 码文件，创建块缓存（用于按需加载） |
| `close_cache()` | — | — | 关闭缓存+fd，不删除磁盘文件 |
| `has_cache()` | — | bool | 缓存是否可用（搜索路径判断条件） |

**PQBlockCache 核心 API**:

| 方法 | 线程安全 | 说明 |
|------|:--:|------|
| `open(path, M, nbits, block_size, max_blocks)` | — | 打开文件 + 验证 32B header |
| `get_code(seq_idx, out)` | ✅ | 单个 PQ 码获取，拷贝到调用方缓冲区 |
| **`prefetch_and_get_codes(seq_indices, out_codes)`** | ✅ | ★ 批量获取：**一次加锁**完成"加载缺失块 + 填充指针 + LRU 更新"。返回的 `out_codes[i]` 指向缓存内部数据（零拷贝），调用方必须在下次缓存修改前使用完毕 |
| `prefetch_blocks(block_ids)` | ✅ | 预热缓存（不返回指针） |
| `stats()` / `reset_stats()` | ✅ | 缓存命中率统计（按 seq_idx 粒度） |

**关键设计决策**:

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| D1 | 缓存粒度 | **块（Block）**，默认4096条/块 | 平衡I/O效率与灵活性；M=16时64KB/块，M=128时512KB/块 |
| D2 | 替换策略 | **LRU** | 适配搜索局部性；实现简单 |
| D3 | 满块分配 | CacheEntry 始终 `block_size×M` 字节 | 偏移计算简单统一，最后一块尾部由 `::pread` 自然零填充 |
| D4 | 码本 | **始终在内存** | 极小（<1MB），每次查询必需 |
| D5 | 缓存线程安全 | 内部 `std::mutex` | batch_query 多线程并发查询必需 |
| D6 | 锁策略 | **组合操作一次加锁** | `prefetch_and_get_codes` 对 128 条向量的 shortlist 仅 1 次加锁（vs 129 次的逐个方案） |
| D7 | 距离表 | **栈上局部变量** | search_one() 中的 `std::vector<float> pq_d_table`，每线程独立，避免 batch_query 数据竞争 |
| D8 | LRU 去重 | 仅 block_id 变化时更新 | 典型 shortlist(128条) 从 128 次 LRU 操作降为 1 次 |

**搜索时 PQ 距离计算流程**（线程安全设计）:

```
多线程并发查询 (batch_query=true):
  Thread 1: search_one() ─┐
  Thread 2: search_one() ─┤
  Thread 3: search_one() ─┼──→ pq_.compute_pq_distances(d_table, vids, output)
                            │         │
                            │         ├── lock(mutex_)         ← 唯一一次加锁
                            │         ├── 扫描 seq_indices，加载缺失块
                            │         ├── 填充所有指针 (零拷贝, 指向缓存内部)
                            │         ├── LRU 更新 (去重)
                            │         ├── unlock(mutex_)
                            │         └── 逐码计算ADC距离（无锁！距离表在线程栈上）
  Thread N: search_one() ─┘
```

**ADC 距离计算公式**:
```
dist = Σ_m d_table[m * 256 + code[m]]   (m = 0..M-1)
```
其中 `d_table[m*256 + k]` = query 子空间 m 与第 k 个码字第 j 维的 L2 距离平方。

**ADC Rerank 流程**（核心加速技术）:
```
1. PQ 阶段: 对所有短列表候选用 compute_pq_distances 计算 PQ 近似距离
2. 候选筛选: RunningList 维护 top-search_ef 个，按距离排序
3. ADC 重排: 取 top (k × rerank_topk_factor) 个，用 compute_exact_batch 做全精度 L2
4. 输出: 精确距离排序后的 top-k
```
→ 为什么有效：PQ 仅需读取 M 字节/向量（vs d×4 字节全精度），大幅减少 I/O 和计算量。

**磁盘持久化格式**:
```
Offset  Size    Field
0       4       Magic = "PQCD" (0x50514344)
4       8       M (子空间数, uint64)
12      8       nbits (量化位数, uint64)
20      8       n_codes (向量数, uint64)
28      4       reserved (0)
32      M       code[0] (向量0的编码)
32+M    M       code[1] 
...
32+N*M  -       code[N-1]
```
> ⚠️ Magic 与 legacy 相同但字段宽度不同（uint64 vs uint32）。当前格式的文件只能由当前代码读取。

#### 5.2 Flash Store（20 分钟）

**核心文件**: [flash_store.h](src/flash_store.h)、[flash_store.cpp](src/flash_store.cpp)（137 行）

**FlashStore 职责**: 仅提供原始文件 I/O。**不持有**树结构映射（映射由 CuratorIndex 管理）。

| 方法 | 说明 |
|------|------|
| `open(path)` | 创建父目录 + `fopen("w+b")`（读写二进制） |
| `close()` / `~FlashStore()` | `fclose`，RAII 自动清理 |
| `write_vector(offset, vec, d)` | fseek + fwrite 单个向量 |
| `write_leaf_region(offset, bytes, vectors, n, d)` | 写一个叶子 region（按 region 对齐写入） |
| `read_vector(offset, d, out)` | fseek + fread 到调用方提供的缓冲区 |
| `read_batch(offsets, d, out)` | **合并读取优化**: 按 offset 排序 → 相邻请求合并为一次 fread（阈值 64KB） |
| `truncate(total)` | `ftruncate` 预分配文件大小 |

**Flash 存储布局**:
```
Flash文件布局 (每个叶子固定 leaf_region_size 字节):
[Leaf 0 region: max_leaf_size × d × 4B]  ← 不足部分为空洞
[Leaf 1 region: max_leaf_size × d × 4B]
...
[Leaf N region: max_leaf_size × d × 4B]

vid → flash_offset 的查找链 (CuratorIndex 持有):
vid → vid_to_leaf_id_[vid]=leaf_node_id
    → leaf_node_id_to_seq_[leaf_node_id]=leaf_seq
    → leaf_seq * leaf_region_size_ + local_idx * d * sizeof(float)
```

**thread_local scratch 优化**: 每次 `compute_vector_distance` 不再 `vector<float>(d)` 堆分配，改用 `static thread_local std::vector<float> tl_scratch`。每线程独立，无同步开销。

**检验标准**:
- 能说出 PQ 训练为什么可以 M 路并行（各子空间独立 K-means）
- 能写出 ADC 距离计算公式（查表累加）
- 能解释 `prefetch_and_get_codes` 为什么一次加锁比逐个 get_code 高效
- 能画出 PQ 码外存架构图（CuratorIndex → PQCodec → PQBlockCache → 磁盘文件）
- 能写出 PQ 磁盘文件的字节布局
- 能解释为什么 `pq_d_table` 放在栈上而非 mutable 成员（batch_query 线程安全）

---

### 阶段 6：临时索引与复杂谓词（预计 1 小时）

#### 6.1 临时索引（30 分钟）

**核心文件**: [temp_index.h](src/temp_index.h)、[temp_index.cpp](src/temp_index.cpp)（192 行）

**设计动机**: 当 bitmap filter 有数千个 qualifier 向量时，直接遍历太慢。构建一个轻量临时树来加速。

**TempIndexNode 结构**:
```cpp
struct TempIndexNode {
    int start, end;              // 在 sorted_qualified_vecs 中的区间
    std::vector<int> children;   // 子节点索引（在 nodes 数组中的下标）
    const float* centroid;       // Non-owning！指向主树 TreeNode::centroid
};
```

**build_temp_index 流程**:
```
输入: sorted_qualified_vecs (已排序的vid列表)

build(start, end, tree_node):
  1. 创建 TempIndexNode{start, end, children=[], centroid=tree_node.centroid}
  2. 若 (end-start) ≤ max_sl_size 或叶子节点 → 返回（不再分裂）
  3. 按 vid 位前缀二分:
     for child_idx = 0..n_clusters-1:
       first = lower_bound(prefix < child_idx)
       last  = lower_bound(prefix < child_idx+1)
     → child_ranges[child_idx] = (first, last)
  4. 对非空子节点递归 build
  5. 记录 children 指向关系
```

**依赖关键不变式**: `tree_node->children.size() == n_clusters`（所有非叶子节点恰好有 n_clusters 个子节点）。因为代码直接做 `curr_node->children[child_idx]`，不检查越界。

**search_temp_index 流程**:
```
1. Beam Search (variance_boost=0，不需要方差修正):
   从根节点开始，展开子节点，保留 top-beam_size 个最佳节点
2. Frontier Search:
   优先队列弹出节点 → 若为"叶"（无子节点或范围≤search_ef）→ 收集该范围内所有 vids
   否则展开子节点并入队
3. 返回 top-k（用 RunningList 维护）

注意：当前 search_temp_index 使用 node_score 作为近似距离放入 RunningList，
实际代码要求调用方后续做精确距离重排。
```

#### 6.2 复杂谓词（30 分钟）

**核心文件**: [complex_predicate.h](src/complex_predicate.h)（257 行）、[complex_predicate.cpp](src/complex_predicate.cpp)（279 行）

**核心设计**: 使用**波兰表示法（Polish Notation）** 解析和求值逻辑表达式。

**两套求值体系**:

| 体系 | 用途 | 关键类型 |
|------|------|---------|
| **符号求值（Abstract）** | 在索引树上执行集合操作优化 | `StateNode(NONE/SOME/MOST/ALL)` + `ExprNode AST` |
| **布尔求值（Concrete）** | 直接在 access_list 上判断 | `evaluate_formula(tokens, access_list)` 模板函数 |

**State 的语义**:
```
NONE  → 无向量满足
SOME  → 少数向量满足（short_list 记录具体 vids）  
MOST  → 多数向量满足（exclude_list 记录少数不满足的 vids）
ALL   → 全部向量满足
UNKNOWN → 未确定
```

**表达式 AST 求值规则**（以 AndNode 为例）:
```
AND(l, r):
  NONE & any    → NONE
  ALL  & ALL    → ALL
  MOST & MOST   → MOST(exclude = l.exclude ∪ r.exclude)
  MOST & SOME   → SOME(short = r.short - l.exclude)
  SOME & SOME   → SOME(short = l.short ∩ r.short)
  otherwise     → UNKNOWN
```

**布尔求值（evaluate_formula）**:
```
输入: "42 AND (7 OR 15)"  ← 波兰表示法
Tokenize: ["42", "7", "15", "OR", "AND"]
栈求值:
  push 42∈access_list? → bool
  push 7∈access_list?  → bool
  push 15∈access_list? → bool
  pop,pop → OR → push
  pop,pop → AND → push
返回栈顶
```

**检验标准**:
- 能解释为什么 build_temp_index 需要 sorted_qualified_vecs 已排序
- 能画出 TempIndexNode 的 children 数组如何映射到主树的 TreeNode::children
- 能手动推演 `"A AND NOT(B OR C)"` 的波兰表达式求值过程

---

### 阶段 7：主编排类 `CuratorIndex`（预计 2.5 小时）

**目标**: 完整掌握所有搜索路径的调用链条和算法细节。这是**最重要的阶段**。

**核心文件**: [curator_index.h](src/curator_index.h)（138 行）、[curator_index.cpp](src/curator_index.cpp)（958 行）

#### 7.1 先读头文件（15 分钟）

阅读 [curator_index.h](src/curator_index.h)，建立类的 5 组 API 心智模型:

| API 组 | 方法 | 角色 |
|--------|------|------|
| Build | `train`, `add_vector`, `grant_access`, `batch_grant_access`, `flush` | 构建 |
| Query | `search`, `search_unfiltered`, `search_with_bitmap`(×2) | 查询 |
| Management | `revoke_access`, `remove_vector`(stub), `build_filter_index`, `get_filter_label` | 管理 |
| Memory | `memory_bytes`, `memory_breakdown`, `print_tree_info` | 诊断 |
| Profiling | `enable_profiling`, `last_profile` | 性能 |

**重点关注 `private` 成员变量的 10 组数据**:
```
1. cfg_                  配置
2. root_                 树根节点
3. vid_map_              外部vid→内部vid 映射
4. tid_map_              外部tid→内部tid 映射（含分配）
5. raw_buffer_ + vid_to_buf_offset_  原始向量缓冲（flush前）
6. vid_to_leaf_id_ + vid_to_local_idx_  向量→叶子映射
7. seq_to_vid_ + vid_to_seq_  序列号↔vid 映射（PQ/Flash 查找核心）
8. leaf_node_id_to_seq_ + num_leaves_ + leaf_region_size_  Flash 元数据
9. pq_ + flash_          子模块实例
10. temp_indexes_ + qualified_cache_  临时索引缓存
```

#### 7.2 构建流程代码精读（45 分钟）

在 curator_index.cpp 中，按以下顺序逐函数读：

| 函数 | 行号 | 关键点 |
|------|------|--------|
| `CuratorIndex(cfg)` | 134-147 | 构造函数：创建 root TreeNode + 设置 PQ 的 seq 映射指针 |
| `train()` | 158-160 | 仅一行：`build_tree(root_, n, x, cfg_)` — 代理给 cluster_tree |
| `add_vector()` | 162-196 | **核心构建函数**：assign_to_leaf → 位编码 vid → 映射管理 → raw_buffer 追加 → variance 向上更新 |
| `grant_access()` | 198-201 | 外部接口：ext_lid→int_lid 映射后调 grant_access_impl |
| `grant_access_impl()` | 204-230 | **递归核心**：叶子直接插入 shortlist；非叶子：短列表超限调 split_shortlist；BF 无此 tenant 则创建新短列表；否则递归下降 |
| `batch_grant_access()` | 232-235 | 批量版，用于 filter index 构建 |
| `flush()` | 238-340 | **最终化**：PQ 训练+编码 → 确定文件路径 → 写入磁盘 → 释放 PQ 码内存 → 打开块缓存 → DFS 遍历收集叶子 → 分配叶子序列号 → 按叶子分组向量 → flash 写入 → 释放 raw_buffer |

**add_vector 的 bit 编码细节**（必须理解）:
```cpp
// 叶子节点 local_vid: 该叶子中第几个向量
// 编码公式:
offset = 64 - leaf->level * 6 - 10;  // MAX_BRANCH_FACTOR_LOG2=6, MAX_LEAF_SIZE_LOG2=10
vid = leaf->node_id | (local_vid << offset);
```

#### 7.3 搜索流程代码精读（75 分钟）

**这是最重要的代码段！** 三种搜索路径按复杂度递增顺序读：

##### 路径 A: search_unfiltered（最简单，20 分钟）

行号: 483-542

```
search_unfiltered(x, k, distances, labels):
  1. node_priority = l2_sqr(query, centroid) - variance_boost * variance_mean
  2. MinHeap 遍历树:
     while pq非空 且 n_cand_vecs < nprobe:
       pop节点 → 若叶子则收集bucket → 若非叶子则展开子节点入队
  3. 按质心距离排序 buckets
  4. prune_thres 剪枝: 只处理 dist ≤ prune_thres × min_bucket_dist 的 bucket
  5. 对幸存 bucket: 遍历 vector_indices → compute_vector_distance → RunningList.insert
  6. 输出 top-k (RunningList 本身有序，直接取前k)
```

**为什么只有 unfiltered 使用 prune_thres？** → 因为无 label 过滤时只能遍历所有叶子，剪枝是必要的加速手段。

##### 路径 B: search/search_one（单租户标准路径，35 分钟）

行号: 319-478

**Phase 1: Beam Search** (行 377-391)
```
匿名 namespace 中的 beam_search():
  根节点 BF 有该 tenant → 加入 beam
  loop:
    对 beam 中每个节点:
      若短列表中有该 tenant → 保留在 next_beam (不再展开)
      若无短列表 → 展开子节点 → 计算 node_score → 全部加入 next_beam
    若没有节点展开过 → break (收敛)
    排序 next_beams → 保留 top-beam_width → 溢出部分推入 unexpanded 向量
  return beam + unexpanded（后续推入 frontier）
```

**Beam Search 的直觉**: 沿树下降时，每层保留 beam_size 个最有前途的节点，而不是只保留一个（避免局部最优）。

**Phase 2: Frontier Search** (行 394-465)
```
MinHeap<Candidate> frontier (初始化自 beam search 结果)

// ★ v2: 判断是否使用 PQ 加速
bool use_pq = cfg_.pq_enabled && pq_.is_trained() && pq_.has_cache();
std::vector<float> pq_d_table;      // 栈上局部变量，每 query 独立，线程安全
bool pq_table_built = false;

while frontier 非空:
  pop 最优节点
  if 节点短列表中有该 tenant:
    if use_pq:  // ══════ PQ 近似距离路径 ══════
      若 pq_table_built==false: build_distance_table (仅首次，构建 [M×256] 查表)
      compute_pq_distances(d_table, vids, output)
        → 内部：vid→seq_idx 转换 → prefetch_and_get_codes (一次加锁获取所有码)
        → ADC 累加 (无锁)
    else:       // ══════ 精确距离回退路径 ══════
      逐 vid 调 compute_vector_distance (FlashStore 读取全精度向量)
    → batch_insert 到 RunningList(top-search_ef)
    若 batch_insert 无更新 → break
  else if BF 中有该 tenant:
    展开子节点 → compute_child_scores_with_prefetch → 入队
```

**PQ 路径 vs 精确路径对比**:
| 维度 | PQ 路径 | 精确路径 |
|------|---------|---------|
| 读取量/向量 | M 字节 (16-128B) | d×4 字节 (512B-1.5KB) |
| 距离计算 | ADC 查表累加 (M 次加) | L2 平方 (d 次乘加) |
| I/O 模式 | 批量预取 (一次加锁) | 逐条 seek+read |
| 典型速度 | ~0.2 ms/shortlist | ~1.5 ms/shortlist |

**Phase 3: ADC Rerank** (行 441-460)
```
if pq_use_adc_rerank 且 PQ 已训练:
  取 RunningList 前 k × rerank_topk_factor 个
  → compute_vector_distance (全精度 L2)
  → 更新为精确距离的 RunningList
```

##### 路径 C: search_with_bitmap（bitmap filter，20 分钟）

行号: 547-601

```
search_with_bitmap(x, k, qualified, n_qualified):
  1. ext_vid → int_vid 转换 + 排序
  2. build_temp_index(root, sorted_vids, n_clusters, max_sl_size)
  3. search_temp_index(temp_nodes, sorted_vids, query, k, d, search_ef, beam_size)
  4. int_vid → ext_vid 输出
```

注意：`build_filter_index` 可以缓存 temp_index（`use_temp_index_caching=true`），下次同一谓词查询直接复用。

#### 7.4 compute_vector_distance（15 分钟）

行号: 607-637。**两级回退机制**:

```
compute_vector_distance(query, vid):
  1. 检查 tl_scratch 是否已 resize (thread_local, 首次调用时 resize)
  2. Flash 路径（优先）:
     vid → vid_to_leaf_id_ → leaf_node_id_to_seq_ → leaf_seq
     offset = leaf_seq * leaf_region_size_ + local_idx * d * sizeof(float)
     flash_.read_vector(offset, d, tl_scratch)
     return l2_sqr(query, tl_scratch, d)
  3. raw_buffer_ 回退（flush 前）:
     return l2_sqr(query, raw_buffer_ + vid_to_buf_offset_[vid], d)
  4. 都找不到 → 抛异常
```

**为什么两级？** → flush 前向量在内存 raw_buffer_，flush 后写入磁盘 flash。`flash_finalized_` 标志控制路径选择。

#### 7.5 管理操作（15 分钟）

| 函数 | 行号 | 逻辑 |
|------|------|------|
| `revoke_access` | 642-669 | 找含此 vid 的短列表 → erase → 若空则重算 BF → try_merge_shortlists 向上合并 |
| `build_filter_index` | 679-709 | 分配虚拟 tenant ID → 构建/缓存 temp_index（或 batch_grant_access 写入主索引） |
| `find_all_qualified_vecs` | 720-749 | 遍历所有向量，对每个向量收集其所属 tenant 集合（遍历叶到根的 shortlist），调用 evaluate_formula 判断 |

**检验标准**:
- 能手动画出 beam_search 在 beam_size=2 时的一轮执行过程（画树图，标注哪些节点被保留/展开/丢弃）
- 能解释 `batch_insert` 返回 false 时 frontier search 为何可以 break
- 能说出三种搜索路径的适用场景和关键参数差异
- 能画出 vid → flash_offset 的完整映射链（4 个 unordered_map）

---

### 阶段 8：CLI 入口 + Python 编排 + 完整调用链（预计 1 小时）

**目标**: 将各模块串联成完整的端到端流程。

#### 8.1 main.cpp（20 分钟）

[main.cpp](src/main.cpp)（325 行）

**bench 模式完整流程**:
```
main():
  1. 解析 CLI 参数 (--train_vecs, --train_access, --queries, --query_labels, --config, --k, etc.)
  2. load_config_from_json(config_path) → CuratorConfig
     // 自定义最小 JSON 解析器（不依赖 nlohmann/json）
  3. cnpy::npy_load 读取所有 .npy 数据
  4. 构建:
     CuratorIndex index(cfg)
     index.train(n, train_vecs)
     for i in ntrain: index.add_vector(vecs[i], i)
     for each (vid,tid) in access_pairs: index.grant_access(vid, tid)
     index.flush()
  5. 搜索:
     if batch_query: #pragma omp parallel for
     for each query: index.search(vec, k, tenant_id)
  6. write_json_results → results.json
  7. 若 --profile: 打印 SearchProfile 各字段
```

#### 8.2 Python 端（15 分钟）

[python/run_experiment.py](python/run_experiment.py)（171 行）

**run_experiment.py 的 4 步流程**:
```
Step 1: preprocess (.pkl → .npy)
  读取 train_mds.pkl → 转换为 [vid, tid] COO pairs → np.save("train_access.npy")

Step 2: 启动 C++ (subprocess.run)
  curator bench --train_vecs ... --config ... --k 10 --output results.json

Step 3: Compute Recall
  加载 results.json + ground_truth.npy → Recall@k 统计 (mean/median/P90/P99/min)

Step 4: Summary 报告
  打印构建时间、内存、查询数等概要信息
```

#### 8.3 CMakeLists.txt（10 分钟）

[CMakeLists.txt](CMakeLists.txt)

```
project(curator)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)

add_executable(curator
    src/main.cpp
    src/curator_index.cpp
    src/cluster_tree.cpp
    src/shortlist.cpp
    src/kmeans.cpp
    src/pq_codec.cpp
    src/pq_block_cache.cpp     # ★ v2: PQ 外存块缓存
    src/flash_store.cpp
    src/temp_index.cpp
    src/complex_predicate.cpp
    src/cnpy.cpp
)

target_link_libraries(curator PRIVATE OpenMP::OpenMP_CXX)
```

仅依赖: **cnpy (.npy I/O) + OpenMP (并行)**，零 FAISS 依赖！

#### 8.4 完整调用链对照（15 分钟）

对照 [REFACTOR_PLAN.md](REFACTOR_PLAN.md) §五的调用链图，在源代码中逐一确认:

**构建链**（见 curator_index.cpp + cluster_tree.cpp):
```
main → CuratorIndex::train → build_tree (递归kmeans)
     → add_vector → assign_to_leaf + 位编码 + variance更新
     → grant_access → grant_access_impl (递归插入, split/merge)
     → flush → pq_.train + pq_.encode_all → pq_.write_to_disk → 释放codes内存
             → pq_.open_cache (开LRU块缓存) → flash_.write
```

**单租户搜索链**（见 curator_index.cpp 第 339-500 行):
```
search → search_one
  → beam_search (匿名namespace)
  → frontier search (MinHeap loop)
    ├── PQ路径: build_distance_table (首次) → compute_pq_distances
    │     → prefetch_and_get_codes (一次加锁获取所有PQ码) → ADC累加 (无锁)
    └── 精确路径: compute_vector_distance (flash→buffer 两级回退)
  → [optional] ADC rerank (pq_use_adc_rerank)
  → int_vid→ext_vid 转换
```

**无过滤搜索链**（见 curator_index.cpp 第 483-542 行):
```
search_unfiltered → MinHeap 节点遍历 + nprobe限制 + prune_thres剪枝
```

**Bitmap Filter 搜索链**（见 curator_index.cpp 第 547-601 行 + temp_index.cpp):
```
search_with_bitmap → build_temp_index → search_temp_index (beam + frontier)
```

---

## 三、总时间预算

| 阶段 | 内容 | 预计时间 | 累计 |
|:---:|------|:---:|:---:|
| 0 | 预备知识确立 | 15min | 15min |
| 1 | 基础工具层 (common.h) | 40min | 55min |
| 2 | 配置与数据结构层 (config + tree_node + bloom) | 30min | 1.5h |
| 3 | 距离计算与 Profiling (distance + profiling) | 20min | 1.8h |
| 4 | 树构建+K-means+短列表 (cluster_tree + kmeans + shortlist) | 1.5h | 3.3h |
| 5 | PQ 与 Flash 存储 (pq_codec + pq_block_cache + flash_store) | 1.2h | 4.5h |
| 6 | 临时索引与复杂谓词 (temp_index + complex_predicate) | 1h | 5.5h |
| 7 | 主编排类 CuratorIndex (curator_index.cpp 全量) | 2.5h | 8.0h |
| 8 | CLI+Python+全调用链串联 | 1h | 9.0h |
| — | **缓冲/复习/验证** | **1-2h** | **~10-11h** |

---

## 四、学习建议

### 4.1 主动学习法

每读完一个阶段，做以下练习：

1. **默写**: 合上代码，默写该模块的核心 API 签名和关键数据结构
2. **画图**: 画出数据流图（如 build_tree 的递归过程、beam_search 的展开模式）
3. **提问**: 对每段代码问"为什么这样设计？"（代码中已有详注的除外）

### 4.2 关键文件阅读顺序（快捷索引）

如果你只有 5 小时而非 10 小时，按此优先级：

| 优先级 | 文件 | 原因 |
|:---:|------|------|
| P0 | [common.h](src/common.h) | 所有代码的类型基础，**必须 100% 掌握** |
| P0 | [curator_index.cpp](src/curator_index.cpp) | 主编排逻辑，包含全部搜索路径 |
| P1 | [cluster_tree.cpp](src/cluster_tree.cpp) | 树构建和遍历 |
| P1 | [tree_node.h](src/tree_node.h) | 核心数据结构 |
| P2 | [pq_codec.cpp](src/pq_codec.cpp) | PQ 训练/编码/距离计算 |
| P2 | [pq_block_cache.cpp](src/pq_block_cache.cpp) | ★ PQ 外存 LRU 块缓存 |
| P2 | [temp_index.cpp](src/temp_index.cpp) | Bitmap Filter 搜索加速 |
| P3 | [shortlist.cpp](src/shortlist.cpp) | 短列表维护逻辑 |
| P3 | [kmeans.cpp](src/kmeans.cpp) | K-means 实现 |
| P4 | [flash_store.cpp](src/flash_store.cpp) | 文件 I/O（较独立） |
| P4 | [complex_predicate.cpp](src/complex_predicate.cpp) | 谓词解析（较独立） |
| P5 | [main.cpp](src/main.cpp) | CLI 入口 |
| P5 | [distance.h](src/distance.h) | 距离函数（简单） |

### 4.3 不变量速查卡片

打印或手写此卡片，放在手边：

```
┌─────────────────────────────────────────────┐
│           Curator 不变量速查卡片              │
├─────────────────────────────────────────────┤
│ 1. 所有非叶子节点恰好有 n_clusters 子节点     │
│ 2. int_vid_t 位编码: 每层6bit + 最后10bit    │
│ 3. 短列表 ≤ max_sl_size（超限触发 split）     │
│ 4. BF 汇总子树所有租户（含子节点 BF 位或）    │
│ 5. Flash 叶子 region 固定大小（含空洞）       │
│ 6. PQ nbits=8 硬约束                         │
│ 7. PQ 码在外存（磁盘），搜索时按需加载        │
│ 8. PQ 块缓存默认 256×4096 条 (~128MB @M=128)  │
│ 9. TempIndexNode::centroid 为非拥有指针       │
├─────────────────────────────────────────────┤
│ ID 类型速记:                                  │
│   ext_vid_t = uint32_t  (用户 label)          │
│   int_vid_t = uint64_t  (路径编码 vid)        │
│   ext_lid_t = int16_t   (用户 tenant ID)      │
│   int_lid_t = int16_t   (内部 tenant ID)      │
├─────────────────────────────────────────────┤
│ 搜索参数对照:                                  │
│   unfiltered: nprobe + prune_thres            │
│   tenant:     search_ef + beam_size           │
│   bitmap:     search_ef + beam_size (temp)    │
└─────────────────────────────────────────────┘
```

### 4.4 验证 100% 掌握度的 10 道自测题

掌握所有阶段后，回答以下问题。若能全部答对，就达到了 100% 掌握度：

1. `add_vector` 中 vid 是如何用位编码的？写出 `offset` 的计算公式
2. `grant_access_impl` 在非叶子节点上如何决定是插入短列表、split 还是递归下降？
3. `beam_search` 的收敛条件是什么？为什么一定能收敛？
4. `search_one` 的 frontier search 在什么条件下 break？
5. `search_unfiltered` 为什么不使用短列表，而直接扫描 leaf vector_indices？
6. `build_temp_index` 为什么需要依赖 `children.size() == n_clusters` 不变式？
7. PQ 的 ADC rerank 为什么使用 `rerank_topk_factor` 倍候选集，而不是全部候选？
8. `compute_vector_distance` 的两级回退逻辑（flash → buffer）是什么？
9. `RunningList::batch_insert` 为什么比逐个 `insert` 更高效？
10. 如果要新增一种搜索模式（如 AND-OR 复合 tenant 过滤），你会修改哪些文件/函数？

---

## 五、代码修改指南

当需要修改代码时，参考以下影响范围评估：

| 修改类型 | 涉及文件 | 风险 |
|---------|---------|:--:|
| 增加/修改配置参数 | `config.h` + `main.cpp`(JSON 解析) + `curator_index.cpp`(使用处) | 低 |
| 修改距离函数 | `distance.h` | 低（header-only，易测试） |
| 修改 K-means | `kmeans.cpp` | 中（影响 build_tree 结果） |
| 修改树构建逻辑 | `cluster_tree.cpp` | 高（影响不变式、temp_index 兼容性） |
| 修改搜索算法 | `curator_index.cpp`（匿名 namespace + search_one/search_unfiltered/search_with_bitmap） | 高（影响 recall 和延迟） |
| 修改 PQ 编码/缓存 | `pq_codec.cpp` + `pq_block_cache.cpp` | 中（影响磁盘格式兼容性；块缓存配置可调） |
| 修改 Flash 存储布局 | `flash_store.cpp` + `curator_index.cpp`（flush 组织逻辑） | 高（Flash 文件无法跨版本兼容） |
| 新增搜索模式 | `curator_index.h/.cpp`（添加新的 search_xxx 方法） | 中 |

---

*文档创建日期: 2026-07-01*  
*预计完整学习时间: 11-13 小时（含缓冲复习）*
