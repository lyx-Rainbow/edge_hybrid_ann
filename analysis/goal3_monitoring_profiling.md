# 目标3：监视 Curator 索引各部分的内存开销和查询各步骤的时间开销

## 一、现有基础分析

### 1.1 现有内存监控能力（经源码核查）

#### 已存在且有价值的基础

| 监控项 | 位置 | 当前状态 | 可利用性 |
|--------|------|----------|----------|
| `get_total_memory_bytes()` | C++ .cpp:2316-2356 | 返回索引总内存（树节点+BF+短列表+PQ码+临时缓存等）。返回 `size_t`，**已有 SWIG 暴露** | **高**：直接可用 |
| `get_cached_temp_index_memory_usage()` | C++ .cpp:1460-1479 | 返回临时索引缓存内存。已有 SWIG 暴露 | **高**：直接可用 |
| `memory_usage()` | C++ .cpp:1604-1712 | **组件级分解**！已统计：树节点数、Bloom Filter 字节、短列表总字节、短列表 payload、向量索引、质心、节点属性。但是：**(a) void 函数，仅 printf 到 stdout**，**(b) 遗漏了 PQ 码、Flash 缓冲、临时索引缓存**，**(c) 未暴露给 Python** | **中高**：代码逻辑成熟（~110 行），可改造为返回结构化数据 |
| `getCurrentRSS()` / `getPeakRSS()` | Python `memory_utils.py` | 跨平台 RSS 获取，已有 | **高**：直接可用 |
| `QueryRSSSampler` | Python `memory_utils.py:138-158` | 查询阶段 RSS 采样器 | **高**：直接可用 |
| `get_page_file_size_bytes()` | Python `curator.py:264-270` | Flash 文件大小 | **高**：直接可用 |

#### `memory_usage()` 的组件分解详情

| 统计项 | 变量名 | 包含内容 |
|--------|--------|----------|
| 树节点数 | `num_tree_nodes` | 所有节点的数量 |
| Bloom Filter 总字节 | `bloom_filter_total_size` | 所有节点的 BF 位数组 |
| 短列表总字节 | `short_lists_total_size` | 含 unordered_map 开销 + ShortList 容量 |
| 短列表 payload 字节 | `short_lists_payload_total_size` | ShortList 中实际数据的字节数 |
| 向量索引字节 | `vector_indices_total_size` | 叶子节点的 vector_indices |
| 质心字节 | `centroid_total_size` | num_nodes × d × sizeof(float) |
| 节点属性字节 | `node_attrs_total_size` | TreeNode 结构体固定部分 |
| ID 分配器字节 | `id_allocator_total_size` | label_to_id + id_to_label 映射表 |
| **遗漏** | — | **PQ 码本（`pq_codebook`）、PQ 编码（`vid_to_pq_code`）、Flash 索引映射（`vid_to_buffer_offset`/`leaf_node_id_to_seq`）、原始向量缓冲（`raw_vectors_buffer`）、临时索引缓存（`cached_temp_indexes`/`cached_qualified_vecs`）** |

#### Python 端已有的构建阶段快照

`run_curator.py:164-248` (`build_curator_index`) 和 `diagnose_build_time.py:87-127` 在以下时间点采集内存：
1. 数据加载后（RSS）
2. 构建完成后（RSS + index 自报内存）
3. 查询峰值（RSS）

粒度尚可，但未分解到 C++ 组件级别。

### 1.2 现有查询时间监控能力（经源码核查）

#### Profiling 的真实覆盖范围

**关键发现**：C++ 端的 profiling 基础设施（`SearchProfilingData`、`enable_profiling`、计时宏）**仅存在于 `search_with_bitmap_filter` 和 `search_with_bitmap_filter_optimized` 两个方法中**。

| 方法 | profiling 覆盖 | 说明 |
|------|:---:|------|
| `search_with_bitmap_filter()` (.cpp:1875-1961) | ✅ 完整 | preproc→sort→build_temp_index→search 四阶段计时 |
| `search_with_bitmap_filter_optimized()` (.cpp:1963-2027) | ✅ 完整 | 跳过 preproc/sort，仅 build_temp_index→search |
| `search_one(tid)` (.cpp:899-1000) | ❌ 无 | **这是标准单标签查询的主路径**，含 beam_search→frontier→PQ/exact→rerank 但无任何计时 |
| `search_one(unfiltered)` (.cpp:1004-1065) | ❌ 无 | 无过滤查询路径 |
| `search_temp_index()` (.cpp:1719-1869) | ❌ 无 | 复杂谓词的临时索引搜索路径 |

**Profiling 变量声明**（.h:377-387）：
```cpp
struct SearchProfilingData {
    double preproc_time_ms;    // 仅 bitmap_filter
    double sort_time_ms;       // 仅 bitmap_filter
    double build_temp_index_time_ms; // 仅 bitmap_filter
    double search_time_ms;     // 仅 bitmap_filter
    size_t qualified_labels_count;
    size_t temp_nodes_count;
};
mutable SearchProfilingData last_search_profile;
mutable bool enable_profiling = false;
```

**结论**：当前 profiling 是为 bitmap_filter 专用设计的，**标准查询路径完全没有时间分解能力**。

#### Python 端的查询计时

`run_curator.py:277-283` 使用 `time.perf_counter()` 测量单次 `query()` 的 wall-clock 总延迟。这是唯一可用于所有查询路径的计时手段，但仅能获得总时间。

### 1.3 现有诊断脚本

| 脚本 | 功能 | 状态 |
|------|------|------|
| `tests/diagnose_build_time.py` | 构建各阶段耗时分解（train/create/grant/pq_train/flash_write） | 可用，调用了 C++ `train_pq_codebook()` + `finalize_flash_storage()` 分开计时 |
| `tests/diagnose_memory.py` | （预期存在，`tests/` 下未找到） | 需创建 |
| `tests/test_diskivf_build_niter.py` | DiskIVF 构建测试 | 仅 DiskIVF |

---

## 二、执行计划

### 阶段 A：内存监控增强

#### A1. 改造 `memory_usage()` → `get_memory_breakdown()`

将现有的 void 打印函数改造为返回结构化数据的方法：

```cpp
// 新增结构体 (在 .h 中)
struct MemoryBreakdown {
    // 树结构
    size_t num_tree_nodes;
    size_t tree_node_attrs_bytes;
    size_t centroids_bytes;
    
    // Bloom Filters
    size_t bloom_filter_bytes;
    
    // 短列表
    size_t shortlists_overhead_bytes;    // unordered_map 开销
    size_t shortlists_payload_bytes;     // 实际 vid 数据
    
    // 向量索引（叶子节点）
    size_t vector_indices_bytes;
    
    // ID 分配器
    size_t id_allocator_bytes;
    size_t tenant_id_allocator_bytes;
    
    // PQ（当前遗漏的重要部分）
    size_t pq_codebook_bytes;
    size_t pq_codes_bytes;              // vid_to_pq_code 总量
    
    // Flash 存储相关
    size_t flash_index_bytes;           // vid_to_buffer_offset 等映射表
    size_t raw_vectors_buffer_bytes;    // flush 前有值，flush 后为 0
    
    // 临时索引缓存
    size_t temp_index_cache_bytes;
    size_t temp_qualified_vecs_bytes;
    
    // 合计
    size_t total_bytes;
};

MemoryBreakdown get_memory_breakdown() const;
```

**实现策略**：90% 的代码已存在于 `memory_usage()` 中，仅需：
1. 返回值改为 `MemoryBreakdown` 结构体
2. 添加 PQ 相关统计（参考 `get_total_memory_bytes()` 中的 PQ 统计逻辑）
3. 添加临时索引缓存统计（复用 `get_cached_temp_index_memory_usage()` 逻辑）
4. 移除 printf 调用

#### A2. Python 端封装

在 `2_Utils/memory_utils.py` 中新增 `MemoryProfiler` 类：

```python
class MemoryProfiler:
    """Curator 索引内存分析器"""
    
    def __init__(self, index: Curator):
        self.index = index
    
    def get_breakdown(self) -> dict:
        """获取组件级内存分解，返回规范化的 dict"""
        raw = self.index.get_memory_breakdown()  # C++ struct → Python dict
        return {
            "tree": {
                "num_nodes": raw.num_tree_nodes,
                "node_attrs_mb": raw.tree_node_attrs_bytes / (1024*1024),
                "centroids_mb": raw.centroids_bytes / (1024*1024),
            },
            "bloom_filters_mb": raw.bloom_filter_bytes / (1024*1024),
            "shortlists": {
                "overhead_mb": raw.shortlists_overhead_bytes / (1024*1024),
                "payload_mb": raw.shortlists_payload_bytes / (1024*1024),
            },
            "vector_indices_mb": raw.vector_indices_bytes / (1024*1024),
            "id_allocators_mb": raw.id_allocator_bytes / (1024*1024),
            "pq": {
                "codebook_mb": raw.pq_codebook_bytes / (1024*1024),
                "codes_mb": raw.pq_codes_bytes / (1024*1024),
            },
            "flash_index_mb": raw.flash_index_bytes / (1024*1024),
            "raw_vectors_buffer_mb": raw.raw_vectors_buffer_bytes / (1024*1024),
            "temp_cache_mb": raw.temp_index_cache_bytes / (1024*1024),
            "total_index_mb": raw.total_bytes / (1024*1024),
        }
    
    def snapshot(self) -> dict:
        """包含 RSS 和 index 内存的完整快照"""
        import memory_utils
        return {
            "rss_mb": memory_utils.getCurrentRSS() / (1024*1024),
            "index_total_mb": self.index.get_index_memory_bytes() / (1024*1024),
            "breakdown": self.get_breakdown(),
            "temp_cache_mb": self.index.get_cached_temp_index_memory_usage() / (1024*1024),
        }
    
    def build_phase_snapshots(self, ...): ...
```

#### A3. 构建阶段自动快照

在 `Curator/run_curator.py` 的 `build_curator_index()` 中，利用 `MemoryProfiler` 在各阶段自动记录：

```python
profiler = MemoryProfiler(index)
snapshots = {}

profiler.snapshot()  # baseline (训练前)
index.train(train_vecs)
snapshots["after_train"] = profiler.snapshot()

# ... add vectors ...
snapshots["after_add"] = profiler.snapshot()

# ... grant access ...
snapshots["after_grant"] = profiler.snapshot()

index.flush()
snapshots["after_flush"] = profiler.snapshot()
```

输出 JSON 中包含完整的内存时间线。

### 阶段 B：查询时间监控增强

#### B1. C++ 端：为标准查询路径添加 profiling

**方案**：扩展现有 `SearchProfilingData` 结构体，并在 `search_one(tid)` 中插入计时点。

```cpp
struct SearchProfilingData {
    // 现有（bitmap_filter 用）
    double preproc_time_ms;
    double sort_time_ms;
    double build_temp_index_time_ms;
    double search_time_ms;
    size_t qualified_labels_count;
    size_t temp_nodes_count;
    
    // 新增（标准单标签查询用）
    double beam_search_time_ms;
    int beam_layers_visited;
    int beam_nodes_scored;
    
    double frontier_search_time_ms;
    int frontier_nodes_popped;
    int frontier_shortlists_scanned;
    int frontier_children_expanded;
    
    double pq_table_build_time_ms;      // 构建距离查找表
    double pq_distance_compute_time_ms; // 查表累加距离
    double exact_distance_compute_time_ms; // 精确 L2 距离计算
    
    double candidate_merge_time_ms;
    double rerank_time_ms;
    int rerank_count;
    
    double total_search_time_ms;
    
    // 查询类型标记：'standard' | 'bitmap_filter' | 'unfiltered' | 'temp_index'
    std::string query_type;
};
```

**计时插入位置**（在 `search_one(tid)` 中）：

```
search_one(tid) {
    [计时] beam_search()          → beam_search_time_ms
    [计时] while (!frontier.empty()) {
        if (has shortlist):
            [计时] compute_pq_distances()  → pq_distance_compute_time_ms
            [计时] batch_insert()          → candidate_merge_time_ms
        else:
            [计时] compute_child_scores()  → frontier expansion
    }
    [计时] rerank loop            → rerank_time_ms
    total = sum of above
}
```

**设计原则**：
- `enable_profiling = true` 时记录，`false` 时跳过（零开销）
- 使用 `std::chrono::high_resolution_clock`
- 避免在每个短列表扫描处都创建 timer 对象——使用手动 tick/tock 宏

#### B2. C++ 端：新增 profiling 访问方法

```cpp
// 通用访问器（替代现有的 bitmap_filter 专用方法）
SearchProfilingData get_last_search_profile() const;  // 已有，保留

// 新增：便捷查询
double get_last_total_search_time_ms() const;
std::string get_last_query_type() const;
```

#### B3. Python 端：查询 Profiler

在 `2_Utils/query_profiler.py` 中新增：

```python
class QueryProfiler:
    """Curator 查询性能分析器"""
    
    def __init__(self, index: Curator):
        self.index = index
    
    def profile_single(self, x, k, tenant_id) -> dict:
        """执行一次查询，返回完整时间分解"""
        self.index.enable_stats_tracking(True)
        result = self.index.query(x, k, tenant_id)
        profile = self.index.get_last_search_profile()
        return {
            "result_ids": result,
            "total_ms": profile.total_search_time_ms,
            "beam_search_ms": profile.beam_search_time_ms,
            "frontier_search_ms": profile.frontier_search_time_ms,
            "pq_table_ms": profile.pq_table_build_time_ms,
            "pq_distance_ms": profile.pq_distance_compute_time_ms,
            "rerank_ms": profile.rerank_time_ms,
            "frontier_stats": {
                "nodes_popped": profile.frontier_nodes_popped,
                "shortlists_scanned": profile.frontier_shortlists_scanned,
            },
        }
    
    def profile_batch(self, queries, k, tenant_ids) -> list[dict]:
        """批量 profile，返回 DataFrame-ready 列表"""
        ...
    
    def profile_by_selectivity(self, queries, query_info, k) -> dict:
        """按选择率分桶统计各步骤耗时分布"""
        ...
```

#### B4. Curator Python 封装增强

修改 `Curator.query()` 添加可选的 profiling 模式：

```python
def query(self, x, k, tenant_id=None, profile=False):
    if profile:
        self.enable_stats_tracking(True)
    # ... 现有查询逻辑 ...
```

### 阶段 C：诊断脚本

基于新的 profiling 基础设施，创建以下诊断脚本：

| 脚本 | 功能 |
|------|------|
| `tests/profile_memory_components.py` | 输出各组件内存占比（表格 + JSON） |
| `tests/profile_query_steps.py` | 按选择率分桶输出查询各步骤耗时 |
| `tests/profile_memory_timeline.py` | 构建过程各阶段内存时间线 |

### 阶段 D：可视化

在 `5_Plot/` 下新增或修改：

| 脚本 | 图表 |
|------|------|
| `fig3_memory.py`（修改） | 增加组件级堆叠柱状图 |
| `fig5_query_time_breakdown.py`（新增） | 查询各步骤耗时堆叠柱状图 |
| `fig6_memory_timeline.py`（新增） | 构建过程内存变化折线图 |

### 阶段 E：Benchmark 集成

修改 `run_curator.py` 和 `run_curator_sweep.py`：
- 增加 `--profile` 标志，启用详细 profiling 输出
- 结果 JSON 中增加 `memory_breakdown` 字段（构建后）
- 结果 JSON 中增加 `query_breakdown` 字段（查询阶段，采样而非全量）

---

## 三、预期效果

### 3.1 内存监控预期输出示例

```
Curator Memory Breakdown (yfcc100m, n=800K, d=192, M=16, nlist=32):
  Component                    Size (MB)    %
  ──────────────────────────────────────────
  Tree Structure
    ├─ Node attributes            1.2       1%
    ├─ Centroids                  0.2       0%
  Bloom Filters                   8.2       5%
  Short Lists
    ├─ Overhead (hash tables)    12.3       8%
    ├─ Payload (vid data)        33.3      22%
  Vector Indices (leaf)          19.2      13%
  ID Allocators                   6.1       4%
  PQ
    ├─ Codebook                   0.3       0%
    ├─ Codes                     12.8       8%
  Flash Index (offset maps)       6.4       4%
  Raw Vectors Buffer              0.0       0%  (flushed)
  Temp Index Cache                0.0       0%
  ──────────────────────────────────────────
  Total (index-reported)        100.0     100%
  RSS (OS-reported)             158.5      —
  Python/OS overhead             58.5      —
```

### 3.2 查询时间监控预期输出示例

```
Query Time Breakdown (yfcc100m, selectivity=0.01, k=10, search_ef=128):
  Phase                       Time (ms)    %
  ──────────────────────────────────────────
  Beam Search                    0.05       2%
    ├─ layers visited: 4
    └─ nodes scored: 128
  PQ Table Build                 0.02       1%
  Frontier Search                1.20      50%
    ├─ nodes popped: 24
    ├─ shortlists scanned: 8
    └─ children expanded: 16
  PQ Distance Compute            0.30      13%
  Candidate Merge                0.08       3%
  ADC Rerank (top-40)            0.55      23%
  Other                          0.20       8%
  ──────────────────────────────────────────
  Total (C++)                    2.40     100%
  Python overhead                0.05       —
  Wall clock                     2.45       —
```

### 3.3 监控的核心价值

1. **识别优化方向**：若短列表 payload 占 50% 内存 → 考虑压缩；若 frontier search 占 50% 时间 → 调整 `search_ef`/`beam_size`
2. **目标1 验证**：`pq_codes_mb` 从 12.8MB → 0MB（PQ 外存后）
3. **目标2 验证**：GIST1M 高维场景下各步骤的时间分布 vs SIFT1M 低维场景
4. **跨参数对比**：不同 PQ_M、nlist、max_sl_size 下的内存/时间 trade-off
5. **异常检测**：RSS 与 index_total 差异过大 → 内存泄漏或碎片

---

## 四、验收方式

### 4.1 内存监控验收

```bash
# 组件分解完整性
python tests/profile_memory_components.py --dataset yfcc100m_small
# 输出所有组件，验证 sum(breakdown) ≈ get_total_memory_bytes() (误差 < 1%)

# 构建阶段快照
python tests/profile_memory_timeline.py --dataset yfcc100m_small
# 验证 flush 后 raw_vectors_buffer = 0，PQ codes > 0

# 跨数据集对比
python tests/profile_memory_components.py --dataset arxiv_small
# 验证 d=384 时 centroids 大小约为 d=192 时的 2 倍
```

### 4.2 查询时间监控验收

```bash
# 时间分解完整性
python tests/profile_query_steps.py --dataset yfcc100m_small --n_queries 200
# 验证各步骤时间之和 ≈ 总查询时间（误差 < 5%）

# 多参数验证
python tests/profile_query_steps.py --dataset yfcc100m_small \
    --sweep '{"search_ef": [64, 128, 256, 512, 1024]}'
# 验证 frontier_search_ms 随 search_ef 增大而增大

# 零开销验证
python tests/profile_query_steps.py --dataset yfcc100m_small \
    --profile-enabled false --n_queries 500
# 验证 disable profiling 时延迟与 baseline 一致（< 1% 差异）
```

### 4.3 集成验收

```bash
# --profile 标志可用
python Curator/run_curator.py --dataset yfcc100m_small --profile
# 输出 JSON 包含 memory_breakdown 和 query_breakdown 字段

# sweep 集成
python Curator/run_curator_sweep.py --dataset yfcc100m_small \
    --config 3_Config/Curator/curator_yfcc100m_small.json \
    --sweep 3_Config/Curator/sweep_test.json --profile
# 每个 sweep point 包含完整 profile
```
