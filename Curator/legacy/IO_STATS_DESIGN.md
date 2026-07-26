# Curator 查询阶段 I/O 统计 — 设计方案 v2 (简化版)

> **目标**: 统计 Curator 索引在查询阶段的 I/O 次数及 I/O 总耗时，两个 I/O 源（PQ Block Cache / Flash Store）分开统计，利用小型数据集 `sift1m_small` 验证。
>
> **设计原则**: 最小侵入性 — 在底层 I/O 网关累计全局统计，查询结束后统一读取；不修改任何搜索函数内部逻辑。

---

## 一、现状分析

### 1.1 查询阶段 I/O 源（共 2 处）

Curator 查询阶段的所有磁盘 I/O 仅经过两个底层网关：

| I/O 源 | 底层调用 | 数据内容 | 触发场景 |
|--------|---------|---------|---------|
| **PQ Block Cache** | `load_block_locked()` → `::pread()` | PQ 压缩码 (每向量 M 字节) | PQ 距离计算时缓存未命中 |
| **Flash Store** | `read_vector()` / `read_batch()` → `::pread()` | 全精度 float32 向量 (每向量 d×4 字节) | 精确距离回退 / ADC Rerank |

> 构建阶段的 I/O 不在统计范围内。本方案仅统计查询阶段的 I/O。

### 1.2 现有统计基础设施

| 组件 | 现有统计 | 缺失 |
|------|---------|------|
| `PQBlockCache::Stats` | ✅ `io_count`, `bytes_read`, `cache_hits`, `cache_misses` | ❌ `io_time_ms` |
| `FlashStore` | ❌ 无任何统计 | ❌ 全部缺失 |
| `CuratorIndex` | ✅ `pq_cache_stats()` | ❌ 无 Flash stats 访问器、无统一 reset 接口 |

---

## 二、设计方案: 全局累计 + 查询前后采样

### 2.1 核心思路

```
main.cpp:
  index.reset_io_stats()           // ① 查询前清零
  ─────────────────────────────────
  for each query:                  // ② 查询循环
      index.search(...)            //    I/O 自动累计到全局计数器
  ─────────────────────────────────
  flash_s = index.flash_io_stats() // ③ 查询后读取
  pq_s    = index.pq_cache_stats()
  print flash_s, pq_s              // ④ 分别输出两个源的数据
```

**为什么不需要 per-query 快照**: 需求是"查询阶段的 I/O 总次数与总耗时"，不是每条查询的分解。全局计数器单调递增，查询前后的差值即为总量。这个方案:

- **自动覆盖所有查询路径**: `search_one()`、`search_unfiltered()`、temp_index 快速路径 — 无论走哪条路，只要调用了 `FlashStore::read_*` 或 `PQBlockCache::load_block_locked`，就会被全局计数器捕获
- **线程安全**: `batch_query` 并发模式下，各线程的 I/O 全部计入全局计数器，reset 和 read 在并行区域外执行，不存在数据竞争
- **零侵入搜索函数**: `search_one()`、`search_unfiltered()` 函数体完全不变

### 2.2 数据结构

#### FlashStore::Stats（新增）

```cpp
// flash_store.h — 在 FlashStore 类 public 区域新增
struct Stats {
    uint64_t read_count  = 0;   // pread() 调用次数
    uint64_t bytes_read  = 0;   // 读取总字节数
    double   io_time_ms  = 0;   // pread() 总耗时 (毫秒)

    void reset() { *this = Stats{}; }
};
```

- `read_vector()`: 每次调用计入 `read_count += 1`
- `read_batch()`: 对每次 `::pread()` (含合并读取) 计入 `read_count += 1`

#### PQBlockCache::Stats（扩展 1 字段）

```cpp
// pq_block_cache.h — 在现有 Stats 中新增最后一行
struct Stats {
    uint64_t cache_hits   = 0;
    uint64_t cache_misses = 0;
    uint64_t bytes_read   = 0;
    uint64_t io_count     = 0;
    double   io_time_ms   = 0;     // ★ 新增

    void reset() { *this = Stats{}; }
    double hit_rate() const { /* 不变 */ }
};
```

### 2.3 各文件修改清单

#### 文件 1: `Curator/src/flash_store.h`

| 修改 | 行数 |
|------|:---:|
| 新增 `#include <mutex>`（`Stats` / `mutable stats_mutex_` 需要） | 1 |
| 在 `FlashStore` 类 public 区域新增 `struct Stats { ... }` | 8 |
| 新增 `Stats stats() const` 声明 | 1 |
| 新增 `void reset_stats()` 声明 | 1 |
| 在 private 区域新增 `mutable std::mutex stats_mutex_` | 1 |
| 在 private 区域新增 `mutable Stats stats_` | 1 |

> **注意**: `<chrono>` **不**加在头文件中——`Stats` 仅用 `uint64_t`/`double`，不需要 chrono 类型。`<chrono>` 仅在 `.cpp` 中包含。

#### 文件 2: `Curator/src/flash_store.cpp`

修改 `read_vector()` 和 `read_batch()`，在每次 `::pread()` 调用处包裹计时。

**`read_vector()` 修改** (在现有的 `::pread()` 前后添加计时):

```cpp
void FlashStore::read_vector(size_t offset, size_t d, float* out) const {
    int fd = fileno(fp_);

    auto t0 = std::chrono::high_resolution_clock::now();
    ssize_t nread = ::pread(fd, out, d * sizeof(float),
                            static_cast<off_t>(offset));
    auto t1 = std::chrono::high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.read_count++;
        stats_.bytes_read += (d * sizeof(float));
        stats_.io_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    if (nread != static_cast<ssize_t>(d * sizeof(float))) {
        throw std::runtime_error("FlashStore: pread failed in read_vector");
    }
}
```

**`read_batch()` 修改** (对每次合并读/单次读的 `::pread()` 包裹计时):

```cpp
// 单次读取 (约第114行):
auto t0 = std::chrono::high_resolution_clock::now();
ssize_t nread = ::pread(fd, out.data() + sorted[i].second * d,
                        d * sizeof(float),
                        static_cast<off_t>(sorted[i].first));
auto t1 = std::chrono::high_resolution_clock::now();
{
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.read_count++;
    stats_.bytes_read += (d * sizeof(float));
    stats_.io_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// 合并读取 (约第125行):
auto t0 = std::chrono::high_resolution_clock::now();
ssize_t nread = ::pread(fd, buf.data(), merge_bytes,
                        static_cast<off_t>(merge_start));
auto t1 = std::chrono::high_resolution_clock::now();
{
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.read_count++;
    stats_.bytes_read += merge_bytes;
    stats_.io_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
}
```

顶部新增 `#include <chrono>` 和 `#include <mutex>`。

新增两个方法:

```cpp
FlashStore::Stats FlashStore::stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void FlashStore::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.reset();
}
```

#### 文件 3: `Curator/src/pq_block_cache.h`

在现有 `Stats` 结构体中新增 1 个字段:

```cpp
double io_time_ms = 0;   // ★ 新增
```

`reset()` 方法已使用 `*this = Stats{}`，自动清零 `io_time_ms`，无需额外修改。

#### 文件 4: `Curator/src/pq_block_cache.cpp`

> **线程安全关键**: `load_block_locked()` 始终在 `mutex_` 已持有的上下文中被调用（3 个调用点均在 `std::lock_guard<std::mutex> lock(mutex_)` 内）。因此在该函数中直接访问 `stats_`，**不能**加新的 `lock_guard`（`std::mutex` 非递归，会导致死锁）。

在 `load_block_locked()` 的 `::pread()` 调用处包裹计时:

```cpp
void PQBlockCache::load_block_locked(size_t block_id) {
    // ... 现有代码（计算 offset, bytes_to_read, 分配 entry）不变 ...

    auto t0 = std::chrono::high_resolution_clock::now();
    ssize_t nread = ::pread(fd_, entry.data.data(), bytes_to_read,
                            static_cast<off_t>(offset));
    auto t1 = std::chrono::high_resolution_clock::now();

    if (nread != static_cast<ssize_t>(bytes_to_read)) {
        fprintf(stderr, "PQBlockCache::load_block_locked: I/O error ...");
    }

    stats_.bytes_read += bytes_to_read;
    stats_.io_count++;
    stats_.io_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ... 现有 LRU 插入代码不变 ...
}
```

顶部新增 `#include <chrono>`。

#### 文件 5: `Curator/src/curator_index.h`

新增两个公共方法声明（在 Profiling 或 Cache 区域）:

```cpp
// ── I/O statistics ──
FlashStore::Stats flash_io_stats() const;
void reset_io_stats();
```

#### 文件 6: `Curator/src/curator_index.cpp`

> **注意**: `reset_io_stats()` 必须是**非 const** 方法——`flash_` 是非 mutable 成员，对其调用非 const 的 `reset_stats()` 需要 this 是非 const。

实现两个方法 (~10 行):

```cpp
FlashStore::Stats CuratorIndex::flash_io_stats() const {
    return flash_.stats();
}

void CuratorIndex::reset_io_stats() {
    flash_.reset_stats();
    pq_.reset_cache_stats();
}
```

> 注意: `search_one()`、`search_unfiltered()` 函数体**完全不变**。

#### 文件 7: `Curator/src/main.cpp`

在查询循环前后添加 (~20 行):

```cpp
// 紧接查询循环之前:
index.reset_io_stats();

// ... 现有查询循环 (完全不变) ...

// 紧接查询循环之后 (profiling 输出之前):
auto flash_s = index.flash_io_stats();
auto pq_s    = index.pq_cache_stats();

printf("\n=== I/O Statistics (Query Phase) ===\n");
printf("  Flash Store:\n");
printf("    reads:       %lu\n", flash_s.read_count);
printf("    bytes:       %lu (%.2f MB)\n",
       flash_s.bytes_read, flash_s.bytes_read / (1024.0 * 1024.0));
printf("    io_time:     %.3f ms\n", flash_s.io_time_ms);
printf("  PQ Block Cache:\n");
printf("    reads:       %lu\n", pq_s.io_count);
printf("    bytes:       %lu (%.2f MB)\n",
       pq_s.bytes_read, pq_s.bytes_read / (1024.0 * 1024.0));
printf("    io_time:     %.3f ms\n", pq_s.io_time_ms);
printf("    cache_hits:  %lu\n", pq_s.cache_hits);
printf("    cache_misses:%lu\n", pq_s.cache_misses);
printf("    hit_rate:    %.1f%%\n", pq_s.hit_rate() * 100.0);
```

---

## 三、I/O 调用链完整映射

### 3.1 标准单租户查询 (query_type: standard)

```
CuratorIndex::search_one(query, k, tid)
│
├─ Phase 1: beam_search()
│   └─ (无 I/O)
│
├─ Phase 2: Frontier Search
│   ├─ [PQ] pq_.compute_pq_distances()
│   │   └─ block_cache_->prefetch_and_get_codes()
│   │       └─ load_block_locked()  →  ::pread()  ← PQ I/O
│   └─ [精确回退] compute_vector_distance()
│       └─ flash_.read_vector()     →  ::pread()  ← Flash I/O
│
├─ Phase 3: ADC Rerank
│   └─ flash_.read_batch()          →  ::pread()  ← Flash I/O
│
└─ Output
```

### 3.2 复杂谓词查询 (query_type: temp_index)

```
search_one() [temp_index 快速路径]
├─ pq_.build_distance_table()       (无 I/O)
├─ pq_.compute_pq_distances()  →  ::pread()  ← PQ I/O
│  或 compute_vector_distance() →  ::pread()  ← Flash I/O
└─ search_temp_index() + output
```

### 3.3 无过滤查询 (search_unfiltered)

```
search_unfiltered()
├─ 优先队列遍历叶子桶               (无 I/O)
└─ compute_vector_distance()
    └─ flash_.read_vector()     →  ::pread()  ← Flash I/O
```

### 3.4 为什么全局统计方案自动覆盖所有路径

所有 I/O 都经过两个底层函数——`FlashStore::read_*` 和 `PQBlockCache::load_block_locked`。统计代码直接嵌入在这些函数中，无论上层是哪个搜索路径触发的，都会被自动计入。**不需要在 `search_one()` 或任何上层函数中做任何修改。**

---

## 四、线程安全分析

| 组件 | 保护机制 | batch_query 安全性 |
|------|---------|:---:|
| `FlashStore::stats_` | `std::mutex` | ✅ |
| `PQBlockCache::stats_` | 已有 `std::mutex` | ✅ |
| `::pread()` | 系统调用，天然线程安全 | ✅ |
| `main.cpp` reset/read | 在 `#pragma omp parallel` 区域外执行 | ✅ |

全局统计方案在 batch_query 模式下完全正确：各线程并发执行 I/O，全局计数器在 mutex 保护下正确累加。reset 在所有线程启动前、read 在所有线程结束后分别执行，无竞争。

---

## 五、验证方案

### 5.1 数据集

**`sift1m_small`** (5K 向量 × 128 维，500 查询) — 构建 < 1 秒，查询 < 1 秒。

### 5.2 验证步骤

```bash
# 0. 编译
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/Curator/build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# 1. 预处理
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
source ~/miniconda3/etc/profile.d/conda.sh && conda activate edge_ann
python Curator/python/preprocess_train.py --dataset sift1m_small

# 2. 默认参数运行（PQ 启用 + Flash 启用）
DS=sift1m_small
DATA=/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/1_Data/ground_truth/$DS
./Curator/build/curator bench \
    --train_vecs   $DATA/train_vecs.npy \
    --train_access $DATA/train_access.npy \
    --queries      $DATA/query_vecs.npy \
    --query_labels $DATA/query_labels.npy \
    --k 10
```

### 5.3 正确性验证矩阵

| # | 实验 | 预期 Flash I/O | 预期 PQ I/O | 验证点 |
|---|------|:---:|:---:|------|
| 1 | 默认参数 (PQ 启用) | > 0 (rerank 阶段) | > 0 (首次缓存未命中) | 两个源均有 I/O |
| 2 | 同一程序运行两次 (不退出) | 相同或不增 | 第 2 次更少 (缓存命中) | 缓存命中率提升 |
| 3 | `"pq_enabled": false` | > 0 (精确距离回退) | = 0 | PQ 零 I/O |
| 4 | `"use_flash_storage": false` | = 0 | = 0 (或仅 PQ I/O) | Flash 零 I/O |
| 5 | 省略 `--query_labels` (无过滤) | > 0 (每候选读一次) | = 0 | 无过滤路径覆盖 |

---

## 六、影响范围

### 6.1 修改汇总

| 文件 | 修改类型 | 估行数 | 风险 |
|------|---------|:---:|:---:|
| `flash_store.h` | 新增 Stats + 成员 | +25 | 低 — 纯新增 |
| `flash_store.cpp` | 包裹 pread 计时 | +35 | 低 — 不改变现有逻辑 |
| `pq_block_cache.h` | 新增 1 字段 | +1 | 极低 |
| `pq_block_cache.cpp` | 包裹 pread 计时 | +5 | 极低 |
| `curator_index.h` | 新增 2 个公共方法声明 | +4 | 极低 |
| `curator_index.cpp` | 实现 2 个方法 | +10 | 极低 |
| `main.cpp` | reset + read + print | +20 | 极低 |

**总计: ~100 行新增，0 行删除，0 个现有接口变更。**

### 6.2 不修改的文件

- `profiling.h` — 不涉及
- `curator_index.cpp` 的 `search_one()` / `search_unfiltered()` — **完全不改**
- `common.h`, `config.h`, `distance.h`, `tree_node.h` — 不涉及
- 所有 Python 脚本 — 不涉及

### 6.3 不影响的现有功能

- ✅ 构建流程 — 完全不涉及
- ✅ 搜索正确性 — 仅增加计时/计数，不改变搜索逻辑
- ✅ 现有 profiling (`--profile`) — 独立运作，互不干扰
- ✅ 内存占用 — `Stats` 结构体 ~40 字节，`mutex` ~40 字节，可忽略
- ✅ 性能 — 每次 I/O 增加 ~100ns (chrono::now + mutex lock)，磁盘 I/O 本身是 μs~ms 级

---

## 七、实现顺序

| 步骤 | 文件 | 内容 |
|:---:|------|------|
| 1 | `pq_block_cache.h` | `Stats` 新增 `io_time_ms` 字段 |
| 2 | `pq_block_cache.cpp` | `load_block_locked()` 包裹 pread 计时 |
| 3 | `flash_store.h` | 新增 `Stats` 结构体 + `stats()` + `reset_stats()` |
| 4 | `flash_store.cpp` | `read_vector()` / `read_batch()` 包裹 pread 计时 + 实现 stats 方法 |
| 5 | `curator_index.h` + `.cpp` | 新增 `flash_io_stats()` 和 `reset_io_stats()` |
| 6 | `main.cpp` | 查询前后 reset/read + 格式化输出 |
| 7 | 编译 + `sift1m_small` 验证 | 完整测试 |

---

*文档更新日期: 2026-07-26 (v2 简化版)*
*设计者: Claude (Claude Code)*
