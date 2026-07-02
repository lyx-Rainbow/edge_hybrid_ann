# Curator PQ码外存改造——详细执行计划（修订版 v5）

> **核心目标**: 将PQ码从内存中移出，存储到外存（磁盘/SSD），在检索需要时按需加载，仅加载实际使用的PQ码，以显著减少索引运行时内存开销。
>
> **修订记录**:
> - v5 (2026-07-03): 第二轮深度源代码审查（逐行对照 curator_index.cpp/h、pq_codec.cpp/h、config.h、profiling.h、common.h、main.cpp、distance.h、flash_store.cpp、temp_index.cpp、tree_node.h）——发现 14 项新问题/改进点，含内部逻辑不一致、跨线程指针安全性、LRU策略优化、边界条件缺失；修正 §4.2/附录A 中 `compute_pq_distances` 的双重加锁冗余；补充完整的内存模型说明
> - v4 (2026-07-03): 对照legacy源码深度对比审查——确认 d_table 重用策略优于legacy；发现 temp_index PQ 集成缺口；标注 header 格式兼容性；补充 legacy 对比分析
> - v3 (2026-07-03): 对照curator源码系统性审查——修复 `pq_d_table_` 线程安全缺陷、`seq_indices` 越界风险、重复转换逻辑；优化锁粒度；补充边界条件处理
> - v2: 系统性审查修正——线程安全、const正确性、指针安全性、接口简化、临时文件管理、profiling接入

---

## 〇-bis、v5 源代码审查发现与修正

> **审查方法**: 第二轮深度审查，逐行对照以下文件的实际代码：`curator_index.cpp/h`、`pq_codec.cpp/h`、`config.h`、`profiling.h`、`common.h`、`main.cpp`、`distance.h`、`flash_store.cpp`、`temp_index.cpp`、`tree_node.h`、`CMakeLists.txt`。重点验证 v4 计划中的接口签名、调用链、线程模型、生命周期、边界条件是否与源码实际行为一致。

### 审查发现 #13 (严重): §4.2 / 附录A `compute_pq_distances` 内 `prefetch_blocks` + `prefetch_and_get_codes` 双重加锁——与 v3 设计目标矛盾

**v4 代码** (§4.2 第 615-619 行，附录A 第 1014-1018 行):
```cpp
// 2. 预取所有需要的块（一次加锁）
block_cache_->prefetch_blocks(needed_blocks);        // ← 加锁 #1

// 3. 批量获取所有码（一次加锁）
std::vector<const uint8_t*> codes;
block_cache_->prefetch_and_get_codes(seq_indices, codes);  // ← 加锁 #2
```

**问题**: 两次调用各获取一次 `mutex_`。而 `prefetch_and_get_codes` (附录B) 内部已经处理了缺失块的加载（第 1048-1055 行），完全涵盖了 `prefetch_blocks` 的功能。第一次 `prefetch_blocks` 调用加载块后释放锁，第二次 `prefetch_and_get_codes` 再次获取锁并重新查找这些块——造成**冗余的锁获取/释放和重复的哈希表查询**。这与 v3 审查发现 #3 的设计目标（"一次锁持有完成所有 I/O + 数据获取"）**直接矛盾**。

**根因**: `prefetch_blocks` 作为独立接口是合理的（用于仅需预热缓存的场景），但 `compute_pq_distances` 的调用模式中不需要它——`prefetch_and_get_codes` 功能上是 `prefetch_blocks` 的超集。

**修正**: 在 `compute_pq_distances` 中**删除 `prefetch_blocks` 调用**，仅保留 `prefetch_and_get_codes`。后者在一次锁持有期间完成：(a) 扫描所有 seq_indices 确定缺失块 → (b) 加载缺失块 → (c) LRU 更新 → (d) 填充所有指针。真正实现 v3 的"一次加锁"设计目标。

**修正后代码**:
```cpp
void PQCodec::compute_pq_distances(
        const std::vector<float>& d_table,
        const std::vector<int_vid_t>& vids,
        std::vector<std::pair<float, int_vid_t>>& output) const {

    if (!block_cache_ || !vid_to_seq_) return;

    // 1. vid → seq_idx 转换（含 sentinel）
    std::vector<size_t> seq_indices;
    seq_indices.reserve(vids.size());
    for (int_vid_t vid : vids) {
        auto it = vid_to_seq_->find(vid);
        if (it != vid_to_seq_->end()) {
            seq_indices.push_back(it->second);
        } else {
            seq_indices.push_back(SIZE_MAX);
        }
    }

    // 2. 批量获取所有码（一次加锁：加载缺失块 + 填充指针 + LRU更新）
    std::vector<const uint8_t*> codes;
    block_cache_->prefetch_and_get_codes(seq_indices, codes);

    // 3. ADC 距离计算（无锁！codes 指针已在锁内填充完毕）
    for (size_t i = 0; i < vids.size(); i++) {
        const uint8_t* code = codes[i];
        if (!code) continue;

        float dist = 0.0f;
        for (size_t m = 0; m < M_; m++) {
            dist += d_table[m * ksub_ + code[m]];
        }
        output.emplace_back(dist, vids[i]);
    }
}
```

> **连带修正**: §4.2 (第 592-633 行) 和附录A (第 991-1032 行) 均需同步修改。

### 审查发现 #14 (中等): `prefetch_and_get_codes` 返回的原始指针在锁释放后可能被其他线程失效（cross-thread eviction）

**机制**: `prefetch_and_get_codes` (附录B) 在 `lock_guard` 作用域内将 `out_codes[i]` 指向缓存内部数据 (`it->second.first.data.data() + block_offset * M_`)，然后释放锁。在后续的 ADC 距离计算期间（锁外），**另一个线程**可能：
1. 调用 `prefetch_and_get_codes` 请求不同的块
2. 触发 LRU 淘汰（若缓存已满 ≥ `max_blocks`）
3. 淘汰的恰好是当前线程正在读取的块
4. → 当前线程的 `codes[i]` 指针悬空 → **读取已释放内存（use-after-free）**

**实际风险评估**:
- **窗口极小**: ADC 距离计算耗时 ~数微秒（128 条向量 × M=128 次查表 = ~16K 次内存访问）
- **触发条件苛刻**: 需同时满足 (a) 缓存满 (b) 另一线程恰好请求新块 (c) LRU 选中正在使用的块
- **默认配置下的缓解**: `max_blocks=256` × `block_size=4096` = 可缓存 1M+ 条 PQ 码；对于 50K 的小数据集，缓存永远不会满（仅 ~13 块）；对于 1.6M 的大数据集，256 块覆盖 1M/1.6M ≈ 62.5% 的数据，LRU 淘汰可能发生但概率低
- **典型访问模式**: 同租户的并发查询倾向于访问相似的 shortlist 集合 → 访问的块高度重叠 → 相互淘汰概率进一步降低

**结论**: 在典型生产负载下实际风险极低。但作为系统健壮性改进，建议在 v5 计划中增加以下缓解措施：

**缓解方案 (3 选 1，推荐方案 A)**:

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A: 文档化 + 保守 sizing** | 明确文档记录此限制；默认 `max_blocks` 足够大覆盖完整工作集 | 零性能开销 | 极端并发+小缓存下仍有理论风险 |
| B: 引用计数 | CacheEntry 增加 `ref_count`，淘汰时跳过 ref_count>0 的块 | 完全安全 | 每次访问需原子操作，~5-10% 性能开销 |
| C: 拷贝输出 | `prefetch_and_get_codes` 将数据拷贝到调用方提供的缓冲区而非返回指针 | 完全安全 | 每次 128×M 字节拷贝，M=128 时 16KB/次 |

**决策**: 采用方案 A。在 `PQBlockCache` 的文档注释中明确标注指针生命周期约束；将 `max_blocks` 默认值设置为能容纳典型工作集的大小。若后续压测发现实际问题，再升级到方案 B。

> **注**: 方案 C 的拷贝开销对 M=16 (256B/shortlist) 可忽略，但对 M=128 (16KB/shortlist) 有影响。若最终选择方案 C，建议仅对大 M 场景 (>64) 启用。

### 审查发现 #15 (中等): `prefetch_and_get_codes` LRU 更新对连续同块 seq_indices 存在冗余操作

**v4 附录B** (第 1067-1070 行):
```cpp
// 对每个 seq_idx 都更新 LRU
lru_list_.erase(it->second.second);
lru_list_.push_front(block_id);
it->second.second = lru_list_.begin();
```

**问题**: 对于 shortlist 内 128 条向量，若它们全部落入同一个块（`block_size=4096` 时 >90% 概率），上述代码对同一 `block_id` 执行 128 次 `erase` + `push_front`——仅第一次有效果（将块移到头部），后续 127 次将已在头部的元素移除再插回头部，纯属浪费。

**修正**: 在填充指针的循环中跟踪上一次访问的 `block_id`，仅当 `block_id` 变化时才更新 LRU：

```cpp
size_t last_block_id = SIZE_MAX;
for (size_t i = 0; i < seq_indices.size(); i++) {
    size_t seq_idx = seq_indices[i];
    if (seq_idx >= n_codes_) continue;

    size_t block_id = get_block_id(seq_idx, cfg_.block_size);
    auto it = cache_.find(block_id);
    if (it == cache_.end()) continue;

    // 仅当切换到不同 block 时才更新 LRU
    if (block_id != last_block_id) {
        lru_list_.erase(it->second.second);
        lru_list_.push_front(block_id);
        it->second.second = lru_list_.begin();
        last_block_id = block_id;
    }

    stats_.cache_hits++;
    size_t block_offset = seq_idx - block_id * cfg_.block_size;
    out_codes[i] = it->second.first.data.data() + block_offset * M_;
}
```

**收益**: 对 block_size=4096 的场景，将 LRU 操作从 O(vids.size()) 降为 O(unique_blocks)，典型 shortlist(128条) 中从 128 次降为 1 次。

### 审查发现 #16 (中等): `prefetch_and_get_codes` 的 cache_hits/cache_misses 语义不精确

**v4 附录B**: 统计在两个独立循环中更新——第一个循环（第 1054 行）统计 `cache_misses`（每缺失块一次），第二个循环（第 1072 行）统计 `cache_hits`（每 seq_idx 一次）。语义混乱：(a) miss 按块计数，hit 按访问计数，分母不一致导致 `hit_rate()` 无意义；(b) 一个被加载的块的 seq_indices 在第二个循环中全部计为 hit（加载后立即可用），但该块本身是由 miss 触发的。

**修正**: 统一为按访问（seq_idx）计数。在 `prefetch_and_get_codes` 内部跟踪每个 seq_idx 是否命中：

```cpp
// 在填充指针的循环中：
for (size_t i = 0; i < seq_indices.size(); i++) {
    // ...
    auto it = cache_.find(block_id);
    if (it == cache_.end()) {
        stats_.cache_misses++;  // 按 seq_idx 计数（不应出现：块已在循环1中加载）
        continue;
    }
    stats_.cache_hits++;        // 按 seq_idx 计数
    // ...
}
```

与此对应，第一个循环中加载缺失块时**不更新统计**（统计统一在第二个循环中按 seq_idx 粒度进行），或在加载块后为该块的所有 seq_indices 各计一次 miss。**推荐做法**: 第一个循环不更新统计；第二个循环中，若块在第一个循环之前已缓存 → hit，若块在第一个循环中刚加载 → 仍计为 hit（因为从调用方视角数据已就绪）。这样 `hit_rate()` 反映的是"访问时数据已在缓存中的比例（含本次调用刚加载的）"，对预热效果评估有意义。

> **另一个视角**: 若需要区分"调用前已缓存"vs"本次调用加载"，可增加 `Stats::prefetch_loads` 字段。当前不需要如此精细的统计。

### 审查发现 #17 (低-中等): 部分最后一块的分配策略未明确

**v4 计划**: `load_block_locked` 的描述为"计算文件偏移 (`HEADER_SIZE + block_id * block_size * M`), `::pread()` 读取，最后一块可能不满"。

**未明确事项**:
1. `CacheEntry::data` 的 `vector<uint8_t>` 应分配 `block_size * M` 字节（满块大小）还是 `actual_codes * M` 字节（实际大小）？
2. 若分配满块大小，最后一块尾部包含未初始化的填充字节，`prefetch_and_get_codes` 中的指针偏移是否会返回指向填充区的指针（对于越界的 seq_idx）？

**修正**: 明确采用**满块分配**策略：
- `CacheEntry::data` 始终分配 `block_size * M_` 字节
- `load_block_locked` 读取 `block_size * M_` 字节（`::pread` 对超出文件末尾的部分返回 0，相当于零填充）
- `prefetch_and_get_codes` 中通过 `seq_idx >= n_codes_` 检查（第 1047 行）确保不会返回指向填充区的指针
- 优点：偏移计算简单统一，无需区分满块/部分块

> **代码注释**: 在 `load_block_locked` 实现中添加注释说明此策略。

### 审查发现 #18 (低): `d_` 字段的生命周期——`build_distance_table` 的前置条件

**现状**: `PQCodec::d_` 在 `train()` 中设置（pq_codec.cpp:34）。`build_distance_table` 使用 `dsub_ = d_ / M_` 进行子空间偏移计算（pq_codec.cpp:95-96）。若在 `train()` 之前调用 `build_distance_table`，`dsub_` 为 0 → 距离表构建错误。

**计划中的流程**: `train()` → `flush()` 中 `open_cache()` → `search_one()` 中 `build_distance_table()`。顺序保证了 `d_` 已设置。但**若 `pq_enabled=true` 而 `ntotal_==0`**（空数据集），`train()` 不会被调用 → `d_` 为 0 → `build_distance_table` 可能被错误调用。

**修正**: 在 `build_distance_table` 开头增加前置条件检查：
```cpp
void PQCodec::build_distance_table(const float* query, std::vector<float>& d_table) const {
    if (!is_trained()) return;  // 或 CURATOR_THROW_IF_NOT
    // ...现有逻辑...
}
```
同时在 `search_one()` 的 PQ 分支中增加 `pq_.is_trained()` 检查（当前 `has_cache()` 已间接保证，但显式检查更安全）。

### 审查发现 #19 (低): `compute_pq_distances` 中 `needed_blocks` 集合收集 + `prefetch_and_get_codes` 内部再次遍历的冗余

**v4 问题**: 删除 `prefetch_blocks` 调用后（参见 #13），`needed_blocks` 集合不再需要。v5 §4.2/附录A 的修正版代码中已删除该集合的构建，简化了 vid→seq_idx 转换循环。

### 审查发现 #20 (低): 默认临时文件路径 `/tmp/curator_pq_<pid>.bin` 的同进程多实例冲突

**场景**: 同一进程创建多个 `CuratorIndex` 实例（库使用场景，当前 CLI 不触发）。两个实例的 PID 相同 → 文件路径冲突 → 后创建的覆盖先创建的。

**修正**: 增加随机后缀或嵌入 `this` 指针：
```cpp
// curator_index.cpp flush():
std::string path = "/tmp/curator_pq_" + std::to_string(getpid()) + "_" +
                   std::to_string(reinterpret_cast<uintptr_t>(this)) + ".bin";
```

### 审查发现 #21 (文档): `use_flash_storage=false` 且 `pq_enabled=true` 下的回退路径行为

**场景分析**:
1. `flush()` 后 `raw_buffer_` 始终被清除（curator_index.cpp:311-313）
2. 若 `use_flash_storage=false`，`flash_finalized_` 保持 `false`
3. `compute_vector_distance` 检查 `flash_finalized_ && flash_.is_open()` → false
4. 回退到 `raw_buffer_` → 但已被清除 → **抛出异常**

**影响**: 当 `pq_enabled=true` 但 `has_cache()` 为 false（如 open_cache 失败）需要走精确距离回退时，若同时 `use_flash_storage=false`，回退路径会崩溃。

**缓解**: 
- 在 `flush()` 中，若 `!cfg_.use_flash_storage && cfg_.pq_enabled`，打印 warning 并不清除 `raw_buffer_`（保留用于回退），或强制开启 flash storage
- 或：在配置验证阶段就拒绝 `use_flash_storage=false, pq_enabled=true` 的组合
- **推荐**: 在 `flush()` 中增加检查：若 `pq_enabled && !use_flash_storage`，自动设置 `use_flash_storage = true` 并打印 info 信息

> 实际使用中，所有实验配置（§三.3）都同时启用了 PQ 和 flash storage，此边界条件在实验中不会触发。

### 审查发现 #22 (文档): `get_code()` 与 `prefetch_and_get_codes()` 的双重内存模型需明确文档化

**两种内存模型**:
| 接口 | 返回指针指向 | 生命周期 | 线程安全性 |
|------|-------------|---------|-----------|
| `get_code(seq_idx)` | `thread_local` 缓冲区 (拷贝) | 下次同线程调用 `get_code` 前有效 | 天然线程安全（thread_local） |
| `prefetch_and_get_codes(seq_indices, out_codes)` | 缓存内部数据 (零拷贝) | 任何缓存修改操作前有效（含其他线程的访问） | 见审查发现 #14 |

**建议**: 在 `pq_block_cache.h` 的接口注释中明确标注两种接口的内存模型差异，防止误用。

### 审查发现 #23 (代码质量): `compute_pq_distances` 调用前的 `d_table` 大小校验缺失

**问题**: `build_distance_table` 分配 `d_table` 为 `M_ × ksub_` 大小。`compute_pq_distances` 中使用 `d_table[m * ksub_ + code[m]]` 查表。若调用方传入大小不匹配的 `d_table`（如未初始化或来自不同 M 的 PQCodec），将导致越界访问。

**修正**: 在 `compute_pq_distances` 开头增加 debug 断言：
```cpp
assert(d_table.size() == M_ * ksub_ && "d_table size mismatch");
```

### 审查发现 #24 (性能): `RunningList::batch_insert` 的 `break` 提前退出可能导致非最优结果

**注**: 这是现有代码的问题（common.h:197），非本次改造引入。但考虑到 PQ 距离不如精确距离准确，`batch_insert` 的行为可能略有影响。标记为已知，不在本次改造中修复。

### 审查发现 #25 (确认): 现有代码中 `seq_to_vid_`/`vid_to_seq_` 指针生命周期安全

**验证**: `PQCodec::set_seq_maps(&seq_to_vid_, &vid_to_seq_)` 存储的是 `CuratorIndex` 成员对象的指针（curator_index.cpp:146）。`std::vector`/`std::unordered_map` 对象本身的地址在 `CuratorIndex` 生命周期内不变（容器对象不移动，仅内部缓冲区可能 reallocate）。通过容器 API（`find()`, `operator[]`）访问数据始终安全。**结论**: 指针生命周期无问题。

### 审查发现 #26 (确认): `profiling.h` 已有 `pq_table_build_ms` 和 `pq_distance_compute_ms` 字段

**验证**: profiling.h:25-26 确认字段已存在。改造直接写入即可。同时 profiling.h:13 已标注多线程下 profiling 非线程安全——与本次改造无关（已知问题）。

---

## 〇、v3 源代码审查发现与修正

> **审查方法**: 对照 `curator_index.cpp`、`pq_codec.cpp/.h`、`main.cpp`、`profiling.h`、`config.h`、`common.h` 的实际代码，逐条验证 v2 计划中的每个设计决策和接口签名。

### 审查发现 #1 (严重): `pq_d_table_` 作为 mutable 成员存在线程安全缺陷

**v2 设计**: `pq_d_table_` 声明为 `mutable std::vector<float>`（CuratorIndex 成员），在 `search_one()`（const 方法）中复用。

**问题**: 当 `batch_query=true` 时，`main.cpp:276-287` 使用 `#pragma omp parallel for` 多线程并发调用 `index.search()` → `search_one()`。所有线程共享同一个 `pq_d_table_` 实例，Thread A 正在读取距离表进行 ADC 累加时，Thread B 可能正在通过 `build_distance_table()` 覆写同一块内存 → **数据竞争（data race）**。

**修正**: `pq_d_table_` **不作为 CuratorIndex 成员**。改为在 `search_one()` 中构建为**栈上局部变量**，或将其构建逻辑移入 `compute_pq_distances()` 内部（与 legacy 代码 `MultiTenantIndexIVFHierarchical.cpp:851-889` 的做法一致）。

**修正后方案**: 采用折中——在 `search_one()` 中维护一个局部 `std::vector<float> pq_d_table`，首次使用时构建，同 query 内复用。这样每个线程有独立的距离表（在线程栈上）。`mutable std::vector<float> pq_d_table_` 从 `CuratorIndex` 中移除。

**代码变更**:
```cpp
// curator_index.h: 移除 mutable std::vector<float> pq_d_table_;
// curator_index.cpp search_one():
void CuratorIndex::search_one(...) const {
    // ... existing code ...
    std::vector<float> pq_d_table;  // 栈上局部变量，每query独立，线程安全
    bool pq_table_built = false;

    while (!frontier.empty()) {
        // ...
        if (cfg_.pq_enabled && pq_.is_trained() && pq_.has_cache()) {
            if (!pq_table_built) {
                pq_.build_distance_table(x, pq_d_table);
                pq_table_built = true;
            }
            pq_.compute_pq_distances(pq_d_table, it->second.data, sorted_cands);
        }
        // ...
    }
}
```

### 审查发现 #2 (严重): §5.3 `compute_pq_distances` 中 `seq_indices` 与 `vids` 索引不对齐

**v2 问题**: §5.3 伪代码中，当某个 vid 在 `vid_to_seq_` 中找不到时，直接 `continue`（不追加到 `seq_indices`），导致 `seq_indices.size() < vids.size()`。后续循环 `for (size_t i = 0; i < vids.size(); i++)` 以 `vids` 为基准访问 `seq_indices[i]` → **越界访问**。

**根因**: v2 §5.3 的伪代码缺少 sentinel 机制。附录A 使用了 `SIZE_MAX` 作为 sentinel（正确），但 §5.3 正文版本遗漏了。

**修正**: 统一使用 sentinel 模式（`SIZE_MAX`），或改为更安全的做法——直接在一次循环中完成 vid→seq_idx 查找 + get_code + ADC 计算，避免中间数组。

**修正后方案** (推荐——消除中间数组):
```cpp
void PQCodec::compute_pq_distances(
        const std::vector<float>& d_table,
        const std::vector<int_vid_t>& vids,
        std::vector<std::pair<float, int_vid_t>>& output) const {

    if (!block_cache_ || !vid_to_seq_) return;

    // 1. 收集唯一 block_id 用于预取
    std::unordered_set<size_t> needed_blocks;
    std::vector<size_t> seq_indices;
    seq_indices.reserve(vids.size());
    for (int_vid_t vid : vids) {
        auto it = vid_to_seq_->find(vid);
        if (it != vid_to_seq_->end()) {
            size_t seq_idx = it->second;
            seq_indices.push_back(seq_idx);
            needed_blocks.insert(PQBlockCache::get_block_id(seq_idx, block_cache_->block_size()));
        } else {
            seq_indices.push_back(SIZE_MAX);  // sentinel: vid not found
        }
    }

    // 2. 预取所有需要的块（内部加锁，仅加载缺失块）
    block_cache_->prefetch_blocks(needed_blocks);

    // 3. 逐码计算 ADC 距离
    for (size_t i = 0; i < vids.size(); i++) {
        size_t seq_idx = seq_indices[i];
        if (seq_idx == SIZE_MAX) continue;  // sentinel check

        const uint8_t* code = get_code(seq_idx);
        if (!code) continue;

        float dist = 0.0f;
        for (size_t m = 0; m < M_; m++) {
            dist += d_table[m * ksub_ + code[m]];
        }
        output.emplace_back(dist, vids[i]);
    }
}
```

同时，`prefetch` 接口改为接收 `unordered_set<size_t>`（block_id 集合），避免 seq_indices 的二次遍历（参见审查发现 #4）。

### 审查发现 #3 (性能): 双重锁定——prefetch + get_code 各自加锁

**v2 问题**: `compute_pq_distances` 中，先调用 `block_cache_->prefetch(seq_indices)`（内部 `lock_guard`），然后循环调用 `get_code(seq_idx)`（每次也 `lock_guard`）。对一个 128 条向量的 shortlist，这意味着 1 + 128 = 129 次加锁/解锁。

**修正**: 提供组合操作 `prefetch_and_get_codes()`，在一次锁持有期间完成"预取 + 批量获取所有码"。

**修正后方案**: 在 `PQBlockCache` 中新增:
```cpp
// 预取并批量获取PQ码（一次加锁，线程安全）
// seq_indices: 需要获取的seq_idx列表（可含SIZE_MAX sentinel）
// out_codes: 输出，out_codes[i] 指向缓存内对应seq_indices[i]的PQ码数据
//            若seq_indices[i]==SIZE_MAX或越界，对应指针为nullptr
void prefetch_and_get_codes(
        const std::vector<size_t>& seq_indices,
        std::vector<const uint8_t*>& out_codes);
```

这样 `compute_pq_distances` 只需一次锁持有完成所有 I/O，后续 ADC 距离计算完全无锁。这对 `batch_query=true` 下的并发性能至关重要。

> **注意**: 返回的指针指向缓存内部数据，在**当前锁释放前**有效。因此 `compute_pq_distances` 必须在解锁前完成所有 ADC 距离计算。这要求 `prefetch_and_get_codes` 返回 `vector<const uint8_t*>`，在函数返回前完成 ADC 求和。由于 ADC 距离计算不涉及任何互斥状态，这是合理的。

### 审查发现 #4 (逻辑冗余): §5.2 中 vid→seq_idx 转换被执行两次

**v2 问题**: §5.2 的 `search_one()` 伪代码中手动做了 vid→seq_idx 转换（Step 1），但该转换结果**并未传递给** `compute_pq_distances()`（Step 3 传入的是原始 `sl_vids`）。而 `compute_pq_distances()` 内部又做了一次 vid→seq_idx 转换。第一次转换的结果被丢弃。

**修正**: 删除 §5.2 中 `search_one()` 的手动转换代码。`compute_pq_distances` 统一负责 vid→seq_idx 转换。`search_one()` 仅负责判断走哪个路径并调用相应函数。

### 审查发现 #5 (边界条件): `flash_finalized_` 与 PQ cache 可用性的关系

**v2 问题**: 计划中 `search_one()` 的 PQ 分支条件为 `cfg_.pq_enabled && pq_.has_cache()`。但 `has_cache()` 检查的是 block_cache 是否已打开——这在 `flush()` 中 PQ 编码写入磁盘后才会发生。如果由于某种原因（如磁盘空间不足）`open_cache()` 失败，`has_cache()` 返回 false，正确回退到全精度距离。

**确认**: 此设计正确。额外建议：在 `flush()` 中 `open_cache()` 失败时打印 warning 到 stderr，帮助诊断。

### 审查发现 #6 (代码现状确认): 当前搜索路径PQ码完全未被使用

**确认 v2 分析正确**: 
- `curator_index.cpp:393-438` (Phase 2 Frontier search) 中所有距离计算走 `compute_vector_distance()` → FlashStore（或 raw_buffer_ 回退）
- `PQCodec::compute_distances()` 已实现但**从未被调用**
- `flush()` 中 `encode_all()` 将 PQ 码写入局部变量 `codes`（第 246 行），**未存入成员 `vid_to_pq_code_`**（该成员始终为空）
- `pq_enabled` 标志仅在 `flush()` 中检查，控制是否训练 PQ；搜索路径中完全忽略 `pq_enabled`
- `memory_breakdown()` 中 `pq_codes_bytes` 遍历 `pq_.codes()` 但始终为 0（因为 `vid_to_pq_code_` 为空）

**结论**: PQ 码被训练和编码但从未在搜索中使用。本改造不仅要实现外存化，还要**首次让 PQ 码在搜索路径中真正生效**。

### 审查发现 #7 (接口清理确认): 可安全移除的接口

以下接口在当前代码中完全未被调用，移除无任何影响：
- `PQCodec::codes()` — 返回 `vid_to_pq_code_`（始终为空）
- `PQCodec::codes_in_memory()` — 检查 `vid_to_pq_code_` 是否为空
- `PQCodec::codes_on_disk()` — 检查 `pq_codes_mmap_`（在未调用 `load_from_disk()` 时为 nullptr）
- `PQCodec::load_from_disk()` — 仅定义，从未被调用
- `PQCodec::free_in_memory_codes()` — 仅在析构函数中调用（清理 mmap）
- `PQCodec::compute_distances(callback)` — 回调版本，从未被调用
- `MemoryBreakdown::pq_codes_bytes` — 统计值始终为 0

### 审查发现 #8 (profiling 字段): `pq_table_build_ms` 和 `pq_distance_compute_ms` 已存在

`profiling.h:25-26` 中这两个字段已定义但从未被写入（因为在当前搜索路径中 PQ 未被使用）。改造后直接写入这些字段即可，无需修改 `profiling.h`（除移除 `pq_codes_bytes` 外）。

### 审查发现 #9 (legacy对比): d_table 构建策略——v3方案优于legacy

**Legacy 做法** (`MultiTenantIndexIVFHierarchical.cpp:851-889`): `compute_pq_distances()` 内部每次调用都重建 d_table（即每访问一个 shortlist 就重新计算 M×256 次 `l2_sqr`）。

**v3 做法**: d_table 在 `search_one()` 中构建一次，通过参数传入 `compute_pq_distances()`，同 query 内所有 shortlist 复用。

**分析**: 对于 M=128 的场景，每次 d_table 构建需要 128×256 = 32768 次 `l2_sqr(dsub)` 运算。若 frontier search 访问 5 个 shortlist，legacy 重建 5 次 d_table（~163840 次 `l2_sqr`），而 v3 仅构建 1 次。**v3 方案在典型查询中可节省数万次距离运算**。

**Legacy profiling 的 20/80 hack**: Legacy 代码用硬编码比例（20% table build, 80% distance compute）拆分首次 `compute_pq_distances` 的耗时（第 1033-1038 行）。v3 通过分离 `build_distance_table` 和 `compute_pq_distances` 调用，实现了**精确计时**，消除了启发式拆分。这是对 legacy 的实质性改进。

### 审查发现 #10 (legacy对比): temp_index 搜索路径的 PQ 集成——当前代码存在缺口

**Legacy 做法** (`MultiTenantIndexIVFHierarchical.cpp:2069-2072`): bitmap-filter 搜索 (`search_temp_index`) 在收集候选向量时也使用 PQ 距离（若 `pq_enabled`）或精确距离（`compute_dists_with_prefetch`）。

**当前代码** (`temp_index.cpp:169-178`): `search_temp_index` 使用 centroid-based 近似距离（节点质心到 query 的距离作为候选向量的近似分数），**既不使用 PQ 也不计算精确向量距离**。这是当前代码与 legacy 之间的一个已知功能差距（非本改造引入）。

**对本次PQ改造的影响**: `search_temp_index` 是独立函数，不依赖 `CuratorIndex` 或 `PQCodec`。要在此路径接入 PQ，需重构 `search_temp_index` 的接口（增加 PQ 相关参数），这会增加模块耦合度。**本次改造暂不覆盖 temp_index 路径**，将搜索路径 PQ 集成限定在 `search_one()`（标准租户搜索）中。`search_temp_index` 的 PQ 化作为后续优化项。

**后续计划**: 可选方案——(a) 在 `search_temp_index` 中增加一个 `std::function<float(int_vid_t)>` 回调参数，由 `CuratorIndex` 注入 PQ 距离计算 lambda；(b) 将 PQ 码指针数组直接传入 `search_temp_index`。方案 (a) 耦合度更低，推荐。

### 审查发现 #11 (legacy对比): PQ 磁盘文件 header 格式差异

**Legacy header** (`MultiTenantIndexIVFHierarchical.cpp:2414-2429`):
```
Offset  Size  Field
0       4     magic = 0x50514344 ("PQCD")
4       4     version = 1
8       4     n_vectors (uint32)
12      4     M (uint32)
16      4     nbits (uint32)
20      4     code_bytes (uint32)
24      8     body_offset = 32
Total:  32 bytes
```

**当前 header** (`pq_codec.cpp:147-158`):
```
Offset  Size  Field
0       4     magic = 0x50514344 ("PQCD")
4       8     M (uint64)
12      8     nbits (uint64)
20      8     n_codes (uint64)
28      4     reserved = 0
Total:  32 bytes
```

**关键发现**: 两者 magic 相同（`0x50514344`）但字段布局完全不同（uint32 vs uint64 宽度，不同语义）。使用相同 magic 但互不兼容的 header 格式**可能导致静默数据损坏**——如果用 legacy 代码读取当前格式文件（或反之），magic 验证通过但后续字段解析错误。

**对本次改造的影响**: v3 方案的 `open_cache()` 读取的是 `write_to_disk()` 写入的当前格式，闭环内无兼容性问题。但建议在 `write_to_disk()` 中改用新 magic（如 `0x43555251` = "CURQ"）以明确区分格式，防止未来混淆。若保持 `0x50514344`，至少应在注释中警告与 legacy 格式不兼容。

> **决策**: 保持 `0x50514344`（向后兼容当前写入的文件），但在 `open_cache()` 和 `write_to_disk()` 的注释中标注与 legacy 格式的差异。

### 审查发现 #12 (legacy对比): PQ 码持久化策略差异

**Legacy flow** (`finalize_flash_storage:2524-2623`):
1. `train_pq_codebook()` → `vid_to_pq_code` 在内存中 (N×M bytes)
2. Flash 最终化
3. 若 `persist_pq_codes=true`: `write_pq_codes_to_disk()` → `free_pq_codes()` → `load_pq_codes_from_disk()` (mmap)
4. 若 `persist_pq_codes=false`: codes **保持在内存中**（搜索时零磁盘 I/O）

**v3 flow**:
1. `encode_all()` → codes 局部变量
2. `write_to_disk()` → **始终写入磁盘**
3. `codes.clear()` → 释放内存
4. `open_cache()` → 按需从磁盘加载

**差异**: v3 方案中 PQ 码**始终在磁盘**（不再有纯内存模式），搜索时总是可能触发磁盘 I/O。这与改造目标一致（减少内存），但需注意：首次访问某块 PQ 码时的冷启动延迟（~100μs-1ms 磁盘读取）会在搜索延迟中体现。LRU 缓存预热后（命中率 >90% 典型场景），后续查询的磁盘 I/O 大幅减少。

---

## 一、现状分析

### 1.1 当前PQ码的生命周期与关键发现

**关键发现1**: 当前新代码（`curator_index.cpp`）搜索路径中**完全不使用PQ码**。
- 所有距离计算走 `compute_vector_distance()` → FlashStore 读取全精度向量
- `PQCodec::compute_distances()` 和 `build_distance_table()` 已实现但未被调用
- `encode_all()` 输出到参数 `codes`（局部变量），**未存入成员 `vid_to_pq_code_`**（`vid_to_pq_code_` 始终为空！）
- 这意味着 PQ 码训练和编码了但立即被丢弃 —— 搜索时根本没用到

**关键发现2**: Legacy 代码（`MultiTenantIndexIVFHierarchical.cpp:851-889`）有完整的 `compute_pq_distances()` ADC 距离计算路径，这是我们改造的参考模板。

**关键发现3**: `batch_query=true` 时，`main.cpp:276` 使用 `#pragma omp parallel for` 多线程并发调用 `index.search()`。这意味着任何新增的共享状态（如 PQ 块缓存）**必须是线程安全的**。

```
BUILD阶段 (CuratorIndex::flush, curator_index.cpp:238-255):
  pq_.train(ntotal_, raw_buffer_, d, M, nbits)
    → 训练 M 个子空间的 K-means 码本 → pq_codebook_ [M × ksub × dsub] floats
    → 设置内部状态: d_=d, M_=M, nbits_=8, ksub_=256, dsub_=d/M
  std::vector<std::vector<uint8_t>> codes;  // 局部变量
  pq_.encode_all(ntotal_, raw_buffer_, codes)
    → 全量向量编码，结果写入 codes 参数（未存入任何成员变量！）
  [可选] pq_.write_to_disk(path, codes, M, nbits)
    → 若 persist_pq_codes=true 且有路径，写入磁盘
  codes 销毁（PQ码丢失！）

SEARCH阶段 (curator_index.cpp:393-438):
  ★ PQ 码在搜索路径中完全未使用 ★
  所有距离计算: compute_vector_distance() → FlashStore
  即使 pq_use_adc_rerank=true，也是 FlashStore 全精度重排
```

### 1.2 内存占用分析

| 数据集 | 向量数 | d | M | PQ codes 理论大小 | pq_codebook 大小 |
|--------|--------|---|---|-------------------|-------------------|
| arxiv_small | 50K | 384 | 128 | 6.4 MB | 384 KB |
| yfcc100m_small | 50K | 128 | 16 | 0.8 MB | 128 KB |
| sift1m_small | 5K | 128 | 16 | 80 KB | 128 KB |
| arxiv (full) | 1.6M | 384 | 128 | **204.8 MB** | 384 KB |
| yfcc100m (full) | 0.8M | 192 | 64 | 51.2 MB | 192 KB |

**结论**: 对于大规模数据集（尤其是高维场景 M=128），PQ码是主要内存消耗者。但当前代码中这些码实际未被使用——它们仅在 `flush()` 中作为局部变量短暂存在后即被释放。本改造**一举两得**：既实现外存化，又让 PQ 码首次在搜索路径中生效。

### 1.3 搜索时PQ码访问模式

在 legacy 的 `compute_pq_distances` 中（需要接入的访问模式）：
1. 对每个命中的 shortlist（大小 ≤ `max_sl_size` = 128-256）
2. 遍历 shortlist 中的所有 vid
3. 通过 `vid_to_seq_` 获取 seq_idx
4. 通过 `get_code(seq_idx)` 获取 PQ 码（M 字节）
5. 用 PQ 码查距离表计算 ADC 近似距离

**访问特征**: 批量性（每次 shortlist 128-256 条）、稀疏性（仅访问极小比例的数据）、局部性（同 shortlist 的向量 seq_idx 有局部性）。

---

## 二、改造目标与设计决策

### 2.1 核心目标

1. **PQ码外存化**: PQ码不常驻内存，存储到磁盘文件中
2. **按需加载**: 检索时仅加载实际访问到的PQ码块
3. **内存可控**: 通过LRU缓存限制PQ码内存占用上限（默认 M=16 时 ~16MB, M=128 时 ~128MB）
4. **线程安全**: 支持 `batch_query=true` 多线程并发查询，**特别关注距离查表的线程安全性**
5. **搜索路径接入**: 参考 legacy 代码，在 `search_one()` 中接入 PQ 近似距离计算

### 2.2 关键设计决策（已确认）

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| D1 | 缓存粒度 | **块（Block）**，默认4096条/块 | 平衡I/O效率与灵活性；M=16时64KB/块，M=128时512KB/块 |
| D2 | 替换策略 | **LRU** | 适配搜索局部性；实现简单 |
| D3 | 文件格式 | **保留现有32字节头+序列编码体** | 完全向后兼容 |
| D4 | 码本(codebook) | **始终在内存** | 极小（<1MB），每次查询必需 |
| D5 | 缓存线程安全 | **内部 `std::mutex`** | `batch_query` 多线程并发查询必需 |
| D6 | 搜索粗排 | **PQ近似距离**（frontier search） | 减少I/O：M字节/向量 vs d×4字节/向量 |
| D7 | 搜索精排 | **全精度距离**（ADC rerank） | 保证最终召回率 |
| D8 | 默认配置 | `block_size=4096`, `max_blocks=256` | M=16→16MB, M=128→128MB, 覆盖典型工作集 |
| D9 | 文件路径 | 优先 `cfg_.pq_codes_path`；为空则自动生成 `/tmp/curator_pq_<pid>.bin` | 灵活且无感知 |
| D10 | 临时文件清理 | `persist_pq_codes=false` 时析构删除；`=true` 时保留 | 复用现有配置语义 |
| **D11** | **距离查表线程安全** | **栈上局部变量**（非 mutable 成员） | v3新增：避免batch_query下的数据竞争 |
| **D12** | **PQ码获取锁策略** | **组合操作** prefetch+get_codes 一次加锁 | v3新增：减少锁竞争，对128条shortlist从129次锁降为1次 |

---

## 三、架构设计

### 3.1 整体架构

```
改造后:
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
│  FlaskStore (磁盘): 全精度向量，仅 rerank 时读取  │
└──────────────────────────────────────────────────┘
```

### 3.2 线程安全设计（v5 修订）

```
多线程并发查询 (batch_query=true):
  Thread 1: search_one() ─┐
  Thread 2: search_one() ─┤
  Thread 3: search_one() ─┼──→ pq_.compute_pq_distances(d_table, vids)
                           │         │
                           │         ├── lock(mutex_)         ← 唯一一次加锁
                           │         ├── 扫描 seq_indices
                           │         ├── 加载缺失块 (::pread)
                           │         ├── 填充所有指针 (零拷贝, 指向缓存内部)
                           │         ├── LRU 更新 (去重: 仅 block_id 变化时)
                           │         ├── unlock(mutex_)
                           │         └── 逐码计算ADC距离（无锁！距离表在线程栈上）
  Thread N: search_one() ─┘

关键变更 (v5):
  ★ pq_d_table 是 search_one() 栈上的局部变量，每个线程独立，零竞争
  ★ prefetch_and_get_codes() 在一次锁持有期间完成：加载缺失块 + 指针填充 + LRU更新
    — 不再有单独的 prefetch_blocks 调用（v5 修正审查发现 #13）
  ★ ADC 距离计算在锁外进行（纯CPU计算，无共享状态）
  ★ codes[i] 指针指向缓存内部数据——调用方必须在指针失效前完成使用（审查发现 #14）
  ★ LRU 更新在填充指针时去重：连续同 block_id 仅更新一次（审查发现 #15）
```

---

## 四、详细模块设计

### 4.1 新增 `PQBlockCache` 类

```cpp
// pq_block_cache.h
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace curator {

class PQBlockCache {
public:
    struct Config {
        size_t block_size = 4096;    // 每块包含的PQ码条数
        size_t max_blocks = 256;     // 最大缓存块数
    };

    PQBlockCache() = default;
    ~PQBlockCache();

    // 不可拷贝
    PQBlockCache(const PQBlockCache&) = delete;
    PQBlockCache& operator=(const PQBlockCache&) = delete;

    // ── 生命周期 ──

    // 打开PQ码磁盘文件（只读），验证header
    // 前置条件: M 和 nbits 必须与写入时一致
    bool open(const std::string& path, size_t M, size_t nbits);

    // 关闭文件并清空缓存。if delete_file=true 则删除磁盘文件
    void close(bool delete_file = false);

    bool is_open() const { return fd_ >= 0; }

    // ── 核心访问接口（线程安全）──

    // 获取指定 seq_idx 的 PQ 码，拷贝到 out（必须至少 M 字节）
    // 若块未缓存则从磁盘加载；若缓存已满则淘汰 LRU 块
    // 返回 true 表示成功，false 表示 seq_idx 越界
    bool get_code(size_t seq_idx, uint8_t* out);

    // ★ v3新增: 批量预取并获取所有码（一次锁持有）
    // seq_indices: 需要的 seq_idx 列表（可含 SIZE_MAX sentinel）
    // out_codes: 输出，out_codes[i] 指向缓存内 seq_indices[i] 的PQ码
    //            若 seq_indices[i] 越界，对应指针为 nullptr
    // 注意: 返回的指针在下次任何 PQBlockCache 方法调用后可能失效（LRU淘汰）
    //       调用方必须在指针失效前完成数据使用
    void prefetch_and_get_codes(
            const std::vector<size_t>& seq_indices,
            std::vector<const uint8_t*>& out_codes);

    // 仅预取指定 block_id 集合中的块（用于需要分批处理时）
    void prefetch_blocks(const std::unordered_set<size_t>& block_ids);

    // ── 查询状态 ──
    size_t n_codes() const { return n_codes_; }
    size_t M() const { return M_; }
    size_t cache_size() const { return cache_.size(); }  // 不加锁，近似值
    size_t max_blocks() const { return cfg_.max_blocks; }
    size_t block_size() const { return cfg_.block_size; }
    size_t cache_bytes() const;  // 当前缓存占用的总字节数

    // ── 统计信息 ──
    struct Stats {
        uint64_t cache_hits = 0;
        uint64_t cache_misses = 0;
        uint64_t bytes_read = 0;     // 从磁盘读取的总字节数
        uint64_t io_count = 0;       // 磁盘I/O次数
        void reset() { *this = Stats{}; }
        double hit_rate() const {
            uint64_t total = cache_hits + cache_misses;
            return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
        }
    };
    Stats stats() const;             // 线程安全拷贝
    void reset_stats();              // 线程安全重置

    // ★ 静态工具方法
    static size_t get_block_id(size_t seq_idx, size_t block_size) {
        return seq_idx / block_size;
    }

private:
    Config cfg_;
    int fd_ = -1;
    size_t n_codes_ = 0;
    size_t M_ = 0;                   // PQ码字节数 (= pq_M, 因为 nbits=8)
    static constexpr size_t HEADER_SIZE = 32;

    // 缓存条目（只读，dirty 始终为 false）
    struct CacheEntry {
        size_t block_id;
        std::vector<uint8_t> data;   // [block_size × M_] 字节（满块分配，审查发现 #17）
    };

    // LRU: list 头部=最新，尾部=最旧
    std::list<size_t> lru_list_;     // block_id 列表
    std::unordered_map<size_t,       // block_id → (entry, lru_iterator)
        std::pair<CacheEntry, std::list<size_t>::iterator>> cache_;

    mutable std::mutex mutex_;       // 保护所有缓存操作
    Stats stats_;                    // 受 mutex_ 保护

    // 内部方法（调用前必须持有 mutex_）
    void load_block_locked(size_t block_id);
    void evict_one_locked();
};

} // namespace curator
```

**关键设计要点 (v5修订)**:
- `prefetch_and_get_codes()` 是推荐的批量访问接口：一次锁持有完成所有 I/O + 指针填充 + LRU 更新。对 128 条向量的 shortlist 仅 1 次加锁。
- `prefetch_blocks()` 用于仅需预热缓存、不需要立即获取所有码的场景（如 flush 后的缓存预热，§九.3）。**不应与 `prefetch_and_get_codes` 串联调用**——后者已涵盖块加载逻辑，串联导致双重加锁（审查发现 #13）。
- 调用方必须注意 `out_codes` 中指针的生命周期——指向缓存内部数据（零拷贝），在下一次任何缓存修改操作后可能失效（审查发现 #14）。
- `CacheEntry::data` 采用**满块分配**策略——始终 `block_size * M_` 字节，简化偏移计算。最后一块尾部字节由 `::pread` 自然零填充（审查发现 #17）。

### 4.2 `PQCodec` 类变更

```cpp
// pq_codec.h 变更摘要

class PQCodec {
public:
    PQCodec() = default;
    ~PQCodec();  // 实现中清理 block_cache_（含临时文件删除）

    // ── 保留不变 ──
    void train(size_t n, const float* x, size_t d, size_t M, size_t nbits);
    void encode(const float* x, std::vector<uint8_t>& code) const;
    void encode_all(size_t n, const float* x,
                    std::vector<std::vector<uint8_t>>& codes) const;
    void build_distance_table(const float* query,
                              std::vector<float>& d_table) const;
    void compute_exact_batch(const float* query,
            const std::vector<int_vid_t>& vids,
            const float* vectors, size_t d,
            std::vector<std::pair<float, int_vid_t>>& output) const;
    void write_to_disk(const std::string& path,
                       const std::vector<std::vector<uint8_t>>& codes,
                       size_t M, size_t nbits);

    // ── 新增: 块缓存接口 ──
    // 打开PQ码磁盘文件用于按需读取（flush后调用）
    bool open_cache(const std::string& path, size_t block_size, size_t max_blocks);
    void close_cache();  // 关闭缓存+文件，不删除磁盘文件
    bool has_cache() const { return block_cache_ != nullptr && block_cache_->is_open(); }

    // ── ★ v3: 核心距离计算（简化签名，内部自行 vid→seq_idx 转换）──
    // 使用内部缓存获取PQ码（线程安全），d_table 由调用方构建（栈上，线程安全）
    void compute_pq_distances(
            const std::vector<float>& d_table,
            const std::vector<int_vid_t>& vids,
            std::vector<std::pair<float, int_vid_t>>& output) const;

    // ── 内部使用 ──
    const uint8_t* get_code(size_t seq_idx) const;  // 内部使用，返回 thread_local 缓冲区指针
    bool is_trained() const { return !pq_codebook_.empty(); }
    const std::vector<float>& codebook() const { return pq_codebook_; }
    size_t M() const { return M_; }
    size_t nbits() const { return nbits_; }
    size_t ksub() const { return ksub_; }
    size_t dsub() const { return dsub_; }
    void set_seq_maps(const std::vector<int_vid_t>* seq_to_vid,
                      const std::unordered_map<int_vid_t, size_t>* vid_to_seq);

    // ── 新增: 统计信息 ──
    PQBlockCache::Stats cache_stats() const;
    void reset_cache_stats();
    size_t cache_bytes() const;

    // ── 新增: 文件路径管理 ──
    const std::string& pq_file_path() const { return pq_file_path_; }
    void set_pq_file_path(const std::string& path) { pq_file_path_ = path; }
    void set_persist(bool persist) { persist_pq_codes_ = persist; }

    // ── 移除 ──
    // ✂ codes_in_memory() — 不再有全量内存码
    // ✂ codes_on_disk() — 改为 has_cache()
    // ✂ load_from_disk() — 被 open_cache() 取代
    // ✂ free_in_memory_codes() — 被 close_cache() 取代
    // ✂ codes() (const vector<vector<uint8_t>>&) — vid_to_pq_code_ 已移除
    // ✂ vid_to_pq_code_ 成员
    // ✂ pq_codes_mmap_ / pq_codes_mmap_size_ 成员
    // ✂ compute_distances(callback) — 被 compute_pq_distances 替代

private:
    size_t M_ = 0, nbits_ = 8, ksub_ = 0, dsub_ = 0, d_ = 0;
    std::vector<float> pq_codebook_;

    // ★ 块缓存（mutable 用于 const 搜索路径中的 LRU 状态更新）
    mutable std::unique_ptr<PQBlockCache> block_cache_;

    // ★ 用于 get_code() 的线程局部缓冲区
    static thread_local std::vector<uint8_t> tl_code_buf_;

    std::string pq_file_path_;     // 磁盘文件路径（用于析构清理）
    bool persist_pq_codes_ = false; // 析构时是否保留文件

    // 外部序列映射指针（指向 CuratorIndex 成员）
    const std::vector<int_vid_t>* seq_to_vid_ = nullptr;
    const std::unordered_map<int_vid_t, size_t>* vid_to_seq_ = nullptr;
};
```

**变更说明 (v3修订)**:

1. **移除 `vid_to_pq_code_`**: 当前代码中该成员始终为空（`encode_all` 写入参数而非成员），移除不影响任何功能。
2. **移除 `pq_codes_mmap_` 和 `load_from_disk()`**: 被 `open_cache()` 的块缓存方式取代。
3. **移除 `compute_distances(callback)`**: 简化为 `compute_pq_distances(d_table, vids, output)`，内部直接访问缓存。
4. **移除 `codes()` / `codes_in_memory()` / `codes_on_disk()`**: 对应的成员已移除。
5. **`get_code()` 改为内部接口**: 使用 `thread_local` 缓冲区，返回的指针仅在线程内短期有效。
6. **新增 `mutable` 和 `thread_local`**: `block_cache_` 为 `mutable`（const 搜索方法可修改缓存）；`tl_code_buf_` 为 `thread_local`（每线程独立缓冲区）。
7. **`compute_pq_distances` 简化签名**: 不再需要 `get_code_ctx` 和回调函数，直接内部使用 `block_cache_` + `vid_to_seq_`。内部仅调用 `prefetch_and_get_codes`（一次加锁），不调用 `prefetch_blocks`（v5 修正审查发现 #13）。

**`compute_pq_distances` v5 实现（修正双重加锁）**:
```cpp
void PQCodec::compute_pq_distances(
        const std::vector<float>& d_table,
        const std::vector<int_vid_t>& vids,
        std::vector<std::pair<float, int_vid_t>>& output) const {

    if (!block_cache_ || !vid_to_seq_) return;
    assert(d_table.size() == M_ * ksub_ && "d_table size mismatch");

    // 1. vid → seq_idx 转换（含 sentinel，不收集 needed_blocks）
    std::vector<size_t> seq_indices;
    seq_indices.reserve(vids.size());
    for (int_vid_t vid : vids) {
        auto it = vid_to_seq_->find(vid);
        if (it != vid_to_seq_->end()) {
            seq_indices.push_back(it->second);
        } else {
            seq_indices.push_back(SIZE_MAX);  // sentinel
        }
    }

    // 2. 批量获取所有码（★ 一次加锁：加载缺失块 + 填充指针 + LRU更新）
    //    prefetch_and_get_codes 内部处理块加载，无需单独 prefetch_blocks
    std::vector<const uint8_t*> codes;
    block_cache_->prefetch_and_get_codes(seq_indices, codes);

    // 3. ADC 距离计算（无锁！codes 指针已在锁内填充完毕）
    //    注意：指针指向缓存内部数据，在下次任何缓存修改前有效（见审查发现 #14）
    for (size_t i = 0; i < vids.size(); i++) {
        const uint8_t* code = codes[i];
        if (!code) continue;  // sentinel 或越界

        float dist = 0.0f;
        for (size_t m = 0; m < M_; m++) {
            dist += d_table[m * ksub_ + code[m]];
        }
        output.emplace_back(dist, vids[i]);
    }
}
```

### 4.3 `CuratorIndex` 类变更

```cpp
// curator_index.h 变更摘要

class CuratorIndex {
public:
    // ... 现有接口不变 ...

    // ★ 新增: PQ 缓存统计
    PQBlockCache::Stats pq_cache_stats() const;
    size_t pq_cache_bytes() const;

private:
    // ★ pq_ 改为 mutable（block_cache_ 的 LRU 状态在 const 搜索方法中需更新）
    mutable PQCodec pq_;
    // ... 其他成员不变 ...

    // ★ v3: 移除 mutable std::vector<float> pq_d_table_ ——
    //   改为 search_one() 中的栈上局部变量，确保 batch_query 下的线程安全
};
```

### 4.4 `CuratorConfig` 新增字段

```cpp
// config.h 新增
struct CuratorConfig {
    // ... 现有字段 ...
    size_t pq_cache_block_size = 4096;   // ★ 新增: 每块缓存 PQ 码条数
    size_t pq_cache_max_blocks = 256;    // ★ 新增: 最大缓存块数
    // 默认容量: M=16 → 4096×256×16 = 16 MB
    //           M=128 → 4096×256×128 = 128 MB
};
```

### 4.5 `MemoryBreakdown` 变更

```cpp
// profiling.h 变更
struct MemoryBreakdown {
    // ... 现有字段 ...
    size_t pq_codebook_bytes = 0;
    // ✂ 移除: size_t pq_codes_bytes = 0;  （始终为0，无实际意义）
    size_t pq_cache_bytes = 0;           // ★ 新增: PQ 块缓存当前占用
    // ... 其余不变 ...
};
```

---

## 五、搜索路径改造（核心变更）

### 5.1 `flush()` 改造

```
改造后的 flush():
  1. pq_.train() → 码本在内存
  2. pq_.encode_all() → codes (局部变量, n×M 字节)
  3. ★ v5: 确定文件路径 (增加 this 指针防同进程多实例冲突):
     if (cfg_.pq_codes_path 非空) → 使用该路径
     else → "/tmp/curator_pq_<pid>_<this_ptr>.bin"  (审查发现 #20)
  4. pq_.write_to_disk(path, codes) → 写入磁盘
  5. codes.clear(); codes.shrink_to_fit() → 释放内存
  6. bool ok = pq_.open_cache(path, cfg_.pq_cache_block_size, cfg_.pq_cache_max_blocks)
     if (!ok) → fprintf(stderr, "WARNING: PQ cache open failed, will use exact distance\n")
  7. pq_.set_pq_file_path(path) → 记录路径供析构清理
  8. pq_.set_persist(cfg_.persist_pq_codes) → 记录持久化策略
  9. ★ v5: use_flash_storage 验证 (审查发现 #21)
     if (cfg_.pq_enabled && !cfg_.use_flash_storage) {
         fprintf(stderr, "WARNING: pq_enabled=true requires use_flash_storage=true for fallback, auto-enabling\n");
         cfg_.use_flash_storage = true;
     }
  10. 继续 FlashStore 写入（不变）
  11. raw_buffer_ 释放（不变）

析构时:
  if (!cfg_.persist_pq_codes) → pq_.close_cache() 内部删除临时文件
  else → pq_.close_cache() 仅关闭不删除
```

### 5.2 `search_one()` Phase 2 改造

> **与 legacy 的对比**: Legacy `compute_pq_distances` (`MultiTenantIndexIVFHierarchical.cpp:851-889`) 在函数内部重建 d_table（每次访问 shortlist 都重建）。本方案将 d_table 构建提升到 `search_one()` 层级，同 query 内复用。对于 M=128 场景，若 frontier 访问 5 个 shortlist，这节省了约 13 万次冗余 `l2_sqr` 运算。此外，分离 `build_distance_table` 和 `compute_pq_distances` 调用消除了 legacy 的 20/80 启发式 profiling hack，实现精确的分阶段计时。详见 §〇 审查发现 #9。

```cpp
// ── Phase 2: Frontier search (curator_index.cpp search_one 内) ──

// ★ v3: pq_d_table 为栈上局部变量（每query独立，线程安全）
std::vector<float> pq_d_table;
bool pq_table_built = false;

while (!frontier.empty()) {
    auto [score, node] = frontier.top();
    frontier.pop();

    auto it = node->shortlists.find(tid);
    if (it != node->shortlists.end()) {
        if (cfg_.pq_enabled && pq_.is_trained() && pq_.has_cache()) {
            // ══════ PQ 近似距离路径 ══════
            // 构建距离查表（整个 query 仅首次构建，在栈上，线程安全）
            if (!pq_table_built) {
                auto t_tbl = profiling_on_ ? now() : time_point{};
                pq_.build_distance_table(x, pq_d_table);
                pq_table_built = true;
                if (profiling_on_) {
                    profile_.pq_table_build_ms = elapsed(t_tbl);
                }
            }

            // 用 PQ 码计算 ADC 距离（线程安全，内部加锁+批量预取）
            auto t_pq = profiling_on_ ? now() : time_point{};
            pq_.compute_pq_distances(pq_d_table, it->second.data, sorted_cands);
            if (profiling_on_) {
                profile_.pq_distance_compute_ms += elapsed(t_pq);
            }
        } else {
            // ══════ 精确距离回退路径（现有逻辑）══════
            for (int_vid_t vid : it->second.data) {
                float dist = compute_vector_distance(x, vid);
                sorted_cands.emplace_back(dist, vid);
            }
        }

        std::sort(sorted_cands.begin(), sorted_cands.end());
        // ... (batch_insert, updated check 不变) ...
    } else if (node->bf.contains(tid)) {
        compute_child_scores_with_prefetch(*this, node, x, frontier);
    }
}
```

### 5.3 关键设计说明

**距离查表生命周期 (v3 修订)**:
- `pq_d_table` 是 `search_one()` 栈上的局部 `std::vector<float>`
- 同 query 内多次进入 PQ 分支时复用（首次 `build_distance_table` 后置 `pq_table_built = true`）
- 不同 query 之间完全隔离（各自在独立线程栈上）
- 避免了 v2 中 `mutable` 成员方案的 `batch_query` 数据竞争

**PQ 距离输出排序**:
- `compute_pq_distances` 按 vids 顺序追加到 `output`，**不做排序**
- 调用方（`search_one()`）在 `batch_insert` 前统一 `std::sort(sorted_cands)`
- 这与精确距离路径的处理一致

**回退路径**:
- 当 `!cfg_.pq_enabled` 或 `!pq_.has_cache()` 时，走精确距离路径（现有逻辑保持不变）
- `has_cache()` 返回 false 的可能原因：flush 未调用、open_cache 失败、或索引尚未最终化

---

## 六、文件变更清单

### 6.1 新增文件

| 文件 | 说明 | 预估行数 |
|------|------|------|
| `src/pq_block_cache.h` | PQ块缓存类声明（含 mutex、LRU、Stats、prefetch_and_get_codes） | ~140 |
| `src/pq_block_cache.cpp` | PQ块缓存实现（open、get_code、prefetch_and_get_codes、prefetch_blocks、load_block） | ~220 |

### 6.2 修改文件

| 文件 | 变更内容 | 变更量 |
|------|----------|--------|
| `src/config.h` | +2字段: `pq_cache_block_size`, `pq_cache_max_blocks` | +5行 |
| `src/pq_codec.h` | 移除 vid_to_pq_code_/mmap；新增 block_cache_ (mutable unique_ptr)；新增 open_cache/close_cache/compute_pq_distances/has_cache；移除 load_from_disk/free_in_memory_codes/codes/codes_in_memory/codes_on_disk/compute_distances(callback) | ~55行变更 |
| `src/pq_codec.cpp` | 重写 get_code（拷贝+thread_local）；新增 compute_pq_distances（v3: sentinel + 批量锁）；新增 open_cache/close_cache；移除 mmap 逻辑；析构中清理缓存 | ~110行变更 |
| `src/curator_index.h` | pq_ 改为 mutable（+）；移除 pq_d_table_（v3）；+pq_cache_stats/pq_cache_bytes | +6行 |
| `src/curator_index.cpp` | flush() 改造（编码→写磁盘→开缓存→释放codes）；search_one() 新增 PQ 分支（v3: 栈上 d_table）；memory_breakdown() 用 pq_cache_bytes 替代 pq_codes_bytes | ~120行变更 |
| `src/main.cpp` | JSON 解析 +2 参数; config 输出 +2 字段 | +12行 |
| `src/profiling.h` | MemoryBreakdown: -pq_codes_bytes, +pq_cache_bytes | ±3行 |
| `CMakeLists.txt` | CURATOR_SOURCES + `src/pq_block_cache.cpp` | +1行 |

### 6.3 不变文件

其余全部模块保持不变：`kmeans`, `cluster_tree`, `shortlist`, `flash_store`, `temp_index`, `complex_predicate`, `distance`, `bloom_filter`, `tree_node`, `cnpy`, `common`。

---

## 七、实施步骤

### Step 1: 新增 `PQBlockCache` 模块

**产出**: `src/pq_block_cache.h` + `src/pq_block_cache.cpp`

**实现要点**:
1. `open()`: 用 `::open()` + `::pread()` 进行随机读取；验证 32 字节 header（magic、M、nbits、n_codes）
2. `get_code()`: 计算 block_id；若在缓存中→memcpy到out；若不在→load_block→移LRU头部→memcpy
3. `prefetch_blocks()`: 对每个未缓存的 block_id 按顺序 load_block（加锁一次）。**注意**: 此接口不与 `prefetch_and_get_codes` 串联使用（避免双重加锁）
4. `prefetch_and_get_codes()` (v5 核心接口): 
   - 加锁一次
   - 第一遍扫描 seq_indices：加载所有缺失块（去重，同一 block_id 仅加载一次）
   - 第二遍扫描 seq_indices：填充指针 + LRU 更新（去重，仅 block_id 变化时更新）+ 统计计数
   - **不更新统计于第一遍扫描**——统计在第二遍按 seq_idx 粒度统一进行（审查发现 #16）
   - 解锁返回
   - 调用方必须在指针失效前使用完毕
5. 所有公开方法内部 `std::lock_guard<std::mutex> lock(mutex_)`
6. `load_block_locked()` (private, 调用前须持锁): 
   - 计算文件偏移 (`HEADER_SIZE + block_id * block_size * M`)
   - **满块分配**: `data.resize(block_size * M, 0)`（始终分配满块大小）
   - `::pread()` 读取 `min(block_size, n_codes - block_start) * M` 字节
   - 最后一块尾部自然零填充（审查发现 #17）
7. `evict_one_locked()`: 取 `lru_list_.back()` 并从 `cache_` 和 `lru_list_` 中移除
8. 析构: `close(false)` — 仅关闭 fd，不删除文件

**验证**: 编译通过；编写简单测试验证 hit/miss/evict 逻辑；验证跨块边界的 seq_idx 访问正确性。

### Step 2: 改造 `PQCodec` 类

**产出**: `src/pq_codec.h` + `src/pq_codec.cpp`

**实现要点**:
1. 移除 `vid_to_pq_code_`, `pq_codes_mmap_`, `pq_codes_mmap_size_` 成员
2. 新增 `mutable std::unique_ptr<PQBlockCache> block_cache_`
3. 新增 `static thread_local std::vector<uint8_t> tl_code_buf_`
4. 重写 `get_code()`: 用 tl_code_buf_ 做缓冲区调用 block_cache_->get_code()
5. 新增 `compute_pq_distances(d_table, vids, output)`: 按 v3 设计——sentinel + prefetch_and_get_codes + ADC 求和
6. 移除 `load_from_disk()`, `free_in_memory_codes()`, `codes()`, `codes_in_memory()`, `codes_on_disk()`, `compute_distances(callback)`
7. 新增 `open_cache()`, `close_cache()`, `has_cache()`, `cache_stats()`, `cache_bytes()`, `reset_cache_stats()`
8. 新增 `pq_file_path_` 和 `persist_pq_codes_` 成员 + setter
9. 析构: 若 `!persist_pq_codes_` → `close_cache()` + `::unlink(pq_file_path_.c_str())`；否则仅 `close_cache()`
10. `encode_all`, `write_to_disk`, `build_distance_table`, `compute_exact_batch`, `train` 保持不变
11. ★ v5: `build_distance_table` 增加前置条件检查：`if (!is_trained()) return;`（审查发现 #18）

**验证**: 编译通过；train+encode+write+open_cache+get_code 手动验证正确性。

### Step 3: 更新配置和CLI

**产出**: `src/config.h` + `src/main.cpp`

**任务**:
1. `config.h` 新增 2 个字段：`pq_cache_block_size` (默认 4096), `pq_cache_max_blocks` (默认 256)
2. `main.cpp`: `load_config_from_json` 新增:
   ```cpp
   get_int("pq_cache_block_size", cfg.pq_cache_block_size);
   get_int("pq_cache_max_blocks", cfg.pq_cache_max_blocks);
   ```
3. `main.cpp`: `write_json_results` 输出配置时包含新字段
4. 更新示例配置文件的参数说明

**验证**: JSON 配置解析+输出正确。

### Step 4: 改造 `CuratorIndex`

**产出**: `src/curator_index.h` + `src/curator_index.cpp`

**任务**:

1. **`curator_index.h`**:
   - `PQCodec pq_` → `mutable PQCodec pq_`
   - **不**新增 `pq_d_table_`（v3: 改为 search_one 局部变量）
   - 新增 `pq_cache_stats()`, `pq_cache_bytes()`

2. **`flush()`**:
   - 确定文件路径（cfg 指定或用 `/tmp/curator_pq_<pid>_<this_ptr>.bin`，v5 防同进程多实例冲突）
   - encode_all → write_to_disk → codes 释放 → open_cache
   - 若 open_cache 失败 → fprintf(stderr, warning)
   - 设置 persist_pq_codes 到 pq_ 中
   - ★ v5: 若 `pq_enabled && !use_flash_storage`，自动启用 flash_storage（审查发现 #21）

3. **`search_one()` Phase 2**:
   - 按 §5.2 的 v3 代码接入 PQ 距离计算分支
   - `pq_d_table` 为栈上局部变量
   - 接入 profiling 计时（pq_table_build_ms, pq_distance_compute_ms）

4. **`memory_breakdown()`**:
   - `mb.pq_cache_bytes = pq_.cache_bytes()` 替代原来的 `mb.pq_codes_bytes`
   - 移除遍历 `pq_.codes()` 的循环

5. **`compute_vector_distance()`**: 保持不变

6. **`search_unfiltered()`**: 当前使用 `compute_vector_distance` → FlashStore。可暂不接入PQ（无过滤搜索的访问模式不同，暂保持现有逻辑）。

7. **`search_with_bitmap()` / `search_temp_index()`**: **本次改造不接入 PQ**。当前 `search_temp_index` 使用 centroid-based 近似距离，legacy 在此路径使用了 `compute_pq_distances`（见 §〇 审查发现 #10）。接入 PQ 需重构 `search_temp_index` 接口，留作后续优化（方案见 §九.1）。

**验证**:
- arxiv_small 端到端 bench 测试
- 对比改造前后 labels/distances 一致性
- memory_bytes 验证下降

### Step 5: 端到端测试

**测试矩阵**:

| 数据集 | 配置 | 验证项 |
|--------|------|--------|
| sift1m_small | M=16, batch_query=false | 正确性 + 内存 |
| arxiv_small | M=128, batch_query=false | 正确性 + 内存 + 延迟 |
| arxiv_small | M=128, batch_query=true | **线程安全** + 正确性 |
| arxiv (full, 如有) | M=128 | 大规模内存收益 |

**验证标准**:
- 搜索结果 labels 与改造前一致（PQ 近似距离可能产生略有不同的排序，但 recall@k 不应退化超过 1%）
- `memory_bytes` 中 PQ 相关部分降至缓存大小上限以下
- batch_query=true 下无 crash、无数据竞争（用 `valgrind --tool=helgrind` 或 ThreadSanitizer 验证）
- 查询延迟增加 < 10%

### Step 6: 清理和文档

- 更新 `EXPERIMENT_GUIDE.md` 参数说明（新增 `pq_cache_block_size`、`pq_cache_max_blocks`）
- 清理无用代码注释
- 确认 `profiling.h` 字段更新

---

## 八、风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| PQ 距离导致召回率下降 | 搜索质量 | 1) ADC rerank 用全精度修正 2) 可禁用 PQ 回退到 FlashStore |
| 缓存抖动（max_blocks 过小） | 频繁磁盘 I/O、延迟增加 | 1) 默认 256 块覆盖典型工作集 2) 可配置调大 3) Stats 暴露命中率 |
| 多线程 mutex 争用 | 并发性能下降 | v5: prefetch_and_get_codes 一次锁持有完成所有操作（加载+填充指针+LRU），后续 ADC 计算无锁 |
| M=128 单块 512KB，I/O 较大 | 偶发延迟尖峰 | SSD 上 512KB 随机读延迟 ~100μs，可接受；block_size 可调小 |
| PQ 码文件与 Flash 文件同目录 | 磁盘空间 | PQ 码文件大小 = n×M，与数据集规模成正比，需预留空间 |
| v3新增: `prefetch_and_get_codes` 返回指针悬空 | crash/错误结果 | 在 `compute_pq_distances` 中立即使用指针完成 ADC 计算，不跨函数传递 |
| v4新增: 冷启动延迟（首次查询时缓存为空） | 首次查询延迟偏高 | 1) 首个 shortlist 仅 ~128 条向量，I/O 1-2 次块读取 2) 可考虑在 `flush()` 后预加载首个块 |
| v4新增: PQ 始终在磁盘（vs legacy可选纯内存） | 磁盘故障导致搜索不可用 | 1) 回退路径：`has_cache()` 为 false 时自动走精确距离 (FlashStore) 2) `pq_enabled=false` 可完全禁用 PQ |
| v4新增: temp_index 搜索路径未接入 PQ | bitmap-filter 查询无法享受 PQ 加速 | 短期：保持现有 centroid-近似距离逻辑 后续：按 §九 方案(a) 通过 lambda 回调接入 |
| **v5新增**: 跨线程指针失效（锁释放后另一线程 LRU 淘汰） | use-after-free / 静默数据错误 | 见审查发现 #14：采用方案 A（文档化+保守 sizing）。ADC 计算窗口~微秒级，max_blocks=256 覆盖典型工作集，实际风险极低。必要时可升级到引用计数方案 |
| **v5新增**: `use_flash_storage=false` + `pq_enabled=true` 回退崩溃 | 特殊配置下搜索失败 | 在 `flush()` 中检测此组合并自动启用 flash storage（或打印 error 拒绝运行）。详见审查发现 #21 |
| **v5新增**: 同进程多实例 PQ 文件路径冲突 | 文件覆盖/数据损坏 | PID + this 指针组合确保唯一性。详见审查发现 #20 |

---

## 九、后续优化项 (Future Work)

以下优化在当前改造范围之外，但在 legacy 代码中有对应实现，作为后续迭代的参考：

### 9.1 temp_index 搜索路径 PQ 集成

**现状**: `search_temp_index` (temp_index.cpp) 使用 centroid-based 近似距离，不使用 PQ。Legacy 代码 (`MultiTenantIndexIVFHierarchical.cpp:2069-2072`) 在此路径中使用了 `compute_pq_distances`。

**方案**: 在 `search_temp_index` 中增加回调参数 `std::function<float(int_vid_t)> distance_fn`，由 `CuratorIndex::search_with_bitmap` 根据 `pq_enabled && pq_.has_cache()` 注入 PQ 距离计算或 FlashStore 精确距离计算。这样 `temp_index` 保持独立性（不直接依赖 `PQCodec`），同时获得准确的向量距离。

### 9.2 exact distance fallback 路径的批量+预取优化

**现状**: 当 PQ 不可用时，fallback 路径逐条调用 `compute_vector_distance`（curator_index.cpp:412-415）。Legacy 代码的 `compute_dists_with_prefetch` 使用 batch-of-4 + prefetch 优化。

**方案**: 在 `distance.h` 中基于已有的 `compute_batch_dists` 模板，封装一个接受 `(CuratorIndex&, vids, x, output)` 的批量距离计算函数，内部使用 `compute_vector_distance` + prefetch。

### 9.3 flush 后缓存预热

**现状**: `flush()` 后缓存为空，首次查询触发冷启动 I/O。

**方案**: 在 `flush()` 完成后，可选调用 `pq_.prefetch_codes(first_N_seq_indices)` 预加载前 N 个块到缓存中（N 可配置），减少首次查询延迟。

### 9.4 `search_unfiltered` 路径 PQ 接入

**现状**: `search_unfiltered` 直接扫描叶子 `vector_indices`，逐条 `compute_vector_distance`。该路径访问模式不同（大量向量，不通过 shortlist），PQ 加速效果有待评估。

**方案**: 在 `search_unfiltered` 的 bucket 扫描循环中，若 `pq_enabled && has_cache()`，批量收集 vids 后调用 `compute_pq_distances` 做粗排，取 top-K×factor 用全精度重排。

---

## 十、实施预估

| 步骤 | 内容 | 预估 |
|------|------|------|
| Step 1 | PQBlockCache 模块 (含 v3 prefetch_and_get_codes) | 3-4 h |
| Step 2 | PQCodec 改造 | 2-3 h |
| Step 3 | 配置和 CLI | 1 h |
| Step 4 | CuratorIndex 搜索路径 | 3-4 h |
| Step 5 | 端到端测试 (含 ThreadSanitizer 验证) | 2-3 h |
| Step 6 | 清理和文档 | 1 h |
| **总计** | | **12-16 h** |

---

## 附录A: v5 `compute_pq_distances` 完整代码

> **v5 修正**: 移除了冗余的 `prefetch_blocks` 调用（审查发现 #13）。`prefetch_and_get_codes` 在一次锁持有期间完成所有 I/O + 指针填充。同时移除了不必要的 `needed_blocks` 集合收集（审查发现 #19）。

```cpp
void PQCodec::compute_pq_distances(
        const std::vector<float>& d_table,       // [M × ksub], 由 build_distance_table 预构建
        const std::vector<int_vid_t>& vids,       // 候选 vid 列表
        std::vector<std::pair<float, int_vid_t>>& output) const {

    if (!block_cache_ || !vid_to_seq_) return;
    assert(d_table.size() == M_ * ksub_ && "d_table size mismatch");

    // 1. vid → seq_idx 转换（含 sentinel）
    std::vector<size_t> seq_indices;
    seq_indices.reserve(vids.size());
    for (int_vid_t vid : vids) {
        auto it = vid_to_seq_->find(vid);
        if (it != vid_to_seq_->end()) {
            seq_indices.push_back(it->second);
        } else {
            seq_indices.push_back(SIZE_MAX);  // sentinel: vid not in seq map
        }
    }

    // 2. 批量获取所有码（★ 一次加锁：加载缺失块 + 填充指针 + LRU更新）
    //    无需单独调用 prefetch_blocks —— prefetch_and_get_codes 已涵盖块加载逻辑
    std::vector<const uint8_t*> codes;
    block_cache_->prefetch_and_get_codes(seq_indices, codes);

    // 3. ADC 距离计算（无锁！codes 指针已在锁内填充完毕）
    //    注意：codes[i] 指向缓存内部数据，在下一次任何 PQBlockCache 修改操作前有效
    //    详细讨论见审查发现 #14
    for (size_t i = 0; i < vids.size(); i++) {
        const uint8_t* code = codes[i];
        if (!code) continue;  // sentinel 或越界

        float dist = 0.0f;
        for (size_t m = 0; m < M_; m++) {
            dist += d_table[m * ksub_ + code[m]];
        }
        output.emplace_back(dist, vids[i]);
    }
}
```

## 附录B: `PQBlockCache::prefetch_and_get_codes` v5 实现概要

> **v5 修正**: (1) LRU 更新去重——连续同 block_id 仅更新一次（审查发现 #15）；(2) 统计按 seq_idx 粒度统一计数（审查发现 #16）；(3) 明确满块分配策略——`CacheEntry::data` 始终分配 `block_size * M_` 字节，最后一块尾部由 `::pread` 自然零填充（审查发现 #17）。

```cpp
void PQBlockCache::prefetch_and_get_codes(
        const std::vector<size_t>& seq_indices,
        std::vector<const uint8_t*>& out_codes) {

    out_codes.resize(seq_indices.size(), nullptr);

    std::lock_guard<std::mutex> lock(mutex_);

    // 1. 收集需要的 block_id 并加载缺失的块（去重：同一 block_id 仅加载一次）
    for (size_t seq_idx : seq_indices) {
        if (seq_idx >= n_codes_) continue;
        size_t block_id = get_block_id(seq_idx, cfg_.block_size);
        if (cache_.find(block_id) == cache_.end()) {
            while (cache_.size() >= cfg_.max_blocks) {
                evict_one_locked();
            }
            load_block_locked(block_id);
            // 统计在步骤2按 seq_idx 粒度统一更新，此处不单独计数
        }
    }

    // 2. 填充指针 + LRU 更新 + 统计（按 seq_idx 粒度）
    size_t last_block_id = SIZE_MAX;
    for (size_t i = 0; i < seq_indices.size(); i++) {
        size_t seq_idx = seq_indices[i];
        if (seq_idx >= n_codes_) continue;  // out_codes[i] already nullptr

        size_t block_id = get_block_id(seq_idx, cfg_.block_size);
        auto it = cache_.find(block_id);
        if (it == cache_.end()) continue;  // shouldn't happen after step 1

        // ★ v5: 仅当切换到不同 block 时才更新 LRU（避免连续同块冗余操作）
        if (block_id != last_block_id) {
            lru_list_.erase(it->second.second);
            lru_list_.push_front(block_id);
            it->second.second = lru_list_.begin();
            last_block_id = block_id;
        }

        // ★ v5: 按 seq_idx 粒度统一统计（hit_rate 分母 = 总访问次数）
        stats_.cache_hits++;

        // 块内偏移（满块分配策略：data 始终为 block_size * M_ 字节）
        size_t block_offset = seq_idx - block_id * cfg_.block_size;
        out_codes[i] = it->second.first.data.data() + block_offset * M_;
    }
    // 锁在此处释放
}
```

### B.1 `load_block_locked` 实现概要（满块分配策略）

```cpp
void PQBlockCache::load_block_locked(size_t block_id) {
    size_t block_start = block_id * cfg_.block_size;
    size_t block_end = std::min(block_start + cfg_.block_size, n_codes_);
    size_t n_in_block = block_end - block_start;

    // ★ 满块分配：始终分配 block_size * M_ 字节
    //    最后一块尾部字节由 ::pread 返回 0（读超出文件末尾）自然填充
    CacheEntry entry;
    entry.block_id = block_id;
    entry.data.resize(cfg_.block_size * M_, 0);  // 零初始化

    size_t offset = HEADER_SIZE + block_id * cfg_.block_size * M_;
    size_t bytes_to_read = n_in_block * M_;
    ssize_t nread = ::pread(fd_, entry.data.data(), bytes_to_read, offset);
    if (nread != static_cast<ssize_t>(bytes_to_read)) {
        // I/O error handling: throw or return false
    }

    // 插入缓存 + LRU 头部（先 push_front 再插入，确保 iterator 正确）
    lru_list_.push_front(block_id);
    cache_[block_id] = {std::move(entry), lru_list_.begin()};
}
```

### B.2 指针生命周期合约

`prefetch_and_get_codes` 返回的 `out_codes[i]` 指针满足以下合约：
- **有效范围**: 指向 `CacheEntry::data` 内部，偏移 = `(seq_idx % block_size) * M_`
- **失效条件**: 任何线程调用以下方法后指针可能失效——`get_code()`、`prefetch_and_get_codes()`、`prefetch_blocks()`（若触发 LRU 淘汰）
- **安全使用模式**: 在调用 `prefetch_and_get_codes` 后、任何其他 `PQBlockCache` 方法调用前，立即使用所有指针完成数据读取
- **不安全使用模式**: 将指针跨函数传递或存储；在指针使用期间并发调用其他缓存方法
- **缓解**: `compute_pq_distances` 在获取指针后立即在无锁代码段中完成 ADC 计算，满足安全使用模式。详见审查发现 #14。

> **设计意图**: 选择零拷贝（返回内部指针）而非拷贝输出是为了避免每次 ~16KB (128 vecs × M=128 bytes) 的额外内存拷贝。若后续压测中发现跨线程淘汰问题，可切换到引用计数方案（审查发现 #14 方案 B）。

## 附录C: 模块依赖关系（改造后）

```
                        ┌──────────────┐
                        │   main.cpp   │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │curator_index │  (编排层: pq_ 改为 mutable)
                        │  .h / .cpp   │
                        └──┬──┬──┬──┬──┘
               ┌───────────┤  │  │  ├───────────┐
               │        ┌──┘  │  └──┐        ┌─┘
         ┌─────▼────┐ ┌▼──────▼──┐ ┌▼──────┐ ┌▼──────────────┐
         │cluster   │ │ shortlist │ │  pq   │ │  flash_store  │
         │_tree     │ │ .h / .cpp │ │_codec │ │  .h / .cpp    │
         │.h / .cpp │ └───────────┘ │.h/cpp │ └──────────────┘
         └──────────┘               └──┬────┘
                                       │ mutable unique_ptr
                                ┌──────▼──────────┐
                                │ pq_block_cache  │  ★ NEW
                                │ .h / .cpp       │  (含 std::mutex)
                                └─────────────────┘
```
