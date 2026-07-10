# Curator 索引 PPT 大纲

> 简约准确风格，结合源码说明算法流程

---

## Slide 1: 标题页

**Curator：基于层次化聚类树的过滤向量检索索引**

副标题：Hierarchical K-means Clustering Tree for Filtered Vector Search

---

## Slide 2: 问题背景

- **过滤向量检索**：在向量数据库中执行带标签过滤的近似最近邻搜索（ANN）
  - 例如："查询与 query 最相似的 10 个向量，且它们必须属于 tenant_id=5"
- **已有基线的不足**：
  - Pre-Filtering：先过滤再搜索，过滤代价高
  - Post-Filtering（DiskIVF/SPANN）：先搜索再过滤，可能漏掉结果
- **Curator 的解决方案**：在索引结构内**原生融合**标签过滤与向量搜索

---

## Slide 3: 核心数据结构总览

```
Curator = 层次化 K-means 聚类树 + Bloom Filter + Shortlist + PQ压缩

                      [Root] (centroid=全局均值)
                  /      |      \     ← n_clusters 路分支
               [L1]     [L1]    [L1]
              /   \     /   \
             ...  ...  ...  ...
            /       \       /
        [Leaf]    [Leaf]  [Leaf]

每个节点包含:
  • centroid + variance    → 聚类信息
  • Bloom Filter           → 子树有哪些租户(tenant)
  • Shortlists             → 租户→向量ID 有序列表
  • vector_indices (叶子)   → 叶子包含的全部向量
```

**源码**: [tree_node.h:16-93](Curator/src/tree_node.h#L16-L93)

---

## Slide 4: 关键类型与不变量

| 类型 | 底层 | 含义 |
|------|------|------|
| `ext_vid_t` | `uint32_t` | 外部向量ID（用户label） |
| `int_vid_t` | `uint64_t` | **位编码路径**的内部vid |
| `ext_lid_t` | `int16_t` | 外部租户label |
| `int_lid_t` | `int16_t` | 内部连续分配的租户ID |

**int_vid_t 位编码**（每层 6bit 分支 + 最后 10bit 局部索引）：
```
[level_0: 6bit | level_1: 6bit | ... | local_vid: 10bit]
                                  ↑ 最多56/6≈9层深度
```

**关键不变量**：
1. 所有非叶子节点恰好有 `n_clusters` 个子节点（含空节点！）
2. 短列表大小 ≤ `max_sl_size`（超限触发向下分裂）
3. Bloom Filter 汇总所有子树中的租户

**源码**: [common.h:50-55](Curator/src/common.h#L50-L55), [cluster_tree.cpp:91](Curator/src/cluster_tree.cpp#L91)

---

## Slide 5: 模块架构 — 星型依赖拓扑

```
                 ┌──────────────┐
                 │  main.cpp    │  CLI 入口
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │curator_index │  主编排类（唯一编排点）
                 │  .h / .cpp   │
                 └──┬──┬──┬──┬─┘
                    │  │  │  │
          ┌─────────┘  │  │  └─────────┐
          ▼            ▼  ▼            ▼
    ┌──────────┐  ┌──────────┐  ┌──────────────┐
    │cluster_  │  │ pq_codec │  │ flash_store  │
    │tree      │  │+block_c. │  │              │
    └────┬─────┘  └──────────┘  └──────────────┘
    ┌────┼─────┐
    ▼    ▼     ▼
┌────┐┌──────┐┌────────┐
│kms ││short ││temp_   │
│    ││list  ││index   │
└────┘└──────┘└────────┘
        ┌───────┴───────┐
        ▼               ▼
  ┌──────────┐   ┌──────────────┐
  │common.h  │   │tree_node.h   │  ← 零相互依赖基础层
  │distance.h│   │bloom_filter.h│
  │config.h  │   │profiling.h   │
  └──────────┘   └──────────────┘
```

**源码**: [CMakeLists.txt](Curator/CMakeLists.txt)（14个编译单元，仅依赖 OpenMP）

---

## Slide 6: 构建流程（一）— 递归建树

```
build_tree(node, n, x):
  ① 计算当前节点 centroid (根节点=全局均值)
  ② 若 n ≤ max_leaf_size 或 level ≥ MAX_TREE_DEPTH → 停止（叶子）
  ③ K-means(n → n_clusters) 聚类
  ④ 按 cluster assignment 重排向量
  ⑤ for clus_id = 0..n_clusters-1:  ← 全部创建，含空节点！
       new TreeNode(level+1, clus_id, node, centroid[clus_id])
       build_tree(child, cluster_size[clus_id], sorted_x[offset])
  ⑥ assert children.size() == n_clusters
```

**源码**: [cluster_tree.cpp:14-94](Curator/src/cluster_tree.cpp#L14-L94)

关键设计：**空子节点也必须保留** — temp_index 的位前缀二分依赖 children 数组完整性

---

## Slide 7: 构建流程（二）— 插入向量 + 授权

```
add_vector(x, label):
  ① assign_to_leaf(root, x) → 沿 centroid 最近邻下降到叶子
  ② 位编码生成 int_vid_t
  ③ 建立 ext→int 映射
  ④ 追加到 raw_buffer_
  ⑤ 向上更新 variance (leaf→root)

grant_access_impl(node, vid, tid):          // 递归核心
  若 node 是叶子:
    → 直接插入 shortlists[tid]，更新 BF
  若 shortlists 中存在 tid:
    → 插入 vid；若 size > max_sl_size → split_shortlist (向下推)
  若 BF 中不存在 tid:
    → 创建新短列表
  否则 (BF 有 tid 但无短列表):
    → 按 vid 路径位提取 child_id → 递归下降
```

**源码**: [curator_index.cpp:162-231](Curator/src/curator_index.cpp#L162-L231)

---

## Slide 8: 构建流程（三）— flush 最终化

```
flush():
  ┌─ PQ 训练 ─────────────────────────────────────┐
  │ ① pq_.train(n, raw_buffer_, d, M, nbits=8)    │
  │    → M 个子空间独立 K-means (OpenMP 并行)      │
  │ ② pq_.encode_all → 全部向量编码为 uint8_t[M]   │
  │ ③ pq_.write_to_disk → 32B Header + N×M bytes  │
  │ ④ codes.clear() → 释放内存中的 PQ 码           │
  │ ⑤ pq_.open_cache → 打开 LRU 块缓存 (按需加载)   │
  └──────────────────────────────────────────────┘
  ┌─ Flash 存储 ──────────────────────────────────┐
  │ ① DFS 遍历收集所有叶子，分配 seq 号            │
  │ ② 按叶子分组向量，写入 flash 文件               │
  │    (每叶子固定 leaf_region_size，含空洞)        │
  │ ③ 释放 raw_buffer_（全精度向量已存盘）          │
  └──────────────────────────────────────────────┘
```

**源码**: [curator_index.cpp:239-344](Curator/src/curator_index.cpp#L239-L344), [pq_codec.cpp:42-277](Curator/src/pq_codec.cpp#L42-L277)

---

## Slide 9: 索引构建完整调用链（一图总结）

```
main() [bench 模式]
  │
  ├── CuratorIndex::CuratorIndex(cfg)
  │     ├─ 验证 cfg.n_clusters ≤ MAX_BRANCH_FACTOR (64)
  │     ├─ 验证 cfg.max_leaf_size ≤ MAX_LEAF_SIZE (1024)
  │     ├─ 验证 cfg.search_ef > 0
  │     ├─ new TreeNode(level=0, centroid=nullptr) → root_
  │     └─ pq_.set_seq_maps(&seq_to_vid_, &vid_to_seq_)  // 非拥有指针
  │
  ├── CuratorIndex::train(n, x)
  │     └─ cluster_tree::build_tree(root_, n, x, cfg_)
  │           │  ┌─ 根节点: centroid ← 全局均值 (n 向量求和÷n)
  │           │  ├─ Guard: n ≤ max_leaf_size? level ≥ MAX_TREE_DEPTH? n < n_clusters?
  │           │  │    → Yes: return (叶子)
  │           │  ├─ kmeans::kmeans(d, n, x, n_clusters, cfg)
  │           │  │    ├─ Random init (shuffle + 抽样 n_clusters 个种子)
  │           │  │    ├─ 迭代 × clus_niter:
  │           │  │    │    ├─ 分配步 (OpenMP parallel for): 每向量→最近质心
  │           │  │    │    ├─ 累积步 (per-thread buffers → reduce)
  │           │  │    │    └─ 更新步: centroid = sum/count; 空簇→全局均值+扰动
  │           │  │    └─ return centroids[n_clusters×d] + assignments[n]
  │           │  ├─ 按 cluster assignment 重排向量→sorted_x[n×d]
  │           │  ├─ for clus_id = 0..n_clusters-1:  // ★ 全部创建，含空节点！
  │           │  │    ├─ new TreeNode(level+1, clus_id, parent,
  │           │  │    │                centroids[clus_id], d, ...)
  │           │  │    │    └─ node_id = parent.node_id | (sibling_id << offset)
  │           │  │    │       (位编码路径: 每层 6bit)
  │           │  │    ├─ build_tree(child, cluster_size, sorted_x[offset], cfg)
  │           │  │    └─ node->children.push_back(child)
  │           │  └─ assert children.size() == n_clusters  // ★ 关键不变量
  │           │
  │           └─ 递归深度: 最多 9 层 (64bit vid / 6bit per level - 10bit leaf)
  │
  ├── for each vector: CuratorIndex::add_vector(x, label) × ntrain
  │     ├─ cluster_tree::assign_to_leaf(root_, x, d)
  │     │    └─ while children 非空:
  │     │         遍历 children → 找 L2(query, child.centroid) 最近的 child
  │     │         curr = best_child → 继续下降 → 直到叶子
  │     ├─ 位编码生成 int_vid_t:
  │     │    offset = 64 - leaf.level × 6 - 10  (MAX_LEAF_SIZE_LOG2)
  │     │    vid = leaf.node_id | (leaf.vector_indices.size() << offset)
  │     ├─ vid_map_.add_mapping(ext_label, vid)     // ext→int 映射
  │     ├─ 存入 raw_buffer_: raw_buffer_ += x[0..d-1]
  │     ├─ 映射表更新:
  │     │    vid_to_buf_offset_[vid] = raw_offset
  │     │    vid_to_leaf_id_[vid]    = leaf->node_id
  │     │    vid_to_local_idx_[vid]  = leaf->vector_indices.size()
  │     │    vid_to_seq_[vid]        = seq_to_vid_.size()   // 全局序号
  │     │    seq_to_vid_.push_back(vid)
  │     ├─ leaf->vector_indices.insert(vid)          // 叶子记录
  │     └─ 向上更新 variance (leaf → root):
  │          curr = leaf
  │          while curr:
  │            dist = l2_sqr(x, curr->centroid)
  │            curr->variance.add(dist)
  │            curr = curr->parent
  │
  ├── for each access pair: CuratorIndex::grant_access(vid, tid) × M
  │     ├─ int_vid_t vid = vid_map_.get_id(label)
  │     ├─ int_lid_t int_tid = tid_map_.get_or_create_id(tenant)
  │     │    └─ IdAllocator: allocate (按需扩 id_to_label)
  │     │       或复用 free_list 中的空洞 ID
  │     └─ grant_access_impl(root_, vid, int_tid)
  │           │
  │           ├── [叶子?] → node.shortlists[tid].insert(vid)
  │           │              node.bf.insert(tid)
  │           │
  │           ├── [shortlists 中已有 tid?]
  │           │    → it->second.insert(vid)
  │           │    → if size > max_sl_size:
  │           │         split_shortlist(node, tid, max_sl_size)
  │           │           ├─ 按 vid 位编码提取 child_id
  │           │           ├─ 将超限的 vid 下推到子节点 shortlists
  │           │           ├─ 若父节点 shortlist 变空 → erase
  │           │           └─ 父节点 BF 重算 (recompute_bloom_filter)
  │           │
  │           ├── [BF 不含 tid?] → 创建新 shortlist = {vid}
  │           │     node.bf.insert(tid)
  │           │
  │           └── [BF 含 tid 但无 shortlist?]
  │                → 按 vid 位编码提取 child_id
  │                → grant_access_impl(children[child_id], vid, tid)
  │                   (递归下降)
  │
  └── CuratorIndex::flush()
        │
        ├── [PQ 阶段] if pq_enabled && ntotal > 0:
        │     ├─ pq_.train(ntotal, raw_buffer_, d, pq_M, pq_nbits=8)
        │     │    └─ M 个子空间各自 kmeans(ksub=256, dsub=d/M) 训练码本
        │     │       → pq_codebook_[M × 256 × dsub] (内存, <1MB)
        │     ├─ pq_.encode_all(ntotal, raw_buffer_, codes)
        │     │    └─ for i in 0..ntotal:  encode(x[i], codes[i])
        │     │       → codes[i] = uint8_t[M]  (每维 1 字节)
        │     ├─ pq_.write_to_disk(path, codes, M, nbits)
        │     │    └─ 32B Header (magic"M"nbits"n_codes) + 逐码 fwrite
        │     ├─ codes.clear() → 释放内存中的 PQ 码
        │     └─ pq_.open_cache(path, block_size, max_blocks)
        │          └─ 打开 PQ 码文件，验证 Header → LRU 块缓存就绪
        │
        └── [Flash 阶段] if use_flash_storage && ntotal > 0:
              ├─ DFS 遍历树收集所有叶子节点 → leaves[]
              ├─ 分配 leaf_seq: leaf_node_id_to_seq_[leaf->node_id] = seq
              ├─ leaf_region_size_ = max_leaf_size × d × sizeof(float)
              ├─ flash_.open(path) + flash_.truncate(num_leaves × region_size)
              ├─ 按 leaf_seq 顺序写入:
              │    for each leaf: 收集 leaf->vector_indices 对应的 raw_buffer_ 向量
              │      → flash_.write_leaf_region(offset, region_bytes, vectors, n, d)
              │      → 内部使用 pwrite (绕过用户态缓冲，直接内核 I/O)
              └─ raw_buffer_.clear() → 释放全精度向量内存
                   (后续通过 FlashStore::read_vector/read_batch 按需读取)
```

**关键模块调用关系**:

| 阶段 | 入口 | 核心模块 |
|------|------|---------|
| 建树 | `build_tree()` | `cluster_tree.cpp`, `kmeans.cpp`, `tree_node.h` |
| 插入 | `add_vector()` | `cluster_tree.cpp` (assign_to_leaf), `common.h` (IdMapping) |
| 授权 | `grant_access()` | `shortlist.cpp` (split), `bloom_filter.h`, `common.h` (IdAllocator) |
| 最终化 | `flush()` | `pq_codec.cpp` + `pq_block_cache.cpp`, `flash_store.cpp` |

**源码**: [curator_index.cpp:135-344](Curator/src/curator_index.cpp#L135-L344), [cluster_tree.cpp:14-94](Curator/src/cluster_tree.cpp#L14-L94), [kmeans.cpp:18-149](Curator/src/kmeans.cpp#L18-L149), [shortlist.cpp:9-83](Curator/src/shortlist.cpp#L9-L83)

---

## Slide 10: 搜索总览 — 三种搜索路径

| 路径 | 入口函数 | 适用场景 | 关键参数 |
|------|---------|---------|---------|
| **标准租户搜索** | `search` → `search_one` | 单租户过滤 | search_ef, beam_size |
| **无过滤搜索** | `search_unfiltered` | 无标签过滤 | nprobe, prune_thres |
| **Bitmap过滤** | `search_with_bitmap` | 复杂谓词 | search_ef (临时索引) |

**源码**: [curator_index.h:33-44](Curator/src/curator_index.h#L32-L44)

---

## Slide 11: 标准搜索 — Phase 1: Beam Search

**目标**：沿树下降，每层保留 `beam_size` 个最有前途的节点

```
Beam Search 流程:
  beam = [(score(root), root)]
  loop:
    for each node in beam:
      if node.shortlists 中有目标 tenant:
        → 保留在 next_beam（不再展开子节点）
      else:
        → 展开所有子节点，计算 node_score，加入 next_beam
    if 没有节点被展开过:
      → break（收敛）
    排序 next_beam，保留 top-beam_width
    溢出部分 → 推入 unexpanded（稍后由 Frontier 处理）
  return beam + unexpanded
```

**node_score 公式**：
```
score = L2_dist(query, centroid) - variance_boost × variance_mean
```
> 高方差节点被降权（更易被优先队列弹出）— **剪枝核心机制**

**源码**: [curator_index.cpp:49-102](Curator/src/curator_index.cpp#L49-L102), [tree_node.h:99-105](Curator/src/tree_node.h#L99-L105)

---

## Slide 12: 标准搜索 — Phase 2: Frontier Search

**目标**：优先队列遍历树，收集候选向量

```
Frontier = MinHeap (初始来自 Beam Search)

while Frontier 非空:
  pop 最优节点 node:
    if node.shortlists 中有目标 tenant:
      ┌─ PQ 路径 (默认) ──────────────────────────┐
      │ build_distance_table(query) ← 仅首次执行    │
      │   → [M×256] 预计算距离查表                 │
      │ compute_pq_distances(d_table, shortlist)    │
      │   → vid→seq_idx 转换                        │
      │   → prefetch_and_get_codes（一次加锁！）     │
      │   → ADC 累加（无锁！）                       │
      │   ADC 公式: dist = Σ d_table[m*256+code[m]]│
      └────────────────────────────────────────────┘
      ┌─ 精确路径 (回退) ──────────────────────────┐
      │ 逐 vid 调用 compute_vector_distance()       │
      │   (FlashStore 读取全精度向量 + L2 计算)     │
      └────────────────────────────────────────────┘
      → batch_insert 到 RunningList(top-search_ef)
      → 若 batch_insert 无更新 → break
    else if node.BF 中有目标 tenant:
      → 展开子节点，计算 score 并入队
```

**源码**: [curator_index.cpp:450-522](Curator/src/curator_index.cpp#L450-L522), [pq_codec.cpp:139-179](Curator/src/pq_codec.cpp#L139-L179)

---

## Slide 13: PQ 距离计算 — 线程安全设计

**核心**：`prefetch_and_get_codes` — 一次加锁完成所有操作

```
多线程并发查询场景:
  Thread 1: search_one() ─┐
  Thread 2: search_one() ─┤     每次 shortlist (128 vectors)
  Thread 3: search_one() ─┼──→  pq_.compute_pq_distances(d_table, vids, out)
                           │         │
                           │         ├── lock(mutex_)         ← 唯一一次加锁
                           │         ├── 扫描所有 seq_indices
                           │         ├── 加载缺失的 block（去重）
                           │         ├── 填充 out_codes[] 指针（零拷贝）
                           │         ├── LRU 更新（去重：同一 block 只更新 1 次）
                           │         ├── unlock(mutex_)
                           │         └── 逐码 ADC 累加（无锁！d_table 在栈上）
  Thread N: search_one() ─┘
```

| 对比 | PQ 路径 | 精确路径 |
|------|---------|---------|
| 读取量/向量 | M 字节 (16-128B) | d×4 字节 (512B-3.8KB) |
| 距离计算 | ADC 查表累加 | L2 平方 (d 次乘加) |
| I/O 模式 | 批量预取 (一次加锁) | 逐条 seek+read |

**源码**: [pq_block_cache.cpp:150-199](Curator/src/pq_block_cache.cpp#L150-L199), [pq_codec.cpp:139-179](Curator/src/pq_codec.cpp#L139-L179)

---

## Slide 14: 标准搜索 — Phase 3: ADC Rerank

**目标**：对 PQ 近似距离的 top 候选做精确 L2 重排

```
if pq_use_adc_rerank:
  ① 取 RunningList 前 k × rerank_topk_factor 个候选
  ② Batch I/O 读取全精度向量:
     收集所有候选的 flash offset
     → flash_.read_batch() 合并相邻 I/O（阈值 64KB）
  ③ 计算精确 L2 距离
  ④ 更新 RunningList 为精确距离
  ⑤ 输出最终 top-k
```

**为什么有效**：PQ 近似距离只需读取 M 字节/向量（vs d×4），大幅减少 I/O；仅对少量候选做精确重排保证精度。

**源码**: [curator_index.cpp:530-587](Curator/src/curator_index.cpp#L530-L587), [flash_store.cpp:85-138](Curator/src/flash_store.cpp#L85-L138)

---

## Slide 15: 短列表的生命周期

```
                   插入向量
                      │
                      ▼
              grant_access_impl
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       叶子节点    非叶+有SL    非叶+有BF无SL
          │           │           │
          ▼           ▼           ▼
     插入shortlist  插入SL   递归到子节点
          │       size>max?        │
          │        │  否            │
          │        ▼               │
          │    split_shortlist     │
          │   (向下推给子节点)       │
          │        │               │
          └────────┴───────────────┘
                      │
                      ▼
              删除向量 revoke_access
                      │
                      ▼
              从 shortlist 中 erase
                      │
               size=0 → BF 重算
                      │
                      ▼
            try_merge_shortlists
            (子节点SL合并回父节点)
```

**源码**: [shortlist.cpp:9-83](Curator/src/shortlist.cpp#L9-L83), [curator_index.cpp:204-231](Curator/src/curator_index.cpp#L204-L231)

---

## Slide 16: 无过滤搜索（search_unfiltered）

```
search_unfiltered(query, k):
  ① node_priority = L2(query, centroid) - var_boost × variance
  ② MinHeap 遍历树:
     while pq 非空 且 n_cand_vecs < nprobe:
       pop 节点
       若非叶子 → 展开子节点入队
       若叶子 → 收集到 buckets 列表
  ③ 按质心距离排序 buckets
  ④ prune_thres 剪枝:
     只扫描 dist ≤ prune_thres × min_bucket_dist 的叶子
  ⑤ 遍历幸存叶子的 vector_indices
     → compute_vector_distance → RunningList
  ⑥ 输出 top-k
```

**与标准搜索的关键差异**：
- 不使用 Bloom Filter / Shortlist（无租户概念）
- 使用 `nprobe` 控制探测范围 + `prune_thres` 剪枝
- 直接扫描叶子 vector_indices

**源码**: [curator_index.cpp:610-669](Curator/src/curator_index.cpp#L610-L669)

---

## Slide 17: Bitmap Filter / 复杂谓词搜索

**场景**：复杂谓词过滤（AND/OR/NOT 布尔组合），支持**全局**和**per-query**两种模式

**CLI 接口**：

| 参数 | 粒度 | 说明 |
|------|------|------|
| `--filter "1 2 AND"` | 全局 | 所有 query 共用同一谓词 |
| `--query_filters filters.txt` | Per-query | 每行一个 RPN 表达式，空行=回退 |

**⚠ 谓词必须使用 Reverse Polish Notation (RPN/后缀表达式)**：

| 含义 | ❌ 中缀 | ✅ RPN |
|------|--------|--------|
| A AND B | `1 AND 2` | `1 2 AND` |
| (A AND B) OR C | `(1 AND 2) OR 3` | `1 2 AND 3 OR` |
| A AND NOT B | `7 AND NOT 2` | `7 2 NOT AND` |

**两阶段架构**：

```
Phase 1 (串行预构建):
  ① 加载 query_filters.txt，收集所有唯一谓词
  ② 对每个新谓词:
     find_all_qualified_vecs(pred) → 全量扫描求值
     get_filter_label(pred) → 去重检查 (filter_to_label_ 映射)
     build_filter_index(pred, qualified, n) → 构建 TempIndex + 缓存
  ③ 映射 per_query_filter_label[q] (正常=label, 空=-2 sentinel, 无=-1)

Phase 2 (搜索, 并行安全):
  每条 query 使用预构建的 filter_label → index.search()
    → search_one() → get_cached_temp_index_data() 命中!
    → search_temp_index() 快速路径
```

**TempIndexNode 结构**：
```cpp
struct TempIndexNode {
    int start, end;              // 在 sorted_qualified_vecs 中的区间
    std::vector<int> children;   // 子节点索引
    const float* centroid;       // Non-owning！指向主树
};
```

**源码**: [temp_index.cpp:15-218](Curator/src/temp_index.cpp#L15-L218), [temp_index.h:20-24](Curator/src/main.cpp#L277-L443)

---

## Slide 18: 伪代码 — find_all_qualified_vecs()

**作用**：对给定的 RPN 谓词表达式，全量扫描所有向量，返回满足条件的 vid 列表。

```
find_all_qualified_vecs(filter="1 2 AND"):
  ① 解析谓词
     tokens = tokenize_formula(filter)
     // "1 2 AND" → tokens = ["1", "2", "AND"]

  ② 全量扫描 seq_to_vid_ (0..ntotal-1):
     for i in 0..seq_to_vid_.size()-1:
       vid = seq_to_vid_[i]

       // ── 构建该 vid 的 access_set (它属于哪些 tenant) ──
       access_set = {}
       leaf = find_assigned_leaf(root_, vid)    // 按位编码定位叶子
       curr = leaf
       while curr != nullptr:
         for (tid, shortlist) in curr->shortlists:
           if shortlist.contains(vid):
             access_set.insert(tid)
         curr = curr->parent                     // 向上遍历到 root
       // 现在 access_set 包含了 vid 所属的所有 tenant ID

       // ── RPN 布尔求值 ──
       if evaluate_formula(tokens, access_set):
         result.push_back(vid)

  ③ return result
```

**关键细节**：
- O(N) 全量扫描，仅在 `build_filter_index` 首次遇到新谓词时执行一次
- `access_set` 通过沿树向上收集：父节点可能有该 vid 的 shortlist（合并上来的）
- `evaluate_formula` 是 header-only 模板（RPN 栈式求值），支持 AND/OR/NOT
- 扫描结果被 `build_filter_index` 缓存，后续同谓词查询不再重复扫描

**源码**: [curator_index.cpp:892-921](Curator/src/curator_index.cpp#L892-L921), [complex_predicate.h:216-246](Curator/src/complex_predicate.h#L216-L246)

---

## Slide 19: 伪代码 — build_filter_index()

**作用**：将 `find_all_qualified_vecs` 的结果构建为可快速搜索的临时索引（或写入主树短列表），并分配一个唯一 filter_label 供后续搜索使用。

```
build_filter_index(predicate="1 2 AND", qualified, n):
  ① 确保 qualified 已排序（调用方保证，此处防御性检查）

  ② 分配 filter tenant ID
     filter_label = tid_map_.allocate_reserved_label()
       // 从 int16_max 向下搜索第一个未占用的 label (如 32767)
     filter_tid = tid_map_.allocate_id(filter_label)
       // 分配连续内部 ID (如 0,1,2... 或复用 free_list)
     filter_to_label_[predicate] = filter_label
       // 注册映射：predicate → label (用于后续 get_filter_label 去重)

  ③ if cfg_.use_temp_index_caching:
       // ── 快速路径（默认）：构建临时索引树 + 缓存 ──
       nodes = temp_indexes_[filter_tid] (清空旧数据)
       build_temp_index(root_, qualified, cfg_.n_clusters,
                        cfg_.max_sl_size, nodes)
         // 详见 Slide 19
       qualified_cache_[filter_tid] = std::move(qualified)
         // 保留 qualified 列表（按需提供 vid 数据）

     else:
       // ── 慢速路径：直接写入主聚类树短列表 ──
       batch_grant_access(qualified, filter_tid)
         // 将 qualified vids 逐一 grant 到主树 (同普通授权流程)

  ④ return filter_label  // 外部 label，传给 search()
```

**关键设计**：

| 步骤 | 说明 |
|------|------|
| `allocate_reserved_label()` | 从 int16 最大值向下搜索，确保不与真实 tenant ID 冲突 |
| `filter_to_label_[pred]` | 全局去重：同一谓词只构建一次，多次调用复用已有 label |
| `use_temp_index_caching=true` | **默认路径**：不污染主树短列表，搜索时秒级命中缓存 |
| 缓存 key = `filter_tid` | `search_one()` 通过 `get_cached_temp_index_data(tid)` 查找 |

**调用链**: main.cpp Phase 1 → `find_all_qualified_vecs` → `build_filter_index` → Phase 2 `search(filter_label)` → `search_one` → 缓存命中 → `search_temp_index`

**源码**: [curator_index.cpp:840-881](Curator/src/curator_index.cpp#L840-L881), [main.cpp:340-399](Curator/src/main.cpp#L340-L399)

---

## Slide 20: 伪代码 — build_temp_index() & search_temp_index()

### build_temp_index — 按位前缀二分构建轻量级临时索引树

```
build_temp_index(root, sorted_qualified_vecs, n_clusters, max_sl_size, nodes):
  递归函数 build(start, end, curr_node) → node_idx:

    ① 创建 TempIndexNode:
       node = {start, end, children=[], centroid=curr_node.centroid.data()}
          // ★ centroid 是 Non-owning 指针！指向主树 TreeNode::centroid
       nodes.push_back(node)
       curr_idx = nodes.size() - 1

    ② 终止条件:
       if (end - start) ≤ max_sl_size  → return   // 足够小，可直接扫描
       if curr_node.children.empty()   → return   // 主树叶子，直接扫描

    ③ 按位前缀二分分组
       offset = 64 - MAX_BRANCH_FACTOR_LOG2 × (level + 1)
       对每个 child_idx = 0..n_clusters-1:
         在 [start, end) 中二分搜索 vid 的位前缀 = child_idx 的起止位置
         → child_ranges[child_idx] = (first, last)

    ④ 递归进入非空子树:
       对每个 child_idx:
         if child_ranges[child_idx].first ≠ last:
           child_idx2 = build(first, last, curr_node.children[child_idx])
           nodes[curr_idx].children.push_back(child_idx2)

    ⑤ return curr_idx
```

### search_temp_index — 两阶段搜索 TempIndexNode 树

```
search_temp_index(nodes, qualified_vecs, query, k, d, search_ef, beam_size,
                  compute_distances, distances, labels):

  ┌─ Phase 1: Beam Search (导航) ─────────────────────────────┐
  │ beam = [(L2(query, root.centroid), root_idx=0)]             │
  │ loop:                                                       │
  │   for each (score, node) in beam:                           │
  │     if node 是终止节点 (无children 或 range ≤ search_ef):   │
  │       → 保留在 beam（传给 Phase 2）                         │
  │     else:                                                   │
  │       → 展开所有子节点，用 L2(query, child.centroid) 评分   │
  │       → 加入 next_beam                                      │
  │   sort next_beam                                            │
  │   保留 top-beam_size → 下一轮 beam                          │
  │   溢出部分 → 推入 frontier 队列                              │
  │   if 没有节点被展开 → break                                  │
  │ 将 beam 全部推入 frontier                                    │
  └────────────────────────────────────────────────────────────┘

  ┌─ Phase 2: Frontier Search (收集候选) ──────────────────────┐
  │ while frontier 非空:                                         │
  │   pop (score, node_idx)                                     │
  │   if node 是终止节点:                                        │
  │     ┌─ ★ 收集向量 (范围 [node.start, node.end)) ─┐         │
  │     │ node_vids = qualified_vecs[start:end]       │         │
  │     │ compute_distances(node_vids, node_dists)    │ ← 回调! │
  │     │   → PQ 路径: build_distance_table (一次)    │         │
  │     │     → compute_pq_distances (ADC 累加)       │         │
  │     │   → 精确路径: compute_vector_distance × N   │         │
  │     │ sort node_dists                             │         │
  │     │ results.batch_insert(node_dists)            │         │
  │     └────────────────────────────────────────────┘         │
  │     // ★ 不展开子节点！(避免重复收集)                        │
  │   else:                                                     │
  │     → 展开所有子节点，评分入队 (L2 query-centroid)          │
  │     // ★ 不收集向量！(range 太大)                            │
  │                                                             │
  │ 输出 results 的前 k 个                                      │
  └────────────────────────────────────────────────────────────┘
```

**关键设计细节**：

| 细节 | 说明 |
|------|------|
| `compute_distances` 回调 | 由 `CuratorIndex` 构造，优先走 PQ 距离（ADC），回退到精确 L2 |
| `search_ef` 终止条件 | 与 `build_temp_index` 的 `max_sl_size` 独立——前者控制搜索扫描量，后者控制构建时的树深度 |
| Phase 2 的 if-else | **终止节点 = 只收集不展开**（避免重复），**非终止 = 只展开不收集**（range太大，继续深入） |
| `TempIndexNode::centroid` | Non-owning 指针 → 指向主树 `TreeNode::centroid`，生命周期绑定主树 |

**源码**: [temp_index.cpp:15-218](Curator/src/temp_index.cpp#L15-L218), [temp_index.h:20-56](Curator/src/temp_index.h#L20-L56)

---

## Slide 21: 外部存储架构

```
┌─────────────────────────────────────────────────┐
│                   CuratorIndex                   │
│                                                  │
│  ┌──────────────┐   ┌─────────────────────────┐ │
│  │  PQCodec     │   │  PQBlockCache           │ │
│  │  codebook    │   │  ┌────┬────┬────┬────┐  │ │
│  │  (内存,<1MB) │   │  │Blk0│Blk5│Blk3│ ...│  │ │
│  │              │   │  └────┴────┴────┴────┘  │ │
│  │  每次查询必需 │   │  默认 256×4096 条/块     │ │
│  └──────────────┘   └───────────┬─────────────┘ │
│                                 │ cache miss     │
│                                 ▼                │
│                      ┌─────────────────────┐     │
│                      │ PQ Codes File (磁盘) │     │
│                      │ 32B Header + N×M B   │     │
│                      └─────────────────────┘     │
│                                                  │
│  ┌──────────────┐                               │
│  │ FlashStore   │  全精度向量磁盘文件              │
│  │ (仅 rerank    │  每个叶子固定 region_size       │
│  │  时读取)      │  vid → offset 由 CuratorIndex   │
│  └──────────────┘  的 4 个 unordered_map 维护      │
└─────────────────────────────────────────────────┘

vid → flash_offset 查找链:
  vid → vid_to_leaf_id_ → leaf_node_id_to_seq_
      → leaf_seq × leaf_region_size_ + local_idx × d × sizeof(float)
```

**源码**: [flash_store.cpp:1-148](Curator/src/flash_store.cpp#L1-L148), [pq_block_cache.h:1-137](Curator/src/pq_block_cache.h#L1-L137)

---

## Slide 22: 完整搜索调用链（一图总结）

```
search(x, k, tenant)
  │
  ▼
search_one(x, k, int_tid)
  │
  ├── [Temp Index Cache Hit?] ──Yes──▶ search_temp_index → output
  │
  ├── Phase 1: beam_search()
  │     └── 每层保留 top-beam_size 节点
  │         溢出节点 → unexpanded → 推入 frontier
  │
  ├── Phase 2: Frontier Search (MinHeap)
  │     ├── node.shortlists 有 tid?
  │     │   ├── PQ路径: build_distance_table → compute_pq_distances
  │     │   │          (prefetch_and_get_codes → ADC累加)
  │     │   └── 精确路径: compute_vector_distance (flash读取)
  │     │   → batch_insert → RunningList
  │     │   → break if no update
  │     └── node.BF 有 tid?
  │         └── 展开子节点 → compute_child_scores → 入队
  │
  ├── Phase 3: ADC Rerank (optional)
  │     └── top k×factor 精确 L2 重排
  │
  └── int_vid → ext_vid 转换 → output
```

**源码**: [curator_index.cpp:369-605](Curator/src/curator_index.cpp#L369-L605)

---

## Slide 23: 核心数据结构速查

| 结构 | 源码位置 | 用途 |
|------|---------|------|
| `TreeNode` | [tree_node.h:16-93](Curator/src/tree_node.h#L16-L93) | 聚类树节点 |
| `RunningList` | [common.h:144-234](Curator/src/common.h#L144-L234) | top-K 有序候选集 |
| `SortedList<T>` | [common.h:94-138](Curator/src/common.h#L94-L138) | 有序向量ID列表 |
| `BloomFilter` | [bloom_filter.h:106-290](Curator/src/bloom_filter.h#L106-L290) | 租户集合摘要 |
| `IdAllocator` | [common.h:240-326](Curator/src/common.h#L240-L326) | 连续ID分配器 |
| `IdMapping` | [common.h:331-363](Curator/src/common.h#L331-L363) | 双向ID映射 |
| `TempIndexNode` | [temp_index.h:20-24](Curator/src/temp_index.h#L20-L24) | 临时索引节点 |
| `PQCodec` | [pq_codec.h:15-128](Curator/src/pq_codec.h#L15-L128) | PQ编解码 |
| `PQBlockCache` | [pq_block_cache.h:35-137](Curator/src/pq_block_cache.h#L35-L137) | LRU块缓存 |
| `FlashStore` | [flash_store.h:13-52](Curator/src/flash_store.h#L13-L52) | 全精度向量磁盘I/O |
| `SearchProfile` | [profiling.h:14-44](Curator/src/profiling.h#L14-L44) | 搜索性能分析 |
| `CuratorConfig` | [config.h:9-42](Curator/src/config.h#L9-L42) | 21个配置参数 |

---

## Slide 24: 关键参数说明

| 分组 | 参数 | 默认值 | 说明 |
|------|------|:---:|------|
| 树结构 | `n_clusters` | 64 | K-means 分支数(≤64) |
| | `max_leaf_size` | 128 | 叶子最大向量数 |
| | `max_sl_size` | 128 | Shortlist 最大长度 |
| 搜索 | `beam_size` | 2 | Beam Search 宽度 |
| | `search_ef` | 128 | Frontier 候选集大小 |
| | `variance_boost` | 0.4 | 方差降权系数 |
| | `nprobe` | 3000 | 无过滤搜索探测数 |
| PQ | `pq_M` | 16 | 子空间数 |
| | `pq_nbits` | 8 | 量化位数 (仅支持8) |
| | `pq_use_adc_rerank` | false | 是否精确重排 |
| Flash | `use_flash_storage` | true | 全精度向量存盘 |
| Cache | `pq_cache_block_size` | 4096 | PQ 码块大小 |
| | `pq_cache_max_blocks` | 256 | 最大缓存块数 |

**源码**: [config.h:9-42](Curator/src/config.h#L9-L42)

---

## Slide 25: 总结

Curator 的核心设计思想：

1. **层次化聚类树**：用 K-means 递归分区向量空间，自然形成多级索引
2. **Bloom Filter + Shortlist 协同**：
   - BF 快速跳过不相关分支（O(1) 判断）
   - Shortlist 批量收集候选（减少随机 I/O）
3. **PQ + 外部存储**：
   - PQ 码写入磁盘，LRU 块缓存按需加载（减少内存）
   - 全精度向量仅在 rerank 时读取
4. **三阶段搜索**：
   - Beam Search → 快速导航到有前途的区域
   - Frontier Search → 优先队列完整搜索
   - ADC Rerank → 精确距离保证精度

**零外部依赖**：独立的 C++ 可执行文件，仅需 OpenMP

**源码总行数**: ~3,500 行 C++（14 个编译单元）

---

## 附录 A: CLI 查询路径选择机制（main.cpp 详解 — v3 更新）

> **说明**: 本页详细解释 `main.cpp` 如何判断走"单租户过滤查询"还是"复杂谓词查询"。v3 新增 `--query_filters` 参数支持 per-query 不同谓词，采用两阶段架构（串行预构建 + 并行搜索）。

### A.1 判断依据

```cpp
// main.cpp:172-173
std::string query_filters_path; // per-query filter file (新增)
std::string filter_expr;        // global complex predicate, empty = simple query

// CLI 解析
else if (arg == "--filter" && i + 1 < argc) filter_expr = argv[++i];
else if (arg == "--query_filters" && i + 1 < argc) query_filters_path = argv[++i];
```

**判断逻辑**：`has_any_predicate = !filter_expr.empty() || !query_filters.empty()`

- `has_any_predicate == false` → **单租户模式**：按 `--query_labels` 每条 query 的 tenant_id 搜索
- `has_any_predicate == true` → **Phase 1 预构建** → **Phase 2 搜索**

### A.2 优先级规则

```
per-query filter (query_filters.txt 非空行)    ← 最高优先级
  └─ fallback → global --filter                ← 中等优先级
       └─ fallback → query_labels[q] 或 -1     ← 最低优先级（无过滤）
```

### A.3 单租户路径（默认路径，无 --filter / --query_filters）

```cpp
// Phase 2 搜索循环中的 fallback 分支
ext_lid_t tid = (q < query_labels.size()) ?
    static_cast<ext_lid_t>(query_labels[q]) : -1;
index.search(query_vecs.data() + q * cfg.d, k, tid,
              all_dists[q].data(), all_labels[q].data());
```

**特点**：
- 每条 query 独立拥有自己的 tenant_id（`-1` = unfiltered）
- 调用 `CuratorIndex::search()` → `search_one()` → 标准三阶段搜索（Beam + Frontier + Rerank）
- 直接利用已构建好的 Cluster Tree 的 Bloom Filter + Shortlist

### A.4 复杂谓词路径（--filter 或 --query_filters 触发）

```
CLI: --query_filters query_filters.txt  (或 --filter "1 2 AND")
         │
         ▼
┌─ Phase 1: 串行预构建 ─────────────────────────────────────────────────┐
│                                                                        │
│ ① 加载 query_filters.txt → std::vector<std::string> query_filters     │
│    空行 = 无复杂谓词（回退到 --filter 或 query_labels）                 │
│                                                                        │
│ ② for q in 0..n_queries-1:                                            │
│      pred = 确定本条查询的谓词 (per-query > global)                     │
│      if pred.empty(): per_query_filter_label[q] = -1; continue         │
│                                                                        │
│      // 去重（两级）                                                    │
│      label = label_cache[pred]       // 本地 unordered_map 缓存        │
│      label = index.get_filter_label(pred)  // CuratorIndex 全局去重    │
│                                                                        │
│      // 首次遇到: 全量扫描 + 构建                                       │
│      qualified = index.find_all_qualified_vecs(pred)  // O(N) 扫描     │
│      if (!qualified.empty()):                                          │
│        label = index.build_filter_index(pred, qualified, n)            │
│        per_query_filter_label[q] = label                               │
│      else:                                                             │
│        per_query_filter_label[q] = -2  // sentinel: 谓词无匹配         │
│                                                                        │
│ 源码: main.cpp:277-379                                                 │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Phase 2: 搜索（并行安全）─────────────────────────────────────────────┐
│                                                                        │
│ for q in 0..n_queries-1:                                               │
│   flabel = per_query_filter_label[q]                                   │
│                                                                        │
│   if flabel == -2:                                                     │
│     → 空结果 (谓词无匹配)                                               │
│   else if flabel >= 0:                                                 │
│     → index.search(query, k, flabel, ...)                              │
│       → search_one() → get_cached_temp_index_data() 命中!              │
│       → search_temp_index() 快速路径                                    │
│   else: // flabel == -1                                                │
│     → 回退: index.search(query, k, query_labels[q], ...)               │
│                                                                        │
│ Phase 2 仅调用 const 方法，对 OpenMP batch_query 并行安全               │
│ 源码: main.cpp:381-443                                                 │
└────────────────────────────────────────────────────────────────────────┘
```

### A.5 两路径对比总结

| 维度 | 单租户路径 | 复杂谓词路径（v3） |
|------|------|------|
| 触发条件 | 无 `--filter` 且无 `--query_filters` | 有至少一个 |
| 过滤粒度 | 每条 query 独立 tenant_id | per-query 谓词 > 全局谓词 > query_labels |
| 查询数据结构 | 主 Cluster Tree (BF+Shortlist) | TempIndexNode 临时索引树（缓存） |
| 构建方式 | 构建时已建好（0 额外开销） | Phase 1 串行预构建（O(N) 扫描 × M 唯一谓词） |
| 去重 | N/A | 两级：本地 label_cache + filter_to_label_ |
| 搜索入口 | `search_one()` 标准三阶段 | `search_temp_index()` 快速路径 |
| 并行安全 | ✅ 始终安全 | ✅ Phase 2 仅 const 方法 |
| 典型用例 | `--query_labels labels.npy` | `--query_filters filters.txt` |

### A.6 search_one 内部的 "Temp Index Cache" 快速路径

即使搜索调用的是 `search(x, k, tenant, ...)`，`search_one` 内部也会**优先检查**该 `tid` 是否对应一个已缓存的 TempIndex：

```cpp
// curator_index.cpp:391-421
const std::vector<TempIndexNode>* temp_nodes = nullptr;
const std::vector<int_vid_t>* qualified_vecs = nullptr;
if (get_cached_temp_index_data(tid, temp_nodes, qualified_vecs)) {
    // ★ 快速路径：直接在 TempIndex 上搜索
    search_temp_index(*temp_nodes, *qualified_vecs, x, k, d,
                      search_ef, beam_sz, batch_dist_fn, distances, labels);
    // Profiling: query_type = "temp_index"
    return;
}
// 否则走标准搜索路径（Beam + Frontier + Rerank）
```

**关键**：`build_filter_index` 时写入 `temp_indexes_[filter_tid]` 和 `qualified_cache_[filter_tid]`，后续 `search_one` 自动命中。这一层对 `main.cpp` 透明——`main.cpp` 始终调 `index.search()`，内部自动分流。

### A.7 完整流程图（v3 更新）

```
main.cpp CLI 启动
  │
  ├── --filter 或 --query_filters 存在?
  │     │
  │     ├── No ──▶ 单租户搜索
  │     │           ├── query_labels[q] == -1? → search_unfiltered()
  │     │           └── query_labels[q] >= 0?  → search() → search_one()
  │     │                                             │
  │     │                                             └── temp_index cache hit?
  │     │                                                   (通常不命中)
  │     │
  │     └── Yes ──▶ 复杂谓词搜索
  │                  │
  │                  ├── Phase 1 (串行):
  │                  │   ① 加载 query_filters.txt
  │                  │   ② 收集唯一谓词 + 两级去重
  │                  │   ③ find_all_qualified_vecs (首次)
  │                  │   ④ build_filter_index → 缓存 TempIndex
  │                  │   ⑤ per_query_filter_label[q] 映射
  │                  │     (-2=无匹配, -1=回退, >=0=filter_label)
  │                  │
  │                  └── Phase 2 (并行安全):
  │                        for each query:
  │                          flabel == -2? → 空结果
  │                          flabel >= 0?  → search() → search_one()
  │                                            └── temp_index cache ★命中!
  │                          flabel == -1? → search(query_labels[q])
```

**源码关键行号**:
- Phase 1 预构建: [main.cpp:277-379](Curator/src/main.cpp#L277-L379)
- Phase 2 搜索: [main.cpp:381-443](Curator/src/main.cpp#L381-L443)
- 快速路径: [curator_index.cpp:391-421](Curator/src/curator_index.cpp#L391-L421)
- Predicate 求值: [curator_index.cpp:891-919](Curator/src/curator_index.cpp#L891-L919)
- Filter index 构建: [curator_index.cpp:854-880](Curator/src/curator_index.cpp#L854-L880)

---
*大纲生成日期: 2026-07-09 | 最后更新: 2026-07-09 (v3)*
*基于 Curator 源码完整阅读及 --query_filters 实施验证*
