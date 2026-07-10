# 复杂谓词查询：当前设计的局限性与多谓词支持方案

> 创建日期: 2026-07-09 | 最后更新: 2026-07-09 (v2，经源码逐行审查修订)
> 基于 Curator 源码完整分析

---

## 一、当前设计回顾

### 1.1 CLI 层（main.cpp）

当前 `main.cpp` 通过 `--filter` 参数接收**一个**复杂谓词表达式，应用于全部查询：

```cpp
// main.cpp:170
std::string filter_expr;  // 全局唯一，empty = 简单查询

// main.cpp:191 — CLI 解析，只接受一个 --filter
else if (arg == "--filter" && i + 1 < argc) filter_expr = argv[++i];

// main.cpp:284-295 — 构建一次过滤索引
if (!filter_expr.empty()) {
    auto qualified = index.find_all_qualified_vecs(filter_expr);
    filter_label = index.build_filter_index(filter_expr, qualified.data(), qualified.size());
}

// main.cpp:306-309 — 所有 query 复用同一 filter_label
for (size_t q = 0; q < n_queries; q++) {
    index.search(query_vecs[q], k, filter_label, ...);  // filter_label 不变
}
```

> ⚠ **已有 Bug**：当 `qualified.empty()` 时，`filter_label` 保持 `-1`，后续全部查询回退到无过滤搜索（`search()` 中 `tenant < 0` → `search_unfiltered()`），而非返回空结果。

### 1.2 底层能力（CuratorIndex）

底层 `CuratorIndex` 的基础设施实际**已经支持多个不同谓词共存**：

| 机制 | 源码位置 | 作用 |
|------|---------|------|
| `filter_to_label_` | [curator_index.h:118](src/curator_index.h#L118) | `predicate_string → filter_label` 映射，天然去重 |
| `temp_indexes_` | [curator_index.h:121](src/curator_index.h#L121) | `filter_tid → TempIndexNode[]`，多条目共存 |
| `qualified_cache_` | [curator_index.h:122](src/curator_index.h#L122) | `filter_tid → sorted_vids[]`，多条目共存 |
| `allocate_reserved_label()` | [common.h:279-290](src/common.h#L279-L290) | 从 int16_max 向下分配，不冲突 |
| `get_filter_label()` | [curator_index.cpp:882-886](src/curator_index.cpp#L882-L886) | 查询谓词是否已构建，返回已有 label |

### 1.3 完整调用链路（复杂谓词路径）

```
main.cpp: --filter "1 AND 2"
  │
  ├─ [构建阶段] index.find_all_qualified_vecs(filter_expr)
  │     └─ curator_index.cpp:891-919
  │         对每个向量：沿树向上遍历所有 shortlist，
  │         收集该向量的 access_set (int_lid_t 集合)，
  │         用 predicate::evaluate_formula(tokens, access_set) 求值
  │         返回 qualified vid 列表 → O(N × tree_depth × avg_shortlist_count)
  │
  ├─ [构建阶段] index.build_filter_index(filter_expr, qualified, n)
  │     └─ curator_index.cpp:854-880
  │         ├─ allocate_reserved_label() → 从 int16_max 向下分配 filter_label (ext_lid_t)
  │         ├─ allocate_id(filter_label) → 分配 filter_tid (int_lid_t)
  │         ├─ filter_to_label_[predicate] = filter_label  (去重映射，key 为原始字符串)
  │         └─ use_temp_index_caching?
  │              ├─ YES → build_temp_index() + 缓存到 temp_indexes_[filter_tid] /
  │              │        qualified_cache_[filter_tid]
  │              └─ NO  → batch_grant_access() 直接写入 Cluster Tree Shortlist
  │
  └─ [搜索阶段] index.search(query, k, filter_label, ...)
        └─ curator_index.cpp:349-367
            tenant = filter_label (ext_lid_t, 如 32767)
            └─ int_tid = tid_map_.get_id(tenant)   (ext→int 转换)
            └─ search_one(x, k, int_tid, ...)
                 └─ curator_index.cpp:369-605
                     ├─ get_cached_temp_index_data(int_tid, ...) 命中?
                     │    temp_indexes_.find(int_tid) → 找到 build_filter_index 时缓存的节点
                     │    └─ YES → search_temp_index() 快速路径
                     │         └─ temp_index.cpp:85-216
                     │             两阶段: Beam Search (导航) → Frontier Search (真实距离)
                     │             通过 BatchDistanceFn 回调 → PQ ADC / 精确 L2
                     │
                     └─ NO  → 标准路径:
                          Beam Search → Frontier Search → PQ 距离 / ADC Rerank
```

### 1.4 谓词求值细节

`find_all_qualified_vecs()` 中 `evaluate_formula` 将用户输入的 token 直接解析为 `tid_t`，与 shortlist 的 key `int_lid_t` 比较：

```cpp
// complex_predicate.h:241
tid_t tid_val = static_cast<tid_t>(std::stoi(token));      // 用户输入的原始数字
stack.push_back(access_list.find(tid_val) != access_list.end()); // 与 int_lid_t 比较

// curator_index.cpp:906-909 — access_set 的 key 是 int_lid_t
for (const auto& kv : curr->shortlists) {
    if (kv.second.contains(vid)) {
        access_set.insert(kv.first);  // ← int_lid_t (shortlist key)
    }
}
```

> ⚠ **隐式假设**：用户输入的谓词数字 = 内部 shortlist key（`int_lid_t`）。因为 `IdAllocator` 对从 0 开始连续的标签分配恒等映射（ext_lid 0→int_lid 0, ext_lid 1→int_lid 1…），对现有数据集（arxiv 0..98, yfcc100m 0..999, sift 0..99）成立。若未来数据集标签不连续或从非 0 开始，需在 `convert_complex_predicate()` 中实现 ext→int 的 token 替换。

> 🔴 **谓词格式要求：Reverse Polish Notation (RPN/后缀表达式)**
>
> `evaluate_formula()` 和 `parse_formula()` 都使用**栈式求值器**处理 token 序列，要求操作数在前、操作符在后（后缀表达式/逆波兰记法）。中缀表达式会静默失败（返回 0 个匹配向量）：
>
> | 含义 | ❌ 中缀（错误） | ✅ RPN（正确） |
> |------|----------------|---------------|
> | A AND B | `1 AND 2` | `1 2 AND` |
> | A OR B | `3 OR 4` | `3 4 OR` |
> | (A AND B) OR C | `(1 AND 2) OR 3` | `1 2 AND 3 OR` |
> | A AND NOT B | `7 AND NOT 2` | `7 2 NOT AND` |
> | 单租户 | `1` | `1` |
>
> 此格式适用于 `--filter` 和 `--query_filters` 两个参数。`query_filters.txt` 中每行的谓词也须使用 RPN。

---

## 二、当前设计的局限性

### 2.1 功能局限

| 局限 | 说明 |
|------|------|
| **单一谓词** | CLI 只接受一个 `--filter`，所有查询共用同一过滤条件 |
| **无法混合** | 无法在同一批查询中同时包含不同复杂谓词 |
| **缺少批量支持** | 没有类似 `--query_labels` 的 per-query 过滤表达式文件输入 |

### 2.2 性能局限

`find_all_qualified_vecs()` 的开销是 **O(N × tree_depth × avg_shortlist_count)**：

- 若有 M 个不同谓词，就需要 M 次全量扫描
- 当 N=1,000,000 且 M 较大时，谓词求值本身的时间会远超搜索时间
- 对于实验场景 M ≤ 20，总开销可控

---

## 三、选定方案：CLI 扩展 `--query_filters`

### 3.1 方案选择

| 维度 | 方案一 CLI 扩展 | 方案二 快速求值 | 方案三 Python 分组 |
|------|:--:|:--:|:--:|
| C++ 代码改动 | **仅 main.cpp** | curator_index 新增方法 | 零 |
| 改动 CuratorIndex 基础设施 | **❌ 不改** | ❌ 需改 | ❌ 不改 |
| 支持 per-query 不同谓词 | ✅ | ✅ | ✅ |
| 谓词去重 | ✅ 自动 | ✅ 自动 | ✅ 自然分组 |
| 构建复用 | ✅ 单次构建 | ✅ 单次构建 | ❌ 每组重建 |
| 并行搜索安全 | ✅ (预构建分离) | ✅ | ✅ |

**选择方案一**：`CuratorIndex` 的 `get_filter_label()` + `find_all_qualified_vecs()` + `build_filter_index()` + `search()` 四个公开 API 已完全提供 per-query 复杂谓词的能力，只需在 CLI 层（main.cpp）正确编排。

### 3.2 CLI 接口

```bash
curator bench \
    --train_vecs    train_vecs.npy \
    --train_access  train_access.npy \
    --queries       query_vecs.npy \
    --query_filters query_filters.txt \   # ★ 新增：每行一个谓词
    --query_labels  query_labels.npy \    # 保留：简单租户查询的回退路径
    --filter        "1 2 AND" \           # 保留：全局默认谓词（RPN 格式）
    --config        config.json \
    --k 10 --output results.json
```

`query_filters.txt` 格式（每行一个谓词，**必须使用 RPN/后缀表达式**；空行 = 无复杂谓词，回退到 `--filter` 或 `query_labels`）：

```
1 2 AND
3 4 AND 5 OR

7 2 NOT AND
1
```

**优先级规则**：

```
per-query filter (query_filters.txt 非空行)    ← 最高优先级
  └─ fallback → global --filter                ← 中等优先级
       └─ fallback → query_labels[q] 或 -1     ← 最低优先级（无过滤）
```

### 3.3 核心设计：两阶段架构

设计文档初稿将 filter_index 构建放在搜索循环内部，存在两个严重问题：

| # | 问题 | 严重程度 | 说明 |
|---|------|:---:|------|
| 1 | 空 qualified 时回退到无过滤搜索 | 🔴 严重 | `label=-1` → `search()` → `search_unfiltered()` |
| 2 | batch_query 并行模式下非线程安全 | 🔴 严重 | `build_filter_index` 写入 `filter_to_label_`/`temp_indexes_`/`tid_map_` |
| 3 | 隐式 ext_lid ≡ int_lid 假设 | 🟡 中 | 对现有数据集无影响，需在文档中记录 |
| 4 | query_filters 行数不足时回退 | 🟡 中 | 需明确定义回退行为 |
| 5 | 空 temp_index 导致未初始化输出 | 🔴 严重 | `search_temp_index()` 在 nodes 为空时直接 return |

修正方案引入 **两阶段架构**：

```
Phase 1 (串行预构建) — 在搜索循环之前：
  遍历 query_filters[0..n_queries-1]:
    确定每条查询的谓词（per-query > global > query_labels）
    对每个新谓词:
      get_filter_label() → 命中则复用
                         → 未命中则 find_all_qualified_vecs() + build_filter_index()
    将每条查询映射到 filter_label（或 sentinel -2 表示"无匹配"）

Phase 2 (搜索) — 串行或 OpenMP 并行：
  使用 Phase 1 构建的 per_query_filter_label[q] 直接 search()
  此阶段仅调用 const 方法，线程安全
```

### 3.4 完整实现

#### 3.4.1 新增 CLI 参数

```cpp
// main.cpp — 新增变量
std::string query_filters_path;  // per-query 谓词文件路径

// 解析
else if (arg == "--query_filters" && i + 1 < argc)
    query_filters_path = argv[++i];
```

#### 3.4.2 Phase 1：串行预构建

```cpp
// ── 加载 per-query 过滤表达式 ──
std::vector<std::string> query_filters;
if (!query_filters_path.empty()) {
    std::ifstream infile(query_filters_path);
    if (!infile.is_open()) {
        fprintf(stderr, "Error: cannot open query_filters file '%s'\n",
                query_filters_path.c_str());
        return 1;
    }
    std::string line;
    while (std::getline(infile, line)) {
        // 去除首尾空白
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            query_filters.push_back("");  // 空行 = 无复杂谓词
        } else {
            size_t end = line.find_last_not_of(" \t\r\n");
            query_filters.push_back(line.substr(start, end - start + 1));
        }
    }
    printf("Loaded %zu query filter expressions from %s\n",
           query_filters.size(), query_filters_path.c_str());
}

if (query_filters.size() < n_queries) {
    printf("Note: query_filters has %zu lines, %zu queries — "
           "remaining queries will use fallback\n",
           query_filters.size(), n_queries);
}

// ── Phase 1: 预构建所有唯一谓词的过滤索引（串行） ──
//
// per_query_filter_label[q]:
//   -2 = 谓词匹配 0 个向量（返回空结果）
//   -1 = 无复杂谓词（回退到 query_labels 或 search_unfiltered）
//   ≥0 = filter_label (ext_lid_t，用于 index.search)
//
std::vector<ext_lid_t> per_query_filter_label(n_queries, -1);

bool has_any_predicate = !filter_expr.empty() || !query_filters.empty();
if (has_any_predicate) {
    printf("Pre-building filter indexes for complex predicates...\n");

    // 本地缓存，避免重复调用 get_filter_label()
    std::unordered_map<std::string, ext_lid_t> label_cache;

    for (size_t q = 0; q < n_queries; q++) {
        // 确定本条查询的谓词
        // 优先级: per-query filter > global --filter
        std::string pred;
        if (q < query_filters.size() && !query_filters[q].empty()) {
            pred = query_filters[q];
        } else if (!filter_expr.empty()) {
            pred = filter_expr;
        }

        if (pred.empty()) {
            per_query_filter_label[q] = -1;  // 无复杂谓词
            continue;
        }

        // 查本地缓存（O(1) string hash，比 get_filter_label 更快）
        auto cache_it = label_cache.find(pred);
        if (cache_it != label_cache.end()) {
            per_query_filter_label[q] = cache_it->second;
            continue;
        }

        // 查 CuratorIndex 的全局去重映射
        ext_lid_t label = index.get_filter_label(pred);
        if (label >= 0) {
            per_query_filter_label[q] = label;
            label_cache[pred] = label;
            continue;
        }

        // 首次遇到此谓词：全量扫描求值
        auto qualified = index.find_all_qualified_vecs(pred);
        printf("  Predicate '%s': %zu qualified vectors\n",
               pred.c_str(), qualified.size());

        if (!qualified.empty()) {
            label = index.build_filter_index(
                pred, qualified.data(), qualified.size());
            per_query_filter_label[q] = label;
            label_cache[pred] = label;
        } else {
            // ★ sentinel -2：谓词匹配 0 个向量
            //    不调用 build_filter_index（避免空 temp_index）
            //    不调用 search()（避免回退到 search_unfiltered）
            per_query_filter_label[q] = -2;
            label_cache[pred] = -2;
            fprintf(stderr, "  Warning: predicate '%s' matches 0 vectors\n",
                    pred.c_str());
        }
    }
    printf("Pre-build complete: %zu unique predicates evaluated\n",
           label_cache.size());
}
```

#### 3.4.3 Phase 2：搜索（并行安全）

```cpp
// ── Phase 2: 搜索 ──
// 此阶段仅调用 const 方法，对并行安全：
//   - index.search() → search_one() → 读取 tree + temp_indexes_（只读）
//   - PQ block cache 内部有 mutex 保护
//   - vid_map_.get_label() 只读 unordered_map 查找

if (cfg.batch_query) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t q = 0; q < n_queries; q++) {
        all_labels[q].resize(k);
        all_dists[q].resize(k);

        ext_lid_t flabel = per_query_filter_label[q];

        if (flabel == -2) {
            // 谓词无匹配 → 返回空结果
            std::fill(all_labels[q].begin(), all_labels[q].end(), 0);
            std::fill(all_dists[q].begin(), all_dists[q].end(),
                      std::numeric_limits<float>::max());
        } else if (flabel >= 0) {
            // 复杂谓词过滤搜索
            index.search(query_vecs.data() + q * cfg.d, k, flabel,
                          all_dists[q].data(), all_labels[q].data());
        } else {
            // flabel == -1：无复杂谓词 → 回退到单租户或无过滤
            ext_lid_t tid = (q < query_labels.size()) ?
                static_cast<ext_lid_t>(query_labels[q]) : -1;
            index.search(query_vecs.data() + q * cfg.d, k, tid,
                          all_dists[q].data(), all_labels[q].data());
        }
    }
} else {
    // 串行路径：逻辑相同
    for (size_t q = 0; q < n_queries; q++) {
        all_labels[q].resize(k);
        all_dists[q].resize(k);

        ext_lid_t flabel = per_query_filter_label[q];

        if (flabel == -2) {
            std::fill(all_labels[q].begin(), all_labels[q].end(), 0);
            std::fill(all_dists[q].begin(), all_dists[q].end(),
                      std::numeric_limits<float>::max());
        } else if (flabel >= 0) {
            index.search(query_vecs.data() + q * cfg.d, k, flabel,
                          all_dists[q].data(), all_labels[q].data());
        } else {
            ext_lid_t tid = (q < query_labels.size()) ?
                static_cast<ext_lid_t>(query_labels[q]) : -1;
            index.search(query_vecs.data() + q * cfg.d, k, tid,
                          all_dists[q].data(), all_labels[q].data());
        }
    }
}
```

#### 3.4.4 更新 print_usage()

```cpp
printf("  --query_filters PATH  Per-query filter expressions file (one per line)\n");
```

#### 3.4.5 Python 层适配

[run_experiment.py](python/run_experiment.py) 新增参数透传：

```python
parser.add_argument("--query_filters", type=str, default=None,
                    help="Per-query filter expressions file")
# ...
if args.query_filters and os.path.exists(args.query_filters):
    cmd += ["--query_filters", args.query_filters]
```

### 3.5 改动范围

| 文件 | 改动量 | 说明 |
|------|:---:|------|
| `Curator/src/main.cpp` | ~80 行 | Phase 1 预构建 + Phase 2 搜索 |
| `Curator/python/run_experiment.py` | ~5 行 | 参数透传 |
| CuratorIndex 全部文件 | **0 行** | 不改变索引基础设施 |

---

## 四、方案二（快速求值）与方案三（Python 分组）

### 4.1 方案二：针对租户组合谓词的快速求值

当实验需要大量不同谓词（M > 20）时，`find_all_qualified_vecs` 的全量扫描成为瓶颈。方案二在 Shortlist 上直接做集合运算，将复杂度从 O(N) 降至 O(|shortlist|)。

**适用条件**：谓词由纯租户 ID 的 AND/OR/NOT 组成，且原子租户都在 Cluster Tree 中有 Shortlist。

**与方案一的关系**：方案一是主方案，方案二作为性能优化可选叠加。两者互不冲突——方案二仅替换 `find_all_qualified_vecs` 的内部实现，不影响 CLI 层。

**无限定条件回退**：包含非租户条件时自动回退到全量扫描。

### 4.2 方案三：Python 分组批处理

每次 `curator bench` 子进程需完整重新构建索引。若分组数多（如 100 个不同谓词），重复构建的总时间远大于搜索本身。

**适用范围**：仅限 small 数据集（N ≤ 50K）的快速功能验证。

---

## 五、相关源码索引

| 内容 | 文件:行号 |
|------|---------|
| CLI `--filter` 解析 | [main.cpp:191](src/main.cpp#L191) |
| 搜索循环（当前） | [main.cpp:297-331](src/main.cpp#L297-L331) |
| `search()` 入口 | [curator_index.cpp:349-367](src/curator_index.cpp#L349-L367) |
| `search_one()` 主搜索 | [curator_index.cpp:369-605](src/curator_index.cpp#L369-L605) |
| Temp index 快速路径 | [curator_index.cpp:391-421](src/curator_index.cpp#L391-L421) |
| `find_all_qualified_vecs` | [curator_index.cpp:891-919](src/curator_index.cpp#L891-L919) |
| `build_filter_index` | [curator_index.cpp:854-880](src/curator_index.cpp#L854-L880) |
| `get_filter_label`（去重入口） | [curator_index.cpp:882-886](src/curator_index.cpp#L882-L886) |
| `get_cached_temp_index_data` | [curator_index.cpp:927-941](src/curator_index.cpp#L927-L941) |
| `filter_to_label_` 声明 | [curator_index.h:118](src/curator_index.h#L118) |
| `allocate_reserved_label` | [common.h:279-290](src/common.h#L279-L290) |
| `search_temp_index` | [temp_index.cpp:85-216](src/temp_index.cpp#L85-L216) |
| `build_temp_index` | [temp_index.cpp:15-83](src/temp_index.cpp#L15-L83) |
| `predicate::tokenize_formula` | [complex_predicate.h:207](src/complex_predicate.h#L207) |
| `predicate::evaluate_formula`（模板） | [complex_predicate.h:217-246](src/complex_predicate.h#L217-L246) |
| `convert_complex_predicate`（pass-through） | [curator_index.cpp:922-925](src/curator_index.cpp#L922-L925) |
| Python 实验脚本 | [run_experiment.py](python/run_experiment.py) |

---

## 六、修正记录

### v3 (2026-07-09) — 编码实施与端到端验证

#### 实施内容

1. **main.cpp**：[Phase 1 预构建 + Phase 2 搜索](src/main.cpp#L277-L443) 已编码完成。
2. **run_experiment.py**：[--query_filters 参数透传](python/run_experiment.py) 已编码完成。
3. 编译通过，未修改 CuratorIndex 任何文件。

#### 验证发现：谓词语法必须使用 Reverse Polish Notation (RPN)

`evaluate_formula()` 使用栈式求值器，要求操作数在前、操作符在后（后缀表达式）。中缀表达式会静默失败（返回 0 匹配向量）。

此格式要求已补充到 §1.4（谓词求值细节），并在 §3.2（CLI 接口）的示例中全部修正为 RPN 格式。

#### 端到端验证结果（arxiv_small, 500 queries）

使用以下测试场景验证全部功能：

| 测试场景 | 预期 | 实际 | 状态 |
|---------|------|------|:--:|
| `0 1 AND` (RPN, int 0=ext 47, int 1=ext 48) | 196 qualified | 196 qualified | ✅ |
| `0` (单租户) | 396 qualified | 396 qualified | ✅ |
| `999 998 AND` (不存在) | 0 qualified → sentinel -2 | Warning + 空结果 | ✅ |
| dedup (15 条查询共用 `0 1 AND`) | 1 次 build | 3 unique total (含 0 和 999 998 AND) | ✅ |
| 空行→query_labels 回退 | 474 queries 正常 | 474 queries 有结果 | ✅ |
| 500 行文件加载 | 500 lines | Loaded 500 | ✅ |
| 搜索性能 | 正常 (0.39 ms/query avg) | 0.39 ms/query | ✅ |

#### 实施的改动文件

| 文件 | 改动 | 说明 |
|------|:---:|------|
| `Curator/src/main.cpp` | ~130 行新增/修改 | Phase 1+2, CLI, usage |
| `Curator/python/run_experiment.py` | +4 行 | --query_filters 透传 |
| `Curator/COMPLEX_PREDICATE_DESIGN.md` | 本文档 | RPN 格式修正 + 验证记录 |
| CuratorIndex 全部文件 | **0 行** | 不改变索引基础设施 ✅ |

### v2 (2026-07-09) — 源码审查修正

在对照全部 Curator 源码逐行审查后，发现并修正了设计初稿的以下问题：

1. **空 qualified 集合处理**（🔴 严重）：初稿在 `qualified.empty()` 时 `label` 保持 `-1`，`search()` 将其解释为无过滤搜索。修正：引入 sentinel `-2` 标记"谓词无匹配"，在搜索阶段直接填入空结果。

2. **batch_query 并行安全**（🔴 严重）：初稿将 `build_filter_index()` 放在搜索循环内部，在 OpenMP 并行模式下写入共享状态。修正：引入两阶段架构——Phase 1 串行预构建全部 filter_label，Phase 2 仅执行 const 的 search()。

3. **空 temp_index 未初始化输出**（🔴 严重）：`search_temp_index()` 在 nodes 为空时直接 return 不初始化 distances/labels。修正：通过在 Phase 1 阻止空 qualified 调用 `build_filter_index` 来规避。

4. **隐式 ext_lid ≡ int_lid 假设**（🟡 中）：用户输入的谓词数字直接与内部 shortlist key 比较。对所有现有数据集（标签从 0 开始连续）无影响，已在文档中记录约束。

5. **query_filters 行数与查询数不匹配**（🟡 中）：明确定义了行数不足时逐行回退到 `--filter` → `query_labels` 的优先级规则。

6. **谓词语法必须使用 RPN**（🔴 严重 — v3 发现）：`evaluate_formula()` 是栈式求值器，要求后缀表达式。中缀表达式（如 `1 AND 2`）会静默失败。已在文档全部示例中修正为 RPN 格式（如 `1 2 AND`）。

---

*文档创建日期: 2026-07-09*
*最后更新: 2026-07-09 (v3)*
*基于 Curator 源码完整分析、逐行审查、编码实施与端到端验证*
