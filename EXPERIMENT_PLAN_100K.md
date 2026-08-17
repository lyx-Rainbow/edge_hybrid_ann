# 100K 数据集对比实验执行计划（v6 — 代码审查修正版）

> 创建: 2026-08-02 | 更新: 2026-08-02 (v6)
> v5 → v6: 全面代码审查，修正 20+ 处问题（详见 §〇）

---

## 〇、v5→v6 代码审查发现的问题及修正

### 关键问题（会导致实验失败或严重低效）

| # | 问题 | 严重性 | 发现来源 | 修正 |
|---|------|:------:|---------|------|
| **C1** | Curator 同样需要 `search` 子命令：`pq_M` 和 `search_ef` 各 4 值 → 16 combo，若不支持 search-only，同一 `pq_M` 下仅 `search_ef` 不同的 4 个 combo 都会**重复构建**（冗余 12 次/数据集） | **高** | [Curator/src/main.cpp:188-536](Curator/src/main.cpp#L188-L536) — `bench` 命令总是先 build 再 search | Curator 也需添加 `search` 子命令（从 flash storage 加载已有索引），或改用 Python API 方案（参考 [legacy/run_curator_sweep.py](Curator/legacy/run_curator_sweep.py) 的 `build_index` once + `run_benchmark` N 次模式） |
| **C2** | DiskIVF 无 `load()` 方法，`search` 子命令实现需要内部重构（当前 `DiskIVFIndex::build()` 耦合了聚类+写盘，无独立加载路径） | **高** | [DiskIVF-PostFiltering/src/main.cpp:262-263](DiskIVF-PostFiltering/src/main.cpp#L262-L263) — `index.build()` 是构建的唯一入口 | 需在 `DiskIVFIndex` 中添加 `load(disk_dir)` 方法；若重构成本过高，备选方案为 CP 实验接受重复构建（DiskIVF 构建快，~5-10s × 7 nprobe × 5 filters ≈ 额外 3-6 min/数据集） |
| **C3** | SPANN 已有 `load()` 方法 | — | [SPANN-PostFiltering/src/spann_pf_index.h:64](SPANN-PostFiltering/src/spann_pf_index.h#L64) — `void load(index_dir, metadata_path)` | SPANN `search` 子命令实现最简单，直接调用 `load()` + `search_with_predicate()` |
| **C4** | Pre-Filtering 无 `run_experiment.py`，只有 `compute_recall.py` / `compute_cp_recall.py` | **高** | [Pre-Filtering/python/](Pre-Filtering/python/) — 仅有 recall 计算脚本 | 需从零创建完整编排脚本（SL + CP + sweep） |

### 中等问题（影响效率或正确性）

| # | 问题 | 严重性 | 发现来源 | 修正 |
|---|------|:------:|---------|------|
| **M1** | `3_Config/SPANN-PostFiltering/sweep.json` 含 `"nprobe"` 参数（DiskIVF 概念），对 SPANN 无意义 | 中 | [3_Config/SPANN-PostFiltering/sweep.json](3_Config/SPANN-PostFiltering/sweep.json) — 内容为 `{"nprobe": [1,2,4,...]}` | SPANN sweep 应使用 `max_check` × `overfetch_factor`；需重写 |
| **M2** | `3_Config/SPANN-PostFiltering/spann_yfcc100m.json` 含 `nlist`/`nprobe` 字段 — copy-paste 错误 | 中 | [3_Config/SPANN-PostFiltering/spann_yfcc100m.json](3_Config/SPANN-PostFiltering/spann_yfcc100m.json#L5-L6) — `"nlist": 256, "nprobe": 64` | SPANN 配置应使用 `max_check`, `hash_exp`, `overfetch_factor` 等 |
| **M3** | sift1m_100k 的 `filters.json` 中 `"AND 2 3"` 重复 3 次（行 13-15） | 中 | [filters.json:13-15](1_Data/ground_truth/sift1m_100k/complex_predicate/filters.json#L13-L15) — 连续 3 行相同表达式 | Curator C++ 自动去重不受影响；DiskIVF/SPANN/Pre-Filtering 脚本需在 filter 循环前 `list(set(filters))` 去重 |
| **M4** | 所有 `*_100k` 数据集的 sweep config JSON 不存在 | 中 | `3_Config/*/sweep_100k.json` 均不存在 | Phase 1 需创建（见 §三 构建参数一览和 §九 sweep config 模板） |
| **M5** | 所有 Python `run_experiment.py` 的 `BUILTIN_DEFAULTS` 均不含 `*_100k` 数据集 | 中 | [DiskIVF run_experiment.py:31-62](DiskIVF-PostFiltering/python/run_experiment.py#L31-L62) / [SPANN run_experiment.py:28-93](SPANN-PostFiltering/python/run_experiment.py#L28-L93) / [Curator run_experiment.py](Curator/python/run_experiment.py) — 无 100k 条目 | Phase 1 扩展脚本时一并添加 |
| **M6** | `pq_M` 是构建参数，每个值需独立构建索引；16 combo = 每个数据集 16 次 C++ bench 调用 | 中 | [Curator/main.cpp:275](Curator/src/main.cpp#L275) — `index.train()` 在每次 bench 调用中执行 | 添加 `search` 子命令后，改为 4 次 build（每 pq_M）+ 4×4 search（每 search_ef），构建从 16 次降至 4 次 |
| **M7** | gist1m_100k 仅 500 条 SL 查询（其他为 1000） | 中 | [DATASETS_100K.md:28](1_Data/ground_truth/DATASETS_100K.md#L28) — "gist1m_100k: 500" | 时间估算已考虑；SL 结果报告时注明查询数差异 |
| **M8** | Curator help text 示例 `"1 AND 2"` 不符合实际解析器（使用 Polish/前缀表示法 `"AND 1 2"`） | 低 | [Curator complex_predicate.cpp:228-275](Curator/src/complex_predicate.cpp#L228-L275) — `parse_formula` 为 RPN 栈求值 | 不影响实验（`filters.json` 使用前缀表示法，与所有方法兼容）；已记录为 Curator 文档 bug |

### 低优先级问题

| # | 问题 | 严重性 | 修正 |
|---|------|:------:|------|
| **L1** | `qps_recall.py:105` 将 QPS × 1.6 硬编码（特定硬件修正因子） | 低 | 新绘图脚本不应沿袭此硬编码；QPS 从 sweep JSON 直接读取 |
| **L2** | `bar1.py` 使用 `plt.style.use("ggplot")` 与计划描述的"白底彩色边框"风格冲突 | 低 | 新 bar 图脚本统一使用白底 + `edgecolor` + `hatch` 方案（bar1.py 的柱体风格是对的，但 `ggplot` style 改了背景） |
| **L3** | Pre-Filtering sweep.json 为空对象 `{}` | 低 | 符合预期（1 combo，无参数） |

---

## 一、各方法 C++ 架构与参数

### 1.1 Curator

- **二进制**: `Curator/build/curator bench`
- **CP 机制**: `--query_filters` 文件（每行一个 Polish 前缀表示法谓词，如 `"AND 0 24"`）。Phase 1 对每个唯一谓词 `find_all_qualified_vecs()` → `build_filter_index()` → 缓存 label；Phase 2 用 label 并行搜索。
- **关键发现**: 所有方法（Curator / DiskIVF / SPANN / Pre-Filtering）使用**统一的 Polish 前缀表示法**（`"AND 0 1"`, `"OR 0 1"`, `"NOT 0"`, `"AND 0 NOT 1"`），`filters.json` 文件可直接用于所有方法。
  - Curator 解析器: [complex_predicate.cpp:228-275](Curator/src/complex_predicate.cpp#L228-L275) — RPN 栈求值，前缀格式
  - DiskIVF/SPANN/Pre-Filtering 解析器: [predicate.h:39-68](DiskIVF-PostFiltering/src/predicate.h#L39-L68) — 右到左栈求值，前缀格式
- **可用搜索参数**: `search_ef`（frontier 队列大小）、`beam_size`、`variance_boost`、`nprobe`（仅 unfiltered）、`prune_thres`（仅 unfiltered）
- **可用构建参数**: `n_clusters`、`pq_M`、`max_sl_size`、`pq_nbits` 等
- **内存统计**: 精确组件级分解（树节点、Bloom Filter、短列表、PQ 缓存、Flash 索引等）

### 1.2 DiskIVF

- **二进制**: `DiskIVF-PostFiltering/build/diskivf bench`
- **CP 机制**: `--filter EXPR`。搜索时在 `process_cluster_predicate()` 中逐候选评估。filter 与索引结构无关（纯后过滤），**天然适合"构建一次 + 多 filter 搜索"**。
- ⚠️ **无 `load()` 方法**：需添加以支持 `search` 子命令。
- **可用搜索参数**: `nprobe`（探测聚类数）
- **可用构建参数**: `nlist`、`clus_niter`

### 1.3 SPANN

- **二进制**: `SPANN-PostFiltering/build/spann_pf bench`
- **CP 机制**: 同 DiskIVF — `--filter EXPR`，在 `search_with_predicate()` 中逐候选评估后过滤。
- ✅ **已有 `load()` 方法** ([spann_pf_index.h:64](SPANN-PostFiltering/src/spann_pf_index.h#L64))：`search` 子命令可直接使用。
- **可用搜索参数**: `max_check`（SPTAG head index 搜索深度）、`overfetch_factor`（后过滤候选放大系数，k' = k × factor）、`overfetch_adaptive`
- **可用构建参数**: `hash_exp`（BKTree 哈希表指数）

### 1.4 Pre-Filtering

- **二进制**: `Pre-Filtering/build/prefiltering bench`
- **CP 机制**: `--filter EXPR`。暴力全量扫描 + 标签过滤。无索引结构，"构建"仅是将向量加载到内存。
- ⚠️ **无 `run_experiment.py`**，需从零创建。
- **配置**: `d`（自动检测）、`k`、`n_labels`（自动检测）、`num_warmup`

---

## 二、Sweep 参数设计

### 2.1 Curator: `search_ef` × `pq_M`

> ⚠️ **v6 重要修正**: `pq_M` 是构建参数，需独立构建。添加 `search` 子命令后，**4 次 build（每 pq_M 一次）+ 16 次 search（每 combo 一次）**，而非 16 次 build+search。

| 数据集 | d | pq_M 取值 | d/pq_M | search_ef 取值 | build 次数 | search 次数 |
|--------|:--:|-----------|:------:|---------------|:--------:|:---------:|
| sift1m_100k | 128 | [16, 32, 64, 128] | 8,4,2,1 | [64, 256, 512, 1024] | 4 | 16 |
| yfcc100m_100k | 192 | [24, 48, 64, 96] | 8,4,3,2 | [64, 256, 512, 1024] | 4 | 16 |
| arxiv_100k | 384 | [48, 96, 128, 192] | 8,4,3,2 | [64, 256, 512, 1024] | 4 | 16 |
| gist1m_100k | 960 | [96, 160, 240, 320] | 10,6,4,3 | [64, 256, 512, 1024] | 4 | 16 |
| wit_100k | 384 | [48, 96, 128, 192] | 8,4,3,2 | [64, 256, 512, 1024] | 4 | 16 |

> 固定: `variance_boost=0.2`, `beam_size=4`, `pq_use_adc_rerank=true`, `use_flash_storage=true`, `n_clusters=32`, `max_sl_size=256`（gist1m: 128）

### 2.2 SPANN: `max_check` × `overfetch_factor`

> ⚠️ **v6 修正**: `max_check` 和 `overfetch_factor` 均为**搜索时参数**（SPTAG 搜索参数），不需要重建索引。1 次 build + 16 次 `search`。

| 参数 | 取值 | 说明 |
|------|------|------|
| `max_check` | [2048, 4096, 8192, 16384] | SPTAG head BKT 搜索深度 |
| `overfetch_factor` | [10, 25, 50, 100] | 后过滤前候选总数 k' = k × factor |

> 4×4 = **16 组合**。固定: `overfetch_adaptive=true`, `hash_exp=8`, `num_threads=1`。

### 2.3 DiskIVF: `nprobe`

各数据集 `nlist` 固定为 32（100k 向量 ÷ 32 ≈ 3,125 向量/分区）。

| 参数 | 取值 |
|------|------|
| `nprobe` | [1, 2, 4, 8, 16, 24, 32] |

> **7 组合**。最大值 32 = nlist → 全分区探测。去除了与原范围 [64, 128] 的重复值（min(nprobe, nlist)=32 使高位值全部等价于 32）。

### 2.4 Pre-Filtering

1 组合（无参数），等价于暴力扫描基线。

---

## 三、构建参数一览

| 数据集 | Curator n_clusters | Curator max_sl_size | DiskIVF nlist | SPANN hash_exp |
|--------|:-----------------:|:------------------:|:------------:|:-------------:|
| sift1m_100k | 32 | 256 | 32 | 8 |
| yfcc100m_100k | 32 | 256 | 32 | 8 |
| arxiv_100k | 32 | 256 | 32 | 8 |
| gist1m_100k | 32 | 128 | 32 | 8 |
| wit_100k | 32 | 256 | 32 | 8 |

---

## 四、执行步骤

### Phase 1: 基础设施

#### Step 1.1: 生成 train_access.npy

所有 100k 数据集目前缺少 `train_access.npy`（只有 `train_mds.pkl`）。

```bash
python -c "
import pickle, numpy as np
for ds in ['arxiv_100k','yfcc100m_100k','sift1m_100k','gist1m_100k','wit_100k']:
    d = f'1_Data/ground_truth/{ds}'
    mds = pickle.load(open(f'{d}/train_mds.pkl','rb'))
    pairs = np.array([[v,t] for v,ts in enumerate(mds) for t in ts], dtype=np.int32)
    np.save(f'{d}/train_access.npy', pairs)
    print(f'{ds}: {len(pairs)} access pairs')
"
```

#### Step 1.2: 添加 C++ `search` 子命令（DiskIVF / SPANN / Pre-Filtering） + Curator Python API sweep

> ⚠️ **v6 终局裁决**: Curator 采用 Python API 方案（§十.10.1），**不添加 C++ `search` 子命令**。其余三个方法添加 C++ `search` 子命令。

**目标**: 消除 CP 实验中的重复索引构建，以及参数 sweep 中的冗余构建。

| 方法 | 实现方案 | 难度 | 预估代码量 | 裁决依据 |
|------|---------|:--:|:-------:|---------|
| **Curator** | **Python API sweep**（`Curator` 类 + `search_params` setter）。`run_experiment.py --sweep` 模式下：每 `pq_M` 构建一次 → 循环修改 `search_ef` → `query()` 收集指标。参考 [legacy/run_curator_sweep.py](Curator/legacy/run_curator_sweep.py)。 | 中 | Python 120-180 行 | C++ 无 save/load（§十.10.1）；Python `search_params` setter 已验证可用 |
| **DiskIVF** | `search` 子命令：build 时新增写入 `metadata.bin`（d, ntotal, nlist）；新增 `DiskIVFIndex::load_metadata(disk_dir)` 方法加载 centroids + cluster_sizes + metadata；CLI `diskivf search` | 中 | C++ ~80 行 | Build 已保存 centroids.bin + cluster_sizes.bin + per-cluster 文件（§十.10.2） |
| **SPANN** | `search` 子命令：已有 `SPANNPostFilterIndex::load()` → CLI `spann_pf search` | 低 | C++ ~50 行 | `load()` 已实现（§十.10.3） |
| **Pre-Filtering** | `search` 子命令：加载向量和标签元数据 → 直接暴力搜索。CLI `prefiltering search` | 低 | C++ ~40 行 | 无索引结构，重载即搜索（§十.10.4） |

**Curator Python API sweep 详细流程**（`run_experiment.py --sweep` 模式）:
```
for pq_m in sweep_config["build_params"]["pq_M"][dataset]:
    index = Curator(d=d, nlist=32, pq_M=pq_m, ...)
    index.train(train_vecs)
    for i in range(n_train):
        index.create(train_vecs[i], label=i)
    for vid, tids in enumerate(train_mds):
        for tid in tids:
            index.grant_access(vid, tid)
    index.flush()
    
    for search_ef in sweep_config["search_params"]["search_ef"]:
        index.search_params = {"search_ef": search_ef}
        run_benchmark(index, query_vecs, query_labels, ...)  # 收集 latency/QPS/recall
        
        if cp_enabled:
            run_cp_benchmark(index, cp_filters, cp_query_vecs, ...)
    
    del index  # 释放内存，为下一个 pq_M 做准备
```

#### Step 1.3: 扩展/创建编排脚本

| 脚本 | 修改内容 |
|------|---------|
| `Curator/python/run_experiment.py` | +`--sweep` 参数（**Python API 模式**：每 `pq_M` 构建一次 + `search_params` setter 循环 `search_ef`）；+BUILTIN_DEFAULTS（5 个 `*_100k` 数据集）；+CP 支持（`index_filter()` + `query_with_complex_predicate()`） |
| `DiskIVF-PostFiltering/python/run_experiment.py` | +`--sweep` 参数；+BUILTIN_DEFAULTS（5 个 `*_100k` 数据集）；CP 改用 `search` 子命令（循环 filter，不重建）；CP filter 列表去重 |
| `SPANN-PostFiltering/python/run_experiment.py` | +`--sweep` 参数；+BUILTIN_DEFAULTS（5 个 `*_100k` 数据集）；CP 改用 `search` 子命令；CP filter 列表去重 |
| `Pre-Filtering/python/run_experiment.py` | **新建**（参照 DiskIVF 模板）；SL + CP；BUILTIN_DEFAULTS；`--sweep` 参数 |

**CP filter 去重说明**: sift1m_100k 的 `filters.json` 中 `"AND 2 3"` 原本重复 3 次，**已在数据层面修复**（`n_filters`: 50 → 48）。其他四个 100k 数据集经检查无重复。为防御性编程，DiskIVF/SPANN/Pre-Filtering 的 Python 编排脚本仍应在 filter 循环前执行 `list(dict.fromkeys(filters))` 去重。

#### Step 1.4: Curator CP 输入文件准备

Curator 的 `--query_filters` 接受每行一个谓词表达式的文件（Polish 前缀表示法，与 `filters.json` 格式一致）。对于 CP 实验（50 filters × 100 queries）：

1. **查询向量**: 将 `complex_predicate/query_vecs.npy`（100 个向量）用 `np.tile` 重复 `n_filters` 次 → `n_filters × 100` 个向量写入临时 `.npy`
2. **Query filters 文件**: 对第 i 个 filter，写入 100 行相同表达式。`n_filters` 个 filter → `n_filters × 100` 行
3. **C++ 调用**: `--queries tiled_queries.npy --query_filters cp_filters.txt`
4. **结果解析**: 输出 `n_filters × 100` 条结果，按 100 条一组切分到各 filter 计算 recall

> C++ Phase 1 自动去重唯一谓词（Curator 内部 `label_cache` + `get_filter_label()` dedup），Phase 2 并行搜索。

#### Step 1.5: CP 代表性 Filter 选取规则

对 DiskIVF/SPANN/Pre-Filtering CP 实验（即使有 search-only 模式也需控制时间），选取规则：

1. 从 `filters.json` 读取所有 filter 及选择率（`selectivities` 字段）
2. **去重** filter 列表（`list(dict.fromkeys(filters))`）
3. 按类型分组（AND / AND_NOT / OR / NOT）
4. 每类选取 1-2 个 filter，覆盖该类内的低/中/高选择率
5. 总数控制在 5-6 个

> 选取在 sweep 编排脚本中自动化完成（读取 filters.json → 按规则挑选 → 写入列表）。

#### Step 1.6: 创建 sweep config JSON

需创建以下配置文件（模板见 §九）：

| 文件 | 内容 |
|------|------|
| `3_Config/Curator/sweep_100k.json` | `search_ef` × `pq_M` 组合（per-dataset 动态展开） |
| `3_Config/DiskIVF-PostFiltering/sweep_100k.json` | `nprobe` [1,2,4,8,16,24,32] |
| `3_Config/SPANN-PostFiltering/sweep_100k.json` | `max_check` × `overfetch_factor` 组合 |
| `3_Config/Pre-Filtering/sweep_100k.json` | 空对象 `{}`（无参数） |

---

### Phase 2: 轻量化验证（sift1m_100k）

1. 确认 4 个 C++ 二进制可执行（含新增 `search` 子命令）
2. 各方法单点 SL 测试（验证 C++ `bench` / `search` → 计算 recall）
3. 各方法单点 CP 测试（验证 `--filter` / `--query_filters` → recall）
4. Sweep 脚本端到端测试（2 组合：如 Curator search_ef=[64,1024], pq_M=[32,64]）
5. 输出 JSON schema 校验（§五）

---

### Phase 3: 正式实验

**执行顺序**（按速度，快→慢）: sift1m → yfcc100m → arxiv → wit → gist1m

#### 单标签 (SL) 实验

```bash
# Curator（Python API sweep 模式：每 pq_M 构建一次 + search_ef 循环）
python Curator/python/run_experiment.py --dataset sift1m_100k --sweep 3_Config/Curator/sweep_100k.json

# DiskIVF（C++ CLI：1 次 build + 7 次 search）
python DiskIVF-PostFiltering/python/run_experiment.py --dataset sift1m_100k --sweep 3_Config/DiskIVF-PostFiltering/sweep_100k.json

# SPANN（C++ CLI：1 次 build + 16 次 search）
python SPANN-PostFiltering/python/run_experiment.py --dataset sift1m_100k --sweep 3_Config/SPANN-PostFiltering/sweep_100k.json

# Pre-Filtering（C++ CLI：1 次）
python Pre-Filtering/python/run_experiment.py --dataset sift1m_100k --sweep 3_Config/Pre-Filtering/sweep_100k.json
```

#### 复杂谓词 (CP) 实验

在 SL sweep 的每个 combo 内部：
- **Curator**: Python API 模式 — SL sweep 的每次 `search_ef` 迭代内，先 `index_filter(predicate, qualified_labels=...)` 对所有 CP filter 构建 filter index，再逐 filter 调用 `query_with_complex_predicate()`（不额外重建索引）
- **DiskIVF/SPANN/Pre-Filtering**: 使用 C++ `search` 子命令，循环 5-6 个代表性 filter（去重后），每个 filter 一次 C++ 调用，无重复构建

---

### Phase 4: 可视化（`5_Plot/`）

三组图表，均参考现有模板：

| 图表脚本 | 对应模板 | 风格要点 |
|---------|---------|---------|
| `fig_sl_qps_recall_100k.py` | [qps_recall.py](5_Plot/qps_recall.py) | 折线图/散点图 + log y 轴 + Pareto frontier + 学术配色 |
| `fig_cp_qps_recall_100k.py` | [qps_recall.py](5_Plot/qps_recall.py) | CP 散点图，按 selectivity bucket 分面 |
| `fig_memory_100k.py` | [bar1.py](5_Plot/bar1.py) | 分组柱状图 + 白底彩色边框 + 斜纹填充 + 单独图例 |

**v6 绘图注意事项**:
- QPS 直接读取 sweep JSON 中的 `qps` 字段，**不要**沿袭 `qps_recall.py:105` 的 `×1.6` 硬编码
- 柱状图脚本应设置 `plt.style.use("default")` 或 `"seaborn-v0_8-white"`（而非 `"ggplot"` 灰色背景），以匹配白底设计
- 颜色/标记方案沿用 [utils.py:INDEX_META](5_Plot/utils.py#L11-L16)

输出 PNG(300 DPI) + SVG 到 `4_Results/fig_100k/`。

---

### Phase 5: 结果整理

---

## 五、Sweep 输出 JSON Schema

```json
{
  "dataset": "sift1m_100k",
  "method": "Curator",
  "build_time_s": 12.5,
  "index_memory_mb": 145.2,
  "rss_after_build_mb": 380.0,
  "sweep_results": [
    {
      "params": {"search_ef": 256, "pq_M": 32},
      "latency_ms": 0.85,
      "p50_ms": 0.78,
      "p95_ms": 1.45,
      "qps": 1176.5,
      "avg_recall": 0.8234,
      "min_recall": 0.6512,
      "empty_results": 0,
      "per_bucket": {
        "low":    {"avg_latency_ms": 0.75, "avg_recall": 0.9012, "count": 200},
        "medium": {"avg_latency_ms": 0.90, "avg_recall": 0.8123, "count": 200},
        "high":   {"avg_latency_ms": 0.81, "avg_recall": 0.7567, "count": 200}
      },
      "rss_peak_query_mb": 195.2,
      "complex_predicate": {
        "n_filters": 48,
        "n_queries": 100,
        "avg_latency_ms": 1.40,
        "avg_recall": 0.832,
        "qps": 714.3,
        "per_filter": {
          "AND 0 24": {"avg_latency_ms": 1.25, "avg_recall": 0.854, "qps": 800.0, "selectivity": 0.00198},
          "...": {}
        }
      }
    }
  ]
}
```

> `rss_peak_query_mb` 由 Python 编排脚本通过 `psutil.Process().memory_info().rss` 在查询循环中采样获取，保证跨方法可比性。
> `n_filters` 为去重后的实际 filter 数量（sift1m_100k 已修复：50 → **48**，其他数据集无重复）。

---

## 六、Sweep 参数总览（v6 修正）

| 方法 | 搜索参数 | 构建参数 | build 次数 | search 次数 | CP 开销 |
|------|---------|:-----:|:--------:|:--------:|:-------:|
| **Curator** | `search_ef` [64,256,512,1024] | `pq_M` [4 per d] | **4** | **16** | 0（Python API：`index_filter()` 一次构建 → 每 search_ef 下逐 filter `query_with_complex_predicate()`） |
| **DiskIVF** | `nprobe` [1,2,4,8,16,24,32] | `nlist=32` | **1** | **7** | 5-6 filters × search-only |
| **SPANN** | `max_check` [2048,4096,8192,16384] × `overfetch_factor` [10,25,50,100] | `hash_exp=8` | **1** | **16** | 5-6 filters × search-only |
| **Pre-Filter** | — | — | **0** | **1** | 5-6 filters × search-only |

---

## 七、时间估算（v6 修正）

> **v6 关键修正**: Curator 搜索参数 sweep 从 16 次 build 降至 4 次 build + 16 次 search（添加 `search` 子命令后）。SL 总时间大幅缩减。

| 数据集 | Curator (4 build + 16 search) | DiskIVF (1 build + 7 search) | SPANN (1 build + 16 search) | Pre-Filter (1×) | 合计 |
|--------|:-----------:|:----------:|:---------:|:-------------:|:----:|
| sift1m_100k | ~3 min | ~0.5 min | ~6 min | ~5s | **~10 min** |
| yfcc100m_100k | ~4 min | ~0.5 min | ~8 min | ~8s | **~13 min** |
| arxiv_100k | ~5 min | ~1 min | ~10 min | ~10s | **~16 min** |
| wit_100k | ~5 min | ~1 min | ~10 min | ~10s | **~16 min** |
| gist1m_100k | ~7 min | ~1.5 min | ~12 min | ~15s | **~21 min** |

> SL 总计 ≈ **1-1.5 小时**（v5 估算为 1.5-2h，v6 因 Curator 减少重复构建而缩减）。CP 额外增加 ~20 min（search-only 模式，无重建开销）。
> 主要瓶颈：SPANN SPTAG 构建（gist1m_100k d=960 最慢）；Curator gist1m_100k K-means（高维聚类慢）。

| Phase | 内容 | 预计 |
|-------|------|:----:|
| Phase 1 | 基础设施（access.npy + C++ search 子命令 + 脚本扩展 + 配置） | 2-3 h |
| Phase 2 | sift1m_100k 验证 | 0.5-1 h |
| Phase 3 | SL + CP 实验（5 数据集 × 4 方法） | 1.5-2.5 h |
| Phase 4 | 可视化 | 2-3 h |
| Phase 5 | 结果整理 | 0.5 h |
| **合计** | | **6.5-10 h** |

---

## 八、运行环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
```

---

## 九、Sweep Config 模板

### 9.1 `3_Config/Curator/sweep_100k.json`

```json
{
  "_comment": "Curator sweep: pq_M (build) × search_ef (search). per-dataset pq_M list resolved by Python script.",
  "sweep_strategy": "build_then_search",
  "build_params": {
    "pq_M": {"sift1m_100k": [16, 32, 64, 128], "yfcc100m_100k": [24, 48, 64, 96], "arxiv_100k": [48, 96, 128, 192], "gist1m_100k": [96, 160, 240, 320], "wit_100k": [48, 96, 128, 192]}
  },
  "search_params": {
    "search_ef": [64, 256, 512, 1024]
  },
  "fixed": {
    "variance_boost": 0.2, "beam_size": 4, "n_clusters": 32,
    "max_sl_size": 256, "gist1m_100k_max_sl_size": 128,
    "pq_use_adc_rerank": true, "use_flash_storage": true
  }
}
```

### 9.2 `3_Config/DiskIVF-PostFiltering/sweep_100k.json`

```json
{
  "nprobe": [1, 2, 4, 8, 16, 24, 32],
  "nlist": 32,
  "clus_niter": 20
}
```

### 9.3 `3_Config/SPANN-PostFiltering/sweep_100k.json`

```json
{
  "max_check": [2048, 4096, 8192, 16384],
  "overfetch_factor": [10, 25, 50, 100],
  "overfetch_adaptive": true,
  "hash_exp": 8,
  "num_threads": 1
}
```

### 9.4 `3_Config/Pre-Filtering/sweep_100k.json`

```json
{}
```

---

## 十、待处理事项 — 终局裁决（v6 已全部解决）

### 10.1 Curator sweep：采用 Python API 方案 ✅

**裁决**: 使用 Python API（[legacy/python/curator.py](Curator/legacy/python/curator.py) 中的 `Curator` 类），不添加 C++ `search` 子命令。

**代码验证依据**:
1. `CuratorIndex` 无 `save()`/`load()` 方法 — [curator_index.h](Curator/src/curator_index.h) 仅有 `flush()` 将向量写入 flash，但树结构、centroids、短列表等核心元数据不支持序列化/反序列化
2. `FlashStore` 只存原始向量（按 leaf 偏移量），不含索引元数据 — [flash_store.h](Curator/src/flash_store.h)
3. Python `Curator` 类通过 `search_params` setter 支持运行时修改 `search_ef` — [curator.py:216-232](Curator/legacy/python/curator.py#L216-L232)：`self.index.search_ef = self.search_ef`
4. 添加 C++ save/load 需序列化整套索引结构（TreeNode、shortlists、Bloom filters、PQ codebook、centroids、ID allocators），属于重大功能（估计 3-5 天），超出实验准备阶段的时间预算

**实现方案**:
- `run_experiment.py --sweep` 模式下：每个 `pq_M` 值构建一次 `Curator` 实例 → 循环设置 `search_params` → 每次调用 `query()` 收集 latency/recall
- 单点 `--config` 模式保持不变（调用 C++ CLI `bench`，用于 Phase 2 验证）
- 具体流程参照 [legacy/run_curator_sweep.py](Curator/legacy/run_curator_sweep.py)（已验证可用）

**影响**: Curator 是唯一不通过 C++ CLI 执行 sweep 的方法。这带来两项注意：
1. Python `Curator` 类底层使用 FAISS C++（`faiss.MultiTenantIndexIVFHierarchical`），与 C++ CLI 的 standalone 实现**可能**有微小性能差异。Phase 2 验证时对比单点结果确认一致性。
2. 内存统计使用 `getCurrentRSS()` + `QueryRSSSampler`（Python 侧 `psutil`），与其他方法一致。

### 10.2 DiskIVF `search` 子命令：确认可实现 ✅

**裁决**: 添加 C++ `search` 子命令，实现成本可控。

**代码验证依据**:
1. `build()` 已将 centroids 和 cluster_sizes 序列化到 `centroids.bin` / `cluster_sizes.bin` — [diskivf_index.cpp:108-122](DiskIVF-PostFiltering/src/diskivf_index.cpp#L108-L122)
2. 每个 cluster 的向量/ID/标签已序列化到 `c*_vecs.bin` / `c*_vids.bin` / `c*_mds.bin`
3. 搜索路径 (`get_nearest_clusters()` → `load_cluster()` → `process_cluster_*()`) 是 const 方法，不依赖构建时的临时数据
4. 缺少的仅是一个 `load_metadata()` 方法 + 写入 `metadata.bin`（d, ntotal, nlist 三个 int32_t）

**实现方案**:
1. 在 `build()` 末尾新增写入 `disk_dir/metadata.bin`（3 个 int32_t: d, ntotal, nlist）
2. 在 `DiskIVFIndex` 新增 `void load_metadata(const std::string& disk_dir)` 方法，读取 metadata.bin + centroids.bin + cluster_sizes.bin
3. `main.cpp` 新增 `search` 子命令：`diskivf search --disk_dir <path> --queries <path> --nprobe <N> --filter <EXPR> --k <K> --output <path>`
4. 预估代码量：C++ 侧 ~80 行（metadata.bin 写入 10 行 + `load_metadata()` 30 行 + `search` CLI 40 行）

### 10.3 SPANN `search` 子命令：直接可用 ✅

**裁决**: 添加 C++ `search` 子命令，实现成本最低。

**代码验证依据**:
- `SPANNPostFilterIndex::load(index_dir, metadata_path)` 已实现 — [spann_pf_index.h:64](SPANN-PostFiltering/src/spann_pf_index.h#L64)
- SPTAG 索引天然支持 load（`SPTAG::VectorIndex::LoadIndex`），构建时已写入磁盘

**实现方案**:
1. `main.cpp` 新增 `search` 子命令：`spann_pf search --index_dir <path> --metadata_path <path> --queries <path> --max_check <N> --overfetch_factor <N> --filter <EXPR> --k <K> --output <path>`
2. 预估代码量：C++ 侧 ~50 行

### 10.4 Pre-Filtering `search` 子命令：重载即搜索 ✅

**裁决**: 添加 C++ `search` 子命令，实现成本最低。

**代码验证依据**:
- Pre-Filtering 无索引结构，"构建" = 加载向量 + 标签到内存
- `search_with_predicate()` 是纯计算，无磁盘 I/O

**实现方案**:
1. `main.cpp` 新增 `search` 子命令：加载向量文件 + metadata → 搜索（跳过 K-means/索引构建）
2. 预估代码量：C++ 侧 ~40 行

### 10.5 SPANN `hash_exp` 参数确认 ✅

**裁决**: 所有 100k 数据集统一使用 `hash_exp=8`。

**依据**:
- SPANN BUILTIN_DEFAULTS 中仅 `*_small`（10k 向量）使用 `hash_exp=6`
- 100k 数据集使用 `hash_exp=8`，与全量数据集（arxiv, yfcc100m, wit）一致
- `hash_exp` 控制 BKTree 哈希表大小（2^hash_exp 个桶），100k 向量需要 ≥256 桶（2^8）以保证分区质量

### 10.6 sift1m_100k filters.json 重复条目 ✅ 已修复

**裁决**: 在数据层面修复。

**问题**: `"AND 2 3"` 在 filters 数组中连续出现 3 次（原位置 13-15），导致 `n_filters=50` 但实际唯一 filter 仅 48 个。

**修复**: 已编辑 [filters.json](1_Data/ground_truth/sift1m_100k/complex_predicate/filters.json)：
- 移除重复的两条 `"AND 2 3"`
- `n_filters`: 50 → **48**
- `selectivities` 字典不变（原仅一个条目）
- GT 文件 `gt_AND_2_3.npy` 不受影响（仅一个文件）

**其他数据集**: 经检查（yfcc100m_100k, arxiv_100k, gist1m_100k, wit_100k）无重复 filter。

---

*计划 v6 — 代码审查修正版。所有待处理事项已通过代码验证裁决，sift1m_100k 数据错误已修复。*
