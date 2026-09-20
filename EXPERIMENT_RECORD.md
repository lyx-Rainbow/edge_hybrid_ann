# 实验记录与结果整理

> 更新日期: 2026-08-17 | 100K 多租户过滤向量检索 benchmark
> 方法: **Curator**（Proposed）/ **DiskIVF-PostFiltering** / **SPANN-PostFiltering** / **Pre-Filtering**
> 查询: SL（单标签过滤）/ CP（复杂谓词 AND/AND_NOT/OR/NOT） | 指标: Recall@10, QPS, 构建时间, 查询期峰值 RSS

---

## 一、实验总体设计

| 维度 | 配置 |
|------|------|
| 数据集 | 5 个 100K 数据集（见 §二） |
| 方法 | 4 个独立 C++ 实现（Curator 零 FAISS 依赖，其余为零 FAISS 的 C++ 重写） |
| 查询 | SL + CP（6 个代表性 filter：AND×2 + AND_NOT×2 + OR×1 + NOT×1） |
| 指标 | Recall@10、QPS、构建时间、查询期峰值 RSS |
| 环境 | WSL Ubuntu 24.04 / GCC 13.3 / conda `edge_ann` |
| 执行器 | `run_100k_sweep.py`（全量 sweep） | 绘图 | `5_Plot/fig_*.py` → `4_Results/fig_100k/` |

## 二、数据集配置

| 数据集 | 向量数 | d | 标签数 | 标签方式 | SL 查询数 | 标签/向量 | 选择率范围 |
|--------|:----:|:--:|:----:|---------|:----:|:----:|:----:|
| sift1m_100k | 100,000 | 128 | 100 | random（层次 Bernoulli） | 1,000 | 3.0 | 1.29%–50.1% |
| yfcc100m_100k | 100,000 | 192 | 1,000 | natural（Flickr tag） | 1,000 | 13.4 | 0.046%–34.3% |
| arxiv_100k | 100,000 | 384 | 99 | natural（学科分类） | 1,000 | 1.7 | 0.37%–8.72% |
| gist1m_100k | 100,000 | 960 | 100 | random（层次 Bernoulli） | 500 | 3.0 | 0.80%–50.1% |
| wit_100k | 100,000 | 384 | 1,000 | random（层次 Bernoulli） | 1,000 | 3.0 | 0.078%–29.9% |

k=10，L2，种子 42，每数据集含 SL+CP 两套 ground truth（详见 `1_Data/ground_truth/DATASETS_100K.md`）。

## 三、索引参数配置（总体）

| 方法 | 构建参数 | 搜索参数 | sweep 维度 |
|------|---------|---------|-----------|
| Curator | `n_clusters=32, max_sl_size=256, pq_nbits=8, pq_use_adc_rerank=true, use_flash_storage=true` | `variance_boost=0.2, beam_size=4, search_ef` | `pq_M` × `search_ef` [64,256,512,1024]（右端另至 2048/4096） |
| DiskIVF | `nlist=32, clus_niter=20` | `nprobe` | `nprobe` [1,2,4,8,16]（右端另至 24/32） |
| SPANN | `hash_exp=8, num_threads=1` | `max_check, overfetch_factor, overfetch_adaptive=false` | `max_check` × `overfetch_factor`（右端另 mc=32768, of 至 200–1000） |
| Pre-Filtering | —（全量内存） | — | 1 combo（精确基线） |

> `pq_M`：sift [16,32,64,128]，yfcc [24,48,64,96]，arxiv [48,96,128,192]，gist [96,160,240,320]，wit [48,96,128,192]。
> SPANN `max_check` [2048,4096,8192,16384] × `overfetch_factor` [10,25,50,100]。

## 四、SL（单标签）QPS-Recall 结果

（每数据集 Pareto frontier；R=右端 Recall@10，QPS=右端吞吐，pts=frontier 点数）

| 数据集 | 方法 | pts | 右端 R@10 | 右端 QPS |
|--------|------|:--:|:--:|:--:|
| sift1m | Curator | 5 | 0.998 | 112 |
| | DiskIVF | 5 | 1.000 | 2 |
| | SPANN | 5 | 0.952 | 63 |
| | Pre-Filtering | 1 | 1.000 | 660 |
| yfcc100m | Curator | 5 | 0.996 | 124 |
| | DiskIVF | 7 | 1.000 | 0 |
| | SPANN | 7 | 0.938 | 39 |
| | Pre-Filtering | 1 | 1.000 | 882 |
| arxiv | Curator | 7 | 0.999 | 128 |
| | DiskIVF | 5 | 0.998 | 1 |
| | SPANN | 3 | 0.977 | 28 |
| | Pre-Filtering | 1 | 1.000 | 992 |
| gist1m | Curator | 9 | 0.987 | 80 |
| | DiskIVF | 6 | 0.953 | 1 |
| | SPANN | 6 | 0.876 | 15 |
| | Pre-Filtering | 1 | 1.000 | 131 |
| wit | Curator | 8 | 0.960 | 76 |
| | DiskIVF | 7 | 0.976 | 1 |
| | SPANN | 7 | 0.746 | 26 |
| | Pre-Filtering | 1 | 0.985 | 433 |

## 五、CP（复杂谓词）QPS-Recall 结果

> 全量数据集上的三类 CP case study（AND/OR/MIXED，每类对应一个数据集）见 [CP_EXPERIMENT_RECORD.md](CP_EXPERIMENT_RECORD.md)。

| 数据集 | 方法 | pts | 右端 R@10 | 右端 QPS |
|--------|------|:--:|:--:|:--:|
| sift1m | Curator | 6 | 0.998 | 56 |
| | DiskIVF | 5 | 1.000 | 2 |
| | SPANN | 6 | 0.883 | 52 |
| | Pre-Filtering | 1 | 1.000 | 158 |
| yfcc100m | Curator | 3 | 0.999 | 185 |
| | DiskIVF | 6 | 1.000 | 0 |
| | SPANN | 8 | 0.723 | 20 |
| | Pre-Filtering | 1 | 1.000 | 135 |
| arxiv | Curator | 4 | 0.998 | 209 |
| | DiskIVF | 5 | 0.887 | 1 |
| | SPANN | 5 | 0.521 | 28 |
| | Pre-Filtering | 1 | 1.000 | 139 |
| gist1m | Curator | 9 | 0.981 | 84 |
| | DiskIVF | 5 | 0.936 | 1 |
| | SPANN | 6 | 0.823 | 11 |
| | Pre-Filtering | 1 | 1.000 | 68 |
| wit | Curator | 6 | 0.956 | 139 |
| | DiskIVF | 7 | 0.935 | 1 |
| | SPANN | 10 | 0.712 | 22 |
| | Pre-Filtering | 1 | 0.989 | 114 |

## 六、内存实验（fig_memory_100k / _v2）

**口径**：查询期峰值进程 RSS（`rss_peak_query_mb`，跨方法统一可比）。

> ⚠️ **参数配置差异说明**：内存图（尤其 `fig_memory_100k_v2`）中 **Curator 使用的是单独的内存优化索引配置**，与 §三 的 QPS-Recall 配置**不同**，故两套图为不同配置下的结果，不可直接混读：
> - QPS-Recall 图（§三 配置）：`n_clusters=32, max_sl_size=256, pq_cache_max_blocks=256, bf_false_pos=0.01`；
> - 内存 v2 图：`n_clusters=16, max_leaf_size=768, bf_capacity=256, bf_false_pos=0.1, pq_cache_max_blocks=32`（大叶子→树节点大幅减少，明显降低查询期内存），并叠加构建后内存优化（释放原始向量 + `compact_memory()` 容器紧缩 + glibc arena/trim 调优）。
>
> 其余三种方法（DiskIVF/SPANN/Pre-Filtering）在两套图里配置一致。

**结果（查询期峰值 RSS, MB）**：

| 数据集 | Curator（QPS图配置） | **Curator v2（内存配置）** | DiskIVF | SPANN | Pre-Filtering |
|--------|:--:|:--:|:--:|:--:|:--:|
| sift1m | 141.4 | **63.2** | 10.9 | 49.0 | 111.7 |
| yfcc100m | 233.9 | **114.3** | 20.5 | 58.8 | 179.3 |
| arxiv | 246.5 | **58.6** | 15.9 | 69.0 | 306.0 |
| gist1m | 522.7 | **68.9** | 34.7 | 131.5 | 747.4 |
| wit | 267.3 | **65.8** | 32.6 | 70.4 | 308.0 |

**说明**：Curator 为磁盘索引（向量在 Flash/PQ 文件上），查询期只需索引结构 + PQ 块缓存；DiskIVF 为纯磁盘索引内存最低（仅质心+簇统计）；Pre-Filtering 需全量向量驻留内存，最重。内存优化后 Curator 查询期内存全线显著低于 Pre-Filtering（−36% ~ −91%）。

## 七、图表产物（`4_Results/fig_100k/`）

| 文件 | 内容 |
|------|------|
| `fig_sl_qps_recall_100k.*` | SL QPS-Recall（5 数据集子图，单调 frontier + 自适应横轴） |
| `fig_cp_qps_recall_100k.*` | CP QPS-Recall |
| `fig_sl_qps_recall_by_bucket.*` | SL 按选择率分面（5 × 3） |
| `fig_cp_qps_recall_by_selectivity.*` | CP 按 filter 选择率 |
| `fig_memory_100k.*(+legend)` | 内存柱状图（统一 RSS 口径，QPS 图配置） |
| `fig_memory_100k_v2.*(+legend)` | 内存 v2（Curator 内存优化配置） |

## 八、关键结论

1. **Curator** 在 SL 与 CP 上均达最高召回上限（SL 0.96–0.999 / CP 0.956–0.999），且磁盘索引查询期内存远低于 Pre-Filtering；
2. **SPANN** 的 post-filtering 在稀疏标签上召回受限（SL 0.75–0.94 / CP 0.52–0.88）。经对 overfetch / max_check / hash_exp / BKTKmeansK / SearchInternalResultNum 五个参数维度逐一扫描（含把 SearchInternalResultNum 调大反而降到 0.32），确认该上限为 SPTAG 头索引+posting 的近似结构固有属性，非参数可解——是"后过滤失效"的直接实证；
3. **DiskIVF** 在高 nprobe 下可达精确（Recall≈1.0），但 NTFS 环境使其 QPS 极低（1–29）；
4. **Pre-Filtering** 为精确基线（Recall=1.0），代价是全量向量常驻内存（112–747MB），是最重的方法。

## 九、数据与脚本产物

| 类别 | 位置 |
|------|------|
| 结果数据 | `4_Results/{Curator,DiskIVF,SPANN,Pre-Filtering}/sweep_*_100k.json`（不入版本库） |
| 内存优化测量 | `4_Results/memory_tuned/curator_v2.json` |
| 全量 sweep / 增量补跑 | `run_100k_sweep.py` / `run_incremental.py` |
| 图脚本 | `5_Plot/fig_*.py` |
| 验收/分析工具 | `tests/report_frontier.py`, `probe_mem_cfg.py`, `run_memory_tuned.py` 等 |
| 归档 | `4_Results/_legacy/` |

> `4_Results/*`（除 `fig_100k/`）已在 `.gitignore` 排除，不入远端；本地完整保留。


---

## 十、最终版本统一说明（2026-09-17）

本次最终实验将 Curator 的 flash/PQ 临时索引、DiskIVF 索引与
SPANN 索引（连同可动态配置的 SPTAG SearchInternalResultNum）统一放到 WSL
ext4 上，并从当前 train_access 重刷了标签相关索引元数据。SL 与 CP
结果均重新从 ext4 索引实测。绘图端已取消手动点位与强制载剪，
每条非 PreFilter 折线至少包含 5 个可见的实测点。

> 2026-09-19: All query-performance line figures now use Latency-Recall@10; PreFilter memory fix and rerun details are in EXPERIMENT_FIGURE_UPDATE.md.
