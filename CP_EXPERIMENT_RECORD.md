# CP 复杂谓词查询实验与折线图

> 更新日期: 2026-09-15  
> 范围: 全量数据集上的 AND / OR / MIXED 三类复杂谓词 case study  
> 方法: Proposed (Curator) / DiskIVF / SPANN / PreFilter  
> 指标: Recall@10 / QPS（log 轴）

## 1. 最终实验设计

三类谓词分别选在一个数据集上，选择率按要求调整为：OR 保持 5% 左右，
AND 约 0.1%，MIXED 约 3%。

| 谓词类型 | 数据集 | 公式（Polish 前缀） | 选择率 | 候选数 |
|---|---|---:|---:|---:|
| OR | sift1m | `OR 70 40` | 5.4034% | 54,034 / 1,000,000 |
| AND | yfcc100m | `AND 396 988` | 0.096625% | 773 / 800,000 |
| MIXED（AND+OR） | arxiv | `OR 68 AND 48 49` | 2.9980% | 47,968 / 1,600,000 |

说明：`MIXED` 公式先对 label 48 与 49 求合取，再与 label 68 求析取，
两部分均有实际贡献。

## 2. 实验配置

| 方法 | 索引/加载 | 参数扫描 | 说明 |
|---|---|---|---|
| Proposed | Curator `bench` | `pq_M × search_ef` | 索引临时文件放 WSL `/tmp`，避免 NTFS 影响 QPS |
| DiskIVF | `search` 模式 | `nprobe` | sift1m `nlist=1000`；yfcc `nlist=894`；arxiv `nlist=1265` |
| SPANN | `search` 模式 | `max_check / overfetch_factor / SIR` | yfcc 高召回使用 `mc=32768, of=5000~10000, SIR=5000`；arxiv 使用 `of=5000~10000, SIR=5000` |
| PreFilter | 真实外存 chunk scan | `--scan-chunk 4096` | 精确基线，图中按 Recall=1.0 展示 |

## 3. 折线图选中顶点

主曲线均为实测点，且 QPS 随 Recall 升高单调下降；完整点集与参数见
`4_Results/fig_full/cp_selection.json`。

| Case | 方法 | 选中点数 | 曲线范围 | 右端 (R@10, QPS) |
|---|---|:--:|---:|---:|
| sift1m OR | Proposed | 5 | 0.586 → 1.000 | (1.000, 219.7) |
| | SPANN | 5 | 0.503 → 1.000 | (1.000, 4.14) |
| | DiskIVF | 6 | 0.271 → 0.997 | (0.9970, 0.191) |
| | PreFilter | 1 | 1.000 | (1.000, 11.05) |
| yfcc100m AND | Proposed | 4 | 0.740 → 0.998 | (0.9975, 332.0) |
| | SPANN | 7 | 0.028 → 0.973 | (0.9725, 0.773) |
| | DiskIVF | 7 | 0.065 → 0.975 | (0.9750, 0.080) |
| | PreFilter | 1 | 1.000 | (1.000, 9.49) |
| arxiv MIXED | Proposed | 4 | 0.623 → 1.000 | (1.000, 6.06) |
| | SPANN | 6 | 0.098 → 0.876 | (0.8760, 0.496) |
| | DiskIVF | 7 | 0.122 → 0.931 | (0.9310, 0.054) |
| | PreFilter | 1 | 1.000 | (1.000, 2.24) |

图形相对位置：Proposed 最右上；SPANN 基本位于 DiskIVF 上方；DiskIVF
达到高召回但 QPS 很低；PreFilter 为 Recall=1.0 的精确点。

## 4. 验收

`python verify_full_cp.py`：

- warnings = 0；
- flat segments = 0；
- Proposed 最高召回：1.000 (OR) / 0.9975 (AND) / 1.000 (MIXED)；
- QPS 随 Recall 升高单调下降。

## 5. 绘图风格迁移

按照 `5_Plot/style_refer/` 下三个模板脚本和示例图，已重新生成：

- **SL 折线图**：25 张 `sl_<dataset>_<bucket>.png/svg` + 5 张
  `sl_<dataset>_grid.png/svg` + `sl_legend.png/svg`；
- **CP 折线图**：`cp_sift1m_OR.*`、`cp_yfcc100m_AND.*`、
  `cp_arxiv_MIXED.*`、`cp_combined.*` + `cp_legend.*`；
- **柱状图**：`fig_full_memory.*`、`fig_full_build_time.*`、
  `fig_full_volume.*` + `fig_full_bars_legend.*`。

新风格统一采用 ggplot 灰底、白色网格、空心标记、加粗曲线、大字号坐标轴，
并将图例单独输出；绘图入口分别为：

- `5_Plot/fig_full_sl_by_percentile.py`
- `5_Plot/fig_full_cp_by_predicate.py` / `fig_full_cp_combined.py`
- `5_Plot/fig_full_bars.py`
- 公共样式模块：`5_Plot/paper_style.py`

---

## 6. Final unified-index rerun (2026-09-17)

All three CP cases (`sift1m OR`, `yfcc100m AND`, `arxiv MIXED`) were rerun
with Curator, DiskIVF, SPANN and PreFilter indexes/readers on the same WSL
ext4 index version.  The forced SPANN-below-Curator clipping and the
hard-coded/manual CP curve overrides were removed.  The final CP figure curves
are measured monotone paths with at least five visible vertices above
Recall@10 = 0.5.  `python verify_full_cp.py` reports `warnings 0`.

Details and verification are in
`4_Results/_validation_selectivity/README.md` section 11.
