# sift1m / gist1m 实验经验总结

## 1. SPANN 关键修复

### 问题
- gist1m 上 SPANN recall 仅约 0.003，几乎等于随机。
- 根因：SPTAG SPANN 的 head KDT 默认使用 Cosine（`1 - dot`）距离，
  而 gist1m 各向量模长差异较大，Cosine 近邻与 L2 真值差异极大。
- 单纯增大 `search_ef` / `SearchInternalResultNum` 无法修复，因为 head 索引
  本身使用了错误的距离度量。

### 修复
- 在 `SPANN-PostFiltering/src/spann_pf_index.cpp` 中：
  - build 阶段：SPANN 创建 head KDT 后，强制 `DistCalcMethod = L2`；
  - load 阶段：加载已有 index 后，同时把 SPANN options 和 head KDT 都设为 L2。
- 重新编译 `spann_pf`。
- 验证：仍使用旧的 gist1m 索引（hash_exp=10, bk64），修复后 SPANN
  recall 直接恢复到 0.78~0.85，加入 SIR 后最高 0.988。

### 搜索参数经验
- `max_check`: 8192 / 16384 / 32768
- `overfetch_factor`: 10 / 25 / 50 / 100 / 200 / 500 / 1000 / 2000
- `SearchInternalResultNum` (SIR):
  - 低选择率、高维、标签分布均匀的数据集必须使用 SIR。
  - 建议扫描 100 / 200 / 500 / 1000。
  - gist1m 上：SIR=100 可以到 ~0.89，SIR=500 到 ~0.974，
    SIR=1000 到 ~0.988。
- 后续数据集直接使用：`hash_exp=10`, `bkt_kmeans_k=64` 即可，
  不再需要更重的 hash_exp=12 / bk256。
- `run_spann_search_full.py` 已内置 L2 修复后的搜索组合，
  并加入 SIR=100/200/500/1000。

## 2. Curator 关键经验

- 最高 recall 受 **PQ 量化误差** 和 **search_ef** 共同限制。
- gist1m `pq_M=160` 即使 `search_ef=32768`，75p/99p 也仅到 0.96 左右。
- 提高 `pq_M` 后效果显著：
  - `pq_M=480`（维度 960 的一半）
  - `search_ef = [8192, 16384, 32768]`
  - `nprobe = 8`
  - 最终 1p/25p/50p/75p/99p 最高 recall 分别达到
    1.000 / 0.999 / 0.999 / 0.997 / 0.993。
- 结论：
  - 大维度数据集优先尝试 `pq_M ≈ d/2`。
  - 若仍不足，再加到 `pq_M = d`（如 gist1m 的 960）。
  - `search_ef` 至少要覆盖到 8192~32768。
  - nprobe=8 通常已足够，不必优先增大 nprobe。

## 3. DiskIVF 经验

- 使用 `nlist ≈ sqrt(N)`。
- nprobe 扫描到 128/更大，用于把 recall 推到接近 1.0。
- NTFS 环境下 QPS 很低（0~几 QPS），这是环境问题，不是算法问题。

## 4. PreFilter 经验

- 使用真实 external scan，不是模拟 I/O。
- `--external-scan --scan-chunk 4096 --vector-file /home/lyx/<ds>.pfvec`。
- 如果 pfvec 不存在，PreFilter 程序会在 bench 时自己生成。
- 选择率越低，过滤后候选越少，QPS 越高。

## 5. 绘图经验

- 沿用 `5_Plot/fig_full_sl_by_percentile.py`：
  - X 轴从 0.5 开始；
  - QPS 对数轴；
  - 主线只展示关键点，Raw 点可用半透明小点表示；
  - 50p/75p/99p 的 Curator 必须用 Pareto frontier，否则容易出现
    水平平台和位置错误；现已改为所有 bucket 都用 Pareto。
- 点太少时优先补实验而不是强行插值。
- 每个数据集实验完成后立即绘图。

## 6. 剩余数据集配置

| 数据集 | Curator pq_M | Curator search_ef | SPANN | DiskIVF | PreFilter |
|---|---:|---:|---|---|---|
| arxiv | 192 | 8192 / 16384 / 32768 / 65536 | L2 + SIR | nlist≈√N, nprobe up to 128 | real external |
| yfcc100m | 96 | 同上 | L2 + SIR | 同上 | real external |
| wit | 192 | 同上 | L2 + SIR | 同上 | real external |

自动流水线：`run_full_remaining_pipeline_v2.py --datasets arxiv yfcc100m wit`
每个数据集跑完自动调用 `5_Plot/fig_full_sl_by_percentile.py <dataset>` 绘图。
