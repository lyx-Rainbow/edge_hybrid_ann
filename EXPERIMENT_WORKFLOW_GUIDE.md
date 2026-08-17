# 100K 实验流程操作指南

> 适用于：手动调整参数后重新运行 SL（单标签）或 CP（复杂谓词）实验

---

## 一、环境准备

```bash
# WSL2 Ubuntu 终端
source ~/miniconda3/etc/profile.d/conda.sh && conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
```

**前置条件**:
- 4 个 C++ 二进制已编译：`{Curator,DiskIVF-PostFiltering,SPANN-PostFiltering,Pre-Filtering}/build/`
- 5 个 100k 数据集已就绪：`1_Data/ground_truth/{sift1m,yfcc100m,arxiv,gist1m,wit}_100k/`
- `train_access.npy` 已生成（Phase 1.1）

---

## 二、配置文件说明

### 2.1 Sweep 参数配置

| 方法 | 配置文件 | 关键参数 |
|------|---------|---------|
| Curator | `3_Config/Curator/sweep_100k.json` | `pq_M` (per-dataset list), `search_ef`, `n_clusters`, `max_sl_size` |
| DiskIVF | `3_Config/DiskIVF-PostFiltering/sweep_100k.json` | `nprobe`, `nlist`, `clus_niter` |
| SPANN | `3_Config/SPANN-PostFiltering/sweep_100k.json` | `max_check`, `overfetch_factor`, `hash_exp` |
| SPANN (reduced) | `3_Config/SPANN-PostFiltering/sweep_100k_reduced.json` | 同上，仅 4 combo (快速测试用) |
| Pre-Filtering | `3_Config/Pre-Filtering/sweep_100k.json` | 空对象 `{}` (无参数) |

### 2.2 修改 Curator sweep 参数示例

编辑 `3_Config/Curator/sweep_100k.json`：

```json
{
  "sweep_strategy": "build_then_search",
  "build_params": {
    "pq_M": {
      "sift1m_100k": [16, 32, 64, 128],
      ...
    }
  },
  "search_params": {
    "search_ef": [64, 256, 512, 1024, 2048]    // 增加 2048
  },
  "fixed": {
    "n_clusters": 64,                            // 从 32 改为 64
    "max_sl_size": 256,
    "variance_boost": 0.2,
    ...
  }
}
```

### 2.3 修改 DiskIVF sweep 参数示例

编辑 `3_Config/DiskIVF-PostFiltering/sweep_100k.json`：

```json
{
  "nprobe": [1, 2, 4, 8, 16],        // 从 [1,2,4,8,16,24,32] 缩减（避免超时）
  "nlist": 64,                        // 从 32 改为 64
  "clus_niter": 20
}
```

### 2.4 修改 SPANN sweep 参数示例

编辑 `3_Config/SPANN-PostFiltering/sweep_100k.json`：

```json
{
  "max_check": [2048, 4096, 8192, 16384],
  "overfetch_factor": [10, 25, 50, 100],
  "fixed": {
    "hash_exp": 6,                    // 从 8 改为 6（更小的哈希表）
    ...
  }
}
```

---

## 三、运行实验

### 3.1 单标签 (SL) 实验

```bash
# 单个方法 × 单个数据集
python3 run_100k_sweep.py --dataset sift1m_100k --method Curator

# 单个方法 × 所有数据集 (逐个运行)
for ds in sift1m_100k yfcc100m_100k arxiv_100k gist1m_100k wit_100k; do
    python3 run_100k_sweep.py --dataset $ds --method Curator
done

# 使用自定义 sweep 配置
python3 run_100k_sweep.py --dataset sift1m_100k --method SPANN \
    --sweep 3_Config/SPANN-PostFiltering/sweep_100k_reduced.json

# 后台运行 (长时间任务)
nohup python3 run_100k_sweep.py --dataset gist1m_100k --method DiskIVF \
    > /tmp/sweep_gist1m_diskivf.log 2>&1 &
```

### 3.2 复杂谓词 (CP) 实验

CP 实验使用 `--run_cp` 参数。脚本自动：
1. 从 `complex_predicate/filters.json` 选取 6 个代表性 filter（AND×2 + AND_NOT×2 + OR×1 + NOT×1）
2. 使用 C++ 批量 `--filters_file` 模式（构建 1 次索引，搜索 6 个 filter）
3. 将结果写入 sweep JSON 的 `complex_predicate` 字段

```bash
# 单数据集 × 单方法 CP
python3 run_100k_sweep.py --dataset sift1m_100k --method Curator --run_cp

# 单数据集 × 所有方法 CP
python3 run_100k_sweep.py --dataset sift1m_100k --method all --run_cp

# 单数据集 × 所有方法 SL+CP 一起跑
python3 run_100k_sweep.py --dataset sift1m_100k --method all --run_sl --run_cp

# 批量所有数据集
for ds in sift1m_100k yfcc100m_100k arxiv_100k gist1m_100k wit_100k; do
    python3 run_100k_sweep.py --dataset $ds --method all --run_sl --run_cp
done

# 后台运行
nohup python3 run_100k_sweep.py --dataset sift1m_100k --method all --run_sl --run_cp \
    > logs/sweep_sift1m_100k_all.log 2>&1 &
```

**CP 单点测试**（快速验证）：

```bash
# 准备 6 个代表性 filter（RPN 格式用于 Curator，Polish 格式用于其他方法）
echo '0 24 AND' > /tmp/cp_filters_rpn.txt
echo '2 3 AND' >> /tmp/cp_filters_rpn.txt

# Curator CP（RPN 格式）
./Curator/build/curator bench \
    --train_vecs 1_Data/ground_truth/sift1m_100k/train_vecs.npy \
    --train_access 1_Data/ground_truth/sift1m_100k/train_access.npy \
    --queries 1_Data/ground_truth/sift1m_100k/complex_predicate/query_vecs.npy \
    --filters_file /tmp/cp_filters_rpn.txt --k 10 --output /tmp/cp_test.json

# DiskIVF CP（Polish 格式）
echo 'AND 0 24' > /tmp/cp_filters_polish.txt
echo 'AND 2 3' >> /tmp/cp_filters_polish.txt
./DiskIVF-PostFiltering/build/diskivf bench \
    --train_vecs 1_Data/ground_truth/sift1m_100k/train_vecs.npy \
    --train_access 1_Data/ground_truth/sift1m_100k/train_access.npy \
    --queries 1_Data/ground_truth/sift1m_100k/complex_predicate/query_vecs.npy \
    --nlist 32 --nprobe 8 --filters_file /tmp/cp_filters_polish.txt \
    --disk_dir /tmp/cp_diskivf_test --k 10 --output /tmp/cp_diskivf_test.json
```

> **注意**: Curator 使用 RPN（Reverse Polish Notation）格式，其他三个方法使用 Polish（前缀）格式。`run_100k_sweep.py` 自动转换。

### 3.3 预计运行时间 (WSL2)

| 方法 | 每数据集 | 5 数据集合计 |
|------|:------:|:----------:|
| Curator | ~8-12 min | ~50 min |
| DiskIVF | ~25-30 min | ~2.5 h |
| SPANN (4 combo) | ~18-22 min | ~1.8 h |
| SPANN (16 combo) | ~70-90 min | ~7 h |
| Pre-Filtering | ~10-30 s | ~2 min |

> DiskIVF 在原生 Linux (ext4) 上预计快 10-50×；SPANN 构建为主要瓶颈。

### 3.4 单点测试 (快速验证)

不运行完整 sweep，仅测试单个参数组合：

```bash
# Curator 单点
./Curator/build/curator bench \
    --train_vecs 1_Data/ground_truth/sift1m_100k/train_vecs.npy \
    --train_access 1_Data/ground_truth/sift1m_100k/train_access.npy \
    --queries 1_Data/ground_truth/sift1m_100k/query_vecs.npy \
    --query_labels 1_Data/ground_truth/sift1m_100k/query_labels.npy \
    --k 10 --output /tmp/curator_test.json

# 检查 recall
python3 tests/check_recall.py /tmp/curator_test.json \
    1_Data/ground_truth/sift1m_100k/ground_truth.npy 10

# DiskIVF 单点
./DiskIVF-PostFiltering/build/diskivf bench \
    --train_vecs 1_Data/ground_truth/sift1m_100k/train_vecs.npy \
    --train_access 1_Data/ground_truth/sift1m_100k/train_access.npy \
    --queries 1_Data/ground_truth/sift1m_100k/query_vecs.npy \
    --query_labels 1_Data/ground_truth/sift1m_100k/query_labels.npy \
    --nlist 32 --nprobe 8 --k 10 \
    --disk_dir /tmp/diskivf_test --output /tmp/diskivf_test.json

# SPANN 单点
./SPANN-PostFiltering/build/spann_pf bench \
    --train_vecs 1_Data/ground_truth/sift1m_100k/train_vecs.npy \
    --train_access 1_Data/ground_truth/sift1m_100k/train_access.npy \
    --queries 1_Data/ground_truth/sift1m_100k/query_vecs.npy \
    --query_labels 1_Data/ground_truth/sift1m_100k/query_labels.npy \
    --k 10 --index_dir /tmp/spann_test --output /tmp/spann_test.json

# Pre-Filtering 单点 (CP 示例)
./Pre-Filtering/build/prefiltering bench \
    --train_vecs 1_Data/ground_truth/sift1m_100k/train_vecs.npy \
    --train_access 1_Data/ground_truth/sift1m_100k/train_access.npy \
    --queries 1_Data/ground_truth/sift1m_100k/complex_predicate/query_vecs.npy \
    --filter "AND 0 24" --k 10 --output /tmp/prefilter_cp_test.json
```

---

## 四、查看实验结果

### 4.1 Sweep 结果 JSON 位置

```
4_Results/
├── Curator/
│   └── sweep_{dataset}.json        # 例: sweep_sift1m_100k.json
├── DiskIVF-PostFiltering/
│   └── sweep_{dataset}.json
├── SPANN-PostFiltering/
│   └── sweep_{dataset}.json
└── Pre-Filtering/
    └── sweep_{dataset}.json
```

### 4.2 JSON 结构

```json
{
  "dataset": "sift1m_100k",
  "method": "Curator",
  "build_time_s": 58.4,
  "index_memory_mb": 22.7,
  "n_combinations": 16,
  "sweep_results": [
    {
      "params": {"search_ef": 64, "pq_M": 16},
      "latency_ms": 10.6,
      "p50_ms": 9.8,
      "p95_ms": 15.2,
      "qps": 94.0,
      "avg_recall": 0.8121,
      "min_recall": 0.5,
      "rss_peak_query_mb": 195.2
    },
    ...
  ]
}
```

### 4.3 命令行快速查看

```bash
# 查看某个 sweep 的最佳 recall
python3 -c "
import json
d = json.load(open('4_Results/Curator/sweep_sift1m_100k.json'))
best = max(d['sweep_results'], key=lambda r: r['avg_recall'])
print(f\"Best R@10: {best['avg_recall']:.4f} @ QPS={best['qps']:.0f} params={best['params']}\")
"

# 查看所有 Curator sweep 结果概览
for f in 4_Results/Curator/sweep_*_100k.json; do
    python3 -c "
import json; d=json.load(open('$f'))
best=max(d['sweep_results'], key=lambda r: r['avg_recall'])
print(f\"{d['dataset']}: R={best['avg_recall']:.4f} QPS={best['qps']:.0f}\")
"
done

# 查看所有方法在某个数据集上的结果
for d in Curator DiskIVF-PostFiltering SPANN-PostFiltering Pre-Filtering; do
    f="4_Results/$d/sweep_sift1m_100k.json"
    [ -f "$f" ] && python3 -c "
import json; d=json.load(open('$f'))
best=max(d['sweep_results'], key=lambda r: r['avg_recall'])
print(f\"{d['method']}: R={best['avg_recall']:.4f} QPS={best['qps']:.0f}\")
"
done
```

---

## 五、绘图

### 5.1 SL QPS-Recall 图

```bash
python3 5_Plot/fig_sl_qps_recall_100k.py
```

输出: `4_Results/fig_100k/fig_sl_qps_recall_100k.png` + `.svg`

**工作原理**:
- 从 `4_Results/{Curator,DiskIVF-PostFiltering,SPANN-PostFiltering,Pre-Filtering}/sweep_{dataset}.json` 读取数据
- 每个数据集一个子图，每个方法一条 Pareto frontier 曲线
- 颜色/标记方案定义在 `5_Plot/utils.py:INDEX_META`

### 5.2 CP QPS-Recall 图

```bash
python3 5_Plot/fig_cp_qps_recall_100k.py
```

输出: `4_Results/fig_100k/fig_cp_qps_recall_100k.png` + `.svg`

从 sweep JSON 的 `sweep_results[].complex_predicate` 字段读取 CP 聚合数据。

### 5.3 选择率分面 SL 图

```bash
python3 5_Plot/fig_sl_qps_recall_by_bucket.py
```

输出: `4_Results/fig_100k/fig_sl_qps_recall_by_bucket.png` + `.svg`

5 数据集行 × 3 选择率列（low/mid/high），展示不同选择率下的 QPS-Recall 变化。

### 5.4 Memory 柱状图

```bash
python3 5_Plot/fig_memory_100k.py
```

输出: `4_Results/fig_100k/fig_memory_100k.png` + `_legend.png`

使用 `rss_peak_query_mb`（查询阶段进程 RSS）作为跨方法可比的统一口径。

### 5.3 自定义绘图

修改 `5_Plot/fig_sl_qps_recall_100k.py` 中的关键参数：

```python
DATASETS_100K = ["sift1m_100k", ...]  # 选择数据集
K = 10                                  # Recall@K

# 在 plot_all() 中：
ax.set_xlim(0.5, 1.005)               # X 轴范围
ax.set_yscale("log")                   # Y 轴 log 尺度
```

颜色/标记在 `5_Plot/utils.py`:

```python
INDEX_META = {
    "curator":   ("Proposed",  "Curator",                 "#1f77b4", "o"),
    "diskivf":   ("DiskIVF",   "DiskIVF-PostFiltering",   "#ff7f0e", "s"),
    "spann":     ("SPANN",     "SPANN-PostFiltering",     "#2ca02c", "D"),
    "prefilter": ("PreFilter", "Pre-Filtering",           "#d62728", "^"),
}
```

---

## 六、添加新数据集 / 新方法

### 6.1 添加新 100k 数据集

1. 准备数据到 `1_Data/ground_truth/{new_dataset}/`（参照 [DATASETS_100K.md](1_Data/ground_truth/DATASETS_100K.md)）
2. 确保包含: `train_vecs.npy`, `train_mds.pkl`, `query_vecs.npy`, `query_labels.npy` (int32!), `ground_truth.npy`, `query_info.json`, `metadata.json`
3. 在 `run_100k_sweep.py` 的 sweep config 中添加 `pq_M` per-dataset 映射（Curator）
4. 在所有 `run_experiment.py` 的 `BUILTIN_DEFAULTS` 中添加条目
5. 在 `5_Plot/fig_sl_qps_recall_100k.py` 的 `DATASETS_100K` 列表中添加

### 6.2 添加新 baseline 方法

1. 在 C++ 侧实现或包装为 CLI（接受 `--train_vecs --queries --k --output` 等标准参数）
2. 在 `run_100k_sweep.py` 中:
   - 添加 `BINARIES` 条目
   - 添加 `METHOD_RUNNERS` 条目（实现 sweep 函数）
   - 添加 `SWEEP_CONFIGS` 条目
3. 在 `5_Plot/utils.py` 的 `INDEX_META` 中添加颜色/标记
4. 在 `5_Plot/fig_*.py` 的方法循环中添加

---

## 七、常见问题

### Q: DiskIVF nprobe≥16 超时

**原因**: WSL2 NTFS I/O 瓶颈。nprobe 越大，加载的 cluster 数据越多。

**解决**:
- 减少 nprobe 范围到 [1,2,4,8]
- 或在原生 Linux 运行
- 或增加 `subprocess.run(timeout=900)` 超时时间

### Q: SPANN 构建极慢 (gist1m >10 min)

**原因**: d=960 高维下 SPTAG 头索引构建耗时长。

**解决**:
- 减小 `hash_exp` (8→6)
- 或接受 gist1m 为慢数据集，适当减少 combo 数

### Q: recall 极低 (如 <0.1)

**检查清单**:
1. `query_labels.npy` dtype 是否为 int32？（yfcc100m/arxiv 曾出现 int64 问题）
2. `query_labels` 中的 label ID 是否存在于 `train_mds.pkl` 中？
3. 是否使用了正确的 `ground_truth.npy`？（单标签用根目录的，CP 用 `complex_predicate/gt_{filter}.npy`）

### Q: 需要重新编译 C++ 二进制

```bash
# Curator
cd Curator/build && make -j$(nproc)

# DiskIVF
cd DiskIVF-PostFiltering/build && make -j$(nproc)

# SPANN (需要先编译 SPTAG 库)
cd SPANN-PostFiltering/SPTAG-main/Release && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
cd ../../build && make -j$(nproc)

# Pre-Filtering
cd Pre-Filtering/build && make -j$(nproc)
```
