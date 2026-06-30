# 目标2：新增数据集 SIFT1M 和 GIST1M

## 一、现有基础分析

### 1.1 现有数据流水线（可复用部分）

当前数据流水线由以下脚本实现，可复用于新数据集：

| 脚本 | 功能 | 复用程度 |
|------|------|----------|
| `1_Data/gt_computing.py` | 完整 GT 计算流水线：加载→划分→构建倒排索引→选择查询→计算 SL GT→计算 CP GT→保存 | **高**：核心函数（`build_inverted_index`、`compute_single_label_gt`、`select_single_label_queries`、`compute_complex_predicate_gt`）均可直接复用 |
| `1_Data/prepare_small_dataset.py` | 从全量数据子采样创建小数据集 | 中：子采样逻辑可复用 |
| `Curator/run_curator.py` | 单次 benchmark | **高**：`compute_recall`、BUILTIN_DEFAULTS 模式可扩展 |
| `Curator/run_curator_sweep.py` | 参数扫描 | **高**：sweep 框架通用 |
| `3_Config/Curator/curator_*.json` | 数据集配置 | 需新增 |

**现有数据加载约定**（从 `run_curator.py:139-152` 和 `gt_computing.py` 推断）：

```python
# 每个数据集的 GT 目录下必须包含：
gt_dir/
├── train_vecs.npy       # float32, [n_train, d]
├── train_mds.pkl        # list[list[int]], 标签访问控制列表
├── query_vecs.npy       # float32, [n_queries, d]
├── query_labels.npy     # int32, [n_queries], 每查询对应的单标签
├── query_info.json      # [{"query_idx", "label", "selectivity", "bucket"}, ...]
├── ground_truth.npy     # int32, [n_queries, k], -1 填充
├── all_labels.json      # 所有标签 ID 列表
├── metadata.json        # {"dim", "n_labels", "dataset", "n_queries", "k", ...}
└── complex_predicate/   # (可选)
    ├── query_vecs.npy
    ├── filters.json     # {"filters": [...], "selectivities": {...}}
    └── gt_{formula}.npy
```

### 1.2 新增数据集的关键差异

#### 与现有数据集的核心不同

| 特性 | 现有数据集 (arxiv/yfcc100m) | SIFT1M/GIST1M |
|------|---------------------------|---------------|
| 标签来源 | 天然多标签（学科分类/用户标签） | **无标签**，需合成 |
| 原始格式 | JSON/自定义二进制 | **fvecs/ivecs 格式** |
| 向量类型 | float32 | float32（但原始是 uint8 SIFT / float32 GIST） |
| 查询数量 | 1000（自定义选择） | 10,000 (SIFT) / 1,000 (GIST)（数据集自带） |
| GT 来源 | 自行计算 | 数据集自带 ivecs GT |
| 用途定位 | 多租户过滤检索 | 通用 ANN benchmark |

#### SIFT1M 数据文件

| 文件 | 格式 | 内容 |
|------|------|------|
| `sift_base.fvecs` | 每向量: int32(d=128) + float32[128] | 1,000,000 基础向量 |
| `sift_query.fvecs` | 同上 | 10,000 查询向量 |
| `sift_learn.fvecs` | 同上 | 100,000 学习向量（用于训练 PQ/聚类） |
| `sift_groundtruth.ivecs` | 每行: int32(k=100) + int32[100] | 10,000 查询的 top-100 GT |

#### GIST1M 数据文件

| 文件 | 格式 | 内容 |
|------|------|------|
| `gist_base.fvecs` | 每向量: int32(d=960) + float32[960] | 1,000,000 基础向量 |
| `gist_query.fvecs` | 同上 | 1,000 查询向量 |
| `gist_learn.fvecs` | 同上 | 500,000 学习向量 |
| `gist_groundtruth.ivecs` | 每行: int32(k=100) + int32[100] | 1,000 查询的 top-100 GT |

#### 关键约束：PQ 的维度整除条件

`ProductQuantizer` 要求 **M 整除 d**：

- **SIFT1M (d=128)**：M ∈ {1, 2, 4, 8, 16, 32, 64, 128}，推荐 M=16（dsub=8）
- **GIST1M (d=960)**：M ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64, 80, 96, 120, 160, 192, 240, 320, 480, 960}，推荐 M=32（dsub=30）或 M=48（dsub=20）

#### 内存估算

| 数据集 | n | d | 全精度大小 | PQ 码 (M=16) | PQ 码 (M=32) |
|--------|---|---|-----------|-------------|-------------|
| SIFT1M | 1M | 128 | 512 MB | 16 MB | 32 MB |
| GIST1M | 1M | 960 | 3,840 MB | M=16 不整除 | 32 MB |

GIST1M 全精度向量 3.8 GB，Flash 文件较大，构建和查询的 I/O 压力显著高于现有数据集。

---

## 二、执行计划

### 阶段 A：数据下载与格式解析

#### A1. 实现 fvecs/ivecs 读取器

新增 `1_Data/io_utils.py`：

```python
import numpy as np

def read_fvecs(path: str) -> np.ndarray:
    """读取 .fvecs 文件。
    
    格式: 每条记录 = int32(d) + float32[d]，连续存储。
    返回: float32 ndarray [n, d]
    """
    with open(path, "rb") as f:
        # 读取第一个向量的维度来确定 d
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        # 计算每条记录大小: 4B(d) + 4B×d
        record_bytes = 4 + 4 * d
        file_size = f.seek(0, 2)
        n = file_size // record_bytes
        f.seek(0)
        data = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            d_read = np.fromfile(f, dtype=np.int32, count=1)[0]
            assert d_read == d, f"Dimension mismatch at vector {i}"
            data[i] = np.fromfile(f, dtype=np.float32, count=d)
    return data

def read_ivecs(path: str) -> np.ndarray:
    """读取 .ivecs 文件 (同 fvecs 但类型为 int32)。
    返回: int32 ndarray [n, k]
    """
    # 实现同上，但 dtype 为 int32
    ...
```

> **fvecs 格式确认**：标准 fvecs 格式为 `[int(dim)][float[dim]]` 重复。文件末尾无额外元数据。`sift_base.fvecs` 大小应为 1,000,000 × (4 + 4×128) = 516,000,004 bytes ≈ 492 MB。

#### A2. 自动下载脚本

新增 `1_Data/download_ann_datasets.py`：

```bash
python download_ann_datasets.py --dataset sift1m --output_dir 1_Data/sift1m
python download_ann_datasets.py --dataset gist1m --output_dir 1_Data/gist1m
```

下载源: http://corpus-texmex.irisa.fr/ （注意：该站点有时不稳定，需备选镜像）

### 阶段 B：合成多租户标签

#### B1. 标签生成策略（推荐方案：K-means + 软分配）

SIFT1M/GIST1M 无天然标签，采用 K-means 聚类合成：

```
算法: synthesize_labels_kmeans
输入: vecs [n, d], n_labels=100, avg_labels_per_vec=3, seed=42
步骤:
  1. 对 vecs 运行 K-means (K=n_labels, 使用 FAISS)
  2. 对每个向量，计算到所有 K 个中心的距离
  3. 取距离最近的 avg_labels_per_vec 个中心 ID 作为该向量的标签
  4. 额外：确保每个标签至少覆盖 min(100, n/n_labels) 个向量（避免空标签）
输出: mds: list[list[int]], len=n
```

**设计理由**：
- K-means 生成的标签具有几何语义——同一标签的向量在空间中相邻，使得过滤检索有意义
- 每向量 3 个标签（类似 arxiv 的 ~2.8）提供适中的标签覆盖
- 标签选择性通过 avg_labels_per_vec 参数可控

**备选方案**：随机分配（`synthesize_labels_random`），用于消融实验验证标签语义对检索性能的影响。

#### B2. 标签合成脚本

新增 `1_Data/synthesize_labels.py`：

```python
def synthesize_labels_kmeans(
    vecs: np.ndarray,
    n_labels: int = 100,
    avg_labels_per_vec: int = 3,
    seed: int = 42,
) -> list[list[int]]:
    """使用 FAISS K-means 聚类 + 软分配生成多标签"""

def synthesize_labels_random(
    n_vectors: int,
    n_labels: int = 100,
    avg_labels_per_vec: int = 3,
    seed: int = 42,
) -> list[list[int]]:
    """随机均匀分配标签，保证每个标签的覆盖量大致相等"""
```

### 阶段 C：Ground Truth 计算

#### C1. 单标签 GT（修改现有逻辑）

现有 `compute_single_label_gt()` 假设候选集通过倒排索引获取。合成标签后该逻辑不变，直接复用。

**SIFT1M**：1M 训练向量 × 128 维，全量 GT 计算可行。

**GIST1M**：1M 训练向量 × 960 维，距离计算量大 7.5 倍。需使用分块策略（`gt_computing.py` 已有 BLOCK_SIZE=50000 的逻辑）。

#### C2. 集成脚本

新增 `1_Data/prepare_ann_dataset.py`，一键完成：

```bash
# 完整流水线
python prepare_ann_dataset.py --dataset sift1m \
    --n_labels 100 --avg_labels 3 --n_queries 1000 --k 10

python prepare_ann_dataset.py --dataset gist1m \
    --n_labels 100 --avg_labels 3 --n_queries 1000 --k 10
```

输出到 `1_Data/ground_truth/sift1m/` 和 `1_Data/ground_truth/gist1m/`。

### 阶段 D：配置与运行脚本

#### D1. 新增配置文件

- `3_Config/Curator/curator_sift1m.json`
- `3_Config/Curator/curator_gist1m.json`
- 各 baseline 方法的对应配置

**参数选择依据**：

| 参数 | SIFT1M (d=128) | GIST1M (d=960) | 理由 |
|------|---------------|----------------|------|
| nlist | 64 | 32 | GIST 高维聚类困难，减少分支因子；树深度受限公式见 .h:32-34 |
| pq_M | 16 (dsub=8) | 32 (dsub=30) | 需整除 d；GIST 高维需更多子空间 |
| pq_nbits | 8 | 8 | 标准配置 |
| max_sl_size | 128 | 128 | 与现有配置一致 |
| max_leaf_size | 128 | 128 | 与现有配置一致 |
| use_flash_storage | true | true | GIST 的全精度向量 3.8 GB，必须外存 |
| search_ef | 128~4096 | 128~4096 | 同现有 sweep 范围 |

#### D2. 更新 BUILTIN_DEFAULTS 和 sweep 脚本

在 `run_curator.py` 的 `BUILTIN_DEFAULTS` 字典中添加 sift1m/gist1m 条目。在 `run_curator_sweep.py` 和 `run_curator_build_sweep.py` 的 `--dataset choices` 中添加选项。

#### D3. 更新绘图脚本

修改 `5_Plot/fig1_sl_latency_recall.py` 等，添加新数据集的数据源路径。

### 阶段 E：测试验证

1. **数据下载完整性**：fvecs 文件 MD5 校验（如可用）
2. **fvecs 解析正确性**：首尾向量维度一致，向量值在合理范围
3. **合成标签质量**：每个标签覆盖 ≥ 100 个向量，标签选择性分布合理
4. **GT 正确性**：单标签 GT recall@10 = 1.0（排除 -1）
5. **全流水线运行**：`run_curator.py --dataset sift1m` 无报错完成
6. **结果合理性**：SIFT1M 基准 recall 应达 0.9+，与文献中 IVFPQ 方法的结果量级一致

---

## 三、预期效果

| 指标 | SIFT1M | GIST1M |
|------|--------|--------|
| 训练向量数 | 1,000,000 | 1,000,000 |
| 维度 | 128 | 960 |
| 合成标签数 | 100 | 100 |
| 查询数（SL） | 1,000（从 10K 中选择） | 1,000（从 1K 中选择） |
| 查询数（CP） | 100 × 50 filters | 100 × 50 filters |
| 数据目录 | 1_Data/ground_truth/sift1m/ | 1_Data/ground_truth/gist1m/ |
| 全精度向量文件大小 | 512 MB | 3,840 MB |
| 预期构建时间 | 2-5 分钟 | 15-45 分钟（取决于 Flash I/O） |
| 预期查询延迟 (P50) | < 5ms | < 50ms（高维距离计算量大） |
| 预期 recall@10 | > 0.90 | > 0.85 |

**扩展价值**：
- SIFT1M (d=128) 和 GIST1M (d=960) 与现有 arxiv (d=384) 和 yfcc100m (d=192) 形成完整的维度谱系
- SIFT1M 和 GIST1M 是 ANN 领域最广泛引用的 benchmark
- GIST1M 的高维场景特别考验 Curator 的剪枝效率和距离计算优化

---

## 四、验收方式

```bash
# 1. 数据准备（含下载、合成标签、GT 计算）
python 1_Data/prepare_ann_dataset.py --dataset sift1m --n_labels 100 --avg_labels 3
python 1_Data/prepare_ann_dataset.py --dataset gist1m --n_labels 100 --avg_labels 3

# 2. 单次 benchmark
python Curator/run_curator.py --dataset sift1m
python Curator/run_curator.py --dataset gist1m

# 3. 参数扫描
python Curator/run_curator_sweep.py --dataset sift1m \
    --config 3_Config/Curator/curator_sift1m.json \
    --sweep 3_Config/Curator/sweep.json

# 4. 交叉验证：各 baseline 方法也需能运行
python DiskIVF-PostFiltering/run_diskivf.py --dataset sift1m
python SPANN-PostFiltering/run_spann.py --dataset sift1m
python Pre-Filtering/run_prefiltering.py --dataset sift1m
```
