# 数据集说明文档

本文档详细描述本项目使用的两个核心数据集：**Arxiv** 和 **YFCC-1M 子集**，包括数据来源、内部结构、预处理流程、读取方式以及如何迁移到另一个项目。

---

## 目录

- [1. 数据格式约定](#1-数据格式约定)
- [2. Arxiv 数据集](#2-arxiv-数据集)
- [3. YFCC-1M 子集数据集](#3-yfcc-1m-子集数据集)
- [4. 预处理缓存格式](#4-预处理缓存格式)
- [5. 数据集迁移指南](#5-数据集迁移指南)
- [6. 完整示例：在新项目中加载数据](#6-完整示例在新项目中加载数据)

---

## 1. 数据格式约定

两个数据集共享统一的 Python 接口：

```python
# 向量数据
train_vecs: np.ndarray  # float32, shape [n_train, d]
test_vecs:  np.ndarray  # float32, shape [n_test,  d]

# 元数据（多标签访问控制列表）
train_mds: list[list[int]]  # 长度 n_train，每个元素是标签 ID 列表
test_mds:  list[list[int]]  # 长度 n_test， 每个元素是标签 ID 列表

# 距离度量
metric: "euclidean"  # L2 距离
```

**元数据语义**：`train_mds[i] = [1, 5, 23]` 表示第 `i` 个向量可以被租户（标签）1、5、23 访问。这是**多标签多租户过滤检索**的核心：查询时限定租户，只搜索该租户有权访问的向量。

---

## 2. Arxiv 数据集

### 2.1 数据集概述

| 属性 | 值 |
|------|-----|
| 全名 | arXiv.org 学术论文元数据 |
| 向量维度 (d) | 384 |
| 向量总数 | 2,000,000 |
| 标签类别数 | 100（arXiv 学科分类） |
| 平均标签数/向量 | ~2-3（一篇论文属于 2-3 个学科） |
| 距离度量 | L2 Euclidean |
| 向量类型 | float32 |
| 训练/测试划分 | 80% / 20%（随机排列后切分） |

### 2.2 原始数据来源

数据集基于 Kaggle 上的 **arXiv Dataset**（Cornell University）：

- **下载地址**：https://www.kaggle.com/datasets/Cornell-University/arxiv
- **原始文件**：`arxiv-metadata-oai-snapshot.json`（约 4.9 GB）
- **内容**：包含 arXiv 上所有论文的元数据，每条记录包含论文 ID、标题、摘要、学科分类（如 `cs.AI`, `math.OC`, `stat.ML` 等）

### 2.3 数据生成流程

整个预处理流程由 `dataset/arxiv_dataset.py` 实现，分为 4 步：

#### 步骤 1：下载原始数据

```bash
# 从 Kaggle 下载 arxiv-metadata-oai-snapshot.json
# 放置到 data/arxiv/ 目录
```

或通过 HuggingFace datasets 库自动下载（代码中已配置）：
```python
from datasets import load_dataset
dataset = load_dataset("arxiv_dataset", data_dir="data/arxiv")
```

#### 步骤 2：筛选和预处理

运行 `preprocess_arxiv_dataset()` 函数：

```python
from dataset.arxiv_dataset import preprocess_arxiv_dataset

preprocess_arxiv_dataset(
    num_categories=100,      # 选取最常见的 100 个学科分类作为标签
    sample_size=2000000,     # 采样 200 万篇论文
    seed=42,
    data_dir="data/arxiv",
    output_path="data/arxiv/processed_user100_vec2e6.pkl",
)
```

**内部逻辑**：
1. 统计所有学科分类的出现次数，取 top-100
2. 遍历所有论文，保留至少拥有一个 top-100 分类标签的论文
3. 每条记录保留字段：`id`（论文 ID）、`categories`（学科分类列表）、`abstract`（摘要文本）
4. 输出为 Python pickle 文件（约 2.0 GB）

**输出文件**：`data/arxiv/processed_user100_vec2e6.pkl`

#### 步骤 3：生成向量嵌入

运行 `gen_arxiv_embeddings()` 函数：

```python
from dataset.arxiv_dataset import gen_arxiv_embeddings

gen_arxiv_embeddings(
    batch_size=32,
    pkl_path="data/arxiv/processed_user100_vec2e6.pkl",
    embed_path="data/arxiv/embeddings_user100_vec2e6.npy",
)
```

**内部逻辑**：
1. 加载步骤 2 的 pkl 文件
2. 使用 `sentence-transformers/all-MiniLM-L6-v2` 模型（384 维）将每篇论文的摘要编码为向量
3. 输出为 `.npy` 文件，shape 为 `[2000000, 384]`，dtype 为 float32

**依赖**：
```bash
pip install torch transformers sentence-transformers
```

**输出文件**：`data/arxiv/embeddings_user100_vec2e6.npy`（约 2.9 GB）

#### 步骤 4：生成元数据（标签访问列表）

运行 `load_arxiv_dataset_mds()` 函数：

```python
from dataset.arxiv_dataset import load_arxiv_dataset_mds

train_mds, test_mds = load_arxiv_dataset_mds(
    pkl_path="data/arxiv/processed_user100_vec2e6.pkl",
    share_degree=None,  # 保持原始平均标签数；传入整数可扩展标签数
    test_size=0.2,
    seed=42,
)
```

**内部逻辑**：
1. 加载 pkl 文件中的预处理记录
2. 将 100 个学科分类映射为整数 ID（0-99）
3. 为每个向量生成标签 ID 列表（论文 → 其所属学科分类的 ID）
4. 如果指定 `share_degree`（如 10），则将平均标签数扩展至目标值（通过复制标签映射实现）
5. 随机排列后切分为 train/test（80/20）
6. **重要**：测试集的元数据会被**随机重新生成**（模拟冷启动场景）

### 2.4 最终数据文件

| 文件 | 大小 | 格式 | 说明 |
|------|------|------|------|
| `arxiv-metadata-oai-snapshot.json` | 4.9 GB | JSON | 原始 arXiv 元数据 |
| `processed_user100_vec2e6.pkl` | 2.0 GB | Pickle | Python `list[dict]`，已筛选的论文条目 |
| `embeddings_user100_vec2e6.npy` | 2.9 GB | NumPy | float32, shape=[2000000, 384] |

### 2.5 读取方式

```python
import numpy as np
from dataset.arxiv_dataset import load_arxiv_dataset_vecs, load_arxiv_dataset_mds

# 加载向量（返回已划分好的 train/test）
train_vecs, test_vecs = load_arxiv_dataset_vecs(
    embed_path="data/arxiv/embeddings_user100_vec2e6.npy",
    test_size=0.2,
    seed=42,
)
# train_vecs: float32, shape [1600000, 384]
# test_vecs:  float32, shape [400000, 384]

# 加载元数据
train_mds, test_mds = load_arxiv_dataset_mds(
    pkl_path="data/arxiv/processed_user100_vec2e6.pkl",
    test_size=0.2,
    seed=42,
)
# train_mds: list[list[int]], len=1600000
# test_mds:  list[list[int]], len=400000

print(f"维度: {train_vecs.shape[1]}")
print(f"训练集向量数: {len(train_vecs)}")
print(f"测试集向量数: {len(test_vecs)}")
print(f"唯一标签数: {len(set().union(*train_mds))}")
print(f"示例元数据: {train_mds[0]}")  # 如 [3, 17, 42]
```

---

## 3. YFCC-1M 子集数据集

### 3.1 数据集概述

| 属性 | 值 |
|------|-----|
| 全名 | Yahoo Flickr Creative Commons 100M（子采样） |
| 向量维度 (d) | 192 |
| 向量总数 | 1,000,000 / 10,000,000（可选） |
| 标签类别数 | 1,000（用户标签/关键词） |
| 平均标签数/向量 | ~20-50 |
| 距离度量 | L2 Euclidean |
| 向量类型 | float32（归一化到 [-0.5, 0.5]） |
| 训练/测试划分 | 80% / 20%（随机排列后切分） |
| 原始格式 | uint8（取值范围 0-255） |

### 3.2 原始数据来源

数据集来自 **Billion-Scale ANN Benchmarks** 项目发布的 YFCC100M 子集：

**下载地址**：
- 向量文件：https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/base.10M.u8bin
- 元数据文件：https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/base.metadata.10M.spmat

**下载命令**：
```bash
mkdir -p data/yfcc100m
cd data/yfcc100m
wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/base.10M.u8bin
wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/base.metadata.10M.spmat
```

### 3.3 原始文件格式

#### base.10M.u8bin（向量文件）

自定义二进制格式：

| 偏移 | 类型 | 内容 |
|------|------|------|
| 0 | uint32 | 向量数量 n（= 10,000,000） |
| 4 | uint32 | 向量维度 d（= 192） |
| 8 | uint8[n*d] | 原始向量数据（uint8，每像素值 0-255） |

```python
def load_vecs(path):
    n, d = map(int, np.fromfile(path, dtype="uint32", count=2))
    vecs = np.memmap(path, dtype="uint8", mode="r", offset=8, shape=(n, d))
    return np.ascontiguousarray(vecs)  # shape [10000000, 192]
```

#### base.metadata.10M.spmat（元数据文件）

自定义 CSR（Compressed Sparse Row）稀疏矩阵格式：

| 偏移 | 类型 | 内容 |
|------|------|------|
| 0 | int64 | 行数 nrow |
| 8 | int64 | 列数 ncol |
| 16 | int64 | 非零元素数 nnz |
| 24 | int64[nrow+1] | CSR indptr 数组 |
| 24 + 8*(nrow+1) | int32[nnz] | CSR indices 数组 |
| ... | float32[nnz] | CSR data 数组（权重，可忽略） |

```python
def read_sparse_matrix(fname):
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype="int64", count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
        indices = np.fromfile(f, dtype="int32", count=nnz)
        data = np.fromfile(f, dtype="float32", count=nnz)
    return csr_matrix((data, indices, indptr), shape=(nrow, ncol))
```

**元数据语义**：稀疏矩阵的每一行对应一个向量，该行所有非零列索引即为该向量所属的用户标签 ID。

### 3.4 数据生成流程

由 `dataset/yfcc100m_dataset.py` 实现：

#### 步骤 1：子采样

```python
from dataset.yfcc100m_dataset import subsample_yfcc_dataset

# 1M 向量子集（733 MB 向量 + 41 MB 元数据）
subsample_yfcc_dataset(n_vectors=1_000_000, n_labels=1000, save_dir="data/yfcc100m")

# 10M 向量子集（7.2 GB 向量 + 194 MB 元数据）
subsample_yfcc_dataset(n_vectors=10_000_000, n_labels=1000, save_dir="data/yfcc100m")
```

**内部逻辑**：
1. 加载原始 `base.10M.u8bin` 和 `base.metadata.10M.spmat`
2. 选取非零元素最多的 n 行和 m 列（确保标签覆盖度最高）
3. 向量类型转换：`vecs_float32 = vecs_uint8 / 255.0 - 0.5`（归一化到 [-0.5, 0.5]）
4. 将 CSR 稀疏矩阵转换为 `list[list[int]]` 格式（每行是一个标签 ID 列表）
5. 保存为 `.npy`（向量）和 `.pkl`（元数据）

#### 步骤 2：划分训练/测试集

由加载函数内部完成：

```python
from dataset.yfcc100m_dataset import load_yfcc_dataset_vecs, load_yfcc_dataset_mds

train_vecs, test_vecs = load_yfcc_dataset_vecs(
    data_path="data/yfcc100m/yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy",
    test_size=0.2,
    seed=42,
)

train_mds, test_mds = load_yfcc_dataset_mds(
    data_path="data/yfcc100m/yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl",
    test_size=0.2,
    seed=42,
)
```

### 3.5 最终数据文件

| 文件 | 大小 | 格式 | 说明 |
|------|------|------|------|
| `base.10M.u8bin` | 1.8 GB | 自定义二进制 | 原始 YFCC-10M uint8 向量 |
| `base.metadata.10M.spmat` | 902 MB | 自定义 CSR | 原始标签稀疏矩阵 |
| `yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy` | 733 MB | NumPy | float32, shape=[1000000, 192] |
| `yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl` | 41 MB | Pickle | list[list[int]], len=1000000 |
| `yfcc_subsampled_nvec_10000000_nlabel_1000_vecs.npy` | 7.2 GB | NumPy | float32, shape=[10000000, 192] |
| `yfcc_subsampled_nvec_10000000_nlabel_1000_mds.pkl` | 194 MB | Pickle | list[list[int]], len=10000000 |

### 3.6 读取方式

```python
import numpy as np
from dataset.yfcc100m_dataset import (
    load_yfcc_dataset_vecs,
    load_yfcc_dataset_mds,
    load_raw_yfcc_dataset,
)

# 方式 A：加载已采样的子集（推荐）
train_vecs, test_vecs = load_yfcc_dataset_vecs(
    data_path="data/yfcc100m/yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy",
    test_size=0.2,
    seed=42,
)
train_mds, test_mds = load_yfcc_dataset_mds(
    data_path="data/yfcc100m/yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl",
    test_size=0.2,
    seed=42,
)

print(f"维度: {train_vecs.shape[1]}")
print(f"训练集向量数: {len(train_vecs)}")
print(f"唯一标签数: {len(set().union(*train_mds))}")

# 方式 B：直接加载原始 YFCC 10M 数据
from scipy.sparse import csr_matrix
vecs, mds_csr = load_raw_yfcc_dataset()
# vecs: np.ndarray, uint8, shape [10000000, 192]
# mds_csr: scipy.sparse.csr_matrix
```

---

## 4. 预处理缓存格式

为了加速 benchmark 迭代，`data/preprocessed/` 目录保存了完整处理后的数据集快照。目录名格式为 `{dataset_key}_test{test_size}` 或 `{dataset_key}_test{test_size}_sub{subsample_ratio}`。

### 4.1 目录内容

以 `data/preprocessed/yfcc100m_test0.001/` 为例：

| 文件 | 大小（示例） | 格式 | 说明 |
|------|------|------|------|
| `train_vecs.npy` | ~732 MB | NumPy | float32 训练向量 |
| `test_vecs.npy` | ~768 KB | NumPy | float32 测试向量 |
| `train_mds.pkl` | ~42 MB | Pickle | list[list[int]] 训练元数据 |
| `test_mds.pkl` | ~42 KB | Pickle | list[list[int]] 测试元数据 |
| `ground_truth.npy` | ~1 MB | NumPy | 预计算的 k-NN 真值（值-1 表示未找到 k 个合格邻居） |
| `all_labels.json` | ~5 KB | JSON | `set[int]`，所有标签 ID 的集合 |
| `metadata.pkl` | ~74 B | Pickle | dict, 包含 `{"metric": ..., "dimension": ...}` |

### 4.2 生成预处理缓存

```python
from dataset import get_dataset, get_metadata
from dataset.utils import compute_ground_truth, load_sampled_metadata

# 加载向量和元数据
train_vecs, test_vecs, meta = get_dataset("yfcc100m", test_size=0.001)
train_mds, test_mds = get_metadata(dataset_name="yfcc100m", test_size=0.001)

# 计算 ground truth
gt, train_cates = compute_ground_truth(
    train_vecs, train_mds, test_vecs, test_mds,
    k=10, multi_tenant=True,
)

# 过滤元数据（只保留在 train 中实际出现的标签）
train_mds, test_mds = load_sampled_metadata(train_mds, test_mds, train_cates)
```

### 4.3 数据集键名对照表

benchmark 中使用下列键名引用数据集：

| 键名 | 向量数 | 标签数 | 维度 | 说明 |
|------|--------|--------|------|------|
| `arxiv-small` | ~40K | 100 | 384 | Arxiv 小样本 |
| `arxiv-large` 或 `arxiv-large-10` | 2M | 100 | 384 | Arxiv 200 万，share_degree=10 |
| `yfcc100m` | 1M | 1,000 | 192 | YFCC 1M 子集 |
| `yfcc100m-10m` | 10M | 1,000 | 192 | YFCC 10M 子集 |

---

## 5. 数据集迁移指南

### 5.1 迁移策略

根据目标项目的需求，有两种迁移方式：

#### 方式 A：复制已处理文件（推荐，快速）

直接将预处理后的最终 `.npy` 和 `.pkl` 文件复制到新项目。不需要重新执行预处理流程。

**最少需要复制的文件**：

```
new_project/data/arxiv/
├── processed_user100_vec2e6.pkl     # 筛选后的论文元数据
└── embeddings_user100_vec2e6.npy   # 向量嵌入

new_project/data/yfcc100m/
├── yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy   # 1M 向量
└── yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl    # 1M 元数据
```

> 如果使用 10M YFCC 子集，复制对应 `nvec_10000000` 文件即可。

**复制命令**：
```bash
# 从 curator-v2 项目复制到新项目
SRC="d:/23235/Documents/Aftergraduate/experiments/curator-v2/data"
DST="/path/to/new_project/data"

mkdir -p "$DST/arxiv" "$DST/yfcc100m"

# Arxiv
cp "$SRC/arxiv/processed_user100_vec2e6.pkl" "$DST/arxiv/"
cp "$SRC/arxiv/embeddings_user100_vec2e6.npy" "$DST/arxiv/"

# YFCC-1M
cp "$SRC/yfcc100m/yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy" "$DST/yfcc100m/"
cp "$SRC/yfcc100m/yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl" "$DST/yfcc100m/"
```

#### 方式 B：从原始数据重新生成（当需要不同采样参数时）

如果需要在不同采样参数下使用数据（如不同数量的标签、不同向量数量），则需下载原始数据并重新走预处理流程。

### 5.2 在新项目中加载数据（无依赖方式）

如果不想复制整套 `dataset/` 模块，以下是最小化自包含加载代码：

```python
import pickle
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

def load_arxiv(data_dir=DATA_DIR):
    """加载 Arxiv 数据集（仅需 .npy 和 .pkl 文件）"""
    data_dir = Path(data_dir) / "arxiv"
    vecs = np.load(data_dir / "embeddings_user100_vec2e6.npy").astype(np.float32)

    with open(data_dir / "processed_user100_vec2e6.pkl", "rb") as f:
        entries = pickle.load(f)

    # 构建标签映射
    categories = sorted(set().union(*(e["categories"] for e in entries)))
    cat2id = {c: i for i, c in enumerate(categories)}

    # 生成元数据
    mds = [[cat2id[c] for c in e["categories"]] for e in entries]

    # 划分 train/test
    n = len(vecs)
    np.random.seed(42)
    idx = np.random.permutation(n)
    train_n = int(n * 0.8)
    train_vecs = vecs[idx[:train_n]]
    test_vecs = vecs[idx[train_n:]]
    train_mds = [mds[i] for i in idx[:train_n]]
    test_mds = [mds[i] for i in idx[train_n:]]

    return train_vecs, test_vecs, train_mds, test_mds, len(categories)


def load_yfcc1m(data_dir=DATA_DIR):
    """加载 YFCC-1M 子集（仅需 .npy 和 .pkl 文件）"""
    data_dir = Path(data_dir) / "yfcc100m"

    vecs = np.load(data_dir / "yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy")
    vecs = vecs.astype(np.float32)

    with open(data_dir / "yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl", "rb") as f:
        mds = pickle.load(f)

    # 划分 train/test
    n = len(vecs)
    np.random.seed(42)
    idx = np.random.permutation(n)
    train_n = int(n * 0.8)
    train_vecs = vecs[idx[:train_n]]
    test_vecs = vecs[idx[train_n:]]
    train_mds = [mds[i] for i in idx[:train_n]]
    test_mds = [mds[i] for i in idx[train_n:]]

    return train_vecs, test_vecs, train_mds, test_mds


# 使用示例
train_vecs, test_vecs, train_mds, test_mds, n_labels = load_arxiv()
print(f"Arxiv: {len(train_vecs)} train, {len(test_vecs)} test, {n_labels} labels, dim={train_vecs.shape[1]}")

train_vecs, test_vecs, train_mds, test_mds = load_yfcc1m()
print(f"YFCC-1M: {len(train_vecs)} train, {len(test_vecs)} test, dim={train_vecs.shape[1]}")
```

### 5.3 Python 依赖

读取已处理文件仅需：
```bash
pip install numpy
```

如需重新生成，额外需要：
```bash
pip install torch transformers sentence-transformers  # Arxiv 嵌入生成
pip install scipy                                      # YFCC 稀疏矩阵读取
pip install datasets                                   # Arxiv 原始数据加载（HuggingFace）
```

---

## 6. 完整示例：在新项目中加载数据

```python
"""新项目中使用 Arxiv / YFCC-1M 数据的完整示例"""

import numpy as np
from pathlib import Path


def load_dataset(dataset_name: str, data_dir: str = "data"):
    """
    统一的数据加载接口。

    Parameters
    ----------
    dataset_name : str
        "arxiv" 或 "yfcc1m"
    data_dir : str
        数据根目录，默认为 "data"

    Returns
    -------
    train_vecs : np.ndarray  (float32, [n_train, d])
    test_vecs  : np.ndarray  (float32, [n_test, d])
    train_mds  : list[list[int]]
    test_mds   : list[list[int]]
    meta       : dict  {"dim", "n_labels", "dataset"}
    """
    if dataset_name == "arxiv":
        return _load_arxiv(data_dir)
    elif dataset_name == "yfcc1m":
        return _load_yfcc1m(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_arxiv(data_dir: str):
    import pickle

    data = Path(data_dir) / "arxiv"
    vecs = np.load(data / "embeddings_user100_vec2e6.npy").astype(np.float32)
    with open(data / "processed_user100_vec2e6.pkl", "rb") as f:
        entries = pickle.load(f)

    cats = sorted(set().union(*(e["categories"] for e in entries)))
    cat2id = {c: i for i, c in enumerate(cats)}
    mds = [[cat2id[c] for c in e["categories"]] for e in entries]

    n = len(vecs)
    np.random.seed(42)
    idx = np.random.permutation(n)
    n_train = int(n * 0.8)

    return (
        vecs[idx[:n_train]],
        vecs[idx[n_train:]],
        [mds[i] for i in idx[:n_train]],
        [mds[i] for i in idx[n_train:]],
        {"dim": 384, "n_labels": len(cats), "dataset": "arxiv"},
    )


def _load_yfcc1m(data_dir: str):
    import pickle

    data = Path(data_dir) / "yfcc100m"
    vecs = np.load(data / "yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy")
    vecs = vecs.astype(np.float32)
    with open(data / "yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl", "rb") as f:
        mds = pickle.load(f)

    n = len(vecs)
    np.random.seed(42)
    idx = np.random.permutation(n)
    n_train = int(n * 0.8)

    return (
        vecs[idx[:n_train]],
        vecs[idx[n_train:]],
        [mds[i] for i in idx[:n_train]],
        [mds[i] for i in idx[n_train:]],
        {"dim": 192, "n_labels": 1000, "dataset": "yfcc1m"},
    )


# ===== 使用示例 =====
if __name__ == "__main__":
    for ds_name in ["arxiv", "yfcc1m"]:
        train_v, test_v, train_m, test_m, meta = load_dataset(ds_name)
        print(f"\n=== {meta['dataset'].upper()} ===")
        print(f"  Vectors:  {len(train_v):,} train + {len(test_v):,} test")
        print(f"  Dimension: {meta['dim']}")
        print(f"  Labels:    {meta['n_labels']}")
        print(f"  Avg labels/vec: {sum(len(m) for m in train_m) / len(train_m):.1f}")
        print(f"  Memory:    {train_v.nbytes / 1e9:.2f} GB (train vecs)")
```

---

## 附录 A：文件大小汇总

| 文件 | 大小 | 是否必须迁移 |
|------|------|:---:|
| `arxiv/arxiv-metadata-oai-snapshot.json` | 4.9 GB | 否 — 原始数据 |
| `arxiv/processed_user100_vec2e6.pkl` | 2.0 GB | **是** — 核心 |
| `arxiv/embeddings_user100_vec2e6.npy` | 2.9 GB | **是** — 核心 |
| `yfcc100m/base.10M.u8bin` | 1.8 GB | 否 — 原始数据 |
| `yfcc100m/base.metadata.10M.spmat` | 902 MB | 否 — 原始数据 |
| `yfcc100m/yfcc_subsampled_nvec_1000000_*_vecs.npy` | 733 MB | **推荐**（1M 子集） |
| `yfcc100m/yfcc_subsampled_nvec_1000000_*_mds.pkl` | 41 MB | **推荐**（1M 子集） |
| `yfcc100m/yfcc_subsampled_nvec_10000000_*_vecs.npy` | 7.2 GB | 可选（10M 子集） |
| `yfcc100m/yfcc_subsampled_nvec_10000000_*_mds.pkl` | 194 MB | 可选（10M 子集） |

## 附录 B：标签分布统计（参考值）

| 数据集 | 标签数 | 平均标签/向量 | 最大标签/向量 | 标签选择性中位数 |
|--------|--------|---------------|---------------|------------------|
| Arxiv (2M) | 100 | ~2.8 | ~15 | ~2.8% |
| YFCC-1M | 1,000 | ~31 | ~200 | ~0.25% |
| YFCC-10M | 1,000 | ~31 | ~200 | ~0.25% |

> **标签选择性** = 拥有该标签的向量数 / 总向量数。选择性越低，过滤检索的搜索空间缩得越小，本论文关注的正是高选择性（低选择性值）场景下的高效检索。
