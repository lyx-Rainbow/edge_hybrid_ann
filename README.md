# Curator 向量检索基准测试项目

基于 FAISS v1.7.4 的 Curator（层次化聚类树索引）过滤向量检索基准测试平台，对比 DiskIVF-PostFiltering、Pre-Filtering 和 SPANN-PostFiltering 三种基线方法。

---

## 目录

- [环境信息](#环境信息)
- [快速开始（已有编译好的环境）](#快速开始已有编译好的环境)
- [第一步：编译 Curator C++ 扩展](#第一步编译-curator-c-扩展)
- [第二步：准备数据集](#第二步准备数据集)
- [第三步：运行基准测试](#第三步运行基准测试)
- [第四步：诊断与分析](#第四步诊断与分析)
- [项目结构](#项目结构)
- [配置参数说明](#配置参数说明)

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11 + WSL Ubuntu 24.04 |
| C++ 编译器 | GCC 13.3.0（需 C++17） |
| CMake | ≥ 3.20 |
| Conda 环境 | `edge_ann`（Python 3.10, faiss 1.7.4, numpy, swig） |
| FAISS 版本 | v1.7.4 |
| WSL 路径 | `/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines` |

**Python 依赖**（`edge_ann` 环境已包含）：
```
numpy, faiss-cpu==1.7.4, swig>=4.0
```

---

## 快速开始（已有编译好的环境）

如果 Curator C++ 扩展已编译安装，直接运行：

```bash
# 激活环境
source /home/lyx/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines

# 小规模快速验证（yfcc100m_small, 50K 向量, ~30 秒）
python Curator/run_curator.py --dataset yfcc100m_small

# 完整数据集（yfcc100m, 800K 向量, ~2 分钟构建）
python Curator/run_curator.py --dataset yfcc100m

# 使用配置文件
python Curator/run_curator.py --config 3_Config/Curator/curator_yfcc100m.json

# 参数扫描
python Curator/run_curator_sweep.py \
    --config 3_Config/Curator/curator_yfcc100m.json \
    --sweep 3_Config/Curator/sweep.json
```

或使用便捷脚本：
```bash
bash run.sh curator yfcc100m              # 单次基准测试
bash run.sh curator yfcc100m_small --sweep # 参数扫描
```

---

## 第一步：编译 Curator C++ 扩展

Curator 以 C++ 扩展形式嵌入 FAISS，需从源码编译。**仅在首次搭建环境或修改 C++ 源码后需要执行。**

### 1.1 一键编译（推荐）

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
chmod +x setup_curator.sh
./setup_curator.sh
```

此脚本自动完成：安装系统依赖 → 克隆 FAISS v1.7.4 → 复制 Curator 源文件 → 修改 CMake/SWIG → 编译 → 安装 Python 包。

### 1.2 分步编译（当一键脚本出错时）

参考 `analysis/goal3_wsl_build_guide.md` 中的详细步骤，关键流程如下：

```bash
# 激活环境
source /home/lyx/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
export PROJ=/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines

# 1. 部署 Curator C++ 源文件到 FAISS 源码树
cd ~/faiss_build/faiss
cp $PROJ/Curator/src/*.cpp   faiss/
cp $PROJ/Curator/src/*.h     faiss/
cp $PROJ/Curator/src/impl/*.h     faiss/impl/
cp $PROJ/Curator/src/impl/*.cpp   faiss/impl/

# 2. 兼容性修复（必须！）
sed -i 's/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD 17)/' CMakeLists.txt
sed -i 's/inline std::size_t hash_count()/inline std::size_t hash_count() const/' faiss/BloomFilter.h
# （完整补丁列表见 analysis/goal3_wsl_build_guide.md 第三步）

# 3. 更新 CMakeLists.txt（添加 Curator 源文件到 FAISS_SRC 和 FAISS_HEADERS）
# 4. 更新 swigfaiss.swig（暴露 Curator 类给 Python）
# 5. 编译
cd build
make -j$(nproc)

# 6. 安装 Python 包
cd faiss/python
SITE_PKGS=/home/lyx/miniconda3/envs/edge_ann/lib/python3.10/site-packages/faiss
rm -f $SITE_PKGS/_swigfaiss*.so
python setup.py install
cp ../libfaiss.so ../libfaiss_avx2.so libfaiss_python_callbacks.so $SITE_PKGS/
```

### 1.3 验证编译

```bash
cd /tmp
python -c "
import faiss
import numpy as np
idx = faiss.MultiTenantIndexIVFHierarchical(128, 16, faiss.METRIC_L2)
idx.set_pq_config(16, 8, True, False, 4)
bd = idx.get_memory_breakdown()
assert hasattr(bd, 'total_bytes')
idx.enable_profiling = True
assert idx.get_last_total_search_time_ms() >= 0
print('Curator build SUCCESS')
"
```

### 1.4 仅修改 Python 脚本时

Python 脚本位于 `/mnt/d/.../new-Baselines/` 下，修改后**直接生效**，无需重编译 C++。

---

## 第二步：准备数据集

### 2.1 已有数据集

以下数据集已预处理并位于 `1_Data/ground_truth/` 下，可直接使用：

| 数据集 | 向量数 | 维度 | 标签数 | 用途 |
|--------|--------|:---:|:---:|------|
| `yfcc100m_small` | 50,000 | 192 | ~200 | 快速验证 |
| `yfcc100m` | 800,000 | 192 | 1,000 | 完整基准 |
| `arxiv_small` | 50,000 | 384 | ~80 | 快速验证 |
| `arxiv` | 1,600,000 | 384 | 100 | 完整基准 |

### 2.2 新增 SIFT1M / GIST1M 数据集

```bash
# 下载原始数据（在 WSL 中执行）
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines

# 下载 SIFT1M（~500 MB）
python 1_Data/download_ann_datasets.py --dataset sift1m --output_dir 1_Data

# 下载 GIST1M（~3.8 GB）
python 1_Data/download_ann_datasets.py --dataset gist1m --output_dir 1_Data

# 预处理：合成标签 + 选取查询 + 计算 Ground Truth
# 小规模验证模式（快速，~30 秒）
python 1_Data/prepare_ann_dataset.py --dataset sift1m --small

# 完整模式
python 1_Data/prepare_ann_dataset.py --dataset sift1m
python 1_Data/prepare_ann_dataset.py --dataset gist1m

# 自定义参数
python 1_Data/prepare_ann_dataset.py --dataset sift1m \
    --n_labels 100 --avg_labels 3 --n_queries 1000
```

预处理输出位于 `1_Data/ground_truth/{sift1m|gist1m}[_small]/`，包含：
- `train_vecs.npy` / `train_mds.pkl` — 训练向量与访问控制列表
- `query_vecs.npy` / `query_labels.npy` / `query_info.json` — 查询向量、单标签和选择率分桶信息
- `ground_truth.npy` — 预计算的单标签 k-NN 真值
- `complex_predicate/` — 复杂谓词（AND/OR/NOT）真值（可选）

---

## 第三步：运行基准测试

### 3.1 Curator 单次运行

```bash
# 使用内置默认参数
python Curator/run_curator.py --dataset yfcc100m_small

# 使用 JSON 配置文件
python Curator/run_curator.py --config 3_Config/Curator/curator_yfcc100m.json

# 启用详细 profiling（内存分解 + 查询步骤耗时）
python Curator/run_curator.py --dataset yfcc100m_small --profile

# 覆盖特定参数
python Curator/run_curator.py --dataset yfcc100m --nlist 64 --k 10
```

输出保存到 `4_Results/Curator/curator_{dataset}.json`。

### 3.2 Curator 参数扫描

```bash
# 搜索参数扫描
python Curator/run_curator_sweep.py \
    --config 3_Config/Curator/curator_yfcc100m.json \
    --sweep 3_Config/Curator/sweep.json

# 构建参数扫描（PQ_M, PQ_nbits）
python Curator/run_curator_build_sweep.py \
    --config 3_Config/Curator/curator_yfcc100m_small.json \
    --sweep 3_Config/Curator/build_sweep.json
```

### 3.3 其他基线方法

```bash
# Pre-Filtering
python Pre-Filtering/run_prefiltering.py --dataset yfcc100m_small

# DiskIVF-PostFiltering
python DiskIVF-PostFiltering/run_diskivf.py --dataset yfcc100m_small

# SPANN-PostFiltering
python SPANN-PostFiltering/run_spann.py --dataset yfcc100m_small
```

### 3.4 便捷脚本

```bash
bash run.sh curator yfcc100m                  # Curator 完整基准
bash run.sh pre-filter yfcc100m_small          # Pre-Filtering 小规模
bash run.sh diskivf arxiv_small --sweep        # DiskIVF 参数扫描
```

---

## 第四步：诊断与分析

### 4.1 内存分析

```bash
# 构建阶段组件级内存分解
python tests/profile_memory_components.py --dataset yfcc100m_small

# 内存正确性诊断
python tests/diagnose_memory.py --dataset yfcc100m_small
```

输出保存到 `4_Results/memory_components/memory_{dataset}.json`。

### 4.2 查询耗时分析

```bash
# 查询各步骤耗时分解（C++ 内部 profiling）
python tests/profile_query_steps.py --dataset yfcc100m_small --n_queries 50
```

输出保存到 `4_Results/query_profile/query_profile_{dataset}.json`。

### 4.3 构建耗时分析

```bash
# 构建各阶段耗时分解
python tests/diagnose_build_time.py --dataset yfcc100m_small
```

输出保存到 `4_Results/build_time_correct/curator_{dataset}.json`。

### 4.4 结果分析文档

详细的实验结果分析见 [4_Results/RESULTS_ANALYSIS.md](4_Results/RESULTS_ANALYSIS.md)。

---

## 项目结构

```
new-Baselines/
├── README.md                         # 本文件
├── setup_curator.sh                  # Curator C++ 一键编译脚本
├── run.sh                            # 便捷运行脚本
│
├── Curator/                          # Curator 索引
│   ├── src/                          #   C++ 源码（需合并到 FAISS）
│   │   ├── MultiTenantIndexIVFHierarchical.h/.cpp  # 核心索引
│   │   ├── BloomFilter.h             #     Bloom Filter（header-only）
│   │   ├── complex_predicate.h/.cpp  #     复杂谓词引擎
│   │   ├── impl/IDSelector.h/.cpp    #     ⚠️ 覆盖 FAISS 原文件
│   │   ├── MetricType.h              #     ⚠️ 覆盖 FAISS 原文件
│   │   └── ...
│   ├── python/                       #   Python 封装
│   │   ├── curator.py                #     Curator API
│   │   ├── base.py                   #     索引基类
│   │   └── hybrid_curator.py         #     HybridCurator API
│   ├── run_curator.py                #   单次基准测试入口
│   ├── run_curator_sweep.py          #   参数扫描入口
│   └── run_curator_build_sweep.py    #   构建参数扫描入口
│
├── DiskIVF-PostFiltering/            # DiskIVF 基线
├── Pre-Filtering/                    # Pre-Filtering 基线
├── SPANN-PostFiltering/              # SPANN 基线
│
├── 1_Data/                           # 数据目录
│   ├── ground_truth/                 #   预处理后的数据集
│   │   ├── yfcc100m/                 #     yfcc100m (800K)
│   │   ├── yfcc100m_small/           #     yfcc100m_small (50K)
│   │   ├── arxiv/                    #     arxiv (1.6M)
│   │   └── arxiv_small/              #     arxiv_small (50K)
│   ├── sift1m/                       #   SIFT1M 原始数据
│   ├── gist1m/                       #   GIST1M 原始数据
│   ├── download_ann_datasets.py      #   数据集下载
│   ├── prepare_ann_dataset.py        #   数据预处理流水线
│   ├── synthesize_labels.py          #   标签合成
│   └── io_utils.py                   #   fvecs/ivecs 读写
│
├── 2_Utils/                          # 工具模块
│   ├── memory_utils.py               #   RSS 测量
│   ├── memory_profiler.py            #   组件级内存分析
│   ├── query_profiler.py             #   查询步骤分析
│   └── predicate.py                  #   复杂谓词解析
│
├── 3_Config/                         # 配置文件
│   └── Curator/                      #   Curator 配置（每数据集一个 JSON）
│
├── 4_Results/                        # 实验结果
│   ├── RESULTS_ANALYSIS.md           #   结果分析文档
│   ├── Curator/                      #   Curator 基准结果
│   ├── memory_components/            #   组件级内存分解
│   ├── query_profile/                #   查询耗时分解
│   ├── memory_correct/               #   内存正确性验证
│   └── build_time_correct/           #   构建耗时分解
│
├── 5_Plot/                           # 可视化脚本
├── analysis/                         # 设计文档与编译指南
│   ├── goal1_pq_external_storage.md  #   PQ 码外存设计
│   ├── goal2_sift1m_gist1m_datasets.md
│   ├── goal3_monitoring_profiling.md #   内存/查询监控设计
│   ├── goal3_verification_plan.md    #   验收计划
│   ├── goal3_wsl_build_guide.md      #   WSL 编译详细指南 ⚠️ 关键
│   └── wsl_environment_reference.md  #   WSL 环境速查
│
└── tests/                            # 诊断脚本
    ├── profile_memory_components.py  #   内存分解
    ├── profile_query_steps.py        #   查询耗时分解
    ├── diagnose_memory.py            #   内存正确性
    └── diagnose_build_time.py        #   构建耗时
```

---

## 配置参数说明

Curator 配置文件（JSON 格式）包含三部分：

### build — 构建参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `nlist` | int | 聚类中心数（树的分支因子） |
| `bf_capacity` | int | Bloom Filter 预期元素数 |
| `bf_error_rate` | float | Bloom Filter 误报率 |
| `max_sl_size` | int | 单节点单租户最大短列表大小 |
| `clus_niter` | int | K-means 聚类迭代次数 |
| `max_leaf_size` | int | 叶子节点最大向量数 |
| `pq_M` | int | PQ 子空间数 |
| `pq_nbits` | int | PQ 每个子空间的量化位数 |
| `pq_enabled` | bool | 是否启用 PQ 压缩 |
| `pq_use_adc_rerank` | bool | 是否启用 ADC 重排序 |
| `pq_rerank_topk_factor` | int | ADC 重排序候选倍数 |
| `use_flash_storage` | bool | 是否将全精度向量持久化到磁盘 |

### search — 搜索参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `variance_boost` | float | 方差提升系数（控制剪枝保守度） |
| `search_ef` | int | 搜索扩展因子（越大召回越高越慢） |
| `beam_size` | int | Beam Search 束宽度 |
| `use_temp_index_caching` | bool | 是否缓存临时索引结构 |

### benchmark — 测试参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `k` | int | 返回的最近邻数量 |
| `n_queries` | int | 查询数量 |
| `num_warmup` | int | 预热查询数 |
