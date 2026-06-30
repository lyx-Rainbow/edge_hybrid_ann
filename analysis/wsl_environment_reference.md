# WSL 环境参考手册

> 本项目在 Windows IDE 中编辑代码，在 WSL-Ubuntu 终端中编译和运行。
> IDE 中的 bash 工具可通过 `wsl bash -c "..."` 调用 WSL 命令。
>
> 最后更新：2026-06-15（编译流程已验证通过）

---

## 基本环境信息

| 项目 | 值 |
|------|-----|
| WSL 用户名 | `lyx` |
| WSL 家目录 | `/home/lyx/` |
| Conda 路径 | `/home/lyx/miniconda3/` |
| **工作环境** | **`edge_ann`**（conda activate edge_ann） |
| 本项目 (WSL 视角) | `/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines` |
| 本项目 (Windows 视角) | `d:\23235\Documents\Aftergraduate\experiments\new-Baselines` |
| GCC 版本 | 13.3.0（默认 C++17） |
| CMake 版本 | 3.28 |

### 可用 Conda 环境

| 环境名 | Python | faiss | numpy | swig | 用途 |
|--------|--------|-------|-------|------|------|
| **edge_ann** | 3.10 | 1.7.4 | 1.26.4 | 4.0.1 | **本项目工作环境** ✅ |
| ann_bench | 3.10 | 1.7.4 | 1.25.0 | 无 | 备用（缺少 swig） |

---

## FAISS 源码与编译路径

```
/home/lyx/faiss_build/faiss/               # FAISS 源码根目录
├── CMakeLists.txt                            # ← 根 CMake（需改 CXX_STANDARD 11→17）
├── faiss/                                    # C++ 源文件目录
│   ├── CMakeLists.txt                        # ← 需添加 Curator 源文件
│   ├── MultiTenantIndexIVFHierarchical.h     # ← Curator 核心头文件（含 Goal3 修改）
│   ├── MultiTenantIndexIVFHierarchical.cpp   # ← Curator 核心实现（含 Goal3 修改）
│   ├── BloomFilter.h                         # ← Curator 头文件（需 const 修复）
│   ├── impl/
│   │   ├── FaissException.h                  # ← 需添加 TransformedVectors
│   │   └── IDSelector.h/.cpp                 # ← Curator 覆盖的 FAISS 文件
│   ├── utils/
│   │   └── prefetch.h                        # ← 从 conda 复制（FAISS 原始版缺失）
│   └── python/
│       ├── swigfaiss.swig                    # ← SWIG 接口（需添加 Curator 类）
│       └── setup.py
├── build/                                    # CMake 构建目录
│   ├── Makefile
│   ├── faiss/libfaiss.so                     # C++ 共享库
│   └── faiss/python/                         # setup.py 安装目录
└── conda/                                    # Conda 打包配置
```

### 关键路径速查

| 用途 | 绝对路径 |
|------|----------|
| 根 CMakeLists.txt | `/home/lyx/faiss_build/faiss/CMakeLists.txt` |
| FAISS CMakeLists.txt | `/home/lyx/faiss_build/faiss/faiss/CMakeLists.txt` |
| C++ 核心头文件 | `/home/lyx/faiss_build/faiss/faiss/MultiTenantIndexIVFHierarchical.h` |
| C++ 核心实现 | `/home/lyx/faiss_build/faiss/faiss/MultiTenantIndexIVFHierarchical.cpp` |
| FaissException.h | `/home/lyx/faiss_build/faiss/faiss/impl/FaissException.h` |
| BloomFilter.h | `/home/lyx/faiss_build/faiss/faiss/BloomFilter.h` |
| SWIG 文件 | `/home/lyx/faiss_build/faiss/faiss/python/swigfaiss.swig` |
| 构建目录 | `/home/lyx/faiss_build/faiss/build/` |
| Python setup | `/home/lyx/faiss_build/faiss/build/faiss/python/setup.py` |
| faiss 安装位置 | `/home/lyx/miniconda3/envs/edge_ann/lib/python3.10/site-packages/faiss/` |

---

## 常用操作命令

### 环境激活

```bash
source /home/lyx/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
```

### 同步 C++ 文件到 FAISS 源码树

```bash
PROJ=/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
FAISS_SRC=/home/lyx/faiss_build/faiss/faiss

# 复制全部 Curator C++ 文件
cp $PROJ/Curator/src/*.cpp     $FAISS_SRC/
cp $PROJ/Curator/src/*.h       $FAISS_SRC/
cp $PROJ/Curator/src/impl/*.h   $FAISS_SRC/impl/
cp $PROJ/Curator/src/impl/*.cpp $FAISS_SRC/impl/

# ⚠️ 复制后必须执行兼容性补丁！详见 analysis/goal3_wsl_build_guide.md 第三步
```

### 完整编译流程（修改 C++ 后）

```bash
source /home/lyx/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
cd /home/lyx/faiss_build/faiss

# 1. 复制源文件（见上）
# 2. 执行兼容性补丁（见 goal3_wsl_build_guide.md 第三步）
# 3. 编译
cd build
make -j$(nproc)

# 4. 安装
cd faiss/python
SITE_PKGS=/home/lyx/miniconda3/envs/edge_ann/lib/python3.10/site-packages/faiss
rm -f $SITE_PKGS/_swigfaiss.cpython-310-x86_64-linux-gnu.so
rm -f $SITE_PKGS/_swigfaiss_avx2.cpython-310-x86_64-linux-gnu.so
python setup.py install
cp ../libfaiss.so ../libfaiss_avx2.so libfaiss_python_callbacks.so $SITE_PKGS/
```

### 仅修改 Python 脚本（无需重编译）

Python 脚本在 `/mnt/d/.../new-Baselines/` 下，修改后直接生效，无需重编译 C++。

### 强制重新生成 SWIG 绑定

```bash
touch /home/lyx/faiss_build/faiss/faiss/python/swigfaiss.swig
cd /home/lyx/faiss_build/faiss/build && make -j$(nproc)
cd faiss/python && python setup.py install
```

### 验证编译

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
print('SUCCESS: Goal3 C++ methods all working')
"
```

### 运行 Goal 3 诊断/测试

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines

# 第 1 层：C++ 方法可用性（30 秒，无需数据集）
python -c "
import faiss; import sys; sys.path.insert(0, 'Curator/python')
from curator import Curator
idx = faiss.MultiTenantIndexIVFHierarchical(128, 16, faiss.METRIC_L2)
idx.set_pq_config(16, 8, True, False, 4)
bd = idx.get_memory_breakdown()
assert hasattr(bd, 'num_tree_nodes') and hasattr(bd, 'total_bytes')
idx.enable_profiling = True
assert idx.get_last_total_search_time_ms() >= 0
c = Curator(d=128, nlist=16)
bd2 = c.get_memory_breakdown()
assert 'total_bytes' in bd2
print('Layer 1 PASSED')
"

# 第 2 层：内存分解正确性（~30 秒）
python tests/profile_memory_components.py --dataset yfcc100m_small

# 第 3 层：查询时间分解（~30 秒）
python tests/profile_query_steps.py --dataset yfcc100m_small --n_queries 50

# 第 4 层：--profile 集成（~15 秒）
python Curator/run_curator.py --dataset yfcc100m_small --profile
```

### 运行完整 benchmark

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines

# 单次运行
python Curator/run_curator.py --dataset yfcc100m
python Curator/run_curator.py --dataset arxiv

# 参数扫描
python Curator/run_curator_sweep.py \
    --config 3_Config/Curator/curator_yfcc100m.json \
    --sweep 3_Config/Curator/sweep.json
```

---

## 数据集速查

### 已有数据集

| 数据集 | 路径 | 向量数 | 维度 | 标签数 |
|--------|------|--------|:---:|:---:|
| arxiv | `1_Data/ground_truth/arxiv/` | 1,600,000 | 384 | 100 |
| arxiv_small | `1_Data/ground_truth/arxiv_small/` | 50,000 | 384 | ~80 |
| yfcc100m | `1_Data/ground_truth/yfcc100m/` | 800,000 | 192 | 1,000 |
| yfcc100m_small | `1_Data/ground_truth/yfcc100m_small/` | 50,000 | 192 | ~200 |

### 新增数据集（Goal 2 用）

| 数据集 | 路径 | 向量数 | 维度 | 状态 |
|--------|------|--------|:---:|------|
| SIFT1M | `1_Data/sift1m/` | 1,000,000 | 128 | ✅ 已下载，待合成标签 |
| GIST1M | `1_Data/gist1m/` | 1,000,000 | 960 | ✅ 已下载，待合成标签 |

---

## IDE ↔ WSL 工作流

```
┌─── Windows IDE ───────────────────────┐
│  编辑 .h / .cpp / .py 文件             │
│  (d:\23235\...\new-Baselines\)        │
└────────────────────┬──────────────────┘
                     │ 文件自动同步（同一磁盘）
                     ▼
┌─── WSL-Ubuntu ────────────────────────┐
│  /mnt/d/23235/.../new-Baselines/      │  ← Python 脚本可直接运行
│                                        │
│  /home/lyx/faiss_build/faiss/faiss/   │  ← C++ 文件需手动 cp + 补丁
│                                        │
│  流程：                                │
│  1. cp .h/.cpp → FAISS 源码树          │
│  2. 执行兼容性补丁                      │
│  3. make -j$(nproc)                   │
│  4. python setup.py install           │
│  5. 删除旧 .so → 复制新 .so            │
│  6. python tests/...py                 │
└────────────────────────────────────────┘
```

> **关键提醒**：
> - Python 文件修改直接生效（`/mnt/d/...` 即时同步）
> - C++ 修改需执行完整流程：复制 → 补丁 → 编译 → 安装
> - 完整编译指南参见 `analysis/goal3_wsl_build_guide.md`
> - `curator.py` 中的方法名已修正（`_c` 后缀移除、`train()` 参数修正）
