# Curator 实验操作指南

> 如何在 WSL-Ubuntu 端手动运行 Curator 索引实验，包括不同数据集、不同参数配置。

---

## 一、环境准备

### 1.1 激活 Conda 环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
```

### 1.2 路径映射

| 视角 | 路径 |
|------|------|
| Windows IDE | `d:\23235\Documents\Aftergraduate\experiments\new-Baselines` |
| WSL 终端 | `/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines` |

建议在 WSL 中设置环境变量方便操作：

```bash
export PROJ=/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
export CURATOR=$PROJ/Curator
```

### 1.3 编译 Curator（仅首次或修改 C++ 后）

```bash
cd $CURATOR
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

编译产物：`$CURATOR/build/curator`（独立可执行文件，无 FAISS 依赖）。

---

## 二、可用数据集

| 数据集 | 向量数 | 维度 | 标签数 | 路径 |
|--------|:---:|:---:|:---:|------|
| arxiv_small | 50,000 | 384 | 99 | `1_Data/ground_truth/arxiv_small/` |
| yfcc100m_small | 50,000 | 192 | 1,000 | `1_Data/ground_truth/yfcc100m_small/` |
| sift1m_small | 5,000 | 128 | 50 | `1_Data/ground_truth/sift1m_small/` |
| gist1m_small | 3,000 | 960 | 50 | `1_Data/ground_truth/gist1m_small/` |
| arxiv（完整） | 1,600,000 | 384 | 99 | `1_Data/ground_truth/arxiv/` |
| yfcc100m（完整） | 800,000 | 192 | 1,000 | `1_Data/ground_truth/yfcc100m/` |
| sift1m（完整） | 1,000,000 | 128 | 100 | `1_Data/ground_truth/sift1m/` |
| gist1m（完整） | 1,000,000 | 960 | 100 | `1_Data/ground_truth/gist1m/` |

### 各数据集必需文件

```
1_Data/ground_truth/<dataset>/
├── train_vecs.npy       # 训练向量 [N, d] float32
├── train_mds.pkl        # 训练访问列表 (Python pickle, 预处理后转为 train_access.npy)
├── query_vecs.npy       # 查询向量 [Q, d] float32
├── query_labels.npy     # 每条查询的 tenant_id [Q] int32（-1 = 无过滤）
└── ground_truth.npy     # Ground Truth 标签 [Q, k] int32（仅 recall 计算需要）
```

---

## 三、实验配置文件

### 3.1 JSON 配置格式

通过 `--config` 参数传入 JSON 文件，所有字段均可选，未指定的使用默认值。

```json
{
  "d": 384,
  "n_clusters": 32,
  "bf_capacity": 1000,
  "bf_false_pos": 0.001,
  "max_sl_size": 256,
  "clus_niter": 20,
  "max_leaf_size": 128,
  "pq_M": 128,
  "pq_nbits": 8,
  "pq_enabled": true,
  "pq_use_adc_rerank": true,
  "pq_rerank_topk_factor": 4,
  "use_flash_storage": true,
  "nprobe": 3000,
  "prune_thres": 1.6,
  "variance_boost": 0.4,
  "search_ef": 1024,
  "beam_size": 4,
  "use_temp_index_caching": true,
  "batch_query": false
}
```

### 3.2 参数说明

#### 构建参数（影响索引结构）

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `d` | 128 | 向量维度（自动从数据推断） |
| `n_clusters` | 64 | K-means 聚类分支数（≤ 64） |
| `bf_capacity` | 1000 | Bloom Filter 预期元素数 |
| `bf_false_pos` | 0.01 | Bloom Filter 假阳性率 |
| `max_sl_size` | 128 | 短列表最大长度（超限触发分裂） |
| `clus_niter` | 20 | K-means 迭代次数 |
| `max_leaf_size` | 128 | 叶子节点最大向量数（超过则继续分裂） |

#### PQ（乘积量化）参数

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `pq_M` | 16 | 子空间数量（d 必须能被 M 整除） |
| `pq_nbits` | 8 | 每子空间量化位数（**当前仅支持 8**） |
| `pq_enabled` | true | 是否启用 PQ 距离计算 |
| `pq_use_adc_rerank` | false | 是否对 top-K×factor 候选做精确重排 |
| `pq_rerank_topk_factor` | 4 | ADC 重排因子 |
| `persist_pq_codes` | false | 是否在析构后保留 PQ 码磁盘文件 |
| `pq_codes_path` | — | 自定义 PQ 码磁盘文件路径（为空则自动生成临时文件） |
| `pq_cache_block_size` | 4096 | ★ PQ 块缓存的每块码数（块 = 连续 block_size 条向量的 PQ 码） |
| `pq_cache_max_blocks` | 256 | ★ PQ 块缓存最大块数（M=16→~16MB, M=128→~128MB 内存上限） |
| `use_flash_storage` | true | 是否将全精度向量写入磁盘 |

> ★ PQ 码自 v2 起采用**外存 + 按需加载**架构：编码写入磁盘后释放内存，检索时通过 LRU 块缓存按需加载。`pq_cache_block_size` 和 `pq_cache_max_blocks` 控制缓存行为。两个参数均有合理默认值，通常无需修改。

#### 搜索参数

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `nprobe` | 3000 | 无过滤搜索的探测桶数（仅 `search_unfiltered`） |
| `prune_thres` | 1.6 | 无过滤搜索的剪枝阈值倍数 |
| `variance_boost` | 0.4 | 方差提升系数（用于节点评分，降低高方差节点优先级） |
| `search_ef` | 128 | Frontier 搜索候选集大小 |
| `beam_size` | 2 | Beam search 宽度（0 = 关闭 beam search） |
| `use_temp_index_caching` | true | 是否缓存 bitmap filter 的临时索引 |
| `batch_query` | false | 是否启用查询间 OpenMP 并行化 |

### 3.3 常用配置模板

#### arxiv / arxiv_small（d=384）

```bash
cat > /tmp/arxiv_config.json << 'JSON'
{
  "d": 384, "n_clusters": 32,
  "bf_capacity": 1000, "bf_false_pos": 0.001,
  "max_sl_size": 256, "clus_niter": 20, "max_leaf_size": 128,
  "pq_M": 128, "pq_nbits": 8,
  "pq_enabled": true, "pq_use_adc_rerank": true, "pq_rerank_topk_factor": 4,
  "pq_cache_block_size": 4096, "pq_cache_max_blocks": 256,
  "use_flash_storage": true,
  "variance_boost": 0.4, "search_ef": 1024, "beam_size": 4,
  "use_temp_index_caching": true
}
JSON
```

#### yfcc100m / yfcc100m_small（d=192）

```bash
cat > /tmp/yfcc_config.json << 'JSON'
{
  "d": 192, "n_clusters": 32,
  "bf_capacity": 1000, "bf_false_pos": 0.001,
  "max_sl_size": 256, "clus_niter": 20, "max_leaf_size": 128,
  "pq_M": 64, "pq_nbits": 8,
  "pq_enabled": true, "pq_use_adc_rerank": true, "pq_rerank_topk_factor": 4,
  "pq_cache_block_size": 4096, "pq_cache_max_blocks": 256,
  "use_flash_storage": true,
  "variance_boost": 0.2, "search_ef": 1024, "beam_size": 4,
  "use_temp_index_caching": true
}
JSON
```

---

## 四、实验流程

### 4.1 方式一：一键运行（推荐）

使用 Python 脚本自动完成预处理 → 构建 → 查询 → recall 计算：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann

cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/Curator

# 使用默认配置
python python/run_experiment.py --dataset arxiv_small

# 使用自定义配置
python python/run_experiment.py --dataset yfcc100m_small \
    --config /tmp/yfcc_config.json \
    --k 10 --profile

# 启用批量查询并行化
python python/run_experiment.py --dataset yfcc100m_small \
    --config /tmp/yfcc_config.json \
    --batch_query
```

### 4.2 方式二：分步手动操作

#### Step 1 — 预处理：生成 `train_access.npy`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann

python3 << 'EOF'
import numpy as np, pickle, os

ds = "arxiv_small"  # 修改为你的数据集名
BASE = "/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/1_Data/ground_truth"
data_dir = os.path.join(BASE, ds)

with open(os.path.join(data_dir, "train_mds.pkl"), "rb") as f:
    mds = pickle.load(f)

pairs = []
for vid, tids in enumerate(mds):
    for tid in tids:
        pairs.append([vid, tid])

np.save(os.path.join(data_dir, "train_access.npy"),
        np.array(pairs, dtype=np.int32))
print(f"Done: {len(pairs)} access pairs saved")
EOF
```

#### Step 2 — 运行 C++ 构建+检索

```bash
# 设置路径
DS=arxiv_small
DATA=/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/1_Data/ground_truth/$DS
CURATOR_BIN=/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/Curator/build/curator

# 运行 bench（构建 + 查询一站式）
$CURATOR_BIN bench \
    --train_vecs     $DATA/train_vecs.npy \
    --train_access   $DATA/train_access.npy \
    --queries        $DATA/query_vecs.npy \
    --query_labels   $DATA/query_labels.npy \
    --config         /tmp/arxiv_config.json \
    --k 10 \
    --output         $DATA/refactored_results.json \
    --profile
```

**CLI 参数速查**：

| 参数 | 必需 | 说明 |
|------|:--:|------|
| `--train_vecs PATH` | ✅ | 训练向量 `.npy` [N, d] float32 |
| `--train_access PATH` | — | 访问对 `.npy` [M, 2] int32（无标签过滤时可省略） |
| `--queries PATH` | ✅ | 查询向量 `.npy` [Q, d] float32 |
| `--query_labels PATH` | — | 查询标签 `.npy` [Q] int32（省略则全部无过滤搜索） |
| `--config PATH` | — | JSON 配置文件（省略则使用 C++ 默认值） |
| `--k K` | — | 返回结果数（默认 10） |
| `--batch-query` | — | 启用查询间 OpenMP 并行 |
| `--output PATH` | — | 结果 JSON 路径（默认 `results.json`） |
| `--profile` | — | 打印最后一个查询的详细计时分解 |

#### Step 3 — 计算 Recall

```bash
python3 << 'EOF'
import json, numpy as np

ds = "arxiv_small"
BASE = "/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/1_Data/ground_truth"
data_dir = f"{BASE}/{ds}"

gt = np.load(f"{data_dir}/ground_truth.npy")
with open(f"{data_dir}/refactored_results.json") as f:
    results = json.load(f)

k = 10
recalls = []
for q, query_result in enumerate(results["queries"]):
    gt_set = set(gt[q, :k])
    res_set = set(query_result["labels"][:k])
    recalls.append(len(gt_set & res_set) / k)

recalls = np.array(recalls)
print(f"Dataset: {ds}")
print(f"Queries: {len(recalls)}")
print(f"Recall@{k}:")
print(f"  Mean:   {np.mean(recalls):.4f}")
print(f"  Median: {np.median(recalls):.4f}")
print(f"  P90:    {np.percentile(recalls, 90):.4f}")
print(f"  P99:    {np.percentile(recalls, 99):.4f}")
print(f"  Min:    {np.min(recalls):.4f}")
print(f"  ==1.0:  {np.sum(recalls == 1.0)}/{len(recalls)}")
EOF
```

---

## 五、参数扫描

### 5.1 手动扫描脚本

以下脚本对搜索参数进行网格扫描（构建一次，多组搜索参数复用）：

```bash
#!/bin/bash
# sweep_search.sh — 搜索参数网格扫描
source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann

DS=arxiv_small
DATA=/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/1_Data/ground_truth/$DS
CURATOR=/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/Curator/build/curator
BASE_CFG=/tmp/arxiv_config.json
RESULTS_DIR=/tmp/sweep_${DS}
mkdir -p $RESULTS_DIR

# 扫描参数空间
for VAR_BOOST in 0.0 0.2 0.4 0.6; do
for SEARCH_EF in 256 512 1024 2048; do
for BEAM in 1 2 4; do
    NAME="vb${VAR_BOOST}_ef${SEARCH_EF}_bm${BEAM}"
    echo "=== Running $NAME ==="

    # 合并参数生成临时配置
    python3 -c "
import json
with open('$BASE_CFG') as f:
    cfg = json.load(f)
cfg['variance_boost'] = $VAR_BOOST
cfg['search_ef'] = $SEARCH_EF
cfg['beam_size'] = $BEAM
with open('$RESULTS_DIR/${NAME}_cfg.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"

    $CURATOR bench \
        --train_vecs   $DATA/train_vecs.npy \
        --train_access $DATA/train_access.npy \
        --queries      $DATA/query_vecs.npy \
        --query_labels $DATA/query_labels.npy \
        --config       $RESULTS_DIR/${NAME}_cfg.json \
        --k 10 \
        --output       $RESULTS_DIR/${NAME}_results.json \
        2>&1 | tail -3
done
done
done

echo "Sweep complete. Results in $RESULTS_DIR/"
```

### 5.2 收集扫描结果

```bash
python3 << 'EOF'
import json, os, glob, numpy as np

RESULTS_DIR = "/tmp/sweep_arxiv_small"
gt = np.load("/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/1_Data/ground_truth/arxiv_small/ground_truth.npy")

rows = []
for f in sorted(glob.glob(f"{RESULTS_DIR}/*_results.json")):
    name = os.path.basename(f).replace("_results.json", "")
    with open(f) as fp:
        r = json.load(fp)

    recalls = []
    for q, qr in enumerate(r["queries"]):
        gt_set = set(gt[q, :10])
        res_set = set(qr["labels"][:10])
        recalls.append(len(gt_set & res_set) / 10)

    rows.append({
        "name": name,
        "build_s": r["build_time_s"],
        "mem_mb": r["memory_bytes"] / 1024**2,
        "recall_avg": np.mean(recalls),
        "recall_min": np.min(recalls),
    })

rows.sort(key=lambda x: x["recall_avg"], reverse=True)
print(f"{'Config':<30s} {'Build(s)':>8s} {'Mem(MB)':>8s} {'Recall@10':>10s} {'Min':>8s}")
print("-" * 70)
for row in rows:
    print(f"{row['name']:<30s} {row['build_s']:8.2f} {row['mem_mb']:8.2f} {row['recall_avg']:10.4f} {row['recall_min']:8.4f}")
EOF
```

### 5.3 构建参数扫描（需每次重建索引）

```bash
#!/bin/bash
# sweep_build.sh — 构建参数网格扫描
for NCLUSTERS in 16 32 64; do
for MAX_SL in 128 256 512; do
for PQ_M in 32 64 128; do
    # 生成配置...
    # 完整 bench 流程（因为构建参数变更需要重建索引）
    $CURATOR bench --config /tmp/build_cfg.json ...
done
done
done
```

---

## 六、结果解读

### 6.1 输出 JSON 格式

```json
{
  "config": { "d": 384, "nlist": 32, "pq_M": 128, "search_ef": 1024 },
  "build_time_s": 18.38,
  "memory_bytes": 11470000,
  "queries": [
    {
      "idx": 0,
      "tenant_id": 48,
      "labels": [37965, 34908, 47118, 28570, 14872, 18150, 2343, 3369, 17616, 5198],
      "distances": [12.34, 15.67, 18.90, ...]
    },
    ...
  ]
}
```

### 6.2 关键指标

| 指标 | 含义 | 来源 |
|------|------|------|
| `build_time_s` | 索引构建总耗时（训练+插入+授权+flush） | JSON |
| `memory_bytes` | 索引内存占用（全部组件） | JSON |
| `avg_latency_ms` | 平均查询延迟 | 需从 `total_search_time / n_queries` 推算 |
| `Recall@k` | 前 k 个结果与 Ground Truth 的交集比例 | Python 计算 |
| `profiling` 输出 | beam / frontier / rerank 各阶段耗时分解 | stdout（需 `--profile`） |

### 6.3 Profiling 输出解读

```
Profiling (last query):
  query_type: standard
  beam_search: 0.015 ms       ← Phase 1: Beam Search 耗时
  frontier_search: 1.295 ms   ← Phase 2: Frontier 优先队列搜索
    (nodes_popped=59, shortlists=56, expanded=2)
  pq_table_build: 0.063 ms    ← ★ PQ 距离查表构建（每 query 仅一次）
  pq_distance_compute: 0.199 ms ← ★ PQ ADC 距离计算（含按需 I/O）
  candidate_merge: 0.075 ms   ← 候选集合并耗时
  rerank: 0.040 ms            ← Phase 3: ADC 精确重排（仅 pq_use_adc_rerank=true）
    (count=40)
  total: 1.350 ms             ← 总查询耗时
```

诊断建议：
- `nodes_popped` 很大 → `beam_size` 太小或 `variance_boost` 不合适
- `shortlists=0` → BF 过于保守，尝试提高 `bf_false_pos`
- `rerank_count=40` (= k×factor) → ADC rerank 正常工作
- `pq_table_build` 有值 → PQ 距离计算已启用
- `pq_distance_compute` > 0 → PQ 码从磁盘按需加载，首次查询可能略慢（冷启动 I/O）
- `total` 过高 → 检查 `search_ef` 或 `max_sl_size`

---

## 七、常用实验配方

### 7.1 快速验证（小型数据）

```bash
# sift1m_small: 5K×128，极快构建，适合验证代码改动
$CURATOR bench \
    --train_vecs   $DATA/sift1m_small/train_vecs.npy \
    --train_access $DATA/sift1m_small/train_access.npy \
    --queries      $DATA/sift1m_small/query_vecs.npy \
    --query_labels $DATA/sift1m_small/query_labels.npy \
    --k 10 --profile
```

### 7.2 中等规模质量评估

```bash
# arxiv_small / yfcc100m_small: 50K 向量，平衡速度与代表性
python python/run_experiment.py --dataset arxiv_small \
    --config /tmp/arxiv_config.json --k 10
```

### 7.3 完整规模性能测试

```bash
# arxiv: 1.6M×384，完整性能评估（构建约 3-5 分钟，取决于 K-means 迭代）
$CURATOR bench \
    --train_vecs   $DATA/arxiv/train_vecs.npy \
    --train_access $DATA/arxiv/train_access.npy \
    --queries      $DATA/arxiv/query_vecs.npy \
    --query_labels $DATA/arxiv/query_labels.npy \
    --config /tmp/arxiv_config.json --k 10 --batch-query
```

### 7.4 消融实验

```bash
# 关闭 PQ：测试纯精确距离搜索
# 修改 config: "pq_enabled": false

# 关闭 ADC Rerank：测试纯 PQ 近似效果
# 修改 config: "pq_use_adc_rerank": false

# 关闭 Flash：测试全内存模式
# 修改 config: "use_flash_storage": false
```

---

## 八、常见问题

### Q1: 编译失败 "OpenMP not found"

```bash
sudo apt install libomp-dev
```

### Q2: `train_access.npy` 未生成

手动执行预处理：
```bash
python python/preprocess_train.py --dataset arxiv_small
```

### Q3: 查询延迟异常高

- 检查 `search_ef` 是否设置过小（建议 ≥ 128）
- 检查 `max_sl_size` 是否过小导致短列表频繁分裂
- 使用 `--profile` 查看各阶段耗时分布

### Q4: 召回率过低

- 增大 `search_ef`（如 2048）
- 增大 `beam_size`（如 4）
- 调整 `variance_boost`（0.2~0.6 范围尝试）
- 检查 `pq_M` 是否匹配维度（d 必须能被 M 整除）

### Q5: 内存不足

- 减小 `max_sl_size`
- 减小 `n_clusters`
- 确保 `use_flash_storage: true`（全精度向量写入磁盘）

### Q6: 构建时间过长

- 减小 `clus_niter`（如 10）
- 减小 `n_clusters`（如 16）
- 检查 OpenMP 是否生效（`nproc` 个核心应全部使用）

### Q7: 使用其他数据集

将数据准备为以下格式的 `.npy` 文件，放入 `1_Data/ground_truth/<your_dataset>/`：

| 文件 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `train_vecs.npy` | [N, d] | float32 | 训练向量 |
| `train_mds.pkl` | Python list | — | `train_mds[vid] = [tid1, tid2, ...]` |
| `query_vecs.npy` | [Q, d] | float32 | 查询向量 |
| `query_labels.npy` | [Q] | int32 | 每条查询的租户 ID（-1=无过滤） |
| `ground_truth.npy` | [Q, 100] | int32 | 暴力搜索 top-100（仅评估需要） |

---

*文档创建日期: 2026-07-01*
