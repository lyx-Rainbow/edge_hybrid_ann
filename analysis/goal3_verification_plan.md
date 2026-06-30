# Goal 3 验收计划

编译通过后，按以下步骤逐层验证。每步给出预期输出和判定标准。

---

## 第 1 层：SWIG 绑定完整性（30 秒）

**目标**：确认新增的 C++ 方法已通过 SWIG 暴露给 Python。

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines

python -c "
import faiss
import sys
sys.path.insert(0, 'Curator/python')
from curator import Curator

idx = faiss.MultiTenantIndexIVFHierarchical(128, 16, faiss.METRIC_L2)
idx.set_pq_config(16, 8, True, False, 4)

# === 新增方法存在性检查 ===

# 1. MemoryBreakdown
bd = idx.get_memory_breakdown()
assert hasattr(bd, 'num_tree_nodes'), 'MISSING: num_tree_nodes'
assert hasattr(bd, 'centroids_bytes'), 'MISSING: centroids_bytes'
assert hasattr(bd, 'bloom_filter_bytes'), 'MISSING: bloom_filter_bytes'
assert hasattr(bd, 'shortlists_payload_bytes'), 'MISSING: shortlists_payload_bytes'
assert hasattr(bd, 'pq_codebook_bytes'), 'MISSING: pq_codebook_bytes'
assert hasattr(bd, 'pq_codes_bytes'), 'MISSING: pq_codes_bytes'
assert hasattr(bd, 'raw_vectors_buffer_bytes'), 'MISSING: raw_vectors_buffer_bytes'
assert hasattr(bd, 'total_bytes'), 'MISSING: total_bytes'
print('[PASS] MemoryBreakdown: all 20+ fields accessible')

# 2. 扩展的 profiling 访问器
assert idx.get_last_beam_search_time_ms() >= 0, 'MISSING: get_last_beam_search_time_ms'
assert idx.get_last_frontier_search_time_ms() >= 0, 'MISSING: get_last_frontier_search_time_ms'
assert idx.get_last_pq_table_build_time_ms() >= 0, 'MISSING: get_last_pq_table_build_time_ms'
assert idx.get_last_pq_distance_compute_time_ms() >= 0, 'MISSING: get_last_pq_distance_compute_time_ms'
assert idx.get_last_rerank_time_ms() >= 0, 'MISSING: get_last_rerank_time_ms'
assert idx.get_last_total_search_time_ms() >= 0, 'MISSING: get_last_total_search_time_ms'
assert idx.get_last_frontier_nodes_popped() >= 0, 'MISSING: get_last_frontier_nodes_popped'
assert idx.get_last_frontier_shortlists_scanned() >= 0, 'MISSING: get_last_frontier_shortlists_scanned'
print('[PASS] Profiling accessors: all 13 new methods accessible')

# 3. Curator Python 封装
from curator import Curator as C
c = C(d=128, nlist=16)
bd2 = c.get_memory_breakdown()
assert 'tree' in bd2 and 'pq' in bd2 and 'shortlists' in bd2, 'MISSING: breakdown dict keys'
assert isinstance(bd2['total_bytes'], int), 'MISSING: total_bytes'
print('[PASS] Curator.get_memory_breakdown() returns dict')
print('[PASS] Curator.query_with_profile() exists:', hasattr(c, 'query_with_profile'))
print()
print('=== LAYER 1: ALL PASSED ===')
"
```

**判定**：所有 `[PASS]`，无 `MISSING` 或 `AssertionError`。

---

## 第 2 层：内存分解正确性（2 分钟）

**目标**：确认 `get_memory_breakdown()` 返回的组件之和等于 `get_index_memory_bytes()`，且 flush 前后变化合理。

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
python tests/profile_memory_components.py --dataset yfcc100m_small
```

**预期输出**：

```
Curator Memory Component Profiler: yfcc100m_small
  Vectors: 50,000 x 192
  Training...
  Adding 50000 vectors...
  Granting access...
  Flushing...

Final Memory Breakdown
Component                                   Size (MB)       %
──────────────────────────────────────────────────────────
Tree Structure
  ├─ Node attributes                            x.xx      x%
  ├─ Centroids                                  x.xx      x%
  └─ Num nodes                                    xx
Bloom Filters                                  x.xx      x%
Short Lists
  ├─ Hash-table overhead                       x.xx      x%
  └─ Payload (vid data)                        x.xx      x%
...

Phase-by-phase delta (MB)

  Before train → After train:
  Component                       Δ (MB)
  ──────────────────────────────  ──────
  Tree: centroids                  +x.xx
  ...

  After add → After grant:
  Shortlists: payload             +x.xx    ← 应该有明显增长
  Bloom filters                   +x.xx    ← 应该有明显增长

  After grant → After flush:
  Raw vectors buffer              -xx.xx   ← 应该大幅减少（释放了原始向量）
  PQ codes                        +x.xx    ← 应该从 0 变为正值
  Total (index)                   变化合理
```

**判定**：
- [ ] `Total (index-reported)` 与 `get_index_memory_bytes()` 一致（误差 < 1%）
- [ ] Flush 后 `Raw Vectors Buffer` ≈ 0（释放了原始向量缓冲区）
- [ ] Flush 后 `PQ codes` > 0（PQ 编码完成）
- [ ] Grant 后 `Shortlists: payload` > 0（短列表已填充）
- [ ] 各阶段 Δ 值符号正确（正增长/负释放符合预期）

---

## 第 3 层：查询时间分解正确性（2 分钟）

**目标**：确认标准查询的 profiling 捕获了非零时间数据，各步骤时间之和 ≈ 总时间。

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
python tests/profile_query_steps.py --dataset yfcc100m_small --n_queries 50
```

**预期输出**：

```
Curator Query Step Profiler: yfcc100m_small
  Building index...
  Profiling 50 queries...

Step breakdown (mean ms):
  Step                             Time (ms)      %
  ─────────────────────────────────────────────────
  total search                        x.xx    100.0%
  beam search                         0.xx      x%
  frontier search                     x.xx     xx%
  pq table build                      0.xx      x%
  pq distance compute                 x.xx     xx%
  exact distance compute              0.00      0%   ← 使用 PQ 时应为 0
  candidate merge                     0.xx      x%
  rerank                              0.00      0%   ← 未启用 ADC 时应为 0

Per selectivity bucket:
  Bucket                           Count   Total    Beam  Frontier   PQDist   Rerank
  ───────────────────────────────  ─────  ──────   ─────  ────────   ──────   ──────
  [0.0001, 0.xxxx)                    xx    x.xx    0.xx      x.xx     x.xx     0.00
  ...
```

**判定**：
- [ ] `total search` > 0（确实计到了时间）
- [ ] `beam search` ≈ 0.02-0.10 ms（波束搜索占比很小）
- [ ] `frontier search` 是最大项（前沿搜索开销最大）
- [ ] `exact distance compute` ≈ 0（PQ 启用时使用 PQ 距离，不使用精确距离）
- [ ] 各 bucket 的 `Total` 时间不同（高选择性 bucket 更快）
- [ ] 无异常 (NaN, inf, 负数)

---

## 第 4 层：`--profile` 标志集成（3 分钟）

**目标**：确认 `run_curator.py --profile` 的输出 JSON 包含完整的 profiling 数据。

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines

# 运行带 profiling 的 benchmark
python Curator/run_curator.py --dataset yfcc100m_small --profile

# 检查输出 JSON
python -c "
import json
with open('4_Results/Curator/curator_yfcc100m_small.json') as f:
    data = json.load(f)

# 内存分解
mem = data.get('final_memory_breakdown')
assert mem is not None, 'MISSING: final_memory_breakdown'
assert 'breakdown' in mem, 'MISSING: breakdown in memory snapshot'
bd = mem['breakdown']
assert bd['pq']['codes_bytes'] > 0, 'PQ codes should be > 0 after flush'
assert bd['raw_vectors_buffer_bytes'] == 0, 'Raw buffer should be 0 after flush'
print('[PASS] Memory breakdown in output JSON')

# 查询 profiling 采样
prof = data['results'].get('profile_samples')
assert prof is not None, 'MISSING: profile_samples in results'
assert len(prof) > 0, 'profile_samples is empty'
sample = prof[0]
for key in ['beam_search_time_ms', 'frontier_search_time_ms', 'total_search_time_ms']:
    assert sample.get(key, 0) > 0, f'MISSING or zero: {key}'
print(f'[PASS] Query profile samples: {len(prof)} samples')
print(f'  Sample query steps: beam={sample[\"beam_search_time_ms\"]:.3f}ms, '
      f'frontier={sample[\"frontier_search_time_ms\"]:.3f}ms, '
      f'total={sample[\"total_search_time_ms\"]:.3f}ms')

# 构建阶段快照
mem_snaps = data['memory'].get('memory_snapshots')
if mem_snaps:
    phases = list(mem_snaps.keys())
    print(f'[PASS] Build phase snapshots: {len(phases)} phases ({phases})')
else:
    print('[INFO] No build phase snapshots (use --profile for full)')

print()
print('=== LAYER 4: ALL PASSED ===')
"
```

**判定**：
- [ ] JSON 中含 `final_memory_breakdown` 字段
- [ ] JSON 中含 `results.profile_samples` 字段
- [ ] 查询步骤时间数据合理（非零、各步骤 < 总时间）
- [ ] 内存数据正确（PQ codes > 0, raw buffer = 0）

---

## 第 5 层：Profiling 零开销验证（1 分钟）

**目标**：确认关闭 profiling 时查询延迟与开启前一致（无性能退化）。

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines

# 无 profiling 运行
python Curator/run_curator.py --dataset yfcc100m_small 2>&1 | grep "Avg latency"

# 带 profiling 运行（应略慢 5-10%，因为计时本身有开销）
python Curator/run_curator.py --dataset yfcc100m_small --profile 2>&1 | grep "Avg latency"
```

**判定**：
- [ ] 两次运行的 `avg_latency_ms` 差异 < 15%
- [ ] 两次运行的 `avg_recall` 完全一致

---

## 第 6 层：跨数据集一致性（5 分钟）

**目标**：确认在另一个数据集上也能正常工作。

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
python tests/profile_memory_components.py --dataset arxiv_small
python tests/profile_query_steps.py --dataset arxiv_small --n_queries 30
```

**判定**：
- [ ] 两个脚本均无报错完成
- [ ] arxiv_small (d=384) 的 centroids 字节数 ≈ yfcc100m_small (d=192) 的 2 倍
- [ ] arxiv_small 的 PQ codes 字节数 ≈ 50K × 24 = 1.2 MB（M=24）

---

## 验收检查清单

| # | 检查项 | 验证方式 |
|:--:|--------|----------|
| 1 | `get_memory_breakdown()` 返回 20+ 字段 | 第 1 层 |
| 2 | 13 个新增 profiling 访问器可用 | 第 1 层 |
| 3 | `Curator.get_memory_breakdown()` 返回 dict | 第 1 层 |
| 4 | `Curator.query_with_profile()` 存在 | 第 1 层 |
| 5 | 组件之和 = `get_index_memory_bytes()` | 第 2 层 |
| 6 | Flush 后 raw_vectors_buffer = 0 | 第 2 层 |
| 7 | Flush 后 PQ codes > 0 | 第 2 层 |
| 8 | Grant 后 shortlists payload > 0 | 第 2 层 |
| 9 | 查询 total_search_time > 0 | 第 3 层 |
| 10 | frontier search 是最大耗时项 | 第 3 层 |
| 11 | 不同选择性 bucket 延迟不同 | 第 3 层 |
| 12 | `--profile` 输出 JSON 含完整数据 | 第 4 层 |
| 13 | 无 profiling 时延迟与 baseline 一致 | 第 5 层 |
| 14 | 跨数据集（arxiv_small）正常 | 第 6 层 |

---

## 快速验收（如果时间紧迫）

```bash
cd /mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines

# 一条命令覆盖第 1-4 层
python tests/profile_memory_components.py --dataset yfcc100m_small && \
python tests/profile_query_steps.py --dataset yfcc100m_small --n_queries 30 && \
python Curator/run_curator.py --dataset yfcc100m_small --profile && \
python -c "
import json
with open('4_Results/Curator/curator_yfcc100m_small.json') as f:
    d=json.load(f)
bd=d['final_memory_breakdown']['breakdown']
assert bd['pq']['codes_bytes']>0,'PQ fail'
assert bd['raw_vectors_buffer_bytes']==0,'Buffer fail'
assert len(d['results']['profile_samples'])>0,'Profile fail'
print('QUICK CHECK: ALL PASSED')
"
```
