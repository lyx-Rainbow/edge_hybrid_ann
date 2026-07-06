---
name: curator-component-analysis-20260706
description: Curator code-review results: component-level memory/time analysis, bugs found, ablation experiment results on small datasets
metadata:
  type: project
---

# Curator 组件级分析 (2026-07-06)

## 实验完成状态
- 在 arxiv_small (50K×384) 和 yfcc100m_small (50K×192) 上完成 7 组实验
- 基线 + 消融实验（无PQ、PQ无Rerank、无Flash）
- 所有配置下代码正确性已验证
- 详细结果见: `4_Results/RESULTS_ANALYSIS.md` 和 `4_Results/experiment_results_20260706.json`

## 核心发现
1. **PQ压缩比**: arxiv 7.66×, yfcc 3.72× — raw_buffer_ 是构建期最大内存消费者(占91.7%/77.1%)
2. **PQ训练瓶颈**: 占构建时间78-83% — PQ train+encode是构建绝对瓶颈
3. **ADC Rerank至关重要**: +0.113~0.124 召回增益, 仅+2.4~9.7% 延迟代价 — 强烈推荐保持启用
4. **PQ距离计算加速比**: 2.07-3.00× faster than exact L2, 且召回率持平(0.9970)
5. **查询延迟模型**: 固定开销(Beam+Table+Rerank) ~0.12ms + 变动开销(PQ_Dist+Merge) ~0.005ms/shortlist

## 已修复的Bug (3个)
1. `flush()` 无条件清理 `raw_buffer_` → 无PQ+无Flash模式崩溃 → 改为仅当 flash_finalized_ 时清理
2. `find_all_qualified_vecs()` 为 private → 编译错误 → 移至 public
3. `build_filter_index` 缺少 int_vid_t* 重载 → 类型不匹配 → 添加重载

## 待完成
- 复杂谓词过滤测试 (AND/OR/NOT)
- 位图过滤搜索测试
- 完整数据集 (arxiv 1.6M / yfcc 800K) 规模测试
- search_ef / beam_size / variance_boost 参数扫描
- PQ SIMD加速实现

**Why:** 实验验证了当前重构版本Curator的正确性和性能特征，为后续优化提供了基线数据。
**How to apply:** 参考 RESULTS_ANALYSIS.md 第六节"关键发现与优化方向"; 对比基线数据进行回归测试。
