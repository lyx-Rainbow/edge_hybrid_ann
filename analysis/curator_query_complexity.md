# Curator 查询时间复杂度分析

## 符号表

| 符号 | 含义 |
|------|------|
| $n$ | 训练向量总数 |
| $d$ | 向量维度 |
| $N_{\text{list}}$ | 层次树的分支因子（nlist） |
| $\mathcal{D}$ | 树深度，$\mathcal{D} \approx \log_{N_{\text{list}}}(n / L_{\max})$ |
| $L_{\max}$ | 最大叶节点容量（max_leaf_size） |
| $B$ | 波束宽度（beam_size） |
| $E$ | 候选列表大小上限（search_ef） |
| $k$ | 返回结果数 |
| $\alpha$ | 重排因子（pq_rerank_topk_factor） |
| $\lambda$ | 方差增强系数（variance_boost） |
| $M$ | 积量化子空间数（pq_M） |
| $\text{nbits}$ | 每子空间编码位数（pq_nbits） |
| $d_{\text{sub}} = d / M$ | 子向量维度 |
| $S_{\max}$ | 短列表容量上限（max_sl_size） |
| $|\text{SL}|$ | 某节点某标签的实际短列表大小，$|\text{SL}| \leq S_{\max}$ |
| $Q$ | 复杂谓词合格向量数量，$Q = \text{selectivity} \times n$ |
| $Q_{\text{SL}}$ | Pre-Filtering 单标签查询中，持有指定标签的向量数量 |
| $|\mathcal{P}|$ | 谓词表达式的词法单元数 |
| $n_{\text{probe}}$ | DiskIVF/SPANN 的查询探头数 |

---

## 1. 单标签查询

### 1.1 节点评分函数

对于查询向量 $q$ 和树节点（其聚类中心为 $c$），节点评分定义为：

$$\delta(c, q) = \|c - q\|_2^2 - \lambda \cdot \sigma_c^2$$

其中 $\sigma_c^2$ 为该簇内所有向量到中心 $c$ 的平方距离的算术均值（即簇内方差）。该评分函数在欧氏距离的基础上减去方差增强项：方差越大的簇越可能包含与 $q$ 更近的向量，因此获得优先探索权。

计算复杂度：$O(d)$（一次 L2 平方距离计算 + 一次标量运算）。

### 1.2 波束搜索

```cpp
// MultiTenantIndexIVFHierarchical.cpp:644-722
inline std::vector<Candidate> beam_search(
    const MultiTenantIndexIVFHierarchical& index,
    const float* x, int_lid_t tid, size_t beam_width,
    std::vector<Candidate>& unexpanded)
```

波束搜索从树根出发，逐层向下展开。对每层：

1. 将当前波束 $B$ 中各节点的**所有子节点**（最多 $B \cdot N_{\text{list}}$ 个）加入候选集
2. 按节点评分排序，保留最优 $B$ 个作为下一层波束
3. 其余节点推入前沿堆 $F$（unexpanded），供后续细粒度搜索

当前波束中的某个节点已包含标签 $t$ 的短列表时，该节点不再展开子节点（标记为 `updated=false`），直接保留在波束中。

**终止条件**：所有波束节点均包含短列表（`updated=false`），或无更多层可展开。

**复杂度**：每层最多计算 $B \cdot N_{\text{list}}$ 次节点评分，共 $\mathcal{D}$ 层，排序操作为 $O(B \cdot N_{\text{list}} \log(B \cdot N_{\text{list}}))$。

$$\text{Beam Search: } O\big(B \cdot N_{\text{list}} \cdot \mathcal{D} \cdot d + \mathcal{D} \cdot B \cdot N_{\text{list}} \log(B \cdot N_{\text{list}})\big)$$

### 1.3 前沿搜索

```cpp
// MultiTenantIndexIVFHierarchical.cpp:953-974
while (!frontier.empty()) {
    auto [score, node] = frontier.top();
    frontier.pop();

    auto it = node->shortlists.find(tid);
    if (it != node->shortlists.end()) {
        // PQ 或精确距离计算
        compute_pq_distances(...);  // 或 compute_dists_with_prefetch(...)
        bool updated = cand_vectors.batch_insert(sorted_cands);
        if (!updated) break;  // 候选列表已满且无更优向量，提前终止
    } else if (node->bf.contains(tid)) {
        // 布隆过滤器命中但无短列表 → 展开子节点
        compute_child_scores_with_prefetch(...);
    }
}
```

前沿堆初始包含波束搜索推入的未展开节点。每次弹出评分最低的节点：

**情形 A：节点包含目标标签的短列表**

从短列表中读取所有内部向量 ID，计算与 $q$ 的距离：

- **PQ 模式**（`pq_enabled = true`）：先构建距离查找表，再查表求和

```cpp
// MultiTenantIndexIVFHierarchical.cpp:825-863
// Step 1: 构建距离查找表 (对 d 的每个子段预计算到各量化中心的距离)
for m in 0..M-1:
    for k in 0..ksub-1:
        d_table[m * ksub + k] = L2(x_sub[m], codebook[m][k], dsub)
// 复杂度: O(M × ksub × dsub) = O(2^nbits × d)

// Step 2: 查表计算每条编码向量的距离
for vid in shortlist:
    dist = Σ(m=0..M-1) d_table[m * ksub + code[m]]
// 复杂度: O(|SL| × M)
```

$$\text{PQ 距离: } O(2^{\text{nbits}} \cdot d + |\text{SL}| \cdot M)$$

- **精确模式**（PQ 未启用）：使用预取优化的 L2 距离计算

$$\text{精确距离: } O(|\text{SL}| \cdot d)$$

距离计算完成后，对 `sorted_cands` 执行 `std::sort`（$O(|\text{SL}| \log |\text{SL}|)$）并通过 `batch_insert`（归并两个有序列表，$O(|\text{SL}| + E)$）将候选批量插入全局候选列表。由于 $|\text{SL}| \leq S_{\max}$ 通常较小（典型值 128），且 $d \gg \log |\text{SL}|$，这两项在数值上远小于距离计算项，后续总复杂度表达式中予以省略。

**情形 B：节点布隆过滤器命中但无短列表**

展开该节点的所有子节点（最多 $N_{\text{list}}$ 个），计算每个子节点的评分并推入前沿堆。

$$\text{子节点展开: } O(N_{\text{list}} \cdot d)$$

### 1.4 重排序

```cpp
// MultiTenantIndexIVFHierarchical.cpp:976-989
if (pq_use_adc_rerank && !vid_to_pq_code.empty()) {
    size_t n_rerank = min(k * pq_rerank_topk_factor, cand_vectors.size());
    for i in 0..n_rerank-1:
        float exact_dist = compute_vector_distance(x, vids[i]);
        // 从闪存读取全精度向量，计算精确 L2
}
```

从 PQ 粗筛得到的候选列表中取前 $k \cdot \alpha$ 个向量，通过 `compute_vector_distance` 读取闪存中的全精度原始向量并计算精确 L2 距离。闪存按叶子分区存储，每叶子区域固定 $L_{\max} \cdot d \cdot 4$ 字节。给定内部 vid，叶子序号和叶内偏移可通过哈希表 $O(1)$ 获得，实现单向量 $O(1)$ 次 `fseek` + $O(d)$ 字节顺序读取。

每次精确距离计算后调用 `RunningList::insert` 将结果插入有序列表，涉及二分查找 $O(\log(k\alpha))$ 与向量元素平移 $O(k\alpha)$，$k\alpha$ 次插入合计 $O((k\alpha)^2)$。由于 $k\alpha$ 较小（典型值 $k=10,\ \alpha=4 \rightarrow 40$），且闪存随机读取的 I/O 开销占主导，总复杂度表达式中予以省略。

$$\text{重排序: } O(k \cdot \alpha \cdot d)$$

### 1.5 总复杂度（单标签查询）

$$O\Big(B \cdot N_{\text{list}} \cdot \mathcal{D} \cdot d \;+\; E \cdot \big(2^{\text{nbits}} \cdot d \;+\; S_{\max} \cdot M \;+\; N_{\text{list}} \cdot d\big) \;+\; k \cdot \alpha \cdot d\Big)$$

**省略项说明**：

- 波束搜索中对 $B \cdot N_{\text{list}}$ 个候选的排序开销 $O(\mathcal{D} \cdot B \cdot N_{\text{list}} \log(B \cdot N_{\text{list}}))$：由于 $d$（典型值 192–384）远大于 $\log(B \cdot N_{\text{list}})$（$B=2,\ N_{\text{list}}=64$ 时约 7），距离计算项 $B \cdot N_{\text{list}} \cdot \mathcal{D} \cdot d$ 占主导，省略对数排序项。
- 前沿搜索中短列表候选排序 $O(|\text{SL}| \log |\text{SL}|)$ 与 `batch_insert` 归并 $O(|\text{SL}| + E)$：同前，$|\text{SL}|$ 较小，距离计算占主导，省略。
- 重排序中 `RunningList::insert` 的 $O((k\alpha)^2)$ 开销：$k\alpha$ 较小且闪存 I/O 占主导，省略。
- 式中 $E$ 项为前沿搜索的宽松上界：每次迭代处理的节点要么计算短列表距离（$2^{\text{nbits}}d + S_{\max}M$），要么展开子节点（$N_{\text{list}}d$），$E$ 近似约束了前沿搜索迭代次数的量级。实际迭代次数由候选列表收敛条件决定，通常远小于 $E$。

关键性质：**各项均与 $n$ 无关**。数据集规模仅通过 $\mathcal{D} \approx \log_{N_{\text{list}}}(n / L_{\max})$ 产生对数级影响。

---

## 2. 复杂谓词查询

### 2.1 预处理

```cpp
// curator.py:49-69 — Python 端布尔求值
def compute_qualified_labels(filter_str, train_mds) -> np.ndarray:
    // 遍历全部 n 条元数据，逐条波兰表达式求值

// MultiTenantIndexIVFHierarchical.cpp:30-110 — C++ 端临时索引构建
void build_temp_index_for_filter(
    const MultiTenantIndexIVFHierarchical* index,
    const std::vector<int_vid_t>& sorted_qualified_vecs,
    std::vector<TempIndexNode>& nodes);
```

1. **布尔求值**（Python 端 `compute_qualified_labels` / `_evaluate_predicate`）：遍历全部 $n$ 条训练元数据，对每条执行谓词表达式的波兰表示法求值。每条求值涉及 $|\mathcal{P}|$ 个词法单元的栈操作。$\Theta(n \cdot |\mathcal{P}|)$。

   > 注：C++ 端另有一树遍历版本 `find_all_qualified_vecs`（[MultiTenantIndexIVFHierarchical.cpp:1282-1369](Curator/src/MultiTenantIndexIVFHierarchical.cpp#L1282-L1369)），利用布隆过滤器剪枝实现亚线性复杂度，但当前 Python 流程（`index_filter` → `compute_qualified_labels`）走的是全量扫描路径。

2. **临时索引构建**（C++ 端 `build_temp_index_for_filter`）：对合格向量集合按内部 vid 排序（vid 的高位比特编码了树路径），利用有序性在层级树结构上二分查找确定各节点的向量归属范围，构建剪枝后的临时搜索树。$O(Q \log Q)$。

### 2.2 搜索

在临时索引上执行与单标签查询类似的波束搜索 + 前沿搜索策略。由于临时索引仅包含合格向量出现过的子树节点，布隆过滤器剪枝不再需要（搜索空间已被限定在相关子树内）。

$$\text{CP 查询: } O\big(n \cdot |\mathcal{P}| \;+\; Q \log Q \;+\; E \cdot (2^{\text{nbits}} \cdot d \;+\; S_{\max} \cdot M)\big)$$

> **省略项说明**：临时索引上的波束搜索同样有 $O(B \cdot N_{\text{list}}' \cdot \mathcal{D}' \cdot d)$ 的开销，但由于临时索引仅包含合格向量涉及的子树分支，节点数远少于完整树，该项在数值上较小，且与单标签查询的同源项数量级相当或更低，故省略。前沿搜索中的排序与归并开销省略原因同 1.5 节。式中 $Q$ 为满足复杂谓词的向量总数（$Q = \text{selectivity} \times n$），$Q \log Q$ 来自合格向量 ID 的排序与临时索引构建中的二分查找。

预处理中 $n \cdot |\mathcal{P}|$ 项在批量执行 $N_q$ 条查询时可被均摊——若 $N_q$ 条查询共享同一谓词，则预处理仅需一次。

---

## 3. 与基线方法的对比

| 方法 | 单标签查询 | 复杂谓词查询 |
|------|-----------|-------------|
| **Curator** | $O(\mathcal{D} \cdot d + E \cdot (2^{\text{nbits}}d + M S_{\max}))$ | $O(n \cdot \|\mathcal{P}\| + Q \log Q + E \cdot 2^{\text{nbits}}d)$ |
| **DiskIVF** | $\Theta(n_{\text{probe}} \cdot n / N_{\text{list}} \cdot d)$ | $\Theta(n_{\text{probe}} \cdot n / N_{\text{list}} \cdot d)$ |
| **SPANN** | $\Theta(n_{\text{probe}} \cdot n / N_{\text{list}} \cdot d)$ | $\Theta(n_{\text{probe}} \cdot n / N_{\text{list}} \cdot d)$ |
| **Pre-Filtering (SL)** | $\Theta(Q_{\text{SL}} \cdot d)$ | — |
| **Pre-Filtering (CP)** | — | $\Theta(n \cdot \|\mathcal{P}\| + Q \cdot d)$ |

> 为简洁起见，表中 Curator 公式省略了常数因子项 $B \cdot N_{\text{list}}$（波束搜索）以及各排序、归并、重排插入等次要项，完整表达式见第 1.5 节与第 2.2 节。$Q_{\text{SL}}$ 为持有指定标签的向量数（由倒排索引 $O(1)$ 获取），$Q$ 为满足复杂谓词的向量数。

**DiskIVF / SPANN** 的复杂度与 $n$ 呈线性关系——查询需从磁盘加载 $n_{\text{probe}}$ 个分区（每分区约 $n / N_{\text{list}}$ 个向量）并在内存中执行暴力 L2 扫描。即便增大 $N_{\text{list}}$ 降低单分区大小，$n_{\text{probe}}$ 也需等比增加以维持召回率，乘积 $n_{\text{probe}} \cdot (n / N_{\text{list}})$ 近似恒定。

**Pre-Filtering** 的单标签查询受益于倒排索引的 $O(1)$ 候选集定位（仅需一次哈希查找即获得全部合格向量），但其复杂谓词查询需遍历全部 $n$ 条元数据逐条执行布尔表达式求值，预处理开销 $\Theta(n \cdot |\mathcal{P}|)$ 无法通过索引结构规避。

**Curator** 通过层次聚类树与布隆过滤器的协同剪枝，将搜索复杂度与数据集规模 $n$ 解耦。对于单标签查询，$\mathcal{D}$ 随 $n$ 呈对数增长（$E=2048$ 时 YFCC-1M 上 $\mathcal{D}=4$）；对于复杂谓词查询，临时索引构建仅在首次遇到某谓词时需要，后续查询直接复用缓存索引，额外开销为零。这使得 Curator 在低至高的选择率区间内均维持亚线性的查询延迟。
