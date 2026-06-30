# 目标1：将 Curator 索引中原本存储在内存的 PQ 码移入外存

## 一、现有基础分析

### 1.1 PQ 码的完整生命周期

通过源码审查，PQ 码在当前系统中的流转如下：

```
train_pq_codebook() [C++, .cpp:2062-2093]
  ├── 使用 FAISS ProductQuantizer 对 raw_vectors_buffer 训练码本
  ├── 将码本存储到 pq_codebook (vector<float>, M × ksub × dsub 个 float)
  ├── 对所有向量逐条编码，存入 vid_to_pq_code (vector<vector<uint8_t>>)
  │     按 **sequence number** 索引（非 vid 直接索引）
  │     通过 vid_to_seq (unordered_map<vid, seq_idx>) 做映射
  └── 打印训练摘要

finalize_flash_storage() [C++, .cpp:2115-2208]
  ├── Step 1: 若 PQ 启用且 vid_to_pq_code 为空 → 调用 train_pq_codebook()
  ├── Step 2: DFS 遍历叶子节点，分配 leaf_node_id_to_seq 序号
  ├── Step 3-5: 将全精度向量按叶子分区写入 Flash 文件
  ├── Step 6: 释放 raw_vectors_buffer 和 vid_to_buffer_offset
  └── 标记 flash_finalized = true

query 时使用 PQ 码 [C++, .cpp:825-863, compute_pq_distances()]
  ├── 构建距离查找表 d_table[M × ksub]
  └── 对每个 vid: vid→seq(via vid_to_seq)→code(via vid_to_pq_code[seq])→查表累加
```

**关键结论**：`finalize_flash_storage()` 释放了全精度向量缓冲区，但 **PQ 码（`vid_to_pq_code`）和码本（`pq_codebook`）始终保留在内存中**。没有任何现有代码将其写入磁盘。

### 1.2 "现有基础"的真实状态

#### 不可用的伪基础

- **`persist_pq_codes` 和 `persist_temp_indexes` 参数**（`curator.py:97-98`）：这两个参数在 Python 构造函数中被声明但**从未赋值给 `self`**，是彻底的 dead code。在 `__init__` 的 143 行代码中找不到 `self.persist_pq_codes` 或 `self.persist_temp_indexes`。C++ 端也没有对应的成员或方法。**结论：不可依赖，需完全从零实现。**

#### 可用的真实基础

| 基础 | 位置 | 可利用性 |
|------|------|----------|
| Flash 存储机制（`set_flash_storage_path` / `finalize_flash_storage` / `load_vector_from_flash`） | C++ .cpp:2099-2310 | **高**：提供了文件 I/O、mmap 参考模式、叶子分区布局 |
| `vid_to_pq_code` 的数据结构和映射关系（`vid_to_seq` / `seq_to_vid`） | C++ .h:344-346 | **高**：文件格式设计需考虑 seq 索引 vs vid 索引 |
| `get_total_memory_bytes()` 已正确统计 PQ 码内存 | C++ .cpp:2335-2337 | 中：可用于验证释放后的内存减少 |
| `compute_pq_distances()` 的查表+累加逻辑 | C++ .cpp:825-863 | **高**：外存模式下仅需改变 code 的来源（内存 → 磁盘） |
| Python 构造函数的 `disk_cache_prefix` 参数（已用于 Flash 路径） | Python curator.py:96 | **高**：可复用为 PQ 码文件路径前缀 |
| PQ 码大小可精确计算：`n × M × nbits / 8` 字节 | — | **高**：用于文件 header 和预分配 |

### 1.3 PQ 码文件格式设计的前提约束

```
vid_to_pq_code 结构:
  - 类型: std::vector<std::vector<uint8_t>>
  - 索引方式: vid_to_pq_code[seq_idx] 返回该向量对应的 M 字节编码
  - seq_idx 通过 seq_to_vid[seq_idx] → vid 或 vid_to_seq[vid] → seq_idx 映射
  - 每个 code 大小: M × nbits/8 = M 字节 (当 nbits=8 时)
  - 总大小: ntotal × M 字节

约束:
  - vid 不是连续整数 (是编码了树路径的 bit-packed ID)
  - seq 是连续整数 (0, 1, 2, ... ntotal-1)，但这是分配顺序，不是逻辑顺序
  - 文件按 seq 顺序存储最紧凑 (与 vid_to_pq_code 内存布局一致)
```

---

## 二、执行计划

### 阶段 A：C++ 核心改造

#### A1. 新增 PQ 码文件 I/O 方法（在 `MultiTenantIndexIVFHierarchical` 类中）

```cpp
// 文件路径管理
void set_pq_codes_path(const std::string& path);
std::string pq_codes_path;

// 写入：将 vid_to_pq_code 序列化到磁盘
void write_pq_codes_to_disk();

// 加载：以只读 mmap 或预读方式打开 PQ 码文件
void load_pq_codes_from_disk();

// 释放内存中的 PQ 码
void free_pq_codes();

// 判断 PQ 码当前在内存还是磁盘
bool pq_codes_in_memory() const;
bool pq_codes_on_disk() const;

// 获取单个向量的 PQ 码（透明处理内存/磁盘）
const uint8_t* get_pq_code_by_seq(size_t seq_idx) const;
```

#### A2. PQ 码文件格式

```
文件: {disk_cache_prefix}/pq_codes.bin

Header (32 bytes):
  offset 0:  uint32_t magic        = 0x50514344 ("PQCD")
  offset 4:  uint32_t version      = 1
  offset 8:  uint32_t n_vectors    = ntotal
  offset 12: uint32_t M            = pq_M
  offset 16: uint32_t nbits        = pq_nbits
  offset 20: uint32_t code_bytes   = M × nbits / 8 (每向量字节数)
  offset 24: uint64_t body_offset  = 32 (body 起始偏移，预留扩展)
  (offset 32 起为 body)

Body: uint8_t[ntotal × code_bytes]
  - 按 seq 顺序连续排列 (与 vid_to_pq_code 内存布局完全一致)
  - 第 i 个 block (code_bytes 字节) 对应 seq=i 的向量
  - 文件总大小 = 32 + ntotal × M × nbits/8 字节
```

**设计理由**：按 seq 顺序而非 vid 顺序，因为：
1. 与内存中 `vid_to_pq_code[seq]` 的索引方式一致，读写无需重排
2. seq 连续（0..ntotal-1），偏移计算简单：`offset = 32 + seq × code_bytes`
3. 查询时从 vid 出发：`vid → vid_to_seq[vid] → seq → 文件偏移`（两次 O(1) 查找）

#### A3. 修改 `compute_pq_distances()` 以支持外存读取

改动最小化的方案：在 `compute_pq_distances` 中，将原来 `index.vid_to_pq_code[seq_idx]` 的直接内存访问改为通过新方法 `index.get_pq_code_by_seq(seq_idx)`，该方法内部判断：
- 若 `vid_to_pq_code` 非空 → 从内存返回
- 若 PQ 码已持久化到磁盘 → 从 mmap 区域或预读缓冲区返回
- 可选优化：添加小容量 LRU 缓存（如 1024 条）减少重复 I/O

#### A4. 在 `finalize_flash_storage()` 中集成 PQ 码持久化

在现有 `finalize_flash_storage()` 的 Step 6（释放 raw_vectors_buffer）之后、`flash_finalized = true` 之前：

```cpp
// Step 7 (新增): Persist PQ codes if configured
if (persist_pq_codes && !pq_codes_path.empty()) {
    write_pq_codes_to_disk();
    free_pq_codes();  // 释放内存中的 vid_to_pq_code
}
```

同时在类中添加 `persist_pq_codes` 成员变量（`bool persist_pq_codes = false`）。

#### A5. 修改 `get_total_memory_bytes()` 

在 PQ 码已持久化且释放后，`vid_to_pq_code` 为空，其大小为 0。该方法现有逻辑已在 2335-2337 行遍历 `vid_to_pq_code`，自然反映释放效果，无需修改。

### 阶段 B：Python API 改造

#### B1. 实现 `persist_pq_codes` 参数

修改 `Curator.__init__`：
```python
# 当前 (curator.py:97-98)：仅声明参数，未赋值
persist_pq_codes: bool = True,   # 参数不生效！
persist_temp_indexes: bool = True,

# 修改后：存储并传递给 C++
self.persist_pq_codes = persist_pq_codes
self.persist_temp_indexes = persist_temp_indexes  # 留待后续实现
# 通过 C++ 方法设置
if persist_pq_codes and disk_cache_prefix is not None:
    self.index.set_pq_codes_path(f"{disk_cache_prefix}/pq_codes.bin")
    self.index.set_persist_pq_codes(True)
```

#### B2. 修改 `flush()` 方法

```python
def flush(self) -> None:
    if not self.index.flash_finalized:
        self.index.finalize_flash_storage()  # 内部已处理 PQ 持久化（若启用）
```

`finalize_flash_storage()` 已在 C++ 层修改，Python 端只需确保调用。

#### B3. 新增查询方法（可选优化）

```python
def preload_pq_codes(self) -> None:
    """将 PQ 码从磁盘重新加载到内存（用于性能敏感场景）"""
    self.index.load_pq_codes_into_memory()

def get_pq_codes_location(self) -> str:
    """返回 'memory' | 'disk' | 'none'"""
    ...
```

### 阶段 C：SWIG 绑定更新

在 `swigfaiss.swig` 中：
1. 新增方法声明（`set_pq_codes_path`、`write_pq_codes_to_disk`、`load_pq_codes_from_disk`、`free_pq_codes`、`set_persist_pq_codes`）
2. 移除或保留对 `vid_to_pq_code` 的 ignore（保持封装，外部通过方法访问）
3. 若新增 mmap 相关成员，也需 ignore 或适配

### 阶段 D：测试验证

1. **正确性测试**：在 `yfcc100m_small` 上，`persist_pq_codes=True/False` 两种模式下的查询 recall 完全一致（逐查询比对 top-10 结果）
2. **内存测试**：`persist_pq_codes=True` 后 `get_index_memory_bytes()` 的内存减少量 = PQ 码理论大小（ntotal × M × nbits/8）
3. **文件正确性**：`pq_codes.bin` 文件大小 = 32 + ntotal × M × nbits/8 字节，magic number 正确
4. **延迟测试**：P50/P95 查询延迟在可接受范围（预计外存模式增加 5-30%，取决于磁盘速度和缓存命中率）
5. **可逆性**：从外存重新加载 PQ 码到内存后，查询结果与原始一致
6. **跨数据集**：在 arxiv (d=384, M=24) 和 yfcc100m (d=192, M=16) 上均通过测试

---

## 三、预期效果

| 指标 | 当前 | 改造后（外存模式） | 
|------|------|-------------------|
| PQ 码内存占用 | ntotal × M × nbits/8 | **0**（或可选 LRU 缓存，如 1MB） |
| 索引总内存减少 | — | ntotal × M × nbits/8（yfcc100m 800K: ~12.8MB, arxiv 1.6M: ~38.4MB） |
| 查询延迟 | 基准 | +5-30%（取决于存储介质和缓存策略） |
| 召回率 | 基准 | **完全一致**（PQ 码本身不变） |
| 磁盘文件大小 | 0 | ntotal × M × nbits/8 + 32B（与内存 PQ 码等大） |

**注意**：当 n 较小时（如 yfcc100m_small: 50K×16=0.8MB），外存模式收益有限；当 n 很大时（如 10M×64=640MB），收益显著。

---

## 四、验收方式

```bash
# 1. 小规模正确性验证
python Curator/run_curator.py --dataset yfcc100m_small \
    --config 3_Config/Curator/curator_yfcc100m_small.json

# 手动修改配置中 persist_pq_codes=true/false，比对两次运行的 recall

# 2. 内存验证
python tests/diagnose_memory.py --dataset yfcc100m_small
# 输出应包含 "PQ codes in memory: 0 MB" (persist=true 时)

# 3. 文件验证
ls -la 4_Results/Curator/disk_data/pq_codes.bin
# 应存在且大小 = 32 + ntrain × M 字节

# 4. 全数据集验证
python Curator/run_curator.py --dataset yfcc100m
python Curator/run_curator.py --dataset arxiv
```
