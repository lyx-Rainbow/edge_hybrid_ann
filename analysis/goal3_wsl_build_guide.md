# Goal 3 C++ 编译指南（已验证通过 · 2026-06-15）

在 WSL-Ubuntu 终端中逐段执行。每次修改 C++ 源码后，只需执行第七、八步。

---

## 前置条件

- WSL Ubuntu 24.04, GCC 13.3.0
- conda 环境 `edge_ann` 已创建，安装了 `faiss` 和 MKL
- FAISS v1.7.4 源码已 clone 到 `~/faiss_build/faiss`
- 本项目位于 `/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines`

---

## 第一步：激活环境

```bash
source /home/lyx/miniconda3/etc/profile.d/conda.sh
conda activate edge_ann
export PROJ=/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines
```

---

## 第二步：部署 Curator + Goal3 源文件

```bash
cd ~/faiss_build/faiss

# 核心 Curator C++ 文件 → faiss/faiss/
cp $PROJ/Curator/src/*.cpp   faiss/
cp $PROJ/Curator/src/*.h     faiss/
cp $PROJ/Curator/src/impl/*.h     faiss/impl/
cp $PROJ/Curator/src/impl/*.cpp   faiss/impl/

# 确认关键文件存在
ls faiss/MultiTenantIndexIVFHierarchical.cpp faiss/BloomFilter.h faiss/complex_predicate.cpp faiss/impl/IDSelector.h && echo "All Curator files OK"
```

---

## 第三步：修复 C++ 源码兼容性问题

**以下修改必须在每次重新部署源码后执行。**

### 3a. 修复 `CMAKE_CXX_STANDARD` —— 根 CMakeLists.txt

FAISS 默认 C++11，但 Curator/Goal3 需要 C++17（`std::filesystem`、structured bindings）。

```bash
sed -i 's/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD 17)/' CMakeLists.txt
```

GCC 13 默认即为 C++17，此设置确保跨编译器一致性。

### 3b. 修复 `BloomFilter::hash_count()` const 正确性

```bash
sed -i 's/inline std::size_t hash_count()/inline std::size_t hash_count() const/' faiss/BloomFilter.h
```

### 3c. 修复泛型 lambda 类型推导（GCC 13 兼容性）

在 `faiss/MultiTenantIndexIVFHierarchical.cpp` 中：

```bash
sed -i 's/\[\](const auto& a, const auto& b) { return a.first < b.first; });/[](const std::pair<size_t, const float*>\& a, const std::pair<size_t, const float*>\& b) { return a.first < b.first; });/' faiss/MultiTenantIndexIVFHierarchical.cpp
```

### 3d. 添加 `TransformedVectors` 到 `FaissException.h`

此结构体是 Curator 代码需要的，但不在 FAISS 原始版本中。

```bash
python3 << 'PYEOF'
path = "faiss/impl/FaissException.h"
with open(path) as f:
    content = f.read()

old = """    ~ScopeDeleter1() {
        delete ptr;
    }
};

/// make typeids more readable"""

new = """    ~ScopeDeleter1() {
        delete ptr;
    }
};

/** RAII object for a set of possibly transformed vectors (deallocated only if
 * they are indeed transformed)
 */
struct TransformedVectors {
    const float* x;
    bool own_x;
    TransformedVectors(const float* x_orig, const float* x) : x(x) {
        own_x = x_orig != x;
    }

    ~TransformedVectors() {
        if (own_x) {
            delete[] x;
        }
    }
};

/// make typeids more readable"""

content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("TransformedVectors added to FaissException.h")
PYEOF
```

### 3e. 将嵌套 struct 移到类外（SWIG 兼容性）

`MemoryBreakdown` 和 `SearchProfilingData` 需要从 `MultiTenantIndexIVFHierarchical` 类内移到 `faiss` 命名空间顶层，否则 SWIG 无法正确暴露成员。

```bash
python3 << 'PYEOF'
path = "faiss/MultiTenantIndexIVFHierarchical.h"
with open(path) as f:
    content = f.read()

# ---- extract MemoryBreakdown ----
mb_start = content.find("    /* ────────────────────────────────────────────\n     * Memory breakdown")
mb_end = content.find("    /* profiling data structures */", mb_start)
memory_breakdown = content[mb_start:mb_end].strip()
memory_breakdown = "\n".join(line[4:] if line.startswith("    ") else line for line in memory_breakdown.split("\n"))

# ---- extract SearchProfilingData ----
sp_start = content.find("    struct SearchProfilingData {", mb_end)
sp_end = content.find("    };", sp_start) + 8
search_profiling = content[sp_start:sp_end].strip()
search_profiling = "\n".join(line[4:] if line.startswith("    ") else line for line in search_profiling.split("\n"))

# ---- remove from inside class ----
content = content[:mb_start] + content[mb_end:]
sp_start2 = content.find("    struct SearchProfilingData {")
sp_end2 = content.find("    };", sp_start2) + 8
content = content[:sp_start2] + content[sp_end2:]

# ---- insert before class ----
class_start = content.find("struct MultiTenantIndexIVFHierarchical : MultiTenantIndex {")
insert_block = "\n" + memory_breakdown + "\n\n" + search_profiling + "\n\n"
content = content[:class_start] + insert_block + content[class_start:]

with open(path, "w") as f:
    f.write(content)
print("MemoryBreakdown and SearchProfilingData moved outside class")
PYEOF

# 修复 .cpp 中的返回类型引用
sed -i 's/MultiTenantIndexIVFHierarchical::MemoryBreakdown MultiTenantIndexIVFHierarchical::get_memory_breakdown/MemoryBreakdown MultiTenantIndexIVFHierarchical::get_memory_breakdown/' faiss/MultiTenantIndexIVFHierarchical.cpp
```

### 3f. 确保 `prefetch.h` 存在

该文件由 Curator 代码引用，FAISS v1.7.4 原始版本不包含。

```bash
if [ ! -f faiss/utils/prefetch.h ]; then
    cp /home/lyx/miniconda3/envs/edge_ann/include/faiss/utils/prefetch.h faiss/utils/prefetch.h
fi
```

---

## 第四步：更新 CMakeLists.txt（添加 Curator 源文件）

```bash
CMAKE_FILE="faiss/CMakeLists.txt"

# 先清理可能残留的旧条目
sed -i '/MultiTenantIndex/d' "$CMAKE_FILE"
sed -i '/complex_predicate/d' "$CMAKE_FILE"
sed -i '/BloomFilter\.h/d' "$CMAKE_FILE"

# 添加 CPP 文件（在 IndexIVFFlat.cpp 之后）
CPP_FILES=(
    "MultiTenantIndex.cpp"
    "MultiTenantIndexIVF.cpp"
    "MultiTenantIndexIVFFlat.cpp"
    "MultiTenantIndexIVFFlatBF.cpp"
    "MultiTenantIndexIVFFlatSep.cpp"
    "MultiTenantIndexIVFHierarchical.cpp"
    "complex_predicate.cpp"
)
for cpp_file in "${CPP_FILES[@]}"; do
    sed -i "/IndexIVFFlat\\.cpp/a\\  ${cpp_file}" "$CMAKE_FILE"
done

# 添加 H 文件（在 IndexIVFFlat.h 之后）
H_FILES=(
    "MultiTenantIndex.h"
    "MultiTenantIndexIVF.h"
    "MultiTenantIndexIVFFlat.h"
    "MultiTenantIndexIVFFlatBF.h"
    "MultiTenantIndexIVFFlatSep.h"
    "MultiTenantIndexIVFHierarchical.h"
    "complex_predicate.h"
    "BloomFilter.h"
)
for h_file in "${H_FILES[@]}"; do
    sed -i "/IndexIVFFlat\\.h/a\\  ${h_file}" "$CMAKE_FILE"
done

# 验证（应各出现恰好 1 次）
echo "CPP files: $(grep -c 'MultiTenant\|complex_predicate' "$CMAKE_FILE")"
echo "H files:   $(grep -c 'BloomFilter' "$CMAKE_FILE")"
```

---

## 第五步：更新 SWIG 接口（暴露 Curator 类给 Python）

```bash
SWIG_FILE="faiss/python/swigfaiss.swig"

# 清理旧条目
sed -i '/MultiTenantIndex/d' "$SWIG_FILE"
sed -i '/IndexHybridCurator/d' "$SWIG_FILE"

# 5a. 在 C++ %{ %} 块中添加 #include（在第一个 %} 之前）
python3 << 'PYEOF'
path = "faiss/python/swigfaiss.swig"
with open(path) as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if not any("MultiTenantIndex" in l for l in new_lines[-10:]):
        if line.strip() == '%}' and 100 < i < 200:
            new_lines.append('\n')
            new_lines.append('#include <faiss/MultiTenantIndex.h>\n')
            new_lines.append('#include <faiss/MultiTenantIndexIVF.h>\n')
            new_lines.append('#include <faiss/MultiTenantIndexIVFFlat.h>\n')
            new_lines.append('#include <faiss/MultiTenantIndexIVFFlatBF.h>\n')
            new_lines.append('#include <faiss/MultiTenantIndexIVFFlatSep.h>\n')
            new_lines.append('#include <faiss/MultiTenantIndexIVFHierarchical.h>\n')
            new_lines.append('\n')
    new_lines.append(line)

with open(path, "w") as f:
    f.writelines(new_lines)
print("C++ includes inserted")
PYEOF

# 5b. 追加 SWIG %include 指令
cat >> "$SWIG_FILE" << 'SWIGEOF'

// ---- Curator additions ----
%include <faiss/MultiTenantIndex.h>
%include <faiss/MultiTenantIndexIVF.h>
%include <faiss/MultiTenantIndexIVFFlat.h>
%include <faiss/MultiTenantIndexIVFFlatBF.h>
%include <faiss/MultiTenantIndexIVFFlatSep.h>

%ignore faiss::MultiTenantIndexIVFHierarchical::flash_fp;
%ignore faiss::MultiTenantIndexIVFHierarchical::pq_codebook;
%ignore faiss::MultiTenantIndexIVFHierarchical::vid_to_pq_code;
%ignore faiss::MultiTenantIndexIVFHierarchical::raw_vectors_buffer;
%ignore faiss::MultiTenantIndexIVFHierarchical::vid_to_buffer_offset;
%ignore faiss::MultiTenantIndexIVFHierarchical::vid_to_leaf_node_id;
%ignore faiss::MultiTenantIndexIVFHierarchical::vid_to_local_index;
%ignore faiss::MultiTenantIndexIVFHierarchical::leaf_node_id_to_seq;
%ignore faiss::MultiTenantIndexIVFHierarchical::seq_to_vid;
%ignore faiss::MultiTenantIndexIVFHierarchical::vid_to_seq;

%ignore bloom_parameters;
%ignore bloom_filter;
%ignore compressible_bloom_filter;

%include <faiss/MultiTenantIndexIVFHierarchical.h>
SWIGEOF

echo "SWIG patched: $(grep -c 'MultiTenant' "$SWIG_FILE") MultiTenant refs"
```

---

## 第六步：cmake 配置

```bash
rm -rf build && mkdir build && cd build

cmake .. \
  -DBUILD_SHARED_LIBS=ON \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=avx2 \
  -DBLA_VENDOR=Intel10_64lp \
  -DCMAKE_PREFIX_PATH=/home/lyx/miniconda3/envs/edge_ann \
  -DPython_EXECUTABLE=/home/lyx/miniconda3/envs/edge_ann/bin/python
```

> **注意**：不再需要 `-DCMAKE_CXX_FLAGS="-std=c++17"`，因为已通过 `CMAKE_CXX_STANDARD 17` 设置，且 GCC 13 默认 C++17。

---

## 第七步：编译

```bash
make -j$(nproc)
```

预期输出：`[100%] Built target swigfaiss`，无 error。

---

## 第八步：安装 Python 包

```bash
cd faiss/python

# 移除旧的 conda .so 文件（这些会被 Python 优先加载）
SITE_PKGS=/home/lyx/miniconda3/envs/edge_ann/lib/python3.10/site-packages/faiss
rm -f $SITE_PKGS/_swigfaiss.cpython-310-x86_64-linux-gnu.so
rm -f $SITE_PKGS/_swigfaiss_avx2.cpython-310-x86_64-linux-gnu.so

# 安装新版本
python setup.py install

# 复制 libfaiss.so 到 site-packages（运行时链接需要）
cp ../libfaiss.so ../libfaiss_avx2.so libfaiss_python_callbacks.so $SITE_PKGS/
```

---

## 第九步：验证

```bash
cd /tmp
python3 -c "
import faiss
import numpy as np

idx = faiss.MultiTenantIndexIVFHierarchical(128, 16, faiss.METRIC_L2)
idx.set_pq_config(16, 8, True, False, 4)

bd = idx.get_memory_breakdown()
assert hasattr(bd, 'num_tree_nodes')
assert hasattr(bd, 'total_bytes')
print(f'MemoryBreakdown: {bd.total_bytes} bytes OK')

idx.enable_profiling = True
assert idx.get_last_total_search_time_ms() >= 0
assert idx.get_last_beam_search_time_ms() >= 0
print('Profiling accessors OK')

print('=== GOAL3 BUILD SUCCESS ===')
"
```

---

## 常见问题排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `ScopeDeleter was not declared` | `FaissException.h` 被 conda 旧版覆盖 | 执行第三步 3d |
| `std::filesystem has not been declared` | C++11 被启用 | 执行第三步 3a |
| `no match for call to lambda` | GCC 13 对 `const auto&` 泛型 lambda 推导失败 | 执行第三步 3c |
| `passing const bloom_filter discards qualifiers` | `hash_count()` 缺少 `const` | 执行第三步 3b |
| `SwigPyObject has no attribute 'num_tree_nodes'` | 嵌套 struct 未被 SWIG 正确解析 | 执行第三步 3e |
| `no attribute 'add_vector_with_ids_c'` | curator.py 中 SWIG 方法名有 `_c` 后缀 | 手动修改 curator.py，去 `_c` 后缀 |
| `flushed but raw_vectors_buffer != 0` | config 中 `use_flash_storage: false` | 设为 `true` 或使用 `--config` 指定正确的 JSON |
| Python 加载旧版 .so | conda 的 `.cpython-310-*.so` 被优先加载 | 第八步删除旧文件 |
