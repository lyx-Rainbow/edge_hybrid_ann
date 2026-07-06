#!/bin/bash
# ============================================================================
# build_sptag.sh — SPTAG-main 一键编译脚本
#
# 功能:
#   1. 自动应用 4 处必要补丁（zstd 子模块、C++17 兼容性、不需要的子目录）
#   2. 编译 SPTAG 静态库（libSPTAGLibStatic.a + libDistanceUtils.a）
#   3. 验证编译产物
#
# 用法:
#   cd SPANN-PostFiltering
#   bash build_sptag.sh
#
# 前置条件:
#   - WSL Ubuntu 24.04
#   - 已安装系统依赖（见 description.md §3.2）
#   - sudo 已配置（用于 apt install，编译本身不需要 sudo）
#
# 预计时间: 首次编译 5-10 分钟（取决于 CPU 核心数）
# ============================================================================
set -e

SPTAG_SRC=/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines/SPANN-PostFiltering/SPTAG-main
SPTAG_BLD=$SPTAG_SRC/Release

echo "============================================"
echo " SPTAG-main Build Script"
echo " Source: $SPTAG_SRC"
echo " Build:  $SPTAG_BLD"
echo "============================================"

# ── Step 0: Verify dependencies ──
echo ""
echo "[0/5] Checking dependencies..."
MISSING=""
for pkg in cmake g++ make; do
    if ! which $pkg > /dev/null 2>&1; then
        MISSING="$MISSING $pkg"
    fi
done
if ! ldconfig -p 2>/dev/null | grep -q libboost_system; then
    MISSING="$MISSING boost"
fi
if ! ldconfig -p 2>/dev/null | grep -q libtbb; then
    MISSING="$MISSING tbb"
fi
if [ -n "$MISSING" ]; then
    echo "ERROR: Missing dependencies:$MISSING"
    echo "Run: sudo apt install -y libboost-system-dev libboost-thread-dev libboost-serialization-dev libboost-filesystem-dev libboost-regex-dev libtbb-dev libzstd-dev libomp-dev cmake g++ make"
    exit 1
fi
echo "  All dependencies found."

# ── Step 1: Apply SPTAG patches ──
echo ""
echo "[1/5] Applying SPTAG build patches..."

PATCHED=0

# Patch 1: Comment out non-essential subdirectories (Test, GPUSupport, Wrappers)
# Reason: Wrappers requires C# which is not installed; GPUSupport requires CUDA
TOP_CMAKE=$SPTAG_SRC/CMakeLists.txt
if grep -q "^add_subdirectory (Test)" "$TOP_CMAKE" 2>/dev/null; then
    cp "$TOP_CMAKE" "$TOP_CMAKE.bak"
    sed -i "s/^add_subdirectory (Test)/#add_subdirectory (Test)/" "$TOP_CMAKE"
    sed -i "s/^add_subdirectory (GPUSupport)/#add_subdirectory (GPUSupport)/" "$TOP_CMAKE"
    sed -i "s/^add_subdirectory (Wrappers)/#add_subdirectory (Wrappers)/" "$TOP_CMAKE"
    echo "  Patch 1: Disabled Test, GPUSupport, Wrappers subdirectories."
    PATCHED=1
else
    echo "  Patch 1: Already applied (subdirectories disabled)."
fi

# Patch 2: Disable vendored zstd (submodule not initialized) → use system zstd
# Reason: ThirdParty/zstd/ is an empty git submodule directory
if grep -q "^add_subdirectory (ThirdParty/zstd/build/cmake)" "$TOP_CMAKE" 2>/dev/null; then
    sed -i "s|^add_subdirectory (ThirdParty/zstd/build/cmake)|#add_subdirectory (ThirdParty/zstd/build/cmake)  # using system zstd|" "$TOP_CMAKE"
    echo "  Patch 2: Disabled vendored zstd, will use system libzstd."
    PATCHED=1
else
    echo "  Patch 2: Already applied (using system zstd)."
fi

# Patch 3: Switch AnnService CMakeLists.txt to system zstd
# Reason: vendored targets libzstd_static/libzstd_shared don't exist
ANN_CMAKE=$SPTAG_SRC/AnnService/CMakeLists.txt
if grep -q "libzstd_static" "$ANN_CMAKE" 2>/dev/null; then
    cp "$ANN_CMAKE" "$ANN_CMAKE.bak"
    sed -i "s|include_directories(\${Zstd}/lib)|#include_directories(\${Zstd}/lib)  # using system zstd|" "$ANN_CMAKE"
    sed -i "s|libzstd_shared|-lzstd|g" "$ANN_CMAKE"
    sed -i "s|libzstd_static|-lzstd|g" "$ANN_CMAKE"
    echo "  Patch 3: Switched AnnService to system -lzstd."
    PATCHED=1
else
    echo "  Patch 3: Already applied (system zstd in use)."
fi

# Patch 4: Fix C++17 compatibility in ConcurrentSet.h
# Reason: 'size_type' is not a standard type; should be 'size_t'
CCT=$SPTAG_SRC/AnnService/inc/Helper/ConcurrentSet.h
if grep -q "size_type size()" "$CCT" 2>/dev/null; then
    cp "$CCT" "$CCT.bak"
    sed -i "s/size_type size()/size_t size()/" "$CCT"
    echo "  Patch 4: Fixed size_type → size_t in ConcurrentSet.h."
    PATCHED=1
else
    echo "  Patch 4: Already applied (size_type fixed)."
fi

if [ "$PATCHED" -eq 0 ]; then
    echo "  All 4 patches already applied — skipping."
fi

# ── Step 2: Clean build directory ──
echo ""
echo "[2/5] Preparing build directory..."
rm -rf "$SPTAG_BLD"
mkdir -p "$SPTAG_BLD"

# ── Step 3: CMake configure (est. 30-60 sec) ──
echo ""
echo "[3/5] CMake configure (est. 30-60s)..."
cmake -S "$SPTAG_SRC" -B "$SPTAG_BLD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBRARYONLY=ON \
    -DROCKSDB=OFF \
    -DSPDK=OFF \
    -DURING=OFF \
    -DTBB=ON \
    2>&1 | grep -E "Configuring|Generating|Build files|Error|error" || true

if [ ! -f "$SPTAG_BLD/Makefile" ]; then
    echo "ERROR: CMake configure failed — no Makefile generated."
    echo "Check full output above for details."
    exit 1
fi
echo "  CMake configure: OK"

# ── Step 4: Build (est. 5-10 min) ──
echo ""
echo "[4/5] Building SPTAG (est. 5-10 min, using $(nproc) cores)..."
make -C "$SPTAG_BLD" -j$(nproc) 2>&1 | tail -20
BUILD_EXIT=${PIPESTATUS[0]}
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "ERROR: Build failed (exit code $BUILD_EXIT)."
    exit 1
fi
echo "  Build: OK"

# ── Step 5: Verify artifacts ──
echo ""
echo "[5/5] Verifying build artifacts..."
ARTIFACTS_OK=1
for lib in libSPTAGLibStatic.a libDistanceUtils.a; do
    if [ -f "$SPTAG_BLD/$lib" ]; then
        SIZE=$(stat -c%s "$SPTAG_BLD/$lib" 2>/dev/null || echo "?")
        echo "  $lib: ${SIZE} bytes — OK"
    else
        echo "  $lib: MISSING"
        ARTIFACTS_OK=0
    fi
done

if [ "$ARTIFACTS_OK" -eq 1 ]; then
    echo ""
    echo "============================================"
    echo " SPTAG build SUCCESS"
    echo ""
    echo " Artifacts:"
    ls -lh "$SPTAG_BLD"/libSPTAGLibStatic.a "$SPTAG_BLD"/libDistanceUtils.a
    echo ""
    echo " Next step:"
    echo "   cd SPANN-PostFiltering"
    echo "   mkdir -p build && cd build"
    echo "   cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "   make -j\$(nproc)"
    echo "============================================"
else
    echo ""
    echo "ERROR: Some artifacts missing — build may have partially failed."
    exit 1
fi
