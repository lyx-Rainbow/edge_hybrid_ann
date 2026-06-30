#!/bin/bash
# ============================================================
# Curator Build Script for WSL Ubuntu
# Builds FAISS v1.7.4 with Curator C++ files merged in.
#
# Usage:
#   chmod +x setup_curator.sh
#   ./setup_curator.sh
#
# If FAISS is already cloned at ~/faiss_build/faiss, only
# steps 3-7 are re-executed (copy + patch + build + install).
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_SRC="$SCRIPT_DIR/Curator/src"
BUILD_DIR="$HOME/faiss_build"

echo "============================================================"
echo "Curator + FAISS Build Script"
echo "============================================================"

# ---- Step 1: Install system dependencies ----
echo ""
echo "[Step 1] Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    python3-dev \
    swig \
    git \
    wget

# ---- Step 2: Clone FAISS v1.7.4 ----
echo ""
echo "[Step 2] Cloning FAISS v1.7.4..."
if [ -d "$BUILD_DIR/faiss" ]; then
    echo "  FAISS already cloned, skipping..."
else
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    git clone --branch v1.7.4 --depth 1 https://github.com/facebookresearch/faiss.git
fi

FAISS_DIR="$BUILD_DIR/faiss"

# ---- Step 3: Copy Curator C++ files into FAISS ----
echo ""
echo "[Step 3] Copying Curator C++ source files into FAISS..."

# Core Curator files -> faiss/faiss/
cp "$CURATOR_SRC/MultiTenantIndex.h"            "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndex.cpp"          "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndexIVF.h"         "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndexIVF.cpp"       "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndexIVFFlat.h"     "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndexIVFFlat.cpp"   "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndexIVFFlatBF.h"   "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndexIVFFlatBF.cpp" "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndexIVFFlatSep.h"  "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndexIVFFlatSep.cpp" "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/BloomFilter.h"                 "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/complex_predicate.h"           "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/complex_predicate.cpp"         "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndexIVFHierarchical.h"  "$FAISS_DIR/faiss/"
cp "$CURATOR_SRC/MultiTenantIndexIVFHierarchical.cpp" "$FAISS_DIR/faiss/"

# ACORN impl files -> faiss/impl/
cp "$CURATOR_SRC/ACORN.h"   "$FAISS_DIR/faiss/impl/"
cp "$CURATOR_SRC/ACORN.cpp" "$FAISS_DIR/faiss/impl/"

# CRITICAL: Overwrite FAISS originals with Curator-modified versions
# MetricType.h    - adds Buffer, tid_t, vid_t, label_t, AccessMap types
# impl/IDSelector.h/.cpp - adds MultiTenantIDSelector
cp "$CURATOR_SRC/MetricType.h"        "$FAISS_DIR/faiss/MetricType.h"
cp "$CURATOR_SRC/impl/IDSelector.h"   "$FAISS_DIR/faiss/impl/IDSelector.h"
cp "$CURATOR_SRC/impl/IDSelector.cpp" "$FAISS_DIR/faiss/impl/IDSelector.cpp"

echo "  Files copied (17 Curator + 3 FAISS overwrites)."

# ---- Step 4: Patch CMakeLists.txt ----
echo ""
echo "[Step 4] Patching CMakeLists.txt..."

CMAKE_FILE="$FAISS_DIR/faiss/CMakeLists.txt"

# Remove stale Curator entries (in case of re-run)
sed -i '/MultiTenantIndex/d' "$CMAKE_FILE"
sed -i '/complex_predicate/d' "$CMAKE_FILE"
sed -i '/MultiTenantIndexIVFHierarchical/d' "$CMAKE_FILE"
sed -i '/BloomFilter\.h/d' "$CMAKE_FILE"

# Add Curator .cpp files to FAISS_SRC (after IndexIVFFlat.cpp, before impl/)
CURATOR_CPP=(
    "MultiTenantIndex.cpp"
    "MultiTenantIndexIVF.cpp"
    "MultiTenantIndexIVFFlat.cpp"
    "MultiTenantIndexIVFFlatBF.cpp"
    "MultiTenantIndexIVFFlatSep.cpp"
    "MultiTenantIndexIVFHierarchical.cpp"
    "complex_predicate.cpp"
)

for cpp_file in "${CURATOR_CPP[@]}"; do
    sed -i "/IndexIVFFlat\\.cpp/a\\  ${cpp_file}" "$CMAKE_FILE"
done

# Add Curator .h files to FAISS_HEADERS (after IndexIVFFlat.h)
CURATOR_H=(
    "MultiTenantIndex.h"
    "MultiTenantIndexIVF.h"
    "MultiTenantIndexIVFFlat.h"
    "MultiTenantIndexIVFFlatBF.h"
    "MultiTenantIndexIVFFlatSep.h"
    "MultiTenantIndexIVFHierarchical.h"
    "complex_predicate.h"
    "BloomFilter.h"
)

for h_file in "${CURATOR_H[@]}"; do
    sed -i "/IndexIVFFlat\\.h/a\\  ${h_file}" "$CMAKE_FILE"
done

echo "  CMakeLists.txt patched (8 .cpp + 8 .h added)."

# ---- Step 5: Patch SWIG bindings ----
echo ""
echo "[Step 5] Patching SWIG bindings..."

SWIG_FILE="$FAISS_DIR/faiss/python/swigfaiss.swig"

# Remove any prior Curator additions (idempotent re-run)
sed -i '/MultiTenantIndex/d' "$SWIG_FILE"
sed -i '/IndexHybridCurator/d' "$SWIG_FILE"
sed -i '/cast_integer_to_uint32_ptr/d' "$SWIG_FILE"

# 5a. Add C++ #include lines inside the %{ %} block.
#     Insert after the last existing #include line before the closing %}
CLOSE_PCT=$(grep -n '^%}$' "$SWIG_FILE" | head -1 | cut -d: -f1)
if [ -n "$CLOSE_PCT" ]; then
    INSERT_LINE=$((CLOSE_PCT - 1))
    sed -i "${INSERT_LINE}a\\
#include <faiss/MultiTenantIndex.h>\\
#include <faiss/MultiTenantIndexIVF.h>\\
#include <faiss/MultiTenantIndexIVFFlat.h>\\
#include <faiss/MultiTenantIndexIVFFlatBF.h>\\
#include <faiss/MultiTenantIndexIVFFlatSep.h>\\
#include <faiss/MultiTenantIndexIVFHierarchical.h>" "$SWIG_FILE"
    echo "  C++ includes inserted."
fi

# 5b. Append SWIG %ignore / %include at end of file
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

echo "  SWIG %include and %ignore appended."

# ---- Step 6: Build FAISS ----
echo ""
echo "[Step 6] Building FAISS with Curator..."
cd "$FAISS_DIR"

rm -rf build
cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DFAISS_OPT_LEVEL=generic \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLA_VENDOR=OpenBLAS .

cmake --build build --config Release -j$(nproc)

# ---- Step 7: Install FAISS Python package ----
echo ""
echo "[Step 7] Installing FAISS Python package..."
cd "$FAISS_DIR/build/faiss/python"
python setup.py install

# ---- Step 8: Verify ----
echo ""
echo "[Step 8] Verifying installation..."
python -c "
import faiss
import numpy as np

# Test Curator
idx = faiss.MultiTenantIndexIVFHierarchical(128, 16, faiss.METRIC_L2)
print('MultiTenantIndexIVFHierarchical: OK')

# Test PQ + Flash
idx.set_pq_config(16, 8, True, False, 4)
idx.set_use_flash_storage(True)
print('PQ + Flash config: OK')

print('')
print('============================================================')
print('Curator installation verified successfully!')
print('============================================================')
"

echo ""
echo "Done! Curator is ready to use."
echo "  Python wrapper is at: $SCRIPT_DIR/Curator/python/"
