---
name: diskivf-phase2-complete
description: DiskIVF-PostFiltering C++ migration completed and verified
metadata:
  type: project
---

DiskIVF-PostFiltering (Phase 2) C++ migration completed on 2026-07-07.

## Design Decisions

1. **Cluster files use raw binary format** (vecs.bin, vids.bin, mds.bin per cluster) — cnpy is read-only, so binary fwrite/fread was mandatory.
2. **K-means seed=42** (matches FAISS Python baseline).
3. **No global label_to_vids_ index** — filtering happens after cluster loading (same as Python version), keeps memory minimal (~0.02 MB for centroids only).
4. **Top-k maintained via sorted vector insert** (not priority_queue) — allows deterministic results.
5. **Flat directory naming** (c0_vecs.bin, c0_vids.bin, c0_mds.bin) — simpler than subdirectories per cluster.
6. **Original Python code preserved** untouched in run_diskivf.py.

## Verified Results (arxiv_small, 50K×384, nlist=16)

| Test | nprobe | Recall@10 | 
|------|:---:|:---:|
| Single-label (exhaustive) | 16 | 1.0000 (500/500) |
| Single-label (approximate) | 8 | 0.9968 (490/500) |
| Complex-predicate (exhaustive) | 16 | 1.0000 (40/40) |
| Unfiltered | 16 | Works correctly |

Build time: ~0.9s, Memory: 0.02 MB, Disk: 73.97 MB.
Search latency: 6.33 ms/query (nprobe=8), 12.80 ms/query (nprobe=16).

## Adaptations from Curator

- kmeans.h/.cpp: removed namespace curator, replaced CURATOR_THROW_MSG → THROW_MSG, changed #include "common.h" → #include "config.h", changed default seed 1234 → 42.
- cnpy.h/.cpp: verbatim copy from Pre-Filtering (which copied from Curator).
- distance.h: same as Pre-Filtering (stripped version of Curator's distance.h).

## Implementation Files

- `src/config.h` — Common macros + DiskIVFConfig struct
- `src/diskivf_index.h` — Class declaration
- `src/diskivf_index.cpp` — build(), search(), search_with_predicate(), search_unfiltered(), cluster I/O
- `src/main.cpp` — CLI entry (bench mode)
- `CMakeLists.txt` — Independent CMake build
- `python/preprocess.py` — train_mds.pkl → train_access.npy
- `python/run_experiment.py` — Experiment orchestration

**Why:** DiskIVF needed C++ migration to match Curator and Pre-Filtering architecture. Python FAISS dependency replaced with adapted Curator K-means. Binary disk I/O for cluster files avoids cnpy write limitation.

**How to apply:** Build with `cd DiskIVF-PostFiltering/build && cmake .. && make -j`, then run via `./build/diskivf bench --train_vecs ... --queries ...` or use `python/python/run_experiment.py --dataset arxiv_small`.
