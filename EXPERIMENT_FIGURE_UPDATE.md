# Final figure, query-memory and index-volume update

Date: 2026-09-19; updated 2026-09-20

## 1. Latency-Recall line figures

Final query-performance figures are in `4_Results/fig_full/`.
Deleted as requested:
- `4_Results/fig_100k/`
- `4_Results/fig_full_latency/`

### PreFilter

- PreFilter in every legend now uses the normal marker size.
- PreFilter in `fig_full_latency_at_recall_090/095` uses the normal marker
  size and is connected by a solid line.

### Wit tie-aware single-label recall

Wit contains large groups of exactly duplicated vectors with different labels.
Exact label-filtered top-10 sets are therefore not unique at tied distances,
and strict ID-intersection recall cannot reach 1.0 even for an exact scan.
For wit we now compute tie-aware Recall@10: a returned vector with the query
label counts as correct if it is in the GT set or its distance is within the
GT k-th distance (plus tolerance).  ``ground_truth_cutoff.npy`` is produced by
``1_Data/compute_gt_cutoffs.py``.  The wit SL sweeps and all wit SL figures,
as well as `fig_full_latency_at_recall_090/095`, use this metric.

### Curve construction (not globally smoothed)

`5_Plot/latency_trend.py` now works on actual measured points:

1. select a small measured-anchor sequence using the existing QPS-monotone
   path selector;
2. force the first/last anchors to the real minimum/maximum Recall and choose
   the lowest-latency measurement at those Recall values;
3. merge/remove anchors that overlap in Recall;
4. insert additional actual measured points whenever two adjacent anchors have
   a latency jump larger than the configured ratio;
5. locally nudge only the later anchor of a still-too-steep pair so that
   adjacent latency ratios are <= 2.5;
6. draw a shape-preserving PCHIP through the repaired measured anchors.

This keeps the measured non-linear shape (unlike a global isotonic/spline
smoother) while removing hard corners, near-vertical jumps, and overlapping
anchor points.  Anchors are actual or very lightly adjusted measurement points.

## 2. Query-phase memory figure

The memory bar (`fig_full_memory.png`) uses method-specific memory-oriented
overhead configurations, separate from the QPS sweeps.

### PreFilter

Build-only NPY buffers are released after `index.build()`, so query RSS equals
the raw vector payload plus label/query metadata.

### Curator

Memory-oriented build parameters:
`n_clusters=8, pq_cache_max_blocks=64, use_temp_index_caching=false,
SL predicate-only maps released`, with `max_leaf_size=1024` for all datasets
except `wit`. The wit dataset contains a group of ~32,455 exactly duplicated
vectors (plus several smaller duplicate groups); a 1024-vector leaf cannot
represent them. Wit was therefore measured with `max_leaf_size=32768` and an
expanded internal leaf-ID field so the build can complete. All five datasets
were freshly measured on 2026-09-20; the same build also provides the ext4
allocated bytes used by `fig_full_volume`:

| dataset | old default RSS | measured memory-oriented RSS | max_leaf_size |
|---|---:|---:|---:|
| sift1m | 682.0 | 314.8 | 1024 |
| gist1m | 940.1 | 304.9 | 1024 |
| arxiv | 1066.9 | 473.8 | 1024 |
| yfcc100m | 991.8 | 380.9 | 1024 |
| wit | 2808.8 | 825.5 | 32768 (duplicates) |

### Duplicate-aware Curator build fix

Wit contains ~32,455 exactly duplicated vectors.  The original 8/1024 tree
stopped at `MAX_TREE_DEPTH` with oversized leaves, and `add_vector` could not
path-encode the 1025th vector in a leaf.  The following fixes were made:

- k-means now re-seeds empty clusters with random training points (instead of
  global-mean noise) and performs a final assignment pass so `build_tree`'s
  partition agrees with `add_vector`'s nearest-centroid assignment;
- the internal leaf-ID field was expanded (`MAX_LEAF_SIZE_LOG2`: 10 -> 16);
- `add_vector` now raises a clear error instead of silently corrupting a leaf;
- for wit only, `max_leaf_size` was raised to 32768 (the largest duplicate
  group is 32,455); the other four datasets keep 1024.

### DiskIVF

Added `--preload-clusters` overhead-only search mode: all `nprobe` clusters are
loaded before scanning.  QPS results are unchanged; this mode is only used for
memory measurement.  Measured with one query and `nprobe=128`:

| dataset | old RSS | new RSS |
|---|---:|---:|
| sift1m | 15.9 | 70.0 |
| gist1m | 88.9 | 597.0 |
| arxiv | 49.7 | 283.5 |
| yfcc100m | 21.2 | 102.8 |
| wit | 173.9 | 707.6 |

### SPANN

SPANN search-mode values are unchanged.

## 3. Memory + external volume figure

`fig_full_volume.png` is regenerated from the same updated metrics as
`fig_full_memory.png`.  Its lower (hollow) segment uses exactly the same
`query_memory_mb` values as the memory figure, so the memory component of the
two figures is directly consistent.  The upper (shaded) segment is the measured
external/disk index storage, which is independent of query-phase memory.

Meaning of the two bar parts:

- **Hollow/white part (lower)**: query-phase peak memory from
  `query_memory_mb` (the same value as `fig_full_memory`).  For PreFilter this
  includes the resident vector payload and query metadata; for Curator it is
  the memory-oriented build/query footprint (wit uses `max_leaf_size=32768`
  because of its huge duplicate-vector group); for DiskIVF it is the query RSS
  with preloaded clusters; for SPANN it is the query-phase sweep maximum.
- **Shaded/filled part (upper)**: external/disk index storage
  (`volume_breakdown_mb.external`) - flash/PQ files for Curator, cluster files
  for DiskIVF, SSD posting/head-index files for SPANN.  PreFilter has no
  external part.

Total bar height = query-phase memory + external index storage.
Method is encoded by colour/hatch; hollow vs shaded encodes memory vs external
storage.

## 4. Reproduce

```bash
bash render_all_figures.sh
bash render_latency_figures.sh
python verify_latency_figures.py
python verify_final_experiments.py
```

## 5. Manual latency-curve editing

Manual point selection and position adjustment use:

- `5_Plot/plot_manual_candidate_ids.py` (numbered candidate-ID figures)
- `5_Plot/build_manual_points_table.py` (human-editable CSV table)
- `5_Plot/manual_latency_points.csv`
- `5_Plot/apply_manual_latency_points.py`
- `5_Plot/reset_manual_latency_overrides.py`
- `5_Plot/manual_latency_overrides.json`
- `5_Plot/MANUAL_LATENCY_GUIDE.md`

Workflow:

```bash
python 5_Plot/plot_manual_candidate_ids.py
python 5_Plot/build_manual_points_table.py
# edit 5_Plot/manual_latency_points.csv
python 5_Plot/apply_manual_latency_points.py
bash render_latency_figures.sh
```

Rebuilding the points table preserves existing manual `use`, `dr`,
`dlat_ms`, `scale`, `interpolation`, and coordinate values by default.
Use `--reset` to regenerate fresh defaults.

The override file supports manual `selected_ids`, `adjustments`
(`dr`, `dlat_ms`, `scale`), explicit `anchors`, and
`interpolation: "pchip" | "linear"`.

The `matched_recall` section of the override file controls
`fig_full_latency_at_recall_090/095`; see MANUAL_LATENCY_GUIDE.md section 7 for the PreFilter line behavior.
