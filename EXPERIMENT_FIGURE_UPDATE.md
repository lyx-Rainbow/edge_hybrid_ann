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

## 6. High-recall supplement (2026-09-22)

The final SL/CP plans had low `pq_M` for Curator and low
`SearchInternalResultNum` (SIR) for SPANN, so several right endpoints saturated
below the desired recall.

- Curator: supplemental ext4 builds with `pq_M=192` for arxiv/wit SL and
  arxiv CP, `pq_M=96`/`pq_M=192` for yfcc100m CP.
- SPANN: supplemental `SearchInternalResultNum=20000/50000` combinations with
  overfetch factors large enough to expose the extra internal candidates.
- Existing manual anchors were preserved exactly; new measured anchors were
  appended only to the right via `5_Plot/high_recall_appends.json` and
  `5_Plot/append_high_recall_anchors.py`. The editable CSV was kept in sync.
- `bash render_latency_figures.sh` and `python verify_latency_figures.py`
  report `errors=0 warnings=0`.

Selected final right endpoints:

| case | Curator | SPANN | DiskIVF |
|---|---|---|---|
| SL arxiv 99p | 1.0000 @ 68.97 ms | 1.0000 @ 913.81 ms | 0.9994 @ 163.00 ms |
| SL wit 99p | 0.9856 @ 214.18 ms | 0.9650 @ 1391.32 ms | 0.9830 @ 651.96 ms |
| CP yfcc100m AND | 0.9975 @ 0.335 ms | 1.0000 @ 1020.12 ms | 1.0000 @ 345.05 ms |
| CP arxiv MIXED | 1.0000 @ 114.63 ms | 1.0000 @ 2701.62 ms | 1.0000 @ 2018.52 ms |
## 7. Manuscript text and split bar figures (2026-09-22)

- Added `5_Plot/figure_text_style.json` as the single text-adjustment
  interface.  Effective size = `base_size * global_scale * <kind>_scale`;
  weights for labels, ticks and titles are also configured there.  The
  `FIGURE_TEXT_CONFIG` environment variable can point to an alternative JSON
  file.  See `5_Plot/FONT_STYLE_GUIDE.md`.
- Legends intentionally keep their previous sizes (`save_legend`,
  `fig_full_bars_legend`).
- `5_Plot/fig_full_bars.py` now supports:
  - one standalone figure per metric per dataset,
    `fig_full_{memory,build_time,volume}_{dataset}.{png,svg}`;
  - one grid per metric, `fig_full_{metric}_grid.{png,svg}`;
  - the legacy combined names `fig_full_{metric}.{png,svg}` as grid aliases.
  CLI: `--metric memory|build_time|volume|all`,
  `--layout separate|grid|both`, `--datasets DATASET...`.
- `render_all_figures.sh` now also runs `verify_figure_outputs.py`.
## 8. SL gist1m high-recall supplement (2026-09-23)

- The final plan had `pq_M=160` for gist1m, giving strict right endpoints
  `0.9756` (75p) and `0.9500` (99p).
- Supplemental ext4 Curator builds:
  - `pq_M=480` with `search_ef` up to `131072`;
  - `pq_M=960` (full-dimensional PQ) with `search_ef` up to `131072`;
  - additional `pq_M=480` / `pq_rerank_topk_factor=100` rerank run.
- All raw outputs are in `4_Results/_final_ext4_raw/curator/gist1m/`; the
  new measured points were appended to the manual gist1m 75p/99p curves
  without changing the existing manual anchors.
- Final strict endpoints:

| bucket | Curator | SPANN | DiskIVF |
|---|---|---|---|
| gist1m 75p | 0.9988 @ 119.91 ms | 0.9935 @ 288.69 ms | 0.9815 @ 502.02 ms |
| gist1m 99p | 0.9917 @ 325.50 ms | 0.9917 @ 204.15 ms | 0.9917 @ 486.97 ms |

- Strict 99p recall stops at 0.9917 because the remaining GT misses are exact
  distance ties, not search-parameter misses.  For example, query 20 returns
  ID 934904 with exact squared-L2 distance 2.536712, exactly the same as GT
  ID 146458; queries 628/661/709 return ID 961300 at the same distance as the
  missed GT ID 708234.  A strict ID-intersection metric therefore cannot
  distinguish equally valid top-10 sets.  The tie-aware Recall@10 already used
  for wit would count these returned results as correct.
## 9. Text wording and content interface (2026-09-23)

- `5_Plot/figure_text_style.json` now contains both numeric style settings and
  a `content` object for user-adjustable visible strings.
- Bar-chart y-axis labels now read from this interface:
  - `bar_memory_ylabel` default `Index Memory`;
  - `bar_build_time_ylabel` default `Index Build Time (s)`;
  - `bar_volume_ylabel` default `Index Volume`;
  - `bar_dataset_xlabel` placeholder `{dataset}`.
- `5_Plot/paper_style.py` exposes `text_content(key, default)` for future
  strings; the style guide is `5_Plot/FONT_STYLE_GUIDE.md`.
- `4_Results/RESULTS_ANALYSIS.md` gist1m notes were corrected: the current
  gist1m labels use `hierarchical_random` (controlled selectivity tiers,
  geometry-independent random assignment); the old K-means coupling note no
  longer applies.