# SIFT1M Full-Scale Experiment Configuration Summary

## 1. Curator
### Fixed parameters
- n_clusters=32, max_sl_size=256, max_leaf_size=128
- clus_niter=20, pq_nbits=8
- pq_enabled=true, pq_use_adc_rerank=true
- pq_rerank_topk_factor=4
- pq_cache_block_size=256, pq_cache_max_blocks=4096
- use_flash_storage=true, nprobe=8
- prune_thres=0.0, variance_boost=0.2
- beam_size=4, use_temp_index_caching=false
- batch_query=false

### Sweep parameters (sift1m)
- pq_M: [32, 64, 128]
- search_ef: fine grid including [1,2,4,8,10,12,14,16,32,40,48,56,64,80,96,112,128,144,160,192,256,384,512,768,1024,1536,2048,4096] plus high-recall fill [850,1000,1200,1400,1800,2200,2500,3000,3500]
- Additional nprobe=4 variants were tested for gap analysis.

## 2. DiskIVF
- nlist=1000 (≈√N, same convention as other full datasets)
- nprobe: [1,2,3,4,6,8,12,16,20,24,32,48,64,80,96,128]
- Build once + search mode, batch query enabled.

## 3. SPANN
### Index fixed
- dist_method=L2, num_threads=1
- hash_exp=10, bkt_kmeans_k=64 (old default index)
- A finer test index was also built: hash_exp=12, bkt_kmeans_k=128
- overfetch_adaptive=false, batch_query=false

### Search parameters
- max_check grid: [64,128,256,512,1024,2048,4096,8192,16384,32768,49152,65536]
- overfetch_factor: dense integer and fractional values
- SearchInternalResultNum probes: [50,100,150,200,250,300,350,400,500,1000,2000,5000]

## 4. PreFilter
- Real external-memory chunked scan, not simulated I/O.
- Reads the entire vector file in fixed-size blocks from disk.
- For each block: load block -> filter candidates -> brute-force distance on filtered vectors -> maintain global top-k.
- sift1m validation uses scan_chunk_vectors=1024 and a 100-query stratified subset (20 per bucket) because full disk scan is very slow in this environment.
- Vector file is stored on WSL local ext4 for real I/O measurement.

## 5. Plotting
- Separate legend figure: `4_Results/fig_full/legend.png/.svg`
- Line plots: no legend, no dataset name in title
- Title: selectivity bucket and median selectivity only
- X-axis starts at 0.5, log-scale QPS
- Axis labels fontsize=20, ticks fontsize=16, title fontsize=18
- Main lines: lw=3.5, marker size=9, marker edge width=2.2
- All raw measured points shown as small semi-transparent markers under the mainline
- Curator 50p/75p/99p drawn from real non-dominated Pareto points, with tiny recall-gap pruning (>0.001) to avoid marker overlap

## 6. Remaining datasets
- arxiv, yfcc100m: natural labels, use actual selectivity distribution for buckets.
- gist1m, wit: synthetic labels; should design selectivity distribution similar to sift1m (wide separation from ~0.05% to ~30%).
- Full sweep configs created under:
  - `3_Config/Curator/sweep_full_<ds>.json`
  - `3_Config/DiskIVF-PostFiltering/sweep_full_<ds>.json`
  - `3_Config/SPANN-PostFiltering/sweep_full_<ds>.json`
  - `3_Config/Pre-Filtering/sweep_full_<ds>.json`
- Long-running pipeline script: `run_full_remaining_pipeline.py`
