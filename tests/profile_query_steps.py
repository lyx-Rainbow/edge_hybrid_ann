"""
Profile Curator query steps (beam search, frontier, PQ, rerank, etc.)

Usage:
    python tests/profile_query_steps.py --dataset yfcc100m_small --n_queries 100
    python tests/profile_query_steps.py --dataset arxiv_small --n_queries 50
    python tests/profile_query_steps.py --dataset yfcc100m_small --n_queries 200 \
        --sweep '{"search_ef": [64, 128, 256, 512]}'
"""

import argparse
import gc
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_Utils"))
sys.path.insert(0, str(PROJECT_ROOT / "Curator" / "python"))
from query_profiler import QueryProfiler  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Profile Curator query steps"
    )
    parser.add_argument("--dataset", required=True,
                        choices=["yfcc100m", "arxiv", "yfcc100m_small", "arxiv_small"])
    parser.add_argument("--data_dir", default="1_Data")
    parser.add_argument("--output_dir", default="4_Results/query_profile")
    parser.add_argument("--n_queries", type=int, default=100,
                        help="Number of queries to profile")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num_warmup", type=int, default=10,
                        help="Warmup queries before profiling")
    parser.add_argument("--sweep", type=str, default=None,
                        help='JSON sweep config, e.g. \'{"search_ef": [64,128,256]}\'')
    args = parser.parse_args()

    try:
        from curator import Curator  # noqa: E402
    except ImportError as e:
        print(f"Curator not available: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Curator Query Step Profiler: {args.dataset}")
    print(f"{'='*60}")

    # ---- load config ----
    config_path = PROJECT_ROOT / f"3_Config/Curator/curator_{args.dataset}.json"
    import json as _json
    if config_path.exists():
        with open(config_path) as f:
            cfg = _json.load(f)
    else:
        from run_curator import BUILTIN_DEFAULTS
        cfg = BUILTIN_DEFAULTS[args.dataset]

    # ---- load data ----
    gt_dir = Path(args.data_dir) / "ground_truth" / args.dataset
    train_vecs = np.load(gt_dir / "train_vecs.npy").astype(np.float32)
    with open(gt_dir / "train_mds.pkl", "rb") as f:
        train_mds = pickle.load(f)
    query_vecs = np.load(gt_dir / "query_vecs.npy").astype(np.float32)
    query_labels = np.load(gt_dir / "query_labels.npy").astype(np.int32)
    with open(gt_dir / "query_info.json", "r") as f:
        query_info = _json.load(f)

    d = cfg["d"]
    build_cfg = cfg["build"]
    search_cfg = cfg.get("search", {})
    print(f"  Vectors: {len(train_vecs):,} x {d}")
    print(f"  Queries: {len(query_vecs)}, k={args.k}")

    # ---- build index ----
    print(f"\n  Building index...")
    index = Curator(
        d=d, nlist=build_cfg["nlist"],
        bf_capacity=build_cfg.get("bf_capacity", 1000),
        bf_error_rate=build_cfg.get("bf_error_rate", 0.001),
        max_sl_size=build_cfg.get("max_sl_size", 128),
        clus_niter=build_cfg.get("clus_niter", 20),
        max_leaf_size=build_cfg.get("max_leaf_size", 128),
        variance_boost=search_cfg.get("variance_boost", 0.2),
        search_ef=search_cfg.get("search_ef", 128),
        beam_size=search_cfg.get("beam_size", 1),
        use_temp_index_caching=search_cfg.get("use_temp_index_caching", True),
        pq_M=build_cfg["pq_M"], pq_nbits=build_cfg["pq_nbits"],
        pq_enabled=build_cfg.get("pq_enabled", True),
        pq_use_adc_rerank=build_cfg.get("pq_use_adc_rerank", False),
        pq_rerank_topk_factor=build_cfg.get("pq_rerank_topk_factor", 4),
        use_flash_storage=build_cfg.get("use_flash_storage", False),
    )

    print("    Training...")
    index.train(train_vecs)
    print(f"    Adding {len(train_vecs)} vectors...")
    for i in range(len(train_vecs)):
        index.create(train_vecs[i], label=i)
    print("    Granting access...")
    for i, labels in enumerate(train_mds):
        for lab in labels:
            index.grant_access(i, int(lab))
    print("    Flushing...")
    index.flush()

    # Free Python-side data
    del train_vecs, train_mds
    gc.collect()

    profiler = QueryProfiler(index)

    # Determine sweep combos
    if args.sweep:
        sweep_cfg = _json.loads(args.sweep)
        keys = list(sweep_cfg.keys())
        from itertools import product
        combos = [dict(zip(keys, vs)) for vs in product(*sweep_cfg.values())]
    else:
        combos = [{}]

    n_q = min(args.n_queries, len(query_vecs))
    all_results = []

    for combo in combos:
        if combo:
            index.search_params = {**search_cfg, **combo}
            print(f"\n  ── Params: {combo} ──")
        else:
            print(f"\n  ── Default params ──")

        # Warmup
        for i in range(min(args.num_warmup, n_q)):
            tid = int(query_labels[i])
            index.query(query_vecs[i], k=args.k, tenant_id=tid)

        # Profile
        print(f"    Profiling {n_q} queries...")
        profiles = []
        for i in range(n_q):
            tid = int(query_labels[i])
            r = profiler.profile_single(query_vecs[i], k=args.k, tenant_id=tid)
            r["query_idx"] = i
            r["label"] = tid
            r["bucket"] = query_info[i].get("bucket", "unknown")
            r["selectivity"] = query_info[i].get("selectivity", 0.0)
            profiles.append(r)

            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{n_q}")

        # Aggregate per step
        steps = [
            "total_search_time_ms",
            "beam_search_time_ms",
            "frontier_search_time_ms",
            "pq_table_build_time_ms",
            "pq_distance_compute_time_ms",
            "exact_distance_compute_time_ms",
            "candidate_merge_time_ms",
            "rerank_time_ms",
        ]
        agg = {"params": combo}
        for s in steps:
            vals = [p.get(s, 0.0) or 0.0 for p in profiles]
            agg[f"{s}_mean"] = float(np.mean(vals))
            agg[f"{s}_p50"] = float(np.percentile(vals, 50))
            agg[f"{s}_p95"] = float(np.percentile(vals, 95))

        # Per-bucket aggregation
        buckets: dict[str, list] = {}
        for p in profiles:
            buckets.setdefault(p["bucket"], []).append(p)
        agg["per_bucket"] = {}
        for bname, bprofs in buckets.items():
            b_agg = {"count": len(bprofs)}
            for s in steps:
                vals = [p.get(s, 0.0) or 0.0 for p in bprofs]
                b_agg[f"{s}_mean"] = float(np.mean(vals))
            agg["per_bucket"][bname] = b_agg

        all_results.append(agg)

        # Print summary
        print(f"\n    Step breakdown (mean ms):")
        print(f"      {'Step':<32s} {'Time':>8s} {'%':>7s}")
        total = agg.get("total_search_time_ms_mean", 1.0) or 1.0
        for s in steps:
            name = s.replace("_search_time_ms", "").replace("_time_ms", "").replace("_", " ")
            val = agg.get(f"{s}_mean", 0.0) or 0.0
            print(f"      {name:<32s} {val:8.2f} {val/total*100:6.1f}%")

        # Print per-bucket
        print(f"\n    Per selectivity bucket:")
        print(f"      {'Bucket':<32s} {'Count':>5s} {'Total':>8s} {'Beam':>8s} {'Frontier':>9s} {'PQDist':>8s} {'Rerank':>8s}")
        for bname in sorted(agg["per_bucket"].keys()):
            b = agg["per_bucket"][bname]
            print(f"      {bname:<32s} {b['count']:5d} "
                  f"{b.get('total_search_time_ms_mean',0):8.2f} "
                  f"{b.get('beam_search_time_ms_mean',0):8.2f} "
                  f"{b.get('frontier_search_time_ms_mean',0):9.2f} "
                  f"{b.get('pq_distance_compute_time_ms_mean',0):8.2f} "
                  f"{b.get('rerank_time_ms_mean',0):8.2f}")

    # ---- save ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": args.dataset,
        "n_queries": n_q,
        "k": args.k,
        "sweep": str(args.sweep) if args.sweep else None,
        "results": all_results,
    }
    out_path = out_dir / f"query_profile_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Saved: {out_path.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
