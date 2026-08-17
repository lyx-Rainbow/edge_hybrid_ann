#!/usr/bin/env python3
"""Quick-check one C++ output JSON: prints SL recall/latency/QPS.
Usage: python tests/quick_check.py <results.json> <dataset> [filter_expr]
"""
import json, sys
import numpy as np

sys.path.insert(0, ".")
import run_100k_sweep as R


def main():
    path, dataset = sys.argv[1], sys.argv[2]
    filter_expr = sys.argv[3] if len(sys.argv) > 3 else None
    res = json.load(open(path))
    gt = R.load_gt(dataset, filter_expr)

    if "filters_results" in res and res["filters_results"]:
        # Batch CP mode: aggregate across filters
        all_recs, all_lats = [], []
        for fr in res["filters_results"]:
            gtf = gt[fr["filter"]]
            for q_idx, qr in enumerate(fr["queries"]):
                all_recs.append(R.compute_recall(qr["labels"], gtf[q_idx], 10))
                st = qr.get("search_time_us", 0)
                if st > 0:
                    all_lats.append(st / 1000.0)
        print(f"CP filters={len(res['filters_results'])} "
              f"recall={np.mean(all_recs):.4f} "
              f"lat_ms={np.mean(all_lats):.2f} qps={1000.0 / np.mean(all_lats):.1f}")
        return

    queries = res["queries"]
    recs, lats = [], []
    for i, q in enumerate(queries):
        recs.append(R.compute_recall(q["labels"], gt[i], 10))
        st = q.get("search_time_us", 0)
        if st > 0:
            lats.append(st / 1000.0)
    avg = float(np.mean(lats)) if lats else 0.0
    print(f"SL n={len(queries)} recall={np.mean(recs):.4f} "
          f"lat_ms={avg:.2f} qps={(1000.0 / avg if avg > 0 else 0):.1f} "
          f"build_time_s={res.get('build_time_s', 0):.1f} "
          f"rss={res.get('rss_peak_query_mb', 0)}")


if __name__ == "__main__":
    main()
