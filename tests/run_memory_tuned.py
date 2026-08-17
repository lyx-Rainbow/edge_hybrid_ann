#!/usr/bin/env python3
"""Run Curator SL benches with the memory-optimized binary and record
query-phase RSS for the memory-v2 figure.

Writes results to 4_Results/memory_tuned/curator_v2.json — does NOT touch
the sweep JSONs used by the other figures.

Usage: python tests/run_memory_tuned.py [dataset1 dataset2 ...]
"""
import json, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_100k_sweep as R

# Best-recall SL operating point per dataset (right end of the QPS-recall
# frontier) — the config a deployment would actually use.
COMBOS = {
    "sift1m_100k":   {"pq_M": 128, "search_ef": 2048},
    "yfcc100m_100k": {"pq_M": 64,  "search_ef": 2048},
    "arxiv_100k":    {"pq_M": 128, "search_ef": 2048},
    "gist1m_100k":   {"pq_M": 320, "search_ef": 4096},
    "wit_100k":      {"pq_M": 128, "search_ef": 4096},
}

OUT_PATH = ROOT / "4_Results/memory_tuned/curator_v2.json"


def main():
    datasets = sys.argv[1:] or list(COMBOS.keys())
    results = {}
    if OUT_PATH.exists():
        results = json.load(open(OUT_PATH))

    sweep_cfg = json.load(open(ROOT / "3_Config/Curator/sweep_100k.json"))
    fixed = sweep_cfg["fixed"]

    for ds in datasets:
        combo = COMBOS[ds]
        print(f"=== {ds} {combo} ===", flush=True)
        base_cfg = dict(fixed)
        base_cfg.update(combo)
        # Memory-optimized index parameters (this figure is allowed to use a
        # different index configuration than the QPS-recall figures):
        base_cfg["n_clusters"] = 16
        base_cfg["bf_false_pos"] = 0.05
        base_cfg["pq_cache_max_blocks"] = 32
        base_cfg["flash_path"] = str(ROOT / "4_Results/Curator"
                                     / f"disk_data_mem_v2_{ds}")
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False, dir="/tmp")
        json.dump(base_cfg, tmp)
        tmp.close()

        out = ROOT / "4_Results/Curator" / f"_mem_v2_{ds}.json"
        gt_dir = ROOT / "1_Data/ground_truth" / ds
        cmd = [str(R.BINARIES["Curator"]), "bench",
               "--train_vecs", str(gt_dir / "train_vecs.npy"),
               "--train_access", str(gt_dir / "train_access.npy"),
               "--queries", str(gt_dir / "query_vecs.npy"),
               "--query_labels", str(gt_dir / "query_labels.npy"),
               "--k", "10", "--output", str(out),
               "--config", tmp.name]
        print("CMD:", " ".join(cmd), flush=True)
        proc = R.subprocess.run(cmd, capture_output=False, text=True,
                                cwd=str(ROOT), timeout=1800)
        os.unlink(tmp.name)
        if proc.returncode != 0 or not out.exists():
            print(f"  FAILED for {ds} rc={proc.returncode}", flush=True)
            continue
        res = json.load(open(out))

        rss = res.get("rss_peak_query_mb", 0)
        total = res.get("memory_breakdown", {}).get("total_bytes", 0)
        results[ds] = {
            "pq_M": combo["pq_M"],
            "search_ef": combo["search_ef"],
            "rss_peak_query_mb": float(rss),
            "index_total_bytes": float(total),
            "build_time_s": res.get("build_time_s", 0),
        }
        print(f"  rss_peak_query_mb={rss:.1f}  "
              f"index_total={total / 1e6:.1f} MB", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT_PATH, "w"), indent=2)
    print(f"saved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
