#!/usr/bin/env python3
"""Fast Curator memory probe: run a SL bench on a SMALL query subset (default
200) with given config overrides and print query-phase peak RSS + index
breakdown. Used to find a memory-lean Curator config for the v2 figure.

Usage:
    python tests/probe_mem_cfg.py sift1m_100k '{"n_clusters":8,"bf_capacity":256,"bf_false_pos":0.1}'
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_100k_sweep as R

DS = sys.argv[1] if len(sys.argv) > 1 else "sift1m_100k"
OVERRIDES = {}
for tok in sys.argv[2:]:
    k, v = tok.split("=", 1)
    OVERRIDES[k] = int(v) if v.lstrip("-").isdigit() else float(v)
N_QUERIES = 200

# Best-recall SL operating point (same as the v2 memory figure)
COMBOS = {
    "sift1m_100k":   {"pq_M": 128, "search_ef": 2048},
    "yfcc100m_100k": {"pq_M": 64,  "search_ef": 2048},
    "arxiv_100k":    {"pq_M": 128, "search_ef": 2048},
    "gist1m_100k":   {"pq_M": 320, "search_ef": 4096},
    "wit_100k":      {"pq_M": 128, "search_ef": 4096},
}

gt_dir = ROOT / "1_Data/ground_truth" / DS
qvecs = np.load(gt_dir / "query_vecs.npy")[:N_QUERIES]
qlabels = np.load(gt_dir / "query_labels.npy")[:N_QUERIES]
tmp = Path(tempfile.mkdtemp(prefix="memprobe_"))
qpath = tmp / "query_vecs.npy"; lpath = tmp / "query_labels.npy"
np.save(qpath, qvecs); np.save(lpath, qlabels)

sw = json.load(open(ROOT / "3_Config/Curator/sweep_100k.json"))
cfg = dict(sw["fixed"])
cfg.update(COMBOS[DS])
# Memory-lean baseline (same as the v2 figure)
cfg["n_clusters"] = 16
cfg["bf_false_pos"] = 0.05
cfg["pq_cache_max_blocks"] = 32
cfg.update(OVERRIDES)
cfg["flash_path"] = str(tmp / "flash")
cp = tmp / "cfg.json"; json.dump(cfg, open(cp, "w"))
out = tmp / "out.json"

cmd = [str(R.BINARIES["Curator"]), "bench",
       "--train_vecs", str(gt_dir / "train_vecs.npy"),
       "--train_access", str(gt_dir / "train_access.npy"),
       "--queries", str(qpath),
       "--query_labels", str(lpath),
       "--k", "10", "--output", str(out),
       "--config", str(cp)]
r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=1200)
if r.returncode != 0:
    print("RC", r.returncode, r.stderr[-300:]); sys.exit(1)
res = json.load(open(out))
mb = res["memory_breakdown"]
print(f"{DS} overrides={OVERRIDES} -> rss_peak={res.get('rss_peak_query_mb'):.1f}  "
      f"idx_total={mb['total_bytes']/1e6:.1f}  nodes={mb['num_tree_nodes']}  "
      f"bloom={mb['bloom_filter_bytes']/1e6:.1f}  shortlists={(mb['shortlists_overhead_bytes']+mb['shortlists_payload_bytes'])/1e6:.1f}")
print("tmp:", tmp)
