#!/usr/bin/env python3
"""Quick hypothesis test: does raising SPTAG SearchInternalResultNum raise
SPANN post-filtering recall? Runs the C++ bench on the first 100 SL queries
of wit_100k with a given search_internal_result_num, then prints recall/qps.

Usage: python tests/test_sir.py <search_internal_result_num>
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_100k_sweep as R

SIR = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
DS = "wit_100k"
N_TEST = 100

gt_dir = ROOT / "1_Data/ground_truth" / DS
# Small query subset
qvecs = np.load(gt_dir / "query_vecs.npy")[:N_TEST]
qlabels = np.load(gt_dir / "query_labels.npy")[:N_TEST]
tmp_dir = Path(tempfile.mkdtemp(prefix="sirtest_"))
qpath = tmp_dir / "query_vecs.npy"
lpath = tmp_dir / "query_labels.npy"
np.save(qpath, qvecs)
np.save(lpath, qlabels)

cfg = {
    "d": 0, "num_threads": 1, "max_check": 16384, "hash_exp": 8,
    "k": 10, "num_warmup": 20, "overfetch_factor": 200,
    "overfetch_adaptive": False, "search_internal_result_num": SIR,
}
cfg_path = tmp_dir / "cfg.json"
json.dump(cfg, open(cfg_path, "w"))

out_path = tmp_dir / "out.json"
idx_dir = tmp_dir / "idx"
cmd = [str(R.BINARIES["SPANN"]), "bench",
       "--train_vecs", str(gt_dir / "train_vecs.npy"),
       "--train_access", str(gt_dir / "train_access.npy"),
       "--queries", str(qpath),
       "--query_labels", str(lpath),
       "--index_dir", str(idx_dir),
       "--config", str(cfg_path),
       "--max_check", "16384", "--overfetch_factor", "200",
       "--output", str(out_path)]
print("CMD:", " ".join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
if res.returncode != 0:
    print("RC", res.returncode, res.stderr[-400:])
    sys.exit(1)

gt = R.load_gt(DS)[:N_TEST]
data = json.load(open(out_path))
recs, lats = [], []
for i, q in enumerate(data["queries"]):
    recs.append(R.compute_recall(q["labels"], gt[i], 10))
    st = q.get("search_time_us", 0)
    if st > 0:
        lats.append(st / 1000.0)
avg = float(np.mean(lats)) if lats else 0.0
print(f"SIR={SIR} n={len(data['queries'])} recall={np.mean(recs):.4f} "
      f"lat_ms={avg:.2f} qps={(1000.0 / avg if avg else 0):.2f}")
print("tmp:", tmp_dir)
