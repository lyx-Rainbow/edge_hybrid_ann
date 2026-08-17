#!/usr/bin/env python3
"""Quick recall checker for a single results.json file."""
import json, sys, numpy as np

results_path = sys.argv[1]
gt_path = sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 10

res = json.load(open(results_path))
gt = np.load(gt_path)

recalls = []
for q, qr in enumerate(res["queries"]):
    gt_set = set(int(i) for i in gt[q][:k] if i >= 0)
    if gt_set:
        recalls.append(len(set(qr["labels"][:k]) & gt_set) / len(gt_set))

recalls = np.array(recalls)
print(f"Recall@{k}: mean={np.mean(recalls):.4f} median={np.median(recalls):.4f} min={np.min(recalls):.4f}")
print(f"Perfect (==1.0): {np.sum(recalls==1.0)}/{len(recalls)}")
print(f"Build time: {res.get('build_time_s', 0):.1f}s")
print(f"Memory: {res.get('memory_bytes', 0)/1024**2:.1f}MB")
