import json, numpy as np, sys

results_path = sys.argv[1]
gt_path = sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 10

with open(results_path) as f:
    results = json.load(f)

gt = np.load(gt_path)
recalls = []
empty_count = 0
for q, qr in enumerate(results["queries"]):
    gt_set = set(int(x) for x in gt[q, :k] if x >= 0)
    res_set = set(qr["labels"][:k])
    if not res_set or res_set == {-1}:
        empty_count += 1
    if gt_set:
        recalls.append(len(gt_set & res_set) / len(gt_set))

recalls = np.array(recalls)
print(f"Recall@{k}:")
print(f"  Mean:   {np.mean(recalls):.4f}")
print(f"  Median: {np.median(recalls):.4f}")
print(f"  P90:    {np.percentile(recalls, 90):.4f}")
print(f"  P99:    {np.percentile(recalls, 99):.4f}")
print(f"  Min:    {np.min(recalls):.4f}")
print(f"  ==1.0:  {np.sum(recalls == 1.0)}/{len(recalls)}")
print(f"  Empty:  {empty_count}/{len(results['queries'])}")
