import json, numpy as np, sys, os

results_path = sys.argv[1]
gt_dir = sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 10

with open(results_path) as f:
    results = json.load(f)

# Auto-detect formula from filter name
formula = results.get("config", {}).get("filter", "")
# Fallback: read filters.json
if not formula:
    with open(os.path.join(gt_dir, "filters.json")) as f:
        filters = json.load(f)
    # Use first filter
    formula = filters["filters"][0]

safe = formula.replace(" ", "_")
gt_path = os.path.join(gt_dir, f"gt_{safe}.npy")

print(f"Filter: {formula}")
print(f"GT path: {gt_path}")

if not os.path.exists(gt_path):
    print(f"ERROR: GT file not found: {gt_path}")
    sys.exit(1)

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
print(f"  Min:    {np.min(recalls):.4f}")
print(f"  ==1.0:  {np.sum(recalls == 1.0)}/{len(recalls)}")
print(f"  Empty:  {empty_count}/{len(results['queries'])}")
