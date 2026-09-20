import json
from pathlib import Path

ROOT = Path('.')
DATASETS = ['sift1m', 'gist1m', 'arxiv', 'yfcc100m', 'wit']
for dataset in DATASETS:
    raw_measure = ROOT / '4_Results/build_measure/raw' / f'prefilter_{dataset}.json'
    if not raw_measure.exists():
        continue
    measured = json.load(open(raw_measure, encoding='utf-8'))
    new_rss = float(measured['rss_peak_query_mb'])
    paths = [
        ROOT / '4_Results/Pre-Filtering' / f'sweep_{dataset}_inmem.json',
        ROOT / '4_Results/Pre-Filtering' / f'_sl_prefiltering_{dataset}_inmem.json',
    ]
    for path in paths:
        if not path.exists():
            continue
        data = json.load(open(path, encoding='utf-8'))
        data['rss_peak_query_mb'] = new_rss
        data['memory_measurement'] = (
            'fresh in-memory build with query-phase RSS sampled after the '
            'build-only NPY buffer is released (duplicate-copy fix)')
        if 'memory' in data and isinstance(data['memory'], dict):
            data['memory']['rss_peak_query_mb'] = new_rss
        for row in data.get('sweep_results', []):
            row['rss_peak_query_mb'] = new_rss
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
        print(f'patched {path} rss={new_rss:.2f} MB')