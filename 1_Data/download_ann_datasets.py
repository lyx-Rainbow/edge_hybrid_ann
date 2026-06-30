"""
Download SIFT1M and GIST1M datasets from corpus-texmex.irisa.fr.

Usage (run in WSL):
    python 1_Data/download_ann_datasets.py --dataset sift1m
    python 1_Data/download_ann_datasets.py --dataset gist1m
    python 1_Data/download_ann_datasets.py --dataset all
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "http://corpus-texmex.irisa.fr/"

# Mirror (if primary is down)
MIRROR_URL = "http://ann-benchmarks.com/"

DATASETS = {
    "sift1m": {
        "files": [
            "sift_base.fvecs",      # 1M × 128d float32 vectors  (~501 MB)
            "sift_query.fvecs",     # 10K × 128d query vectors   (~5 MB)
            "sift_learn.fvecs",     # 100K × 128d learn vectors  (~50 MB)
            "sift_groundtruth.ivecs", # 10K × 100 int32 GT       (~4 MB)
        ],
        "expected_sizes": {  # approximate, in bytes
            "sift_base.fvecs": 516000004,
            "sift_query.fvecs": 5160004,
            "sift_learn.fvecs": 51600004,
            "sift_groundtruth.ivecs": 4000400,
        },
    },
    "gist1m": {
        "files": [
            "gist_base.fvecs",      # 1M × 960d float32 vectors (~3.8 GB)
            "gist_query.fvecs",     # 1K × 960d query vectors   (~4 MB)
            "gist_learn.fvecs",     # 500K × 960d learn vectors (~1.9 GB)
            "gist_groundtruth.ivecs", # 1K × 100 int32 GT       (~400 KB)
        ],
        "expected_sizes": {
            "gist_base.fvecs": 3840000004,
            "gist_query.fvecs": 3840004,
            "gist_learn.fvecs": 1920000004,
            "gist_groundtruth.ivecs": 400400,
        },
    },
}


def download_file(url: str, dest: str, expected_size: int | None = None) -> bool:
    """Download a file with progress display. Returns True on success."""
    dest = Path(dest)
    if dest.exists():
        actual = dest.stat().st_size
        if expected_size and actual == expected_size:
            print(f"  ✓ {dest.name} already exists (size OK)")
            return True
        else:
            print(f"  ⚠ {dest.name} exists but size mismatch "
                  f"({actual} vs expected {expected_size}), re-downloading...")

    print(f"  ↓ Downloading {dest.name} from {url} ...")
    start = time.perf_counter()
    try:
        urlretrieve(url, dest)
        elapsed = time.perf_counter() - start
        actual = dest.stat().st_size
        print(f"    Done in {elapsed:.0f}s ({actual / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"    Failed: {e}")
        # Clean partial file
        if dest.exists():
            dest.unlink()
        return False


def download_dataset(dataset: str, output_dir: str, use_mirror: bool = False) -> bool:
    """Download all files for a dataset. Returns True if all succeeded."""
    info = DATASETS[dataset]
    url_base = MIRROR_URL if use_mirror else BASE_URL
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {dataset}")
    print(f"  Output: {out.resolve()}")
    print(f"  Source: {url_base}")
    print(f"{'='*60}")

    all_ok = True
    for filename in info["files"]:
        url = url_base + filename
        dest = out / filename
        expected = info["expected_sizes"].get(filename)
        ok = download_file(url, str(dest), expected)
        all_ok = all_ok and ok

    # Verify
    if all_ok:
        print(f"\n  ✓ All files for {dataset} downloaded OK")
    else:
        print(f"\n  ✗ Some files for {dataset} failed — try --mirror flag")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download SIFT1M / GIST1M ANN benchmark datasets"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["sift1m", "gist1m", "all"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Output directory (default: current dir)",
    )
    parser.add_argument(
        "--mirror", action="store_true",
        help="Use mirror URL if primary is down",
    )
    args = parser.parse_args()

    datasets = ["sift1m", "gist1m"] if args.dataset == "all" else [args.dataset]

    all_success = True
    for ds in datasets:
        out = Path(args.output_dir) / ds
        ok = download_dataset(ds, str(out), use_mirror=args.mirror)
        all_success = all_success and ok

    if all_success:
        print(f"\n{'='*60}")
        print("All downloads complete!")
        print(f"Files are in: {Path(args.output_dir).resolve()}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("Some downloads failed. Try:")
        print(f"  python {__file__} --dataset ... --mirror")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
