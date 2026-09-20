#!/usr/bin/env python3
"""Restart remaining-dataset pipeline using search-mode / build-once runners."""
import argparse
import subprocess
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
PY = sys.executable
DATASETS = ["gist1m", "arxiv", "yfcc100m", "wit"]
STEPS = [
    ("Curator", "run_curator_full.py"),
    ("DiskIVF", "run_diskivf_search_full.py"),
    ("SPANN", "run_spann_search_full.py"),
    ("PreFilter", "run_prefilter_real_full.py"),
    ("PreFilter-inmem", "run_prefilter_inmemory_full.py"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()
    for ds in args.datasets:
        for name, script in STEPS:
            log = PROJ_ROOT / "logs" / f"v2_{name}_{ds}.log"
            log.parent.mkdir(exist_ok=True)
            print(f"=== {ds} {name} ===", flush=True)
            cmd = [PY, str(PROJ_ROOT / script), "--dataset", ds]
            if args.quick:
                cmd.append("--quick")
            with open(log, "w") as f:
                subprocess.run(cmd, cwd=str(PROJ_ROOT), stdout=f, stderr=subprocess.STDOUT)
        # After all methods for this dataset, render figures with the same
        # sift1m plotting style.
        plot_log = PROJ_ROOT / "logs" / f"v2_plot_{ds}.log"
        print(f"=== Plotting {ds} ===", flush=True)
        with open(plot_log, "w") as f:
            subprocess.run(
                [PY, str(PROJ_ROOT / "5_Plot/fig_full_sl_by_percentile.py"), ds],
                cwd=str(PROJ_ROOT), stdout=f, stderr=subprocess.STDOUT)
    with open(PROJ_ROOT / "logs" / "v2_plot_matched_recall.log", "w") as f:
        subprocess.run(
            [PY, str(PROJ_ROOT / "5_Plot/fig_full_sl_qps_at_recall.py")],
            cwd=str(PROJ_ROOT), stdout=f, stderr=subprocess.STDOUT)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()

