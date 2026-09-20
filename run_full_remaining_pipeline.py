#!/usr/bin/env python3
"""Background pipeline placeholder for full-scale remaining datasets.

Run sequentially to avoid memory pressure. This script is intended to be
launched in a long-running session.
"""
import subprocess
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
PY = sys.executable
DATASETS = ["gist1m", "arxiv", "yfcc100m", "wit"]
METHODS = ["Curator", "DiskIVF", "SPANN", "Pre-Filtering"]
CONFIG_BASE = {
    "Curator": "3_Config/Curator/sweep_full_{ds}.json",
    "DiskIVF": "3_Config/DiskIVF-PostFiltering/sweep_full_{ds}.json",
    "SPANN": "3_Config/SPANN-PostFiltering/sweep_full_{ds}.json",
    "Pre-Filtering": "3_Config/Pre-Filtering/sweep_full_{ds}.json",
}

def main():
    for ds in DATASETS:
        for method in METHODS:
            cfg = CONFIG_BASE[method].format(ds=ds)
            log = PROJ_ROOT / "logs" / f"full_{method}_{ds}.log"
            log.parent.mkdir(exist_ok=True)
            print(f"=== {ds} {method} ===", flush=True)
            with open(log, "w") as f:
                subprocess.run(
                    [PY, str(PROJ_ROOT / "run_100k_sweep.py"),
                     "--dataset", ds, "--method", method,
                     "--sweep", str(PROJ_ROOT / cfg),
                     "--output_dir", str(PROJ_ROOT / "4_Results"),
                     "--run_sl"],
                    cwd=str(PROJ_ROOT), stdout=f, stderr=subprocess.STDOUT)
    print("all done", flush=True)

if __name__ == "__main__":
    main()
