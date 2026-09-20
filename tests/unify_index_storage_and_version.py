#!/usr/bin/env python3
"""Unify full-scale index storage on WSL ext4 and refresh label metadata.

This performs a cheap, non-destructive migration/version alignment:
  * mirror SPANN and DiskIVF index directories to /home/lyx (ext4)
  * rewrite label-dependent index metadata (SPANN metadata.bin and
    DiskIVF c*_mds.bin) from the current train_access.npy
  * write a manifest with data/index hashes so all later figures can be
    checked against one index version

The vector indexes themselves are label-independent; therefore no expensive
index rebuild is required for this version alignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "1_Data"))
from repair_duplicate_labels import (  # noqa: E402
    labels_by_vid_from_access,
    patch_diskivf_index,
    write_spann_metadata,
)

DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]
SPANN_SRC = ROOT / "4_Results/SPANN-PostFiltering/spann_index"
DISKIVF_SRC = ROOT / "4_Results/DiskIVF-PostFiltering/disk_data"
SPANN_DST_ROOT = Path("/home/lyx/spann_index")
DISKIVF_DST_ROOT = Path("/home/lyx")
MANIFEST = ROOT / "4_Results/_validation_selectivity/index_migration_manifest.json"


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def mirror(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"missing index source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mirror] {src} -> {dst}", flush=True)
    subprocess.run(["rsync", "-a", "--delete", f"{src}/", f"{dst}/"], check=True)


def patch_spann_ini(index_dir: Path) -> None:
    ini = index_dir / "indexloader.ini"
    if not ini.exists():
        raise FileNotFoundError(f"missing {ini}")
    lines = ini.read_text(encoding="utf-8", errors="replace").splitlines()
    replaced = False
    out = []
    for line in lines:
        if line.startswith("IndexDirectory="):
            out.append(f"IndexDirectory={index_dir}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"IndexDirectory={index_dir}")
    ini.write_text("\n".join(out) + "\n", encoding="utf-8")
    head_ini = index_dir / "HeadIndex/indexloader.ini"
    if head_ini.exists():
        text = head_ini.read_text(encoding="utf-8", errors="replace")
        if "IndexDirectory=" in text:
            lines = text.splitlines()
            head_ini.write_text(
                "\n".join(
                    f"IndexDirectory={index_dir / 'HeadIndex'}"
                    if line.startswith("IndexDirectory=") else line
                    for line in lines
                ) + "\n",
                encoding="utf-8",
            )


def current_labels(dataset: str, n: int):
    access_path = ROOT / "1_Data/ground_truth" / dataset / "train_access.npy"
    import numpy as np
    access = np.load(access_path).astype(np.int32)
    unique = np.unique(access, axis=0)
    labels = labels_by_vid_from_access(n, unique)
    return labels, unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--skip-spann", action="store_true")
    parser.add_argument("--skip-diskivf", action="store_true")
    parser.add_argument("--no-patch", action="store_true")
    args = parser.parse_args()

    manifest = {"created": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                "datasets": {}}
    for ds in args.datasets:
        print(f"\n===== {ds} =====", flush=True)
        meta = json.load(open(ROOT / "1_Data/ground_truth" / ds / "metadata.json"))
        n = int(meta["train_size"])
        labels = None
        record = {"train_size": n}
        access_path = ROOT / "1_Data/ground_truth" / ds / "train_access.npy"
        record["train_access_sha256"] = sha256_file(access_path)
        if not args.skip_spann:
            src = SPANN_SRC / ds
            dst = SPANN_DST_ROOT / ds
            mirror(src, dst)
            patch_spann_ini(dst)
            if labels is None:
                labels, _ = current_labels(ds, n)
            if not args.no_patch:
                write_spann_metadata(dst, labels)
            record["spann_dst"] = str(dst)
            record["spann_metadata_sha256"] = sha256_file(dst / "metadata.bin")
            record["spann_index_bytes"] = sum(
                p.stat().st_size for p in dst.rglob("*") if p.is_file())
        if not args.skip_diskivf:
            src = DISKIVF_SRC / f"{ds}_build_sqrt"
            dst = DISKIVF_DST_ROOT / f"diskivf_{ds}_build_sqrt"
            mirror(src, dst)
            if labels is None:
                labels, _ = current_labels(ds, n)
            if not args.no_patch:
                patch_diskivf_index(dst, labels)
            record["diskivf_dst"] = str(dst)
            record["diskivf_index_bytes"] = sum(
                p.stat().st_size for p in dst.rglob("*") if p.is_file())
        manifest["datasets"][ds] = record
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2), flush=True)
    print(f"\nwrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())