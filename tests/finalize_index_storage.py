#!/usr/bin/env python3
"""Reset ext4 SPANN search caps and refresh the version manifest hashes."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path

ROOT = Path("/mnt/d/23235/Documents/Aftergraduate/experiments/new-Baselines")
MANIFEST = ROOT / "4_Results/_validation_selectivity/index_migration_manifest.json"
DATASETS = ["sift1m", "gist1m", "arxiv", "yfcc100m", "wit"]

def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
manifest["finalized_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
manifest["spann_search_cap_default"] = 64
manifest["note"] = (
    "SPANN sweep points vary SearchInternalResultNum dynamically; the runners "
    "patch indexloader.ini before each LoadIndex and this finalizer restores "
    "the saved default 64 afterwards. Sweep JSON params record the value used."
)
for ds in DATASETS:
    spann = Path(f"/home/lyx/spann_index/{ds}")
    disk = Path(f"/home/lyx/diskivf_{ds}_build_sqrt")
    ini = spann / "indexloader.ini"
    bak = ini.with_suffix(".ini.bak_sir64")
    if bak.exists():
        ini.write_text(bak.read_text(encoding="utf-8"), encoding="utf-8")
    rec = manifest.setdefault("datasets", {}).setdefault(ds, {})
    rec.update({
        "spann_dst": str(spann),
        "diskivf_dst": str(disk),
        "spann_metadata_sha256": sha256_file(spann / "metadata.bin"),
        "spann_indexloader_sha256": sha256_file(ini),
        "spann_index_bytes": sum(p.stat().st_size for p in spann.rglob("*") if p.is_file()),
        "diskivf_index_bytes": sum(p.stat().st_size for p in disk.rglob("*") if p.is_file()),
    })
    access = ROOT / "1_Data/ground_truth" / ds / "train_access.npy"
    rec["train_access_sha256"] = sha256_file(access)
    print("finalized", ds, "SearchInternalResultNum=", next(
        (line.split("=", 1)[1] for line in ini.read_text(encoding="utf-8").splitlines()
         if line.startswith("SearchInternalResultNum=")), "?"))
MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("wrote", MANIFEST)