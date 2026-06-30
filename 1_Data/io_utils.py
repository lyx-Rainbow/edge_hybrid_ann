"""
I/O utilities for ANN benchmark datasets (fvecs/ivecs formats).

The .fvecs format stores float32 vectors:
  Each record = int32(dimension) + float32[dimension]
  Records are concatenated without separators.

The .ivecs format is identical but stores int32 vectors (used for ground truth).
"""

import numpy as np


def read_fvecs(path: str, max_vectors: int | None = None) -> np.ndarray:
    """Read a .fvecs file and return float32 ndarray of shape [n, d].

    Format: repeated [int32(d), float32[d]] blocks.
    All vectors in the file must have the same dimension.

    Parameters
    ----------
    max_vectors : int, optional
        If set, read at most this many vectors from the file (efficient:
        only the requested records are read, not the whole file).
    """
    # Read first record to determine dimension
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=1)
        if len(header) == 0:
            raise ValueError(f"Empty file: {path}")
        d = int(header[0])
        if d <= 0 or d > 100000:
            raise ValueError(f"Unexpected dimension {d} in {path}")

    record_ints = 1 + d  # int32(d) + float32[d] stored as d int32s (same byte width)

    if max_vectors is not None:
        # Read only max_vectors records — efficient for large files
        total_ints = max_vectors * record_ints
        raw = np.fromfile(path, dtype=np.int32, count=total_ints)
        expected_per_record = record_ints
    else:
        raw = np.fromfile(path, dtype=np.int32)
        expected_per_record = record_ints
        if len(raw) % expected_per_record != 0:
            raise ValueError(
                f"File {path}: total int32 values {len(raw)} not divisible by "
                f"expected_per_record={expected_per_record}"
            )

    n = len(raw) // expected_per_record

    # Reshape: each row has [d_value, vec_0, vec_1, ..., vec_{d-1}]
    # where vec_i are the float32 bits reinterpreted as int32
    raw_reshaped = raw.reshape(n, expected_per_record)
    # Drop the first column (dimension headers), keep the vector data
    int_data = raw_reshaped[:, 1:]  # [n, d] as int32
    # Reinterpret int32 bits as float32
    vecs = int_data.view(np.float32).copy()
    return vecs


def read_ivecs(path: str, max_vectors: int | None = None) -> np.ndarray:
    """Read a .ivecs file and return int32 ndarray of shape [n, k].

    Format is identical to fvecs but with int32 values.
    Used for ground truth files (each row = top-k neighbor IDs).

    Parameters
    ----------
    max_vectors : int, optional
        If set, read at most this many vectors from the file.
    """
    if max_vectors is not None:
        # Read first record to determine k
        with open(path, "rb") as f:
            k = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        total_ints = max_vectors * (1 + k)
        raw = np.fromfile(path, dtype=np.int32, count=total_ints)
        expected_per_record = 1 + k
    else:
        raw = np.fromfile(path, dtype=np.int32)
        k = int(raw[0])
        expected_per_record = 1 + k
        if len(raw) % expected_per_record != 0:
            raise ValueError(
                f"File {path}: total int32 values {len(raw)} not divisible by "
                f"expected_per_record={expected_per_record}"
            )

    n = len(raw) // expected_per_record
    raw_reshaped = raw.reshape(n, expected_per_record)
    # Drop the first column (k headers), keep the neighbor IDs
    return raw_reshaped[:, 1:].copy()


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent

    for ds_name in ["sift1m", "gist1m"]:
        ds_dir = data_dir / ds_name
        if not ds_dir.is_dir():
            print(f"  {ds_name}: directory not found, skipping")
            continue

        base_file = ds_dir / f"{'sift' if 'sift' in ds_name else 'gist'}_base.fvecs"
        query_file = ds_dir / f"{'sift' if 'sift' in ds_name else 'gist'}_query.fvecs"
        gt_file = ds_dir / f"{'sift' if 'sift' in ds_name else 'gist'}_groundtruth.ivecs"

        if base_file.exists():
            base = read_fvecs(str(base_file))
            print(f"  {ds_name} base:   {base.shape}  dtype={base.dtype}  "
                  f"min={base.min():.3f} max={base.max():.3f}")

        if query_file.exists():
            queries = read_fvecs(str(query_file))
            print(f"  {ds_name} query:  {queries.shape}  dtype={queries.dtype}  "
                  f"min={queries.min():.3f} max={queries.max():.3f}")

        if gt_file.exists():
            gt = read_ivecs(str(gt_file))
            print(f"  {ds_name} GT:     {gt.shape}  dtype={gt.dtype}  "
                  f"min={gt.min()} max={gt.max()}")
