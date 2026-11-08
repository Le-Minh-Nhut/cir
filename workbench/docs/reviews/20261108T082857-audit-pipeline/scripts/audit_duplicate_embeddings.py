"""Count exact-duplicate gallery rows/embeddings: the systematic-tie check for `_ranks_from_scores`.

`_ranks_from_scores` breaks ties by gallery position (1 + #greater + #earlier-equal), so any pair of
gallery rows with bit-identical scores is a systematic tie. This script measures whether that happens
on real data, and whether the released gallery list itself repeats an image identity.

Read-only. CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def digest_rows(matrix: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Per-row sha1 of the exact float32 bytes, so duplicates are byte-identical, not merely close."""
    out = []
    for start in range(0, len(matrix), chunk):
        block = np.ascontiguousarray(matrix[start:start + chunk], dtype=np.float32)
        for row in block:
            out.append(hashlib.sha1(row.tobytes()).hexdigest())
    return np.asarray(out, dtype=object)


def report_duplicates(name: str, matrix: np.ndarray) -> dict:
    digests = digest_rows(matrix)
    unique, counts = np.unique(digests, return_counts=True)
    duplicated = counts > 1
    excess = int((counts[duplicated] - 1).sum())
    # Group sizes among the duplicated identities, to say how bad the tie is.
    sizes = sorted(int(value) for value in counts[duplicated])[-10:]
    return {
        "array": name,
        "rows": int(len(matrix)),
        "distinct_row_bytes": int(len(unique)),
        "rows_in_exact_duplicate_groups": int(counts[duplicated].sum()),
        "excess_duplicate_rows": excess,
        "largest_duplicate_group_sizes": sizes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True,
                        help="directory holding *.npy memmaps (gallery_<scope>.npy)")
    parser.add_argument("--gallery-jsonl", type=Path, default=None,
                        help="optional M-BEIR candidate pool, to count repeated released identities")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result: dict = {"feature_dir": str(args.features), "arrays": []}
    for path in sorted(args.features.glob("*.npy")):
        matrix = np.load(path, mmap_mode="r")
        if matrix.ndim != 2:
            continue
        result["arrays"].append(report_duplicates(path.name, np.asarray(matrix)))

    if args.gallery_jsonl and args.gallery_jsonl.is_file():
        identities, basenames = [], []
        with args.gallery_jsonl.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                identities.append(str(row.get("img_path") or row.get("did")))
                basenames.append(Path(identities[-1]).stem)
        result["released_identities"] = {
            "rows": len(identities),
            "distinct_img_path": len(set(identities)),
            "distinct_basename_stem": len(set(basenames)),
        }

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
