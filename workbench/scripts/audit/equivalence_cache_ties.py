"""G4b + G3c: cache-identity audit and tie-policy census. Static/descriptive, no verdict tolerance.

G4b: enumerate every input that can change a CACHED FEATURE MATRIX, and check which of them appear
in the run identity that keys the cache. The cache is a `.npy` plus a `.json` sidecar holding only
`fingerprint`, `shape`, `dtype`; `runner._load_memmap` trusts the sidecar fingerprint and never
hashes the array contents. So a cache can be reused whenever the sidecar can be made to match --
including by copying a directory and rewriting the sidecar.

G3c: `_ranks_from_scores` counts exact score equality (`row[:index] == row[index]`), so a tied
gallery changes ranks by ORDER. Census the real gallery + real query features for:
  * byte-identical duplicate gallery feature ROWS (the strongest tie source),
  * duplicate score values within a single query's score row (the ties the rank rule counts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))
RUN = REPO / "workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
FEATURES = RUN / "work/habit/features"
OUT = REPO / "workbench/docs/reviews/20261108T153045-audit-equivalence"

#: Every input that can change a cached feature matrix, and where it is (or is not) recorded.
CACHE_INPUTS = [
    ("gallery ordered identities", "identity.gallery_identity", True,
     "fingerprint of protocol id + ordered gallery + candidate names"),
    ("gallery file content", "identity.gallery_image_identity.content_digest", True,
     "sha256 per resolved file, rolled up"),
    ("query ordered identities", "identity.query_order_identity", True, ""),
    ("query reference file content", "identity.query_image_identity.content_digest", True, ""),
    ("condition set", "identity.conditions", True,
     "changes which query matrices are produced"),
    ("ablation policy (black image / empty text)", "identity.condition_transform_source_hash", True,
     "hashes workbench/composition/inputs.py"),
    ("adapter module source", "identity.adapter_module_source_hash", True, ""),
    ("adapter method source", "identity.adapter_source_hashes", True,
     "encode_queries/encode_targets/score/rank/mode_supported"),
    ("shared decode + scoring helpers", "identity.scoring_module_source_hash", False,
     "ABSENT from run 497c25c57fbeba885ecfb9d1: the field was added after this run was written. "
     "Present in later identities, but hashes only workbench/composition/adapters.py"),
    ("model checkpoint bytes", "identity.checkpoint_identity", True, ""),
    ("model interpreter path + env fingerprint", "identity.runtime_identity", True, ""),
    ("torch / torchvision / numpy versions", "identity.runtime_identity.environment_fingerprint", True,
     "indirect: a fingerprint of the environment record, not of the imported module versions"),
    ("BLAS / OMP intra-op thread count", "(none)", False,
     "NOT recorded by name. Measured to change FAST-PATH score bits (ein_threads.txt: "
     "maxdelta 2.384e-07 at 10 threads vs 3.576e-07 elsewhere), so it can change a score-derived "
     "rank. The shipped `old` einsum was bit-identical across 1..10 threads, so today it does not "
     "move the cached MATRIX, but it does move a fast-path score."),
    ("GPU model", "identity.gpu_identity", True, "device NAME only"),
    ("GPU compute capability / driver / CUDA version", "identity.runtime_identity (partial)", False,
     "the device capability that decides kernel selection is not a named field; only the name is"),
    ("tf32 / cudnn benchmark / deterministic flags", "(none)", False,
     "NOT recorded; can change GPU forward bits"),
    ("batch size for the FORWARD pass", "identity.batch_size, identity.adapter_batch_size", True,
     "changes run id, so it invalidates the cache"),
    ("decode worker count", "identity (absent)", False,
     "deliberately NOT in the identity because decode is bit-identical; measured, and re-measured "
     "in this audit across 2/4/8 workers for two transform families"),
    ("transform implementation used per model", "identity.preprocessing.registry_values.transform", True,
     "a LABEL from the registry, not the transform's source"),
    ("the transform's actual code/parameters", "identity.preprocessing.registry_values.image_size", False,
     "recorded as a registry LABEL; the code that performs it is covered only by the adapter "
     "module hash, so a change inside a third-party transform function is not detected"),
    ("feature dtype", "sidecar `dtype`", True, "cached, but not compared on load"),
    ("feature array CONTENT", "(none)", False,
     "the sidecar holds a fingerprint of the IDENTITY, not of the array bytes; nothing verifies the "
     "stored numbers"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--conditions", default="multimodal,text_only_black_image,image_only_empty_text")
    parser.add_argument("--out", default=str(OUT / "cache_and_ties.json"))
    args = parser.parse_args()

    identity = json.loads((RUN / "ranks/habit_fiq_n02.json").read_text())["identity"]
    identity_fields = set(_flatten(identity))

    # ---------- G4b ----------
    rows = []
    for label, recorded_at, present, note in CACHE_INPUTS:
        record = {"input": label, "recorded_at": recorded_at,
                  "in_identity": bool(present), "note": note}
        rows.append(record)
    missing = [row for row in rows if not row["in_identity"]]

    # Any extra identity field that is a source hash we did not model?
    source_hash_fields = sorted(field for field in identity_fields
                                if field.endswith("_hash") or field.endswith("_hashes")
                                or "source_status" in field)

    # ---------- G3c ----------
    gallery = np.load(FEATURES / "gallery_global.npy", mmap_mode="r")
    gallery_count = gallery.shape[0]
    row_hashes: dict[bytes, list[int]] = {}
    for start in range(0, gallery_count, 2048):
        block = np.asarray(gallery[start:start + 2048], dtype=np.float32)
        for offset, row in enumerate(block):
            digest = hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
            row_hashes.setdefault(digest, []).append(start + offset)
    duplicate_groups = [positions for positions in row_hashes.values() if len(positions) > 1]
    duplicate_positions = sum(len(group) for group in duplicate_groups)

    manifest = json.loads((OUT / "query_manifest.json").read_text())
    qids = manifest["query_order"][:args.queries]
    positions = [tuple(manifest["positive_positions"][qid]) for qid in qids]
    tie_report = {}
    for condition in args.conditions.split(","):
        features = np.load(FEATURES / f"query_{condition}_global.npy", mmap_mode="r")
        queries = np.asarray(features[:len(qids)], dtype=np.float32)
        tie_exact_counts: Counter = Counter()
        queries_with_tie = 0
        positive_tie_queries = 0
        ranks_affected = 0
        top1_ties = 0
        examples = []
        for index in range(0, len(qids), 32):
            block = queries[index:index + 32]
            scores = np.einsum("nd,gqd->ngq", block, gallery).max(axis=-1)
            for local, row in enumerate(scores):
                unique, counts = np.unique(row, return_counts=True)
                ties = counts[counts > 1]
                if ties.size:
                    queries_with_tie += 1
                    tie_exact_counts[int(ties.sum())] += 1
                    if counts.max() > 1 and unique[np.argmax(counts)] == row.max():
                        top1_ties += 1
                best_position = None
                best = -np.inf
                for position in positions[index + local]:
                    if row[position] > best:
                        best, best_position = row[position], position
                if best_position is None:
                    continue
                multiplicity = int(np.count_nonzero(row == row[best_position]))
                if multiplicity > 1:
                    positive_tie_queries += 1
                    above = int(np.count_nonzero(row > row[best_position]))
                    before = int(np.count_nonzero(row[:best_position] == row[best_position]))
                    if before:
                        ranks_affected += 1
                        if len(examples) < 10:
                            examples.append({"qid": qids[index + local],
                                             "positive_position": best_position,
                                             "positive_score": float(row[best_position]),
                                             "tie_multiplicity": multiplicity,
                                             "rows_strictly_greater": above,
                                             "equal_rows_before": before,
                                             "rank_if_earliest_first": 1 + above + before,
                                             "rank_if_tie_broken_differently": 1 + above})
        tie_report[condition] = {
            "queries": len(qids),
            "queries_with_any_exact_duplicate_score": queries_with_tie,
            "queries_whose_top1_is_tied": top1_ties,
            "queries_whose_positive_is_tied": positive_tie_queries,
            "queries_whose_rank_depends_on_tie_order": ranks_affected,
            "tie_size_histogram": dict(sorted(tie_exact_counts.items())),
            "examples": examples,
        }

    payload = {
        "cache_identity": {
            "run": RUN.name,
            "identity_has_scoring_module_source_hash": "scoring_module_source_hash" in identity,
            "identity_source_hash_fields": source_hash_fields,
            "inputs": rows,
            "inputs_not_in_identity": [row["input"] for row in missing],
            "sidecar_fields": ["dtype", "fingerprint", "shape"],
            "sidecar_verifies_array_bytes": False,
            "sidecar_note": "gallery_global.npy.json and the query sidecars each hold exactly "
                            "dtype/fingerprint/shape. The fingerprint is of the run identity, so a "
                            "copied array with a rewritten sidecar is accepted without any check of "
                            "its contents (runner._load_memmap).",
        },
        "duplicate_gallery_rows": {
            "gallery_count": gallery_count,
            "duplicate_groups": len(duplicate_groups),
            "positions_in_duplicate_groups": duplicate_positions,
            "largest_group": max((len(group) for group in duplicate_groups), default=0),
            "examples": duplicate_groups[:10],
        },
        "tie_census": tie_report,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(json.dumps({"out": args.out,
                      "inputs_not_in_identity": payload["cache_identity"]["inputs_not_in_identity"],
                      "duplicate_groups": len(duplicate_groups),
                      "duplicate_positions": duplicate_positions}, indent=1))
    return 0


def _flatten(value, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            out.extend(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out.append(prefix)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
