#!/usr/bin/env python3
"""Read-only HABIT scorer counterfactual over retained Adapter feature arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from workbench.composition.model_habit import Adapter

ROOT = Path(__file__).resolve().parents[1] / "raw" / "third_model"
OUT = Path(__file__).resolve().parents[1] / "raw" / "habit-score-counterfactual-20261120.json"
CATEGORIES = ("dress", "shirt", "toptee")
CUTS = (1, 10, 50)
CHUNK = 31


def digest_ids(values):
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def rank_targets(scores, target_positions):
    ranks = np.empty(scores.shape[0], dtype=np.int32)
    for i, row in enumerate(scores):
        order = np.argsort(-row)
        ranks[i] = int(np.flatnonzero(order == target_positions[i])[0]) + 1
    return ranks


def counts(ranks):
    return {f"r{k}": int(np.count_nonzero(ranks <= k)) for k in CUTS}


def score_variants(query, gallery, references, target_positions):
    adapter_masked = np.empty((len(query), len(gallery)), dtype=np.float32)
    adapter_eligible = np.empty_like(adapter_masked)
    native_masked = np.empty_like(adapter_masked)
    native_eligible = np.empty_like(adapter_masked)
    adapter = Adapter.__new__(Adapter)
    for start in range(0, len(query), CHUNK):
        end = min(start + CHUNK, len(query))
        q = query[start:end]
        actual = adapter.score(q, gallery)
        native = np.matmul(q[:, None, None, :], gallery.transpose(0, 2, 1)).squeeze().max(-1)
        # Rank-3 retained targets are (gallery, token, embedding); upstream arithmetic's
        # equivalent contraction yields (query, gallery, token) before max(-1).
        if actual.shape != native.shape or actual.shape != (end - start, len(gallery)):
            raise AssertionError(f"score shape differs: adapter={actual.shape}, native={native.shape}")
        adapter_eligible[start:end] = actual
        native_eligible[start:end] = native
        adapter_masked[start:end] = actual
        native_masked[start:end] = native
        for local, ref in enumerate(references[start:end]):
            if ref is not None:
                adapter_masked[start + local, ref] = -10e10
                native_masked[start + local, ref] = -10e10
    target = np.asarray(target_positions, dtype=np.int64)
    ranks = {
        "adapter_masked": rank_targets(adapter_masked, target),
        "adapter_reference_eligible": rank_targets(adapter_eligible, target),
        "native_arithmetic_masked": rank_targets(native_masked, target),
        "native_arithmetic_reference_eligible": rank_targets(native_eligible, target),
    }
    max_abs = float(np.max(np.abs(adapter_eligible - native_eligible)))
    mismatches = int(np.count_nonzero(adapter_eligible != native_eligible))
    return ranks, {
        "score_max_abs_adapter_vs_native_arithmetic": max_abs,
        "score_elementwise_equal_adapter_vs_native_arithmetic": mismatches == 0,
        "score_differing_elements": mismatches,
        "scores_compared": int(adapter_eligible.size),
    }


def main():
    result = {
        "schema": "habit-score-counterfactual/v1",
        "purpose": "same retained Adapter feature arrays; not independent native-vs-Adapter parity",
        "numpy": np.__version__,
        "chunk_queries": CHUNK,
        "native_arithmetic": "np.matmul(query_chunk[:, None, None, :], gallery.transpose(0, 2, 1)).squeeze().max(-1)",
        "native_arithmetic_note": "native gallery output is (gallery, embedding, token); broadcasted matmul gives (query, gallery, 1, token), squeeze then max(-1) returns query-major scores",
        "rank_tie_rule": "np.argsort(-scores), as upstream HABIT/test.py",
        "categories": {},
    }
    before = {}
    for category in CATEGORIES:
        path = ROOT / f"population_habit_{category}.npz"
        inputs_path = ROOT / f"sample_habit_{category}_inputs.json"
        before[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs = json.loads(inputs_path.read_text())
        with np.load(path, allow_pickle=False) as data:
            query = data["queries"]
            gallery = data["gallery"]
            gallery_ids = list(map(str, inputs["gallery_ids"]))
            refs = list(map(str, inputs["full_reference_ids"]))
            targets = list(map(str, inputs["full_target_ids"]))
            if len(gallery_ids) != gallery.shape[0] or len(refs) != query.shape[0] or len(targets) != query.shape[0]:
                raise AssertionError(f"{category}: ordered IDs do not match feature rows")
            if len(set(gallery_ids)) != len(gallery_ids):
                raise AssertionError(f"{category}: gallery IDs are not unique")
            positions = {image_id: index for index, image_id in enumerate(gallery_ids)}
            missing_targets = [image_id for image_id in targets if image_id not in positions]
            if missing_targets:
                raise AssertionError(f"{category}: target IDs absent from gallery: {missing_targets[:3]}")
            ref_positions = [positions.get(image_id) for image_id in refs]
            target_positions = [positions[image_id] for image_id in targets]
            ranks, score = score_variants(query, gallery, ref_positions, target_positions)
            category_result = {
                "feature_file": str(path),
                "feature_sha256_before": before[str(path)],
                "query_shape": list(query.shape),
                "gallery_shape": list(gallery.shape),
                "query_ids": {"count": len(refs), "sha256": digest_ids(refs)},
                "ordered_gallery_ids": {"count": len(gallery_ids), "sha256": digest_ids(gallery_ids)},
                "target_ids": {"count": len(targets), "sha256": digest_ids(targets)},
                "reference_ids_in_gallery": sum(position is not None for position in ref_positions),
                "reference_ids_absent_from_gallery": sum(position is None for position in ref_positions),
                "variants": {},
                "score_arithmetic": score,
            }
            for name, values in ranks.items():
                category_result["variants"][name] = {
                    "target_hit_counts": counts(values),
                    "target_rank_min": int(values.min()),
                    "target_rank_max": int(values.max()),
                }
            for left, right in (("adapter_masked", "adapter_reference_eligible"),
                                ("adapter_masked", "native_arithmetic_masked"),
                                ("native_arithmetic_masked", "native_arithmetic_reference_eligible")):
                a, b = ranks[left], ranks[right]
                category_result.setdefault("comparisons", {})[f"{left}_vs_{right}"] = {
                    "query_rank_flips": int(np.count_nonzero(a != b)),
                    "query_target_hit_flips": {
                        f"r{k}": int(np.count_nonzero((a <= k) != (b <= k))) for k in CUTS
                    },
                }
            result["categories"][category] = category_result
    after = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before}
    result["input_hashes_after"] = after
    result["inputs_unchanged"] = before == after
    if not result["inputs_unchanged"]:
        raise AssertionError("protected Adapter feature NPZ changed during read-only experiment")
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(OUT), "inputs_unchanged": True,
                      "categories": {k: v["comparisons"] for k, v in result["categories"].items()}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
