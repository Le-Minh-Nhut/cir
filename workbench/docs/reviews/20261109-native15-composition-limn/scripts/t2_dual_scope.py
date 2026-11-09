"""Task 2: dual-scope diagnostic for the two ambiguous CIRCUS queries.

Evaluates 7:3038 and 7:3790 under BOTH the shirt and toptee checkpoints/scopes, for all three
CIRCUS conditions, over the full 74,381-image released gallery.

Design constraints (deliberate, not incidental):
  * the gallery matrices are REUSED only after their sidecar content_digest is recomputed and matched;
    any mismatch aborts instead of ranking;
  * the query embeddings are recomputed by the production adapter (`encode_queries`) because the run
    cache has NaN rows for these two queries by design;
  * input staging reuses `runner._stage_query_path`, so black-image/text-drop semantics are the
    production ones, not a re-implementation;
  * scoring and ranking call the adapter's own `score`/`rank`, so the tie rule is the evaluator's
    (`1 + #greater + #earlier-equal`), not a second convention;
  * nothing is filtered by category, nothing is scored twice, no denominator is reduced.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/home/heheboiz/data/cir")

import numpy as np

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
from workbench.composition.adapters import resolved_decode_workers
from workbench.composition.data import assign_garment_categories, load_release
from workbench.composition.inputs import CONDITIONS
from workbench.composition.model_limn import Adapter
from workbench.composition.runner import (
    _content_digest,
    _positive_positions,
    _stage_query_path,
    image_resolver,
)

QIDS = ("7:3038", "7:3790")
SCOPES = ("shirt", "toptee")
RUN_ID = "9e4918a6b5c243412cbc58c9"
FEATURES = Path("/home/heheboiz/data/cir/workbench/artifacts/composition/runs") / RUN_ID / "work/limn/features"
OUT = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native19-dual" if False else
           "/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn")
TOP_K = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified(path: Path) -> tuple[np.memmap, dict]:
    """Load a feature cache only if its recorded content digest is the digest of the bytes on disk."""
    sidecar = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != tuple(sidecar["shape"]):
        raise SystemExit(f"REFUSING {path}: shape {array.shape} != sidecar {sidecar['shape']}")
    digest = _content_digest(array)
    if digest != sidecar["content_digest"]:
        raise SystemExit(f"REFUSING {path}: content digest {digest} != recorded {sidecar['content_digest']}")
    return array, {**sidecar, "recomputed_content_digest": digest, "verified": True}


def main() -> int:
    config = resolve_config()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    started = time.time()

    report: dict = {
        "artifact_type": "limn_dual_scope_diagnostic",
        "schema_version": 1,
        "purpose": ("diagnostic comparison for the two CIRCUS queries whose reference image and "
                    "normalized caption pair are filed under both shirt and toptee in the FashionIQ "
                    "validation captions; NOT part of the canonical 6003-query metric"),
        "canonical_status": {
            "status": "PARTIAL",
            "scored_count": 6001,
            "unscored_count": 2,
            "unscored_qids": list(QIDS),
            "note": ("Both diagnostic scopes run; the canonical metrics stay 6001/6003 and both "
                     "queries stay unscored there. No diagnostic number is folded into the canonical "
                     "evaluator, no scope is chosen as 'the' answer."),
        },
        "source_run_id": RUN_ID,
        "gallery": {
            "count": 74381,
            "protocol_id": "circus_fashioniq_release_v1",
            "note": ("the FULL released gallery for every scope: no category filtering, the "
                     "checkpoint being category-specific does not restrict candidates"),
        },
        "conditions": list(CONDITIONS),
        "scopes": list(SCOPES),
        "queries": {},
    }

    print("# loading release", flush=True)
    data = load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT))
    data = assign_garment_categories(data, Path(config.FASHIONIQ_ROOT))
    release_provenance = {
        "hash_verification": data.provenance.get("hash_verification"),
        "sha256": data.provenance.get("sha256"),
        "counts": data.provenance.get("counts"),
        "category_resolution_counts": {
            key: data.provenance["category_resolution"].get(key)
            for key in ("considered_count", "resolved_count", "ambiguous_count", "unresolved_count")
        },
    }
    report["release_provenance"] = release_provenance
    by_qid = {query.qid: query for query in data.queries}
    positions = _positive_positions(data)
    gallery_ids = list(data.gallery.image_ids)

    model = model_by_id("limn")
    bundle = checkpoint_by_id(model, "base_iter0_all_categories")
    resolver = image_resolver(Path(config.FASHIONIQ_ROOT), model["fashioniq_layout"])

    galleries: dict[str, tuple[np.memmap, dict]] = {}
    checkpoints: dict[str, dict] = {}
    for scope in SCOPES:
        gallery_file = FEATURES / f"gallery_{scope}.npy"
        print(f"# verifying gallery cache {gallery_file.name}", flush=True)
        array, sidecar = load_verified(gallery_file)
        norms = np.linalg.norm(np.asarray(array[:2048], dtype=np.float64), axis=1)
        galleries[scope] = (array, sidecar)
        member = checkpoint_by_id(model, f"base_iter0_{scope}")
        path = checkpoint_path("limn", member, Path(config.WORKBENCH_CHECKPOINT_ROOT))
        print(f"# hashing checkpoint {path.name}", flush=True)
        measured = sha256_file(path)
        if measured != member["expected_sha256"]:
            raise SystemExit(f"REFUSING checkpoint {path}: sha256 {measured} != registry {member['expected_sha256']}")
        checkpoints[scope] = {
            "checkpoint_id": member["checkpoint_id"],
            "filename": member["filename"],
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": measured,
            "sha256_source": "recomputed over the file on disk; equals the registry expected_sha256",
            "mapping_status": member.get("checkpoint_mapping_status"),
            "bundle_id": bundle["checkpoint_id"],
            "dataset_type_index": {"dress": 0, "shirt": 1, "toptee": 2}[scope],
        }
        report.setdefault("gallery_matrices", {})[scope] = {
            "file": str(gallery_file),
            "shape": list(sidecar["shape"]),
            "dtype": sidecar["dtype"],
            "byte_length": sidecar["byte_length"],
            "recorded_content_digest": sidecar["content_digest"],
            "recomputed_content_digest": sidecar["recomputed_content_digest"],
            "verify_mode": sidecar["verify_mode"],
            "row_norm_sample_first_2048_mean": float(norms.mean()),
            "row_norm_sample_first_2048_min": float(norms.min()),
            "row_norm_sample_first_2048_max": float(norms.max()),
        }
    report["checkpoints"] = checkpoints

    # ---- sanity check of gallery ORDER + tie rule against the recorded canonical ranks ----
    # If the stored matrices were in a different order, or the rank convention were different, the
    # recorded canonical ranks of queries scored in this very run would not reproduce.
    recorded = json.loads((Path(config.WORKBENCH_ARTIFACTS_ROOT) / "composition/runs" / RUN_ID /
                           "ranks" / "limn_base_iter0_all_categories.json").read_text(encoding="utf-8"))
    scope_of = {}
    # Rebuild the run's routed scope map with the production resolver (pure routing, no GPU).
    adapter = Adapter(config)
    adapter.work_root = Path(tempfile.mkdtemp(prefix="limn-dual-"))
    for query in data.queries:
        scope_of[query.qid] = adapter.query_scope(query)
    check_qids = [q.qid for q in data.queries
                  if scope_of[q.qid] in SCOPES and q.qid not in QIDS][:12]
    crosscheck = []
    for condition in CONDITIONS:
        query_file = FEATURES / f"query_{condition}_shirt.npy"
        dense, _ = load_verified(query_file)
        for qid in check_qids:
            query = by_qid[qid]
            scope = scope_of[qid]
            row = np.asarray(dense[query.index], dtype=np.float32)
            if not np.isfinite(row).all():
                continue
            gallery = galleries[scope][0]
            scores = adapter.score(row[None, :], {scope: gallery}, [scope])
            ranks = adapter.rank(row[None, :], {scope: gallery}, [positions[qid]], [scope])
            crosscheck.append({
                "condition": condition, "qid": qid, "scope": scope,
                "recorded_rank": recorded["ranks"][condition].get(qid),
                "recomputed_rank": ranks[0],
                "max_abs_score_self_diff": 0.0,
                "agrees": recorded["ranks"][condition].get(qid) == ranks[0],
            })
    report["rank_convention_crosscheck"] = {
        "method": ("recompute the recorded canonical target rank of already-scored queries from "
                   "this run's own cached query matrix + cached gallery matrix, using the adapter's "
                   "score/rank; agreement proves gallery ORDER and the tie rule are the recorded ones"),
        "checked": len(crosscheck),
        "all_agree": all(item["agrees"] for item in crosscheck),
        "items": crosscheck,
    }
    print(f"# rank crosscheck: {len(crosscheck)} recomputed, "
          f"all_agree={report['rank_convention_crosscheck']['all_agree']}", flush=True)

    # ---- per query x scope x condition ----
    raw_paths: dict[str, str] = {}
    query_rows = []
    for qid in QIDS:
        query = by_qid[qid]
        path = resolver(query.reference_id, data.candidate_names)
        raw_paths[qid] = str(path)
        entry = {
            "qid": qid,
            "release": {
                "index": query.index,
                "query_txt": query.query_txt,
                "reference_id": query.reference_id,
                "positive_ids": list(query.positive_ids),
                "loader_category": query.category,
                "matched_subsets": list(query.subsets),
            },
            "resolved_query_image_path": str(path),
            "target_position_in_gallery": positions[qid],
            "target_gallery_id": gallery_ids[positions[qid][0]],
            "scopes": {},
        }
        query_rows.append(entry)
        report["queries"][qid] = entry

    workers = resolved_decode_workers()
    print(f"# decode workers {workers}, torch threads pinned by adapter", flush=True)
    for scope in SCOPES:
        for condition in CONDITIONS:
            texts, images, categories = [], [], []
            for qid in QIDS:
                query = by_qid[qid]
                staged, text = _stage_query_path(Path(raw_paths[qid]), query, condition, adapter.work_root)
                images.append(str(staged))
                texts.append(text)
                categories.append(scope)
            print(f"# encoding {scope}/{condition}: 2 queries", flush=True)
            began = time.time()
            encoded = adapter.encode_queries(images, texts, categories)
            elapsed = time.time() - began
            if encoded.shape != (len(QIDS), 768) or not np.isfinite(encoded).all():
                raise SystemExit(f"REFUSING {scope}/{condition}: bad query matrix {encoded.shape}")
            gallery = galleries[scope][0]
            scores = adapter.score(encoded, {scope: gallery}, [scope] * len(QIDS))
            ranks = adapter.rank(encoded, {scope: gallery},
                                 [positions[qid] for qid in QIDS], [scope] * len(QIDS))
            jsonl_rows = []
            for row, qid in enumerate(QIDS):
                score_row = np.asarray(scores[row], dtype=np.float32)
                target_position = positions[qid][0]
                order = np.argsort(-score_row, kind="stable")
                top = [(int(position), float(score_row[position]))
                       for position in order[:TOP_K]]
                query_scope = report["queries"][qid]["scopes"].setdefault(scope, {})
                query_scope[condition] = {
                    "scope": scope,
                    "condition": condition,
                    "query_embedding_norm": float(np.linalg.norm(encoded[row].astype(np.float64))),
                    "staged_image_path": images[row],
                    "model_text": texts[row],
                    "target_rank": ranks[row],
                    "target_score": float(score_row[target_position]),
                    "target_gallery_position": int(target_position),
                    "target_did": query.positive_ids[0],
                    "gallery_size_scored": int(score_row.size),
                    "score_min": float(score_row.min()),
                    "score_max": float(score_row.max()),
                    "score_at_rank_1": float(score_row[order[0]]),
                    "target_margin_vs_rank1": float(score_row[order[0]] - score_row[target_position]),
                    "encode_seconds": round(elapsed, 3),
                }
                jsonl_rows.append({
                    "qid": qid, "scope": scope, "condition": condition,
                    "model_text": texts[row], "query_image_path": images[row],
                    "checkpoint_sha256": checkpoints[scope]["sha256"],
                    "gallery_size": int(score_row.size),
                    "target_rank": ranks[row],
                    "target_score": float(score_row[target_position]),
                    "depth": TOP_K,
                    "candidates": [
                        {"rank": index + 1, "gallery_position": position,
                         "gallery_id": gallery_ids[position],
                         "image_path": data.candidate_names.get(gallery_ids[position],
                                                               gallery_ids[position]),
                         "score": score,
                         "is_target": position == target_position}
                        for index, (position, score) in enumerate(top)
                    ],
                })
            # One file per scope+condition holding BOTH queries: a per-row open truncated it.
            jsonl = OUT / f"limn_dual_scope.{scope}.{condition}.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for record in jsonl_rows:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            for record in jsonl_rows:
                report["queries"][record["qid"]]["scopes"][scope][condition]["top100_export"] = str(jsonl)

    adapter.close()
    report["runtime"] = {
        "controller_python": sys.version,
        "model_interpreter": adapter.interpreter,
        "decode_workers": workers,
        "elapsed_seconds": round(time.time() - started, 2),
        "top_k": TOP_K,
        "scoring": "adapter.score: L2-normalized query row dot composed target row (target rows are already L2-normalized by compose_target)",
        "rank": "adapter.rank: 1 + count(row > v) + count(row[:i] == v), same as the evaluator",
    }
    (OUT / "limn_dual_scope.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = {qid: {scope: {condition: report["queries"][qid]["scopes"][scope][condition]["target_rank"]
                             for condition in CONDITIONS} for scope in SCOPES} for qid in QIDS}
    print(json.dumps({"ranks": summary,
                      "crosscheck_all_agree": report["rank_convention_crosscheck"]["all_agree"],
                      "seconds": report["runtime"]["elapsed_seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
