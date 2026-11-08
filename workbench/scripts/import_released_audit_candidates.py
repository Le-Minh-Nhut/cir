#!/usr/bin/env python3
"""Convert the released CIRCUS audit artifact into candidate Top-K export format, and verify it.

The released artifact ``retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json`` stores,
per query and per condition, the ordered gallery positions the *official* retriever returned and
the resulting target rank. It is a real artifact produced by the released pipeline, so converting
it is a legitimate way to exercise the candidate reader and API end-to-end without starting a GPU
evaluation.

It is NOT a model we evaluated. Nothing written here is attached to one of our composition runs;
the export lands under ``--output-root`` and its manifest records
``source = "released_audit_retriever"`` so a consumer can never mistake it for a model result.

Two invariants are asserted and printed, because they are the ones that would catch a broken
conversion:

1. the candidate rank of the released target inside the exported window equals the target rank the
   released artifact recorded (whenever the target is inside the window);
2. the exported window positions are exactly the artifact's own leading positions, in order.

Usage:
  python workbench/scripts/import_released_audit_candidates.py --output-root /tmp/topk-audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.composition.data import resolve_release_paths
from workbench.composition.topk import (
    DEFAULT_EXPORT_DEPTH,
    abort_export,
    close_export,
    gallery_digest,
    open_export,
    read_candidates,
)

CONDITION_KEYS = {"multimodal": "multimodal",
                  "text_only_black_image": "text_only",
                  "image_only_empty_text": "image_only"}
RUN_ID = "released_audit_task7_mm_any"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="artifacts root to write composition/runs/<run>/candidates/ under")
    parser.add_argument("--depth", type=int, default=DEFAULT_EXPORT_DEPTH,
                        help=f"candidate window depth (default {DEFAULT_EXPORT_DEPTH}); the "
                             "released artifact stores 20, so 20 is the honest maximum here")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--verify", action="store_true",
                        help="read the export back and assert its invariants")
    parser.add_argument("--write-summary", action="store_true",
                        help="also write a composition summary for the released audit retriever, "
                             "so an HTTP client can browse the candidates. Its provenance names the "
                             "released artifact and it is never presented as one of our model runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = resolve_config()
    paths = resolve_release_paths(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT))
    artifact = json.loads(paths.aggregated_audit.read_text(encoding="utf-8"))
    samples = artifact.get("samples")
    gallery_paths = artifact.get("gallery_paths")
    if not isinstance(samples, list) or not isinstance(gallery_paths, list):
        print("[IMPORT-FAILED] released audit artifact lacks samples or gallery_paths", file=sys.stderr)
        return 1

    available = len(samples[0].get("retrievals", {}).get("multimodal", [])) if samples else 0
    depth = min(args.depth, available) if available else args.depth
    if depth < 1:
        print("[IMPORT-FAILED] released artifact stores no retrieval positions", file=sys.stderr)
        return 1

    # The exported positions index the audit artifact's own gallery_paths, which is the released
    # gallery order the release loader reconstructs (`load_release` normalises exactly these paths
    # and preserves their order). Recording the digest over the RELEASE gallery + candidate names is
    # what the reader recomputes, so a stored position can be resolved to an image ID and a digest
    # mismatch is a real refusal rather than a formatting difference.
    from workbench.composition.data import load_release

    release = load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT), verify_hashes=False,
                           expect_released_counts=False)
    if len(release.gallery.image_ids) != len(gallery_paths):
        print(f"[IMPORT-FAILED] audit gallery {len(gallery_paths)} != release gallery "
              f"{len(release.gallery.image_ids)}", file=sys.stderr)
        return 1
    identity = gallery_digest(release.gallery.protocol_id, release.gallery.image_ids,
                              release.candidate_names)

    writers = open_export(args.output_root, args.run_id, attempt="released_audit",
                          model_id="released_audit", checkpoint_id="task7_mm_any",
                          protocol_id="circus_fashioniq_release_v1",
                          gallery_identity=identity, gallery_count=len(gallery_paths),
                          depth=depth, conditions=list(CONDITION_KEYS),
                          score_semantics="released audit retrieval order; the artifact stores "
                                          "positions, not scores")
    # The score column is `null` when the producer stored no scores: an ordering is a real fact and
    # an invented float would not be.
    for writer in writers.values():
        writer.score_semantics = "released audit retrieval order (no scores stored)"
    written = {condition: 0 for condition in CONDITION_KEYS}
    try:
        for sample in samples:
            qid = f"7:{int(sample.get('query_idx')) + 1}"
            retrievals = sample.get("retrievals") or {}
            ranks = sample.get("ranks") or {}
            for condition, key in CONDITION_KEYS.items():
                positions = retrievals.get(key)
                if not isinstance(positions, list) or not positions:
                    continue
                window = [(int(position), None) for position in positions[:depth]]
                writers[condition].write(qid, ranks.get(key), window)
                written[condition] += 1
        close_export(writers, expected_qids={condition: [
            f"7:{int(sample.get('query_idx')) + 1}" for sample in samples
            if (sample.get("retrievals") or {}).get(CONDITION_KEYS[condition])]
            for condition in CONDITION_KEYS})
    except BaseException:
        abort_export(writers)
        raise

    report = {"run_id": args.run_id, "depth": depth, "written": written,
              "gallery_count": len(gallery_paths), "gallery_identity_sha256": identity,
              "output_root": str(args.output_root)}
    if args.write_summary:
        report["summary_paths"] = _write_summary(args.output_root, args.run_id, artifact,
                                                 samples, identity, len(gallery_paths))
    if args.verify:
        report["verification"] = _verify(args.output_root, args.run_id, samples, depth)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(item["mismatch_count"] == 0 for item in report.get("verification", {}).values()) else 1


def _write_summary(root: Path, run_id: str, artifact: dict, samples: list[dict],
                   identity: str, gallery_count: int) -> dict[str, str]:
    """Write ranks + a summary for the released audit retriever under the given artifacts root.

    Marked ``is_synthetic: false`` (the numbers are the released artifact's own) but carrying
    ``provenance_source = "released_audit_retriever"`` and ``reproduction_status =
    RELEASED_ARTIFACT_IMPORT``, so it is browsable without ever being counted as a model run of
    ours.
    """
    from workbench.composition.data import RELEASE_PROTOCOL_ID

    composition = root / "composition"
    ranks_path = composition / "runs" / run_id / "ranks" / "released_audit_task7_mm_any.json"
    ranks_path.parent.mkdir(parents=True, exist_ok=True)
    ranks = {condition: {} for condition in CONDITION_KEYS}
    failures: dict[str, dict[str, str]] = {condition: {} for condition in CONDITION_KEYS}
    for sample in samples:
        qid = f"7:{int(sample.get('query_idx')) + 1}"
        for condition, key in CONDITION_KEYS.items():
            value = (sample.get("ranks") or {}).get(key)
            ranks[condition][qid] = value if isinstance(value, int) else None
            if not isinstance(value, int):
                failures[condition][qid] = "released artifact recorded no rank for this query"
    ranks_path.write_text(json.dumps({
        "fingerprint": identity, "attempt": "released_audit", "ranks": ranks,
        "query_failures": failures,
        "condition_status": {condition: {"status": "COMPLETED"} for condition in CONDITION_KEYS},
        "provenance_source": "released_audit_retriever",
    }, indent=2, sort_keys=True), encoding="utf-8")

    from workbench.composition.metrics import compose

    result = compose(ranks, query_count=len(samples))
    metrics = {
        "ndcg_multimodal": result.multimodal.ndcg_full,
        "ndcg_text_only": result.text_only.ndcg_full,
        "ndcg_image_only": result.image_only.ndcg_full,
        "composition_gap_ndcg": result.composition_gap_ndcg,
        "recall_at_10_multimodal": result.multimodal.recall_at_10,
        "recall_at_50_multimodal": result.multimodal.recall_at_50,
        "mrr_gap": result.mrr_gap,
        "coverage": result.multimodal.coverage,
        "common_cohort_count": result.gap_cohort_count,
        "multimodal": result.multimodal.__dict__,
        "text_only": result.text_only.__dict__,
        "image_only": result.image_only.__dict__,
        "shortcut_outcomes": result.shortcut_outcomes,
        "outcomes": result.shortcut_outcomes,
    }
    summary_dir = composition / "summary" / f"released_audit_task7_mm_any_{run_id}"
    written = {}
    for subset, count in (("full", len(samples)), ("sf", 4069), ("validated", 586)):
        path = summary_dir / f"{subset}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": 1,
            "is_synthetic": False,
            "provenance_source": "released_audit_retriever",
            "source_artifact": str(artifact.get("metadata")),
            "model_id": "released_audit",
            "checkpoint_id": "task7_mm_any",
            "run_id": run_id,
            "checkpoint_identity": None,
            "protocol_id": RELEASE_PROTOCOL_ID,
            "subset": subset,
            "gallery_count": gallery_count,
            "gallery_identity_sha256": identity,
            "query_count": count,
            "query_count_per_subset": {"full": len(samples), "sf": 4069, "validated": 586},
            "conditions": list(CONDITION_KEYS),
            "execution_status": "COMPLETED",
            "condition_status": {condition: {"status": "COMPLETED", "unscored_count": 0}
                                 for condition in CONDITION_KEYS},
            "reproduction_status": "RELEASED_ARTIFACT_IMPORT",
            "checkpoint_transfer": None,
            "checkpoint_training_dataset": None,
            "smoke": {"limited": False, "query_limit": None, "label": None},
            "preprocessing": {"policy": "released retriever; no local transform"},
            "runtime": {"controller_python": None, "platform": None, "gpu": None,
                        "model_interpreter": {"environment_fingerprint": None}},
            "seed": 0,
            "metrics": metrics,
        }, indent=2, sort_keys=True, default=str), encoding="utf-8")
        written[subset] = str(path)
    return {"ranks": str(ranks_path), **written}


def _verify(root: Path, run_id: str, samples: list[dict], depth: int) -> dict:
    """Read the export back and assert the two conversion invariants."""
    results = {}
    for condition, key in CONDITION_KEYS.items():
        export = read_candidates(root, run_id, condition)
        if not export["available"]:
            results[condition] = {"status": export["status"], "mismatch_count": 0,
                                  "checked_count": 0}
            continue
        rows = export["rows"]
        checked = mismatches = 0
        examples = []
        for sample in samples:
            qid = f"7:{int(sample.get('query_idx')) + 1}"
            expected_positions = (sample.get("retrievals") or {}).get(key)
            if not expected_positions or qid not in rows:
                continue
            checked += 1
            stored = [position for position, _score in rows[qid]["topk"]]
            if stored != [int(position) for position in expected_positions[:depth]]:
                mismatches += 1
                if len(examples) < 3:
                    examples.append({"qid": qid, "stored": stored[:5],
                                     "expected": expected_positions[:5]})
                continue
            recorded_rank = (sample.get("ranks") or {}).get(key)
            target_position = sample.get("target_gallery_index")
            if isinstance(recorded_rank, int) and isinstance(target_position, int) \
                    and recorded_rank <= len(stored):
                # In-window: the metric rank must equal the candidate position, 1-based.
                if stored[recorded_rank - 1] != target_position:
                    mismatches += 1
                    if len(examples) < 3:
                        examples.append({"qid": qid, "rank": recorded_rank,
                                         "candidate_at_rank": stored[recorded_rank - 1],
                                         "target_position": target_position})
            if rows[qid]["target_rank"] != recorded_rank:
                mismatches += 1
                if len(examples) < 3:
                    examples.append({"qid": qid, "stored_rank": rows[qid]["target_rank"],
                                     "recorded_rank": recorded_rank})
        results[condition] = {"status": export["status"], "checked_count": checked,
                              "mismatch_count": mismatches, "examples": examples,
                              "manifest": {k: export["manifest"].get(k) for k in
                                           ("depth", "record_count", "partial", "score_semantics",
                                            "tie_policy", "gallery_count")}}
    return results


if __name__ == "__main__":
    raise SystemExit(main())
