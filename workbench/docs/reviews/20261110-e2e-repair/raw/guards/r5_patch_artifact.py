"""Regenerate `limn_dual_scope.json`'s six wrong `target_did` cells, keeping everything else.

This is the R5 artifact repair. It is deliberately narrow:

* ONLY the six `target_did` cells belonging to `7:3038` change (`7:42304` -> `7:3416`). The six cells
  of `7:3790` already carry their own positive and are not written at all.
* `target_rank`, `target_score`, `target_gallery_position`, `score_at_rank_1`,
  `target_margin_vs_rank1`, `top100_export`, `gallery_size_scored`, `model_text`, `encode_seconds`,
  `query_embedding_norm`, `score_min`, `score_max` are proven correct (see `r5_reconcile.json`,
  9 checks x 12/12) and are asserted unchanged. So are `target_gallery_id`,
  `target_position_in_gallery`, `checkpoints`, `gallery`, `canonical_status`, `canonical_cohort`,
  `release_provenance`, `rank_convention_crosscheck`, `runtime`, and every other top-level field.
* The original is preserved byte-for-byte as `limn_dual_scope.json.pre-r5` (created once; a second run
  refuses rather than overwriting it).

Nothing is re-measured and no producer is run: the value written is the query's own positive id,
taken from the release loader -- the same lookup the serve layer now performs.

Usage:
  workbench/envs/ctl312h/bin/python \
    workbench/docs/reviews/20261110-e2e-repair/raw/guards/r5_patch_artifact.py [--dry-run]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
REVIEW = REPO / "workbench" / "docs" / "reviews" / "20261109-native15-composition-limn"
SUMMARY = REVIEW / "limn_dual_scope.json"
BACKUP = REVIEW / "limn_dual_scope.json.pre-r5"
GUARDS = REPO / "workbench" / "docs" / "reviews" / "20261110-e2e-repair" / "raw" / "guards"
sys.path.insert(0, str(REPO))

from workbench.backend.operator_config import resolve_config
from workbench.composition.data import load_release

#: Cells whose target_did is rewritten: the six of 7:3038. 7:3790's six are correct and untouched.
QID_TO_FIX = "7:3038"
SCOPES = ("shirt", "toptee")
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")

#: Never written by this script, at any depth below a condition block.
UNTOUCHED_CELL_FIELDS = (
    "target_rank", "target_score", "target_gallery_position", "score_at_rank_1",
    "target_margin_vs_rank1", "top100_export", "gallery_size_scored", "model_text",
    "encode_seconds", "query_embedding_norm", "score_min", "score_max", "scope", "condition",
    "staged_image_path", "query_embedding_norm",
)

#: Never written at the query record or top level.
UNTOUCHED_QUERY_FIELDS = ("target_gallery_id", "target_position_in_gallery", "release",
                          "resolved_query_image_path", "qid")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    dry_run = "--dry-run" in (argv if argv is not None else sys.argv[1:])
    config = resolve_config()
    data = load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT))
    by_qid = {query.qid: query for query in data.queries}
    did_to_asset = dict(data.candidate_names)

    original_bytes = SUMMARY.read_bytes()
    before = sha256(original_bytes)
    summary = json.loads(original_bytes)
    # A deep snapshot of every field this script must not touch, for the post-write assertion.
    untouched_snapshot = {
        qid: {
            "top": {key: value for key, value in record.items() if key not in ("target_did", "scopes")},
            "cells": {
                scope: {
                    condition: {key: value for key, value in block.items()
                                if key not in UNTOUCHED_CELL_FIELDS[:0] + ("target_did",)}
                    for condition, block in record["scopes"][scope].items()
                } for scope in record["scopes"]
            },
        } for qid, record in summary["queries"].items()
    }
    top_level_snapshot = {key: value for key, value in summary.items() if key != "queries"}

    own = [str(value) for value in by_qid[QID_TO_FIX].positive_ids]
    if not own:
        raise SystemExit(f"REFUSING: the release states no positive id for {QID_TO_FIX}")
    correct_did = own[0]
    expected_asset = Path(str(did_to_asset.get(correct_did, ""))).name
    declared_asset = Path(str(summary["queries"][QID_TO_FIX]["target_gallery_id"])).name
    if expected_asset != declared_asset:
        raise SystemExit(f"REFUSING: {QID_TO_FIX} positive {correct_did} resolves to "
                         f"{expected_asset} but the summary names {declared_asset}")

    changes = []
    for scope in SCOPES:
        for condition in CONDITIONS:
            block = summary["queries"][QID_TO_FIX]["scopes"][scope][condition]
            was = block.get("target_did")
            if was == correct_did:
                continue
            changes.append({"qid": QID_TO_FIX, "scope": scope, "condition": condition,
                            "was": was, "now": correct_did})
            block["target_did"] = correct_did
    if not changes:
        raise SystemExit("nothing to change: every cell already records its own query's positive")

    summary["queries"][QID_TO_FIX]["target_did"] = correct_did
    summary["provenance_correction_r5"] = {
        "defect": "R5",
        "what": ("`target_did` repeated one query's positive id across every cell because the "
                 "producer read a shared loop variable (t2_dual_scope.py:273, fixed there too)"),
        "fix": ("each cell's target_did is that query's own positive id, cross-checked against the "
                "release loader; the serve layer additionally refuses a stale value and reports it "
                "as a defect"),
        "cells_changed": len(changes),
        "changed": changes,
        "unchanged": list(UNTOUCHED_CELL_FIELDS),
        "source_sha256_before": before,
        "original_preserved_at": BACKUP.name,
        "rescored": False,
        "reranked": False,
        "verification": ("rank/score/position/asset of all 12 cells re-derived from each scope's own "
                         "export plus the release loader; raw/guards/r5_reconcile.json"),
    }

    # The payload this script must not have altered, checked before anything is written.
    for qid, snapshot in untouched_snapshot.items():
        record = summary["queries"][qid]
        top_after = {key: value for key, value in record.items()
                     if key not in ("target_did", "scopes", "provenance_correction_r5")}
        if top_after != snapshot["top"]:
            raise SystemExit(f"REFUSING: top-level fields of {qid} changed")
        for scope, cells in snapshot["cells"].items():
            for condition, block in cells.items():
                after = {key: value for key, value in record["scopes"][scope][condition].items()
                         if key != "target_did"}
                if after != block:
                    raise SystemExit(f"REFUSING: measurements of {qid}/{scope}/{condition} changed")
    # The one field this script ADDS is excluded by name; every pre-existing top-level field must be
    # byte-identical, so a measurement cannot move underneath the provenance note.
    if ({key: value for key, value in summary.items()
         if key not in ("queries", "provenance_correction_r5")} != top_level_snapshot):
        raise SystemExit("REFUSING: a top-level artifact field changed")

    payload = json.dumps(summary, indent=2) + "\n"
    after = sha256(payload.encode("utf-8"))
    report = {
        "artifact": str(SUMMARY),
        "backup": str(BACKUP),
        "sha256_before": before,
        "sha256_after": after,
        "cells_changed": len(changes),
        "changes": changes,
        "rescored": False,
        "reranked": False,
        "dry_run": dry_run,
    }
    if not dry_run:
        if BACKUP.exists():
            if sha256(BACKUP.read_bytes()) != before:
                raise SystemExit(f"REFUSING: {BACKUP.name} exists and is not this artifact's "
                                 "original bytes; not overwriting")
            report["backup_preexisting"] = True
        else:
            BACKUP.write_bytes(original_bytes)
            report["backup_preexisting"] = False
        SUMMARY.write_text(payload, encoding="utf-8")
        if sha256(SUMMARY.read_bytes()) != after:
            raise SystemExit("REFUSING: written bytes do not match the recorded digest")
        report["verified_on_disk"] = sha256(SUMMARY.read_bytes())
    GUARDS.mkdir(parents=True, exist_ok=True)
    (GUARDS / "r5_artifact_patch.json").write_text(json.dumps(report, indent=1) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("sha256_before", "sha256_after", "cells_changed", "rescored", "dry_run")},
                     indent=1))
    for change in changes:
        print(f"  {change['qid']} {change['scope']} {change['condition']} "
              f"{change['was']} -> {change['now']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
