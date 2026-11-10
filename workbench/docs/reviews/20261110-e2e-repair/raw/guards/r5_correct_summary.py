"""R5 remedy: rebuild the diagnostic summary from the evidence that is already proven correct.

The producer (`t2_dual_scope.py:273`) read `query.positive_ids[0]` from a loop variable that had
already advanced, so all twelve rows recorded `7:42304` (the positive of `7:3790`). Nothing else
leaked: rank, score, gallery position, target asset and candidate ids in every export row agree with
the summary and with the query's own release row.

This script does NOT overwrite the historical artifact. It writes a corrected copy beside its own
digest, so the old file stays exactly as it was measured and the correction is auditable.

Read-only with respect to the reviewed evidence.
Usage: workbench/envs/ctl312h/bin/python .../raw/guards/r5_correct_summary.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
REVIEW = REPO / "workbench" / "docs" / "reviews" / "20261109-native15-composition-limn"
OUT = REPO / "workbench" / "docs" / "reviews" / "20261110-e2e-repair" / "raw" / "guards"
sys.path.insert(0, str(REPO))

from workbench.backend.operator_config import resolve_config
from workbench.composition.data import load_release

QIDS = ("7:3038", "7:3790")
SCOPES = ("shirt", "toptee")
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def main() -> int:
    config = resolve_config()
    data = load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT))
    by_qid = {query.qid: query for query in data.queries}
    did_to_asset = dict(data.candidate_names)

    source = REVIEW / "limn_dual_scope.json"
    original_bytes = source.read_bytes()
    summary = json.loads(original_bytes)
    changed = []

    for qid in QIDS:
        own = [str(value) for value in by_qid[qid].positive_ids]
        record = summary["queries"][qid]
        expected_asset = did_to_asset.get(own[0])
        # The query record itself carried no target_did; the leaked value lives on each condition.
        record["target_did"] = own[0]
        if Path(str(expected_asset)).name != Path(str(record["target_gallery_id"])).name:
            raise SystemExit(f"REFUSING: {qid} positive {own[0]} -> {expected_asset} but summary says "
                             f"{record['target_gallery_id']}")
        for scope in SCOPES:
            for condition in CONDITIONS:
                block = record["scopes"][scope][condition]
                if block.get("target_did") != own[0]:
                    changed.append({"qid": qid, "scope": scope, "condition": condition,
                                    "was": block.get("target_did"), "now": own[0]})
                    block["target_did"] = own[0]

    summary["provenance_correction"] = {
        "defect": "R5",
        "what": ("target_did repeated one query's positive id across every row because the producer "
                 "read a shared loop variable (t2_dual_scope.py:273)"),
        "fix": "each row's target_did is that query's own positive id, cross-checked against the release",
        "changed_records": len(changed),
        "source_file": str(source),
        "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "unchanged_fields": ["target_rank", "target_score", "target_gallery_position",
                             "score_at_rank_1", "target_margin_vs_rank1", "top100_export",
                             "gallery_size_scored", "model_text", "encode_seconds",
                             "query_embedding_norm", "score_min", "score_max"],
        "verification": ("rank/score/position/asset of all 12 combinations re-derived from each "
                         "scope's own export and the release loader; see r5_reconcile.json"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "limn_dual_scope.corrected.json"
    payload = json.dumps(summary, indent=2) + "\n"
    target.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    report = {
        "corrected_file": str(target),
        "corrected_sha256": digest,
        "source_file": str(source),
        "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "source_untouched": source.read_bytes() == original_bytes,
        "records_changed": len(changed),
        "changes": changed,
        "rescored": False,
        "reranked": False,
        "note": ("No rank, score, asset or candidate was recomputed. Only the mis-recorded identity "
                 "field changed; the historical file is preserved byte-for-byte."),
    }
    (OUT / "r5_correction.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("corrected_sha256", "source_sha256", "records_changed", "rescored")}, indent=1))
    for change in changed:
        print(" ", change["qid"], change["scope"], change["condition"],
              f"{change['was']} -> {change['now']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
