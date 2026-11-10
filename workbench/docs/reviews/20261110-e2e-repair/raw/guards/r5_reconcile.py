"""R5: reconcile qid -> positive DID -> gallery asset -> rank across all 12 diagnostic combinations.

Read-only. Prints one line per (qid, scope, condition) and a summary verdict.
Usage: workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261110-e2e-repair/raw/guards/r5_reconcile.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
REVIEW = REPO / "workbench" / "docs" / "reviews" / "20261109-native15-composition-limn"
sys.path.insert(0, str(REPO))

from workbench.backend.operator_config import resolve_config
from workbench.composition.data import assign_garment_categories, load_release

QIDS = ("7:3038", "7:3790")
SCOPES = ("shirt", "toptee")
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def main() -> int:
    config = resolve_config()
    data = assign_garment_categories(load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT)),
                                     Path(config.FASHIONIQ_ROOT))
    by_qid = {query.qid: query for query in data.queries}
    # canary id -> gallery image path, exactly as the release loader resolved it
    did_to_asset = {did: path for did, path in data.candidate_names.items()}

    summary = json.loads((REVIEW / "limn_dual_scope.json").read_text(encoding="utf-8"))
    rows = []
    problems = []
    for qid in QIDS:
        query = by_qid[qid]
        positive = query.positive_ids[0]
        expected_asset = did_to_asset.get(positive)
        record = summary["queries"][qid]
        print(f"# {qid}: reference={query.reference_id} positive_did={positive} "
              f"-> asset={expected_asset} | summary target_gallery_id={record['target_gallery_id']}")
        if Path(str(expected_asset)).name != Path(str(record["target_gallery_id"])).name:
            problems.append(f"{qid}: positive {positive} resolves to {expected_asset} but summary "
                            f"declares target_gallery_id {record['target_gallery_id']}")
        for scope in SCOPES:
            for condition in CONDITIONS:
                declared = record["scopes"][scope][condition]
                export = REVIEW / Path(declared["top100_export"]).name
                export_rows = [json.loads(line) for line in export.read_text(encoding="utf-8").splitlines()
                               if line.strip()]
                export_row = next(row for row in export_rows if row["qid"] == qid)
                targets = [c for c in export_row["candidates"] if c.get("is_target")]
                row = {
                    "qid": qid, "scope": scope, "condition": condition,
                    "positive_did": positive,
                    "expected_asset": Path(str(expected_asset)).name,
                    "summary_target_did": declared.get("target_did"),
                    "summary_target_gallery_id": record["target_gallery_id"],
                    "summary_target_position": declared["target_gallery_position"],
                    "summary_target_rank": declared["target_rank"],
                    "summary_target_score": declared["target_score"],
                    "export_target_count": len(targets),
                    "export_target_gallery_id": targets[0]["gallery_id"] if targets else None,
                    "export_target_position": targets[0]["gallery_position"] if len(targets) == 1 else None,
                    "export_target_rank": targets[0]["rank"] if len(targets) == 1 else None,
                    "export_target_score": targets[0]["score"] if len(targets) == 1 else None,
                    "export_row_target_rank": export_row["target_rank"],
                    "export_row_target_score": export_row["target_score"],
                    "export_checkpoint_sha256": export_row.get("checkpoint_sha256"),
                    "summary_checkpoint_sha256": (summary.get("checkpoints") or {}).get(scope, {}).get("sha256"),
                    "export_candidate_ids": [c["gallery_id"] for c in export_row["candidates"]],
                    "export_candidate_count": len(export_row["candidates"]),
                    "export_depth": export_row.get("depth"),
                }
                rows.append(row)
                # A target ranked outside the stored window has no `is_target` row at all: that is
                # expected (rank 6891 vs depth 100), not a defect. The asset is then checked against
                # the summary's own target_gallery_id, which is what the window endpoint serves.
                flagged = row["export_target_gallery_id"] is not None
                checks = {
                    "did_matches_own_query": row["summary_target_did"] == positive,
                    "asset_matches_own_positive": (
                        Path(str(row["export_target_gallery_id"] or
                                 row["summary_target_gallery_id"])).name == row["expected_asset"]),
                    "target_flag_consistent_with_window": (
                        flagged == (1 <= row["summary_target_rank"] <= row["export_candidate_count"])),
                    "rank_summary_eq_export": (
                        row["summary_target_rank"] == row["export_row_target_rank"]
                        and (not flagged or row["summary_target_rank"] == row["export_target_rank"])),
                    "score_summary_eq_export": (
                        row["summary_target_score"] == row["export_row_target_score"]
                        and (not flagged or row["summary_target_score"] == row["export_target_score"])),
                    "position_summary_eq_export": (not flagged
                                                   or row["summary_target_position"] == row["export_target_position"]),
                    "checkpoint_summary_eq_export": row["export_checkpoint_sha256"] == row["summary_checkpoint_sha256"],
                    "no_duplicate_candidate_ids": len(set(row["export_candidate_ids"])) == len(row["export_candidate_ids"]),
                    "export_depth_is_candidate_count": row["export_depth"] == row["export_candidate_count"],
                }
                failed = [name for name, ok in checks.items() if not ok]
                print(f"  {scope:6s} {condition:22s} rank={row['summary_target_rank']:>5} "
                      f"score={row['summary_target_score']:.6f} did={row['summary_target_did']} "
                      f"asset={Path(str(row['export_target_gallery_id'] or row['summary_target_gallery_id'])).name} "
                      f"-> {'OK' if not failed else 'FAIL ' + ','.join(failed)}")
                if failed:
                    problems.append(f"{qid}/{scope}/{condition}: {failed}")

    out = REPO / "workbench/docs/reviews/20261110-e2e-repair/raw/guards/r5_reconcile.json"
    out.write_text(json.dumps({
        "combinations": len(rows),
        "rows": [{key: value for key, value in row.items() if key != "export_candidate_ids"}
                 for row in rows],
        "problems": problems,
        "verdict": "R5_CONFIRMED" if problems else "R5_NOT_REPRODUCED",
        "note": ("candidate ids / rank / score are compared between the summary, the export row and the "
                 "export's own is_target candidate; target_did is compared against the query's own "
                 "positive id and against the release mapping positive->gallery asset"),
    }, indent=1) + "\n", encoding="utf-8")
    print(f"\nVERDICT {('R5_CONFIRMED' if problems else 'R5_NOT_REPRODUCED')} problems={len(problems)}")
    for problem in problems:
        print("  -", problem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
