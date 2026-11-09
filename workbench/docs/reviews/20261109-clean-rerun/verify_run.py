"""Step 3 invariants for one composition run artifact, read from disk only.

    verify_run.py <run_id> --model <id> --checkpoint <id> [--config /path/cfg.yaml]

Checks, per condition:
  1. every in-scope query has a rank, or an explicit unscored reason;
  2. the candidate export exists at the claimed depth, its positions are valid gallery indices,
     each row is strictly ordered by score (ties by earlier position) and target_rank agrees with
     the metric path;
  3. the identity records the effective BLAS pin and identifies as identity schema v2.

Exit 0 = all invariants held. Prints a JSON verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", type=Path, default=Path.home() / "cir_rerun_20261109.yaml")
    args = parser.parse_args()

    from workbench.backend import autopilot
    from workbench.composition import data as composition_data

    config = autopilot._workbench_config(autopilot.load_config(args.config))
    data = composition_data.load_release(
        Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT), verify_hashes=False,
        expect_released_counts=False)
    data = composition_data.assign_garment_categories(data, config.FASHIONIQ_ROOT)
    gallery = list(data.gallery.image_ids)
    gallery_set = set(gallery)
    scope_qids = {query.qid for query in composition_data.select_queries(data, "full")}

    run_dir = Path(config.WORKBENCH_ARTIFACTS_ROOT) / "composition" / "runs" / args.run_id
    payload = json.loads((run_dir / "ranks" / f"{args.model}_{args.checkpoint}.json").read_text())
    ranks = payload["ranks"]
    failures = payload.get("query_failures") or {}
    identity = payload.get("identity") or {}
    cand_dir = run_dir / "candidates" / f"{args.model}_{args.checkpoint}"

    report: dict = {"run_id": args.run_id, "model": args.model,
                    "identity": {
                        "blas_threads_pinned": identity.get("blas_threads_pinned"),
                        "blas_pin_capability": identity.get("blas_pin_capability"),
                        "cache_verify_mode": identity.get("cache_verify_mode"),
                        "identity_schema_version": identity.get("identity_schema_version"),
                        "has_v2_markers": all(identity.get(key) is not None for key in
                                              ("blas_threads_pinned", "query_scope_resolution",
                                               "query_order_identity")),
                    },
                    "conditions": {}, "failed": []}

    for condition, rank_map in ranks.items():
        reasons = failures.get(condition) or {}
        missing = [qid for qid in scope_qids if qid not in rank_map and qid not in reasons]
        unsure = [qid for qid, value in rank_map.items() if value is None and qid not in reasons]
        scored = sum(1 for value in rank_map.values() if value is not None)
        entry = {"scored": scored, "query_count": len(rank_map), "unscored_with_reason": len(reasons),
                 "missing_any_record": len(missing), "rank_none_without_reason": len(unsure)}
        if missing:
            report["failed"].append(f"{condition}: {len(missing)} in-scope queries have neither a "
                                    f"rank nor a reason, e.g. {missing[:5]}")
        if unsure:
            report["failed"].append(f"{condition}: {len(unsure)} queries have rank None with no "
                                    f"reason, e.g. {unsure[:5]}")

        manifest_path = cand_dir / f"{condition}.manifest.json"
        if not manifest_path.is_file():
            entry["candidates"] = "ABSENT"
            report["conditions"][condition] = entry
            continue
        manifest = json.loads(manifest_path.read_text())
        rows_path = cand_dir / manifest["file"]
        depth = manifest["depth"]
        entry.update(candidates={"depth": depth, "record_count": manifest.get("record_count"),
                                 "expected_record_count": manifest.get("expected_record_count"),
                                 "gallery_identity_matches_rank": True,
                                 "partial": manifest.get("partial")})
        bad_pos, bad_order, bad_target, bad_gallery = [], [], [], []
        seen = 0
        with rows_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                seen += 1
                qid = row["qid"]
                positions = [position for position, _score in row["topk"]]
                scores = [score for _position, score in row["topk"]]
                if len(positions) != depth:
                    bad_pos.append((qid, "depth", len(positions)))
                for position in positions:
                    if not isinstance(position, int) or not 0 <= position < len(gallery):
                        bad_pos.append((qid, "range", position))
                    elif gallery[position] not in gallery_set:  # pragma: no cover - defensive
                        bad_gallery.append((qid, position))
                for (p1, s1), (p2, s2) in zip(zip(positions, scores), zip(positions[1:], scores[1:])):
                    if s2 > s1 or (s2 == s1 and p2 < p1):
                        bad_order.append((qid, p1, s1, p2, s2))
                        break
                if qid in rank_map and rank_map[qid] is not None and row["target_rank"] != rank_map[qid]:
                    bad_target.append((qid, row["target_rank"], rank_map[qid]))
        entry["candidates"]["rows_read"] = seen
        if seen != manifest.get("record_count"):
            report["failed"].append(f"{condition}: manifest says {manifest.get('record_count')} rows, "
                                    f"file has {seen}")
        if bad_pos:
            report["failed"].append(f"{condition}: invalid/short candidate rows {bad_pos[:3]}")
        if bad_order:
            report["failed"].append(f"{condition}: candidates not ordered by score {bad_order[:3]}")
        if bad_target:
            report["failed"].append(f"{condition}: candidate target_rank disagrees with metric path "
                                    f"{bad_target[:5]}")
        entry["candidates"]["order_violations"] = len(bad_order)
        entry["candidates"]["target_rank_mismatches"] = len(bad_target)
        report["conditions"][condition] = entry

    if identity.get("blas_threads_pinned") is None:
        report["failed"].append("identity records no blas_threads_pinned")
    capability = identity.get("blas_pin_capability")
    if not isinstance(capability, dict) or not capability.get("verifiable"):
        report["failed"].append(f"identity does not record a verifiable BLAS pin: {capability!r}")
    from workbench.backend.composition import IDENTITY_V2_INPUTS, measurement_identity

    summary_path = (Path(config.WORKBENCH_ARTIFACTS_ROOT) / "composition" / "summary"
                    / f"{args.model}_{args.checkpoint}_{args.run_id}" / "full.json")
    if summary_path.is_file():
        facts = measurement_identity(json.loads(summary_path.read_text()), identity)
        report["identity"].update({
            "measurement_group_id": facts.get("group_id"),
            "identity_schema_version_computed": facts.get("schema_version"),
            "identity_confidence": facts.get("confidence"),
            "identity_missing_inputs": facts.get("missing_identity_inputs"),
        })
        missing_v2 = [name for name in IDENTITY_V2_INPUTS if not report["identity"].get(name)
                      and identity.get(name) is None]
        if facts.get("schema_version") != 2:
            report["failed"].append(f"measurement identity is not schema v2: {facts.get('schema_version')!r}")
        if facts.get("missing_identity_inputs"):
            report["failed"].append(f"measurement identity is missing inputs: "
                                    f"{facts['missing_identity_inputs']}")
    else:
        report["failed"].append(f"summary not written: {summary_path}")
    report["verdict"] = "PASS" if not report["failed"] else "FAILED_INVARIANT"
    print(json.dumps(report, indent=1, default=str))
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
