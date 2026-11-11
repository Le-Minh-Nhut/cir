"""Publish the derived LIMN subset runs through the runner's own summary/artifact layout.

The derivation is a SELECTION + RE-AGGREGATION of the parent's per-qid exact ranks: no image is
decoded, no forward pass runs, no score is recomputed, no GPU is used. What this script writes is the
same shape the runner writes, so the existing API readers pick it up without a new route:

  workbench/artifacts/composition/summary/<model>_<checkpoint>_<derived_run_id>/<subset>.json
  workbench/artifacts/composition/results/<protocol>/<model>/<checkpoint>/<derived_run_id>/<subset>/<condition>.jsonl
  workbench/artifacts/composition/runs/<derived_run_id>/ranks/<model>_<checkpoint>.json

Every derived artifact states its parentage inside its own manifest (`derived_from`), so a reader
never has to open a review file to learn that this is not a fresh measurement. The parent's own files
are only READ.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))

from workbench.backend.operator_config import resolve_config  # noqa: E402
from workbench.composition.data import (  # noqa: E402
    EXPECTED_QUERY_COUNTS, assign_garment_categories, load_release, select_queries)
from workbench.composition.metrics import compose  # noqa: E402

PARENT = "9e4918a6b5c243412cbc58c9"
MODEL = "limn"
CHECKPOINT = "base_iter0_all_categories"
PROTOCOL = "circus_fashioniq_release_v1"
SCHEMA = "circus-derived-subset/v1"
#: Written where an unscored qid has no parent reason, so a reader can tell "the parent recorded
#: nothing" apart from "the reason was not carried".
PARENT_RECORDED_NO_REASON = "parent_recorded_no_reason"
ARTIFACTS = REPO / "workbench/artifacts"
COMPOSITION = ARTIFACTS / "composition"
EVIDENCE = REPO / "workbench/docs/reviews/20261111-master-audit/raw/m03"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    tmp.replace(path)


def main() -> int:
    config = resolve_config()
    data = assign_garment_categories(
        load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT)), Path(config.FASHIONIQ_ROOT))
    rank_path = COMPOSITION / "runs" / PARENT / "ranks" / f"{MODEL}_{CHECKPOINT}.json"
    ranks_payload = json.loads(rank_path.read_text(encoding="utf-8"))
    parent_ranks = ranks_payload["ranks"]
    parent_summary_path = COMPOSITION / "summary" / f"{MODEL}_{CHECKPOINT}_{PARENT}" / "full.json"
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    parent_row_path = (COMPOSITION / "results" / PROTOCOL / MODEL / CHECKPOINT / PARENT / "full")
    parent_rows = {condition: [json.loads(line) for line in
                               (parent_row_path / f"{condition}.jsonl").read_text(
                                   encoding="utf-8").splitlines() if line.strip()]
                   for condition in parent_ranks}

    all_parent_qids = set(parent_ranks["multimodal"])
    published = {}
    for subset in ("sf", "validated"):
        declared = {query.qid for query in select_queries(data, subset)}
        membership_sha256 = digest_bytes("\n".join(sorted(declared)).encode())
        identity = {"parent_run_id": PARENT,
                    "parent_rank_artifact_sha256": sha256_file(rank_path),
                    "subset": subset, "membership_sha256": membership_sha256, "schema": SCHEMA}
        derived_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]

        subset_ranks = {condition: {qid: rank for qid, rank in parent_ranks[condition].items()
                                    if qid in declared}
                        for condition in parent_ranks}
        result = compose(subset_ranks, query_count=len(declared))
        metrics = asdict(result)

        # The failure map is carried BY MEMBERSHIP: a qid outside this subset must not appear at
        # all, and every unscored qid must say WHY it is unscored. Writing `{condition: {}}` made a
        # reader report `failure=null, unscored_kind=null`, which is indistinguishable from "the
        # pipeline recorded nothing" -- so an unscored qid with no parent reason gets an explicit
        # sentinel instead of null.
        parent_failures = ranks_payload.get("query_failures") or {}
        subset_failures = {}
        for condition in parent_ranks:
            reasons = parent_failures.get(condition) or {}
            carried = {}
            for qid, rank in subset_ranks[condition].items():
                reason = reasons.get(qid)
                if reason is None and rank is None:
                    reason = PARENT_RECORDED_NO_REASON
                if reason is not None:
                    carried[qid] = reason
            unscored_without_reason = [qid for qid, rank in subset_ranks[condition].items()
                                       if rank is None and qid not in carried]
            outside = set(carried) - set(subset_ranks[condition])
            if unscored_without_reason or outside:
                raise SystemExit(
                    f"REFUSING {subset}/{condition}: unscored qids without a carried reason="
                    f"{unscored_without_reason}, failure qids outside the membership={sorted(outside)}")
            subset_failures[condition] = carried
        parent_topk = dict(parent_summary.get("topk_export") or {})

        parent_scope_resolution = dict(parent_summary.get("adapter_query_scope_resolution") or {})
        # `unscoped_qids` = un-SCOPED (no checkpoint could be chosen). Every such qid in this subset's
        # membership is also a query this subset cannot score, so `still_unscoped` and the per-condition
        # `unscored_count` must agree; the invariant is asserted below, not assumed.
        parent_unscoped = [str(qid) for qid in (parent_scope_resolution.get("unscoped_qids") or ())]
        subset_unscoped = [qid for qid in parent_unscoped if qid in declared]

        # THIS subset's own completeness, computed once and used both here and in the summary.
        own_unscored = len(declared) - metrics["multimodal"]["scored_count"]
        own_coverage = metrics["multimodal"]["coverage"]

        derived_from = {
            "schema": SCHEMA,
            "derivation": ("select the parent's per-qid exact ranks for the declared subset "
                           "membership and re-aggregate with workbench.composition.metrics.compose; "
                           "no image decoded, no forward pass, no score recomputed"),
            "derived_run_id": derived_id,
            "parent_run_id": PARENT,
            "parent_rank_artifact": str(rank_path.relative_to(REPO)),
            "parent_rank_artifact_sha256": identity["parent_rank_artifact_sha256"],
            "parent_summary": str(parent_summary_path.relative_to(REPO)),
            "parent_summary_sha256": sha256_file(parent_summary_path),
            # The PARENT's own candidate export, so a reader can say whether a parent window
            # exists at all before it looks for one, and label it as the parent's.
            "parent_topk_export": {
                "requested": bool(parent_topk.get("requested")),
                "status": parent_topk.get("status"),
                "depth": parent_topk.get("depth"),
                "conditions": list(parent_topk.get("conditions") or ()),
                "partial": bool(parent_topk.get("partial")),
            },
            "parent_topk_note": ("derived runs store no candidates of their own; the candidate "
                                 "window for a derived query is the PARENT run's export, served "
                                 "only when this lineage verifies"),
            "subset": subset,
            "membership_definition": (f"workbench.composition.data.select_queries(subset={subset!r}) "
                                      "over the pinned CIRCUS release membership columns"),
            "membership_sha256": membership_sha256,
            "membership_count": len(declared),
            "declared_query_count": EXPECTED_QUERY_COUNTS[subset],
            "coverage": metrics["multimodal"]["coverage"],
            "scored_count": metrics["multimodal"]["scored_count"],
            "denominator": metrics["multimodal"]["scored_count"],
            "denominator_note": ("metrics are means over the SCORED queries of this subset; the "
                                 "declared membership is the larger number in declared_query_count"),
            "gpu_used": False,
            "re_encoded": False,
            "rescored": False,
            # The parent's partiality is recorded as the PARENT's, never reused as this run's status:
            # a subset can be complete (validated 586/586) while its parent is not (6001/6003).
            "parent_execution_status": parent_summary.get("execution_status"),
            "parent_coverage": ((parent_summary.get("metrics") or {}).get("multimodal") or {}).get("coverage"),
            "parent_unscoped_qids": parent_unscoped,
            "parent_unscoped_count": len(parent_unscoped),
            "parent_unscoped_qids_in_this_subset": subset_unscoped,
            # Aliases kept so the existing regression (`tests/test_derived_subset_status.py`, written
            # before the spelling was corrected) keeps asserting the same facts; that file is not
            # editable while the release sweep measures the tree.
            "parent_unscored_qids": parent_unscoped,
            "parent_unscored_qids_in_this_subset": subset_unscoped,
            "parent_unscored_count": parent_summary.get("unscored_category_query_count"),
            # THIS subset's own completeness. Deliberately NOT a new `execution_status` value: that
            # enum is read as "did this run finish" by scope.ts:151, OverviewPage.tsx:251,
            # EvaluationPage.tsx:259 and documented at backend/composition.py:189, so widening it for
            # one label moves a contract other code depends on. `derived_from` is already the field
            # that means "not a fresh measurement", and unknown keys there are tolerated.
            "own_coverage": own_coverage,
            "own_scored_count": metrics["multimodal"]["scored_count"],
            "own_unscored_count": own_unscored,
            "own_complete": own_unscored == 0,
            "own_status_label": ("DERIVED_COMPLETE" if own_unscored == 0 else "DERIVED_PARTIAL"),
            "own_status_note": ("read this, not execution_status: execution_status is inherited from "
                                "the parent so it stays inside COMPLETED|PARTIAL|REUSED|FAILED"),
        }

        # Status is computed from THIS subset's own numbers, per condition, never inherited.
        subset_condition_status = {}
        for condition, label in (("multimodal", "multimodal"), ("text_only_black_image", "text_only"),
                                ("image_only_empty_text", "image_only")):
            bucket = metrics[label]
            complete = bucket["scored_count"] == bucket["query_count"]
            subset_condition_status[condition] = {
                "status": "COMPLETED" if complete else "PARTIAL",
                "unscored_count": bucket["query_count"] - bucket["scored_count"],
                "category_scope_failure_count": 0,
                "fallback_used": False,
            }
        subset_complete = all(row["status"] == "COMPLETED" for row in subset_condition_status.values())

        # The parent's scope blocks describe the PARENT's query set. Inheriting them verbatim made a
        # `DERIVED_COMPLETE` row assert "2 unscoped queries" beside "unscored_count: 0" for two qids
        # that are provably outside this subset's membership. They are recomputed against THIS
        # subset's membership, and the parent's own values are kept under `parent_*` names so the
        # pathology stays visible without contradicting the row it is printed on.
        parent_scope_failures = dict(parent_summary.get("category_scope_failures") or {})
        parent_ambiguous = set((parent_summary.get("category_resolution") or {}).get("ambiguous_qids")
                               or ())
        subset_ambiguous = sorted(qid for qid in parent_ambiguous if qid in declared)
        subset_loader_scoped = len(declared - parent_ambiguous)
        subset_category_resolution = dict(parent_summary.get("category_resolution") or {})
        subset_category_resolution.update({
            "considered_count": len(declared),
            "resolved_count": subset_loader_scoped,
            "ambiguous_count": len(subset_ambiguous),
            "ambiguous_qids": subset_ambiguous,
            "ambiguous_examples": [row for row in
                                   (subset_category_resolution.get("ambiguous_examples") or ())
                                   if row.get("qid") in set(subset_ambiguous)],
            "unresolved_count": len(subset_unscoped),
            "unresolved_qids": subset_unscoped,
            # Every qid the adapter could not route is also a query this subset cannot score, so the
            # invariant `still_unscoped == unscored_count` must hold per condition.
            "invariant": "still_unscoped == condition_status[*].unscored_count",
            "scope_note": ("counts recomputed over THIS subset's membership; the parent's own "
                           "counts are under parent_category_resolution"),
            "parent_category_resolution": parent_summary.get("category_resolution"),
        })

        # The row must not undercut itself: what it says about routing has to agree with what it says
        # about scoring, per condition.
        for condition, row in subset_condition_status.items():
            if row["unscored_count"] != len(subset_unscoped):
                raise SystemExit(
                    f"REFUSING {subset}/{condition}: condition_status.unscored_count="
                    f"{row['unscored_count']} but adapter routing says {len(subset_unscoped)} "
                    f"unscoped qid(s) {subset_unscoped}; one of the two is wrong")

        summary = dict(parent_summary)
        summary.update({
            "run_id": derived_id,
            "subset": subset,
            "query_count": metrics["multimodal"]["query_count"],
            # Inherited verbatim from the parent, so the value stays inside the enum the rest of the
            # system understands. This subset's own completeness is in `derived_from.own_*`; a reader
            # must never infer "this subset has unscored queries" from this field alone.
            "execution_status": parent_summary.get("execution_status"),
            "condition_status": subset_condition_status,
            "derived_complete": subset_complete,
            "metrics": metrics,
            "derived_from": derived_from,
            "category_resolution": subset_category_resolution,
            "unresolved_category_query_count": len(subset_unscoped),
            "category_scope_failures": {condition: len(subset_unscoped)
                                        for condition in parent_scope_failures},
            "adapter_query_scope_resolution": {
                "resolver": parent_scope_resolution.get("resolver"),
                "loader_scoped": subset_loader_scoped,
                "adapter_routed": len(declared) - len(subset_unscoped),
                "still_unscoped": len(subset_unscoped),
                "unscoped_qids": subset_unscoped,
                "scope_note": ("counts recomputed over THIS subset's membership"),
                "parent_adapter_query_scope_resolution": parent_scope_resolution,
            },
            "parent_category_scope_failures": parent_scope_failures,
            "smoke": {"limited": False, "query_limit": None, "label": None},
            "feature_storage": {"format": "not stored (derived: parent feature cache reused)",
                                "arrays_removed_after_rank": True,
                                "projected_bytes": {}},
            "reused_query_result_count": 0,
            "topk_export": {"requested": False, "depth": None, "conditions": [],
                            "partial": False, "status": "NOT_EXPORTED_FOR_DERIVED_RUN",
                            "reason": "a derived run re-aggregates the parent's ranks; the parent's "
                                      "own candidate export is the window evidence"},
        })
        summary["resume_key"] = derived_id
        summary["query_count_per_subset"] = dict(parent_summary.get("query_count_per_subset") or {})

        summary_path = (COMPOSITION / "summary" / f"{MODEL}_{CHECKPOINT}_{derived_id}" /
                        f"{subset}.json")
        atomic_json(summary_path, summary)

        for condition in parent_ranks:
            rows = [row for row in parent_rows[condition] if row.get("qid") in declared]
            atomic_jsonl(COMPOSITION / "results" / PROTOCOL / MODEL / CHECKPOINT / derived_id /
                         subset / f"{condition}.jsonl", rows)

        rank_record = {
            "attempt": derived_id,
            "condition_status": subset_condition_status,
            "fingerprint": derived_id,
            "identity": dict(ranks_payload.get("identity") or {}, derived_from=derived_from),
            "query_failures": subset_failures,
            "ranks": subset_ranks,
            "derived_from": derived_from,
        }
        atomic_json(COMPOSITION / "runs" / derived_id / "ranks" / f"{MODEL}_{CHECKPOINT}.json",
                    rank_record)

        published[subset] = {
            "derived_run_id": derived_id,
            "summary_path": str(summary_path.relative_to(REPO)),
            "metrics_denominator": derived_from["denominator"],
            "coverage": derived_from["coverage"],
            "multimodal_recall_at_10": metrics["multimodal"]["recall_at_10"],
            "membership_count": len(declared),
            "membership_sha256": membership_sha256,
            "parent_run_id": PARENT,
            "per_query_rows": {condition: len([row for row in parent_rows[condition]
                                               if row.get("qid") in declared])
                               for condition in parent_ranks},
        }

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "published_derived_runs.json").write_text(
        json.dumps({"parent_run_id": PARENT, "parent_bytes_touched": False,
                    "published": published}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(published, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
