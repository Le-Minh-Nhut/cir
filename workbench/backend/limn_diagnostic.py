"""Diagnostic evidence for LIMN's two ambiguous CIRCUS queries, served read-only.

The two queries `7:3038` and `7:3790` cannot be routed canonically: their reference image and
normalized caption pair are filed under BOTH `shirt` and `toptee`, so the adapter refuses instead of
guessing and the canonical run stays at 6001/6003. The evaluator was asked to show both scopes
anyway, as a diagnostic comparison.

This module is the boundary between that diagnostic and everything a reader could mistake for the
canonical metric, so it enforces three rules mechanically:

* the canonical status is returned in the same payload as the comparison, with the two unscored
  qids, so no view can render the diagnostic without it;
* each scope's evidence is refused unless its own file is present AND its `canonical_status`
  agrees with the record on disk (no scope can be silently dropped from the comparison);
* a scope's rank is only ever read from the file belonging to that scope. Nothing here chooses a
  scope, averages them, or folds either into a metric.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workbench.backend.operator_config import resolve_config

#: Where the dual-scope evidence lives. A fixed repo-relative path (never request-derived), so this
#: endpoint cannot be pointed at another file by its caller.
EVIDENCE_RELATIVE = ("workbench", "docs", "reviews", "20261109-native15-composition-limn")

SCOPES = ("shirt", "toptee")
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
QUERIES = ("7:3038", "7:3790")

#: The scratch query images the diagnostic staged (black images). Refused for display: they are
#: evaluator INPUTS, not dataset images, and rendering one beside a real candidate would imply a
#: corpus identity it does not have.
STAGED_PREFIX = "queries"


class DiagnosticEvidenceError(RuntimeError):
    """The diagnostic artifact is absent or inconsistent; report it rather than serve partial data."""


def evidence_root(config: Any | None = None) -> Path:
    config = config or resolve_config()
    return Path(config.CIR_REPO_ROOT).joinpath(*EVIDENCE_RELATIVE)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DiagnosticEvidenceError(f"unreadable diagnostic evidence {path.name}: {error}") from error


def load_diagnostic(config: Any | None = None) -> dict[str, Any]:
    """The full diagnostic payload, or a named absence.

    The canonical status is never inferred from the diagnostic: it is read from the record the run
    wrote, and the caller is told the diagnostic did not resolve it.
    """
    root = evidence_root(config)
    path = root / "limn_dual_scope.json"
    if not path.is_file():
        return {"available": False,
                "status": "DIAGNOSTIC_NOT_PUBLISHED",
                "reason": f"dual-scope evidence not published at {path}",
                "canonical_status": None,
                "queries": {}}
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("artifact_type") != "limn_dual_scope_diagnostic":
        raise DiagnosticEvidenceError("dual-scope evidence has an unexpected artifact type")

    canonical = payload.get("canonical_status") or {}
    if set(canonical.get("unscored_qids") or ()) != set(QUERIES):
        raise DiagnosticEvidenceError(
            "dual-scope evidence does not declare the two canonical unscored queries; refusing to "
            "serve it beside a canonical count it may contradict")

    # Evidence that cannot be opened is not evidence. The summary can be complete while every
    # per-scope export is gone, and the panel would then render twelve ranks whose every "check this
    # cell" link is dead -- while the notice advertises six per-scope files. Stat the exact expected
    # names (not a bare glob, which silently accepts a partially-deleted set) before serving.
    expected_exports = [path.parent / f"limn_dual_scope.{scope}.{condition}.jsonl"
                        for scope in SCOPES for condition in CONDITIONS]
    missing_exports = [export.name for export in expected_exports if not export.is_file()]
    if missing_exports:
        raise DiagnosticEvidenceError(
            f"per-scope export(s) missing: {missing_exports}; a comparison whose evidence files are "
            "absent cannot be checked and must not be served as complete")

    cohort = payload.get("canonical_cohort")
    if not isinstance(cohort, dict) or not cohort.get("subset") or not cohort.get("query_count"):
        raise DiagnosticEvidenceError(
            "dual-scope evidence does not declare which cohort its canonical counts belong to; a "
            "count that does not name its population reads as belonging to whatever page shows it")

    queries: dict[str, Any] = {}
    for qid in QUERIES:
        record = (payload.get("queries") or {}).get(qid)
        if not isinstance(record, dict):
            raise DiagnosticEvidenceError(f"diagnostic evidence is missing query {qid}")
        scopes: dict[str, Any] = {}
        for scope in SCOPES:
            conditions = (record.get("scopes") or {}).get(scope)
            if not isinstance(conditions, dict):
                raise DiagnosticEvidenceError(
                    f"diagnostic evidence is missing scope {scope} for {qid}; a one-scope "
                    "comparison would read as a resolved category")
            scopes[scope] = {
                condition: {
                    "target_rank": conditions[condition]["target_rank"],
                    "target_score": conditions[condition]["target_score"],
                    "score_at_rank_1": conditions[condition].get("score_at_rank_1"),
                    "target_margin_vs_rank1": conditions[condition].get("target_margin_vs_rank1"),
                    "gallery_size_scored": conditions[condition].get("gallery_size_scored"),
                    "model_text": conditions[condition].get("model_text"),
                    "export": Path(str(conditions[condition].get("top100_export") or "")).name or None,
                }
                for condition in CONDITIONS if condition in conditions
            }
            missing = [condition for condition in CONDITIONS if condition not in scopes[scope]]
            if missing:
                raise DiagnosticEvidenceError(
                    f"{qid}/{scope} is missing condition(s) {missing}; a partial comparison would "
                    "understate the scope's behaviour")
        queries[qid] = {
            "qid": qid,
            "release": record.get("release"),
            "target_gallery_id": record.get("target_gallery_id"),
            "target_position_in_gallery": record.get("target_position_in_gallery"),
            "scopes": scopes,
        }

    return {
        "available": True,
        "status": "DIAGNOSTIC_COMPLETE",
        "canonical_status": {
            "status": canonical.get("status"),
            "scored_count": canonical.get("scored_count"),
            "unscored_count": canonical.get("unscored_count"),
            "unscored_qids": list(canonical.get("unscored_qids") or ()),
            "note": canonical.get("note"),
        },
        "label": "ambiguous category — diagnostic comparison",
        # WHICH population the canonical counts count. The panel is rendered on every subset page,
        # and the `sf` page's row shows 4,000/4,069 while `validated` shows 581/586: a bare
        # "6,001/6,003" beside those reads as this page's number. Naming the cohort is what lets the
        # UI say "full cohort ... this page is the <subset> subset" instead of implying a per-page
        # figure it never measured. Read from the record, never inferred here.
        "canonical_cohort": {
            "subset": cohort.get("subset"),
            "query_count": cohort.get("query_count"),
            "query_count_per_subset": dict(cohort.get("query_count_per_subset") or {}),
            "scored_count": canonical.get("scored_count"),
            "unscored_count": canonical.get("unscored_count"),
            "multimodal_recall_at_10": cohort.get("multimodal_recall_at_10"),
            "source": cohort.get("source"),
        },
        "reason": ("each query's reference image and normalized caption pair appear in BOTH the "
                   "shirt and toptee FashionIQ validation captions, so the release determines no "
                   "single checkpoint; both scopes are scored separately and NEITHER replaces the "
                   "canonical unscored state"),
        "gallery": payload.get("gallery"),
        "conditions": list(CONDITIONS),
        "scopes": list(SCOPES),
        "checkpoints": payload.get("checkpoints"),
        "release_provenance": payload.get("release_provenance"),
        "rank_convention_crosscheck": {
            key: value for key, value in (payload.get("rank_convention_crosscheck") or {}).items()
            if key in ("checked", "all_agree", "method")},
        "runtime": payload.get("runtime"),
        "queries": queries,
        "evidence_files": {"summary": str(path),
                           # The full validated set -- never a glob, so this can never be empty or
                           # short while `available` is True.
                           "per_scope": [export.name for export in expected_exports]},
    }


def load_candidates(qid: str, scope: str, condition: str, depth: int = 100,
                    config: Any | None = None) -> dict[str, Any]:
    """Ordered candidates for one query/scope/condition, from that scope's own export.

    ``depth`` bounds how many stored rows are returned; the target's rank is the rank recorded for
    that scope and is never recomputed from the returned window.
    """
    if qid not in QUERIES:
        raise DiagnosticEvidenceError(f"{qid} is not one of the ambiguous queries")
    if scope not in SCOPES:
        raise DiagnosticEvidenceError(f"{scope} is not a diagnostic scope")
    if condition not in CONDITIONS:
        raise DiagnosticEvidenceError(f"{condition} is not a supported condition")
    root = evidence_root(config)
    path = root / f"limn_dual_scope.{scope}.{condition}.jsonl"
    if not path.is_file():
        raise DiagnosticEvidenceError(f"no per-scope export at {path.name}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    record = next((row for row in rows if row.get("qid") == qid), None)
    if record is None:
        raise DiagnosticEvidenceError(f"{path.name} has no row for {qid}")
    limit = max(1, min(int(depth), 200))
    return {
        "qid": qid, "scope": scope, "condition": condition,
        "target_rank": record["target_rank"],
        "target_score": record["target_score"],
        "checkpoint_sha256": record.get("checkpoint_sha256"),
        "gallery_size": record.get("gallery_size"),
        "model_text": record.get("model_text"),
        "depth_stored": record.get("depth"),
        "candidates": record["candidates"][:limit],
        "staged_query_image_is_evaluator_input": STAGED_PREFIX in str(record.get("query_image_path") or ""),
        "canonical_status_note": ("diagnostic ranking for a query the canonical evaluation leaves "
                                 "unscored; not a canonical per-query row"),
    }
