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

One validated record, shared by every reader
--------------------------------------------
`load_diagnostic` (summary) and `load_candidates` (one window) used to parse the files
independently, so the two endpoints could disagree about the same combination and neither noticed a
summary field that contradicted the export it cites. Both now read ONE `_record`, built by
`_load_record` and cached by the content digest of the seven files it consumes:

* the manifest names every file and its sha256, so a response says exactly which bytes it describes;
* every (qid, scope, condition) becomes a validated `combination`: the export row must agree with
  the summary on rank, score, checkpoint digest, gallery position and target asset, its candidate
  positions must be unique, its ranks must be the ordered window it claims, and its depth must cover
  the rows it stores;
* the target's data identifier is taken from THAT query's own record at serialize time. The producer
  that wrote the artifact leaked a loop variable here (`t2_dual_scope.py:273` read the loop's last
  query for every row), so all twelve rows recorded `7:42304` -- the positive of `7:3790` -- even
  for `7:3038`. Reading the value from the query record is the fix at the layer that serves it: a
  later producer cannot re-introduce a shared variable into this reader;
* `complete` is True only when no defect was found. A defect never deletes the evidence it
  describes: the ranks, scores and candidate windows stay served, and the specific reason is named.
  "No data" and "data that contradicts itself" are different answers and are reported differently.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
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

#: Identity of the validation contract below. A response names the contract it was checked under, so
#: a reader can tell "these bytes passed v2" from "nobody checked these bytes".
VALIDATION_STANDARD = "limn-diagnostic-evidence/v2"

#: Statuses. `COMPLETE` is the ONLY one that means "every combination validated".
DIAGNOSTIC_COMPLETE = "DIAGNOSTIC_COMPLETE"
DIAGNOSTIC_EVIDENCE_DEFECT = "DIAGNOSTIC_EVIDENCE_DEFECT"
DIAGNOSTIC_NOT_PUBLISHED = "DIAGNOSTIC_NOT_PUBLISHED"

#: Upper bound on a served candidate window. The export stores 100; the bound exists so a caller
#: cannot ask this endpoint to materialise the whole 74,381-row gallery.
MAX_DEPTH = 200


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _defect(code: str, detail: str, *, qid: str | None = None, scope: str | None = None,
            condition: str | None = None) -> dict[str, Any]:
    return {"code": code, "qid": qid, "scope": scope, "condition": condition, "detail": detail}


def _evidence_files(root: Path) -> tuple[list[Path], list[str]]:
    """The exact seven files the record consumes, and the names that are absent."""
    expected = [root / "limn_dual_scope.json"] + [
        root / f"limn_dual_scope.{scope}.{condition}.jsonl"
        for scope in SCOPES for condition in CONDITIONS]
    return expected, [path.name for path in expected if not path.is_file()]


_RECORD_CACHE: dict[str, dict[str, Any]] = {}


def _load_record(config: Any | None = None) -> dict[str, Any]:
    """The one validated record every reader uses, cached by the content identity of its files.

    The cache key is a digest over (name, sha256) of every consumed file, so an edited, truncated or
    swapped artifact is a different key: a stale validation can never be served for new bytes.
    """
    root = evidence_root(config)
    expected, missing = _evidence_files(root)
    if not (root / "limn_dual_scope.json").is_file():
        return {"available": False, "status": DIAGNOSTIC_NOT_PUBLISHED,
                "reason": f"dual-scope evidence not published at {root / 'limn_dual_scope.json'}",
                "defects": [], "manifest": None, "queries": {}, "combinations": {}, "payload": None}
    present = [path for path in expected if path.is_file()]
    manifest_files = [{"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
                      for path in present]
    digest = hashlib.sha256(json.dumps(manifest_files, sort_keys=True).encode("utf-8")).hexdigest()
    cached = _RECORD_CACHE.get(digest)
    if cached is not None:
        return cached

    payload = _read_json(root / "limn_dual_scope.json")
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
    if missing:
        raise DiagnosticEvidenceError(
            f"per-scope export(s) missing: {missing}; a comparison whose evidence files are "
            "absent cannot be checked and must not be served as complete")

    cohort = payload.get("canonical_cohort")
    if not isinstance(cohort, dict) or not cohort.get("subset") or not cohort.get("query_count"):
        raise DiagnosticEvidenceError(
            "dual-scope evidence does not declare which cohort its canonical counts belong to; a "
            "count that does not name its population reads as belonging to whatever page shows it")

    defects: list[dict[str, Any]] = []
    checks = payload.get("checkpoints") or {}
    # Load each export once: the summary and the window endpoint must describe the same bytes, and
    # re-reading per combination is how they came to disagree.
    exports: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for scope in SCOPES:
        for condition in CONDITIONS:
            path = root / f"limn_dual_scope.{scope}.{condition}.jsonl"
            rows = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError as error:
                    raise DiagnosticEvidenceError(
                        f"{path.name}:{line_number} is not valid JSON: {error}") from error
            exports[(scope, condition)] = rows
    # A second row for the same qid is not a longer export, it is two claims about one query. The
    # reader below takes the FIRST row by qid, so without this the second row is silently dropped.
    for (scope, condition), rows in exports.items():
        seen: dict[str, int] = {}
        for row in rows:
            row_qid = str(row.get("qid"))
            seen[row_qid] = seen.get(row_qid, 0) + 1
        for row_qid, count in sorted(seen.items()):
            if count > 1:
                defects.append(_defect(
                    "export_duplicate_qid",
                    f"limn_dual_scope.{scope}.{condition}.jsonl stores {count} rows for {row_qid}; "
                    "only one window per query can be the recorded evidence",
                    qid=row_qid, scope=scope, condition=condition))

    queries: dict[str, Any] = {}
    combinations: dict[tuple[str, str, str], dict[str, Any]] = {}
    for qid in QUERIES:
        record = (payload.get("queries") or {}).get(qid)
        if not isinstance(record, dict):
            raise DiagnosticEvidenceError(f"diagnostic evidence is missing query {qid}")
        release = record.get("release") or {}
        # The target's identity belongs to THIS query, read from THIS query's record. Never from a
        # value shared with another row (see the module docstring: the producer leaked one).
        own_positives = [str(value) for value in (release.get("positive_ids") or ())]
        declared_asset = record.get("target_gallery_id")
        declared_did = record.get("target_did")
        target_did = own_positives[0] if own_positives else declared_did
        if own_positives and declared_did and str(declared_did) not in own_positives:
            defects.append(_defect(
                "target_did_not_own_query",
                f"recorded target_did {declared_did} is not a positive id of {qid} "
                f"(own positives: {own_positives}); serving the query's own id instead", qid=qid))
        if not own_positives:
            defects.append(_defect(
                "target_did_missing",
                f"{qid} states no positive id, so its target cannot be identified", qid=qid))

        scopes: dict[str, Any] = {}
        for scope in SCOPES:
            conditions = (record.get("scopes") or {}).get(scope)
            if not isinstance(conditions, dict):
                raise DiagnosticEvidenceError(
                    f"diagnostic evidence is missing scope {scope} for {qid}; a one-scope "
                    "comparison would read as a resolved category")
            readings: dict[str, Any] = {}
            for condition in CONDITIONS:
                if condition not in conditions:
                    continue
                declared = conditions[condition] or {}
                window = _validate_combination(
                    qid=qid, scope=scope, condition=condition, declared=declared,
                    export=exports[(scope, condition)], root=root, checks=checks,
                    target_did=target_did, declared_asset=declared_asset, defects=defects)
                combinations[(qid, scope, condition)] = window
                readings[condition] = {
                    "target_rank": declared["target_rank"],
                    "target_score": declared["target_score"],
                    "score_at_rank_1": declared.get("score_at_rank_1"),
                    "target_margin_vs_rank1": declared.get("target_margin_vs_rank1"),
                    "gallery_size_scored": declared.get("gallery_size_scored"),
                    "model_text": declared.get("model_text"),
                    "export": Path(str(declared.get("top100_export") or "")).name or None,
                    # Identity of the row the rank came from, so a cell can be checked against the
                    # same validated combination the window endpoint serves.
                    "target_did": target_did,
                    "label": f"diagnostic — {scope} scope, {condition} (not canonical)",
                }
            missing_conditions = [condition for condition in CONDITIONS
                                  if condition not in readings]
            if missing_conditions:
                raise DiagnosticEvidenceError(
                    f"{qid}/{scope} is missing condition(s) {missing_conditions}; a partial "
                    "comparison would understate the scope's behaviour")
            scopes[scope] = readings
        queries[qid] = {
            "qid": qid,
            "label": f"diagnostic for ambiguous query {qid} (not canonical)",
            "release": record.get("release"),
            "target_did": target_did,
            "declared_target_did": declared_did,
            "target_gallery_id": declared_asset,
            "target_position_in_gallery": record.get("target_position_in_gallery"),
            "scopes": scopes,
        }

    record_out = {
        "available": True,
        "status": DIAGNOSTIC_COMPLETE if not defects else DIAGNOSTIC_EVIDENCE_DEFECT,
        "complete": not defects,
        "standard": VALIDATION_STANDARD,
        "defects": defects,
        "manifest": {"standard": VALIDATION_STANDARD, "digest": digest,
                     "files": manifest_files,
                     "declared_root": str(root)},
        "payload": payload,
        "queries": queries,
        "combinations": combinations,
        "root": root,
    }
    _RECORD_CACHE.clear()  # one live record per process is enough; the key is the content identity
    _RECORD_CACHE[digest] = record_out
    return record_out


def _validate_combination(*, qid: str, scope: str, condition: str, declared: dict[str, Any],
                          export: list[dict[str, Any]], root: Path, checks: dict[str, Any],
                          target_did: str | None, declared_asset: str | None,
                          defects: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate ONE summary record against the export row that is supposed to back it."""
    where = {"qid": qid, "scope": scope, "condition": condition}
    path_name = f"limn_dual_scope.{scope}.{condition}.jsonl"
    row = next((item for item in export if item.get("qid") == qid), None)
    if row is None:
        raise DiagnosticEvidenceError(f"{path_name} has no row for {qid}")
    if row.get("scope") != scope or row.get("condition") != condition:
        defects.append(_defect(
            "export_row_identity",
            f"{path_name} row for {qid} declares scope={row.get('scope')!r} "
            f"condition={row.get('condition')!r}", **where))

    candidates = row.get("candidates") or []
    depth_stored = row.get("depth")
    if not isinstance(depth_stored, int) or isinstance(depth_stored, bool) or depth_stored < 0:
        defects.append(_defect("export_depth_invalid",
                               f"{path_name} declares depth={depth_stored!r}", **where))
        depth_stored = len(candidates)
    elif depth_stored < len(candidates):
        defects.append(_defect(
            "export_depth_short",
            f"{path_name} stores {len(candidates)} candidates but declares depth={depth_stored}; "
            "a window longer than its own declared depth cannot be reproduced from it", **where))
    # The opposite direction is the truncation this whole file exists to catch: a row that declares
    # the full depth while storing fewer rows. The expected window is bounded by the gallery the row
    # itself states, so a genuinely short gallery is not reported as a truncation.
    gallery_size = row.get("gallery_size")
    expected_stored = depth_stored if not isinstance(gallery_size, int) else min(depth_stored, gallery_size)
    if candidates and len(candidates) < expected_stored:
        defects.append(_defect(
            "export_depth_truncated",
            f"{path_name} declares depth={depth_stored} over a gallery of {gallery_size!r} "
            f"(expected {expected_stored} rows) but stores {len(candidates)}; the missing window "
            "cannot be recovered from the file", **where))
    if not candidates:
        defects.append(_defect("export_candidates_empty",
                               f"{path_name} row for {qid} stores no candidates", **where))

    # Rank/score must be one ordered, duplicate-free window.
    ranks = [candidate.get("rank") for candidate in candidates]
    if ranks != list(range(1, len(candidates) + 1)):
        defects.append(_defect(
            "export_rank_not_sequential",
            f"{path_name} ranks are {ranks[:8]}{'…' if len(ranks) > 8 else ''}, not 1..{len(candidates)}",
            **where))
    scores = [candidate.get("score") for candidate in candidates]
    if any(a < b for a, b in zip(scores, scores[1:])):
        defects.append(_defect(
            "export_scores_not_descending",
            f"{path_name} candidate scores increase somewhere in the stored window", **where))
    ids = [candidate.get("gallery_id") for candidate in candidates]
    if len(set(ids)) != len(ids):
        duplicated = sorted({value for value in ids if ids.count(value) > 1})
        defects.append(_defect("export_duplicate_candidate",
                               f"{path_name} repeats candidate id(s) {duplicated[:5]}", **where))
    positions = [candidate.get("gallery_position") for candidate in candidates]
    if len(set(positions)) != len(positions):
        duplicated = sorted({value for value in positions if positions.count(value) > 1})
        defects.append(_defect("export_duplicate_position",
                               f"{path_name} repeats gallery position(s) {duplicated[:5]}", **where))

    # The summary and the export must state the same target.
    summary_rank = declared.get("target_rank")
    summary_score = declared.get("target_score")
    if not isinstance(summary_rank, int) or isinstance(summary_rank, bool) or summary_rank < 1:
        defects.append(_defect("summary_rank_invalid",
                               f"summary target_rank={summary_rank!r}", **where))
    if summary_rank != row.get("target_rank"):
        defects.append(_defect(
            "rank_summary_vs_export",
            f"summary target_rank={summary_rank!r} but {path_name} states "
            f"{row.get('target_rank')!r}", **where))
    if summary_score != row.get("target_score"):
        defects.append(_defect(
            "score_summary_vs_export",
            f"summary target_score={summary_score!r} but {path_name} states "
            f"{row.get('target_score')!r}", **where))

    # Checkpoint identity: the rank is only meaningful for the checkpoint named beside it.
    summary_ckpt = str((checks.get(scope) or {}).get("sha256") or "")
    export_ckpt = str(row.get("checkpoint_sha256") or "")
    if not export_ckpt:
        defects.append(_defect("export_checkpoint_missing",
                               f"{path_name} states no checkpoint_sha256", **where))
    elif summary_ckpt and summary_ckpt != export_ckpt:
        defects.append(_defect(
            "checkpoint_summary_vs_export",
            f"summary checkpoint {summary_ckpt[:16]}… but {path_name} states {export_ckpt[:16]}…",
            **where))

    # The window's flagged target, when the window reaches it. A window that stops before the target
    # is not a defect -- it is reported as `target_in_window: False` so a view can say so.
    flagged = [candidate for candidate in candidates if candidate.get("is_target")]
    in_window = isinstance(summary_rank, int) and 1 <= summary_rank <= len(candidates)
    if len(flagged) > 1:
        defects.append(_defect(
            "export_multiple_targets",
            f"{path_name} flags {len(flagged)} candidates as the target", **where))
    if in_window and not flagged:
        defects.append(_defect(
            "export_target_not_flagged",
            f"target rank {summary_rank} is inside the stored window but no candidate is flagged "
            "is_target", **where))
    if flagged:
        target_row = flagged[0]
        if target_row.get("rank") != summary_rank:
            defects.append(_defect(
                "export_target_rank",
                f"flagged target sits at rank {target_row.get('rank')!r}, summary says {summary_rank!r}",
                **where))
        if target_row.get("score") != summary_score:
            defects.append(_defect(
                "export_target_score",
                f"flagged target score {target_row.get('score')!r}, summary says {summary_score!r}",
                **where))
        asset = Path(str(target_row.get("gallery_id") or "")).name
        if declared_asset and asset and asset != Path(str(declared_asset)).name:
            defects.append(_defect(
                "target_asset_mismatch",
                f"flagged target is {asset} but the query record names {Path(str(declared_asset)).name}",
                **where))
    else:
        target_row = None

    summary_position = declared.get("target_gallery_position")
    if target_row is not None and summary_position is not None:
        if summary_position != target_row.get("gallery_position"):
            defects.append(_defect(
                "target_position_mismatch",
                f"summary target_gallery_position={summary_position!r} but the flagged row is at "
                f"{target_row.get('gallery_position')!r}", **where))

    if target_did is None:
        defects.append(_defect("target_did_unavailable",
                               f"{qid} has no positive id to name as the target", **where))
    elif declared.get("target_did") not in (None, target_did):
        # This is R5's shape: the row recorded another query's positive id. The served value comes
        # from this query's own record, and the contradiction is named rather than repeated.
        defects.append(_defect(
            "target_did_not_own_query",
            f"recorded target_did {declared.get('target_did')} is not a positive id of {qid} "
            f"(own: {target_did}); serving the query's own id", **where))

    return {
        "qid": qid, "scope": scope, "condition": condition,
        "label": f"diagnostic — {scope} scope, {condition} (not canonical)",
        "root": root, "export_name": path_name, "row": row,
        "candidates": candidates, "depth_stored": depth_stored,
        "target_did": target_did,
        "target_asset": (Path(str(declared_asset)).name if declared_asset else None),
        "target": {
            "rank": summary_rank, "score": summary_score,
            "gallery_id": (target_row or {}).get("gallery_id") or declared_asset,
            "gallery_position": (target_row or {}).get("gallery_position", summary_position),
            "in_window": bool(flagged),
            "delta_to_rank_1": declared.get("target_margin_vs_rank1"),
        },
        "checkpoint_sha256": (export_ckpt or summary_ckpt or None),
        "gallery_size": row.get("gallery_size", declared.get("gallery_size_scored")),
        "model_text": row.get("model_text", declared.get("model_text")),
        "staged_query_image_is_evaluator_input": STAGED_PREFIX in str(row.get("query_image_path") or ""),
    }


def load_diagnostic(config: Any | None = None) -> dict[str, Any]:
    """The full diagnostic payload, or a named absence.

    The canonical status is never inferred from the diagnostic: it is read from the record the run
    wrote, and the caller is told the diagnostic did not resolve it.
    """
    record = _load_record(config)
    if not record["available"]:
        return {"available": False, "status": record["status"], "reason": record["reason"],
                "canonical_status": None, "queries": {}, "complete": False,
                "defects": [], "manifest": None}

    payload = record["payload"]
    canonical = payload.get("canonical_status") or {}
    cohort = payload.get("canonical_cohort") or {}
    root = record["root"]
    expected_exports = [f"limn_dual_scope.{scope}.{condition}.jsonl"
                        for scope in SCOPES for condition in CONDITIONS]
    return {
        "available": True,
        "status": record["status"],
        # `complete` is the ONLY field a view may read as "every combination validated". A response
        # that contradicts itself keeps its evidence and names its defects instead.
        "complete": record["complete"],
        "standard": record["standard"],
        "defects": record["defects"],
        "manifest": record["manifest"],
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
        "queries": record["queries"],
        "evidence_files": {"summary": str(root / "limn_dual_scope.json"),
                           # The full validated set -- never a glob, so this can never be empty or
                           # short while `available` is True.
                           "per_scope": expected_exports},
    }


@lru_cache(maxsize=8192)
def _resolvable(root: str, category: str, stem: str, suffix: str) -> bool:
    """Does the served image endpoint have this file? Same resolver, so the URL always works."""
    from workbench.backend.main import resolve_image

    return resolve_image(category, f"{stem}{suffix}", Path(root)) is not None


def _image_url(image_id: str | None, scope: str,
               fashioniq_root: Path) -> tuple[str | None, str | None]:
    """The one URL that actually resolves for this image, and the category it resolved under.

    Reuses the served image endpoint's own resolver rather than a second path convention: a URL this
    function returns is one the app can serve. Imported lazily because `main` imports this module.
    """
    if not image_id or fashioniq_root is None:
        return None, None
    path = Path(str(image_id))
    stem, suffix = path.stem, path.suffix or ".jpg"
    for category in (scope, "dress", "shirt", "toptee"):
        if _resolvable(str(fashioniq_root), category, stem, suffix):
            return f"/api/images/{category}/{stem}{suffix}", category
    return None, None


def load_candidates(qid: str, scope: str, condition: str, depth: int | None = None,
                    config: Any | None = None) -> dict[str, Any]:
    """Ordered candidates for one query/scope/condition, from that scope's own export.

    ``depth`` bounds how many stored rows are returned. It is validated, not clamped silently: an
    out-of-range or non-positive depth used to be absorbed into a 1-row window with nothing in the
    response saying so. The target's rank is the rank recorded for that scope and is never recomputed
    from the returned window.
    """
    if qid not in QUERIES:
        raise DiagnosticEvidenceError(f"{qid} is not one of the ambiguous queries")
    if scope not in SCOPES:
        raise DiagnosticEvidenceError(f"{scope} is not a diagnostic scope")
    if condition not in CONDITIONS:
        raise DiagnosticEvidenceError(f"{condition} is not a supported condition")
    if depth is not None:
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise DiagnosticEvidenceError(f"depth must be an integer, got {depth!r}")
        if depth < 1:
            raise DiagnosticEvidenceError(
                f"depth must be at least 1, got {depth}; a zero or negative depth cannot return a "
                "window and silently returning one row would misstate the request")
        if depth > MAX_DEPTH:
            raise DiagnosticEvidenceError(
                f"depth {depth} exceeds the {MAX_DEPTH}-row limit this endpoint serves")

    record = _load_record(config)
    if not record["available"]:
        raise DiagnosticEvidenceError(f"no per-scope export at {evidence_root(config)}")
    combination = record["combinations"].get((qid, scope, condition))
    if combination is None:
        raise DiagnosticEvidenceError(
            f"no per-scope export row for {qid}/{scope}/{condition}")

    stored = combination["depth_stored"]
    requested = stored if depth is None else depth
    # K comes from the actual export depth: what the file stores bounds what can be served.
    served = min(requested, stored, len(combination["candidates"]))
    window = combination["candidates"][:served]
    defects = [item for item in record["defects"]
               if item["qid"] in (None, qid)
               and item["scope"] in (None, scope)
               and item["condition"] in (None, condition)]

    # A config that declares no FashionIQ root cannot resolve an image; the window still serves, with
    # `image_url: None`, and no tile is invented.
    fashioniq_root = getattr(config or resolve_config(), "FASHIONIQ_ROOT", None)
    candidate_rows = []
    for candidate in window:
        image_id = candidate.get("gallery_id")
        url, category = _image_url(image_id, scope, fashioniq_root)
        candidate_rows.append({
            "rank": candidate.get("rank"),
            "gallery_position": candidate.get("gallery_position"),
            "gallery_id": image_id,
            "image_path": candidate.get("image_path"),
            "image_stem": Path(str(image_id or "")).stem,
            "image_url": url,
            "image_category": category,
            "score": candidate.get("score"),
            "is_target": bool(candidate.get("is_target")),
        })

    target = combination["target"]
    target_url, target_category = _image_url(target["gallery_id"], scope, fashioniq_root)
    # W-03: presence must be computed on the window THIS response serves, not on the whole stored
    # export. `combination["target"]["in_window"]` was decided against all 100 stored rows, so a
    # depth=10 request for a target at rank 43 answered `in_window: true` and a view printed "the
    # target is inside the 10 candidates shown" over ten rows that do not contain it.
    served_ranks = {row.get("rank") for row in window}
    target_rank = target["rank"]
    in_window = isinstance(target_rank, int) and not isinstance(target_rank, bool) \
        and target_rank in served_ranks
    return {
        "qid": qid, "scope": scope, "condition": condition,
        "label": combination["label"],
        # Request and response identity are the same triple, echoed from the validated record.
        "identity": {"qid": qid, "scope": scope, "condition": condition,
                     "standard": record["standard"],
                     "manifest_digest": (record["manifest"] or {}).get("digest")},
        "complete": record["complete"] and not defects,
        "defects": defects,
        "manifest": record["manifest"],
        "target_rank": target["rank"],
        "target_score": target["score"],
        # The query's own positive id, from its own record: the summary's identity field, so a caller
        # can check qid -> positive DID -> asset -> rank without opening the summary.
        "target_did": combination["target_did"],
        "target_asset": combination["target_asset"],
        # The target is ALWAYS returned separately from the window, because it frequently is not in
        # it: at depth 100 the rank-6891 target is not a candidate row, and a view that can only show
        # the target when it happens to be in the top K cannot show the finding.
        "target": {**target, "in_window": in_window,
                   "in_stored_window": bool(target.get("in_window")),
                   "window_served": served, "window_stored": stored,
                   "image_url": target_url, "image_category": target_category,
                   "image_stem": Path(str(target["gallery_id"] or "")).stem},
        "checkpoint_sha256": combination["checkpoint_sha256"],
        "gallery_size": combination["gallery_size"],
        "model_text": combination["model_text"],
        "depth_requested": requested,
        "depth_served": served,
        "depth_stored": stored,
        "depth_truncated": served < requested,
        "candidates": candidate_rows,
        "staged_query_image_is_evaluator_input": combination["staged_query_image_is_evaluator_input"],
        "canonical_status_note": ("diagnostic ranking for a query the canonical evaluation leaves "
                                  "unscored; not a canonical per-query row"),
    }
