from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, median
from typing import Any

from workbench.backend.errors import WorkbenchError
from workbench.backend.index import configured_database_path


def _database_path(path: Path | None) -> Path:
    path = path or configured_database_path()
    if not path.is_file():
        raise WorkbenchError("index_unavailable", "DuckDB index is unavailable. Rebuild it from canonical result JSON.", {"database_path": str(path)})
    return path


def _connect(database_path: Path | None = None):
    import duckdb

    return duckdb.connect(str(_database_path(database_path)), read_only=True)


def _placeholders(values: list[str]) -> str:
    return ", ".join("?" for _ in values)


def list_runs(protocol_id: str | None = None, database_path: Path | None = None) -> list[dict[str, Any]]:
    connection = _connect(database_path)
    try:
        sql = "SELECT metadata, evidence_status FROM runs"
        params: list[str] = []
        if protocol_id is not None:
            sql += " WHERE protocol_id = ?"
            params.append(protocol_id)
        sql += " ORDER BY run_id"
        return [{**json.loads(metadata), "evidence_status": evidence_status}
                for metadata, evidence_status in connection.execute(sql, params).fetchall()]
    finally:
        connection.close()


def get_run(run_id: str, database_path: Path | None = None) -> dict[str, Any]:
    connection = _connect(database_path)
    try:
        row = connection.execute("SELECT metadata FROM runs WHERE run_id = ?", [run_id]).fetchone()
    finally:
        connection.close()
    if row is None:
        raise WorkbenchError("run_not_found", "Run was not found.", {"run_id": run_id})
    return json.loads(row[0])


def get_runs(run_ids: list[str], database_path: Path | None = None) -> list[dict[str, Any]]:
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise WorkbenchError("run_not_found", "Run selection must contain distinct run IDs.", {"requested_run_ids": run_ids})
    connection = _connect(database_path)
    try:
        rows = connection.execute(
            f"SELECT run_id, metadata, evidence_status FROM runs WHERE run_id IN ({_placeholders(run_ids)})",
            run_ids).fetchall()
    finally:
        connection.close()
    # `evidence_status` is an INDEX column, not part of the stored metadata, so reading only
    # `metadata` silently dropped it -- the API served run records with the evidence state missing,
    # and the frontend then fell back to trusting `data_kind`. Merged here so the truth the index
    # computed actually reaches the consumer.
    records = {run_id: {**json.loads(metadata), "evidence_status": evidence_status}
               for run_id, metadata, evidence_status in rows}
    missing = [run_id for run_id in run_ids if run_id not in records]
    if missing:
        raise WorkbenchError("run_not_found", "One or more run IDs were not found.", {"requested_run_ids": run_ids, "missing_run_ids": missing})
    return [records[run_id] for run_id in run_ids]


def _query_filters(run_id: str, query_id: str | None, category: str | None, failed_at: int | None, min_rank: int | None, max_rank: int | None, passed_at: int | None = None) -> tuple[str, list[Any]]:
    filters = ["run_id = ?"]
    params: list[Any] = [run_id]
    for column, value in (("query_id", query_id), ("category", category)):
        if value is not None:
            filters.append(f"{column} = ?")
            params.append(value)
    # `failed_at`/`passed_at` compare the AUTHORITATIVE `target_rank`, which is a position in the
    # complete eligible ranking -- never the length of a saved Top-K list. A run that stores fewer
    # retrieved images than another still classifies every query identically.
    if failed_at is not None:
        filters.append("target_rank > ?")
        params.append(failed_at)
    if passed_at is not None:
        filters.append("target_rank <= ?")
        params.append(passed_at)
    if min_rank is not None:
        filters.append("target_rank >= ?")
        params.append(min_rank)
    if max_rank is not None:
        filters.append("target_rank <= ?")
        params.append(max_rank)
    return " AND ".join(filters), params


def _top_results(connection, run_id: str, query_ids: list[str], k: int | None = None) -> dict[str, list[dict[str, Any]]]:
    if not query_ids:
        return {}
    sql = f"SELECT query_id, rank, image_id, score FROM top_results WHERE run_id = ? AND query_id IN ({_placeholders(query_ids)})"
    params: list[Any] = [run_id, *query_ids]
    if k is not None:
        sql += " AND rank <= ?"
        params.append(k)
    sql += " ORDER BY query_id, rank"
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_id, rank, image_id, score in connection.execute(sql, params).fetchall():
        output[query_id].append({"rank": rank, "image_id": image_id, "score": score})
    return output


def query_page(run_id: str, *, query_id: str | None = None, category: str | None = None, failed_at: int | None = None, min_rank: int | None = None, max_rank: int | None = None, order: str = "worst", limit: int = 50, offset: int = 0, database_path: Path | None = None, passed_at: int | None = None) -> dict[str, Any]:
    if not 1 <= limit <= 200 or offset < 0:
        raise WorkbenchError("invalid_pagination", "limit must be 1..200 and offset must be non-negative.", {"limit": limit, "offset": offset})
    if order not in {"worst", "best"}:
        raise WorkbenchError("invalid_query_order", "order must be worst or best.", {"order": order})
    # The two threshold filters are complements: "fails at K" is exactly "does not pass at K", so
    # accepting both would express an empty or contradictory selection rather than a question.
    if failed_at is not None and passed_at is not None:
        raise WorkbenchError("conflicting_rank_filter",
                             "failed_at and passed_at are mutually exclusive; a query cannot both "
                             "fail and pass at a threshold.",
                             {"failed_at": failed_at, "passed_at": passed_at})
    for name, value in (("failed_at", failed_at), ("passed_at", passed_at)):
        if value is not None and not 1 <= value <= 10000:
            raise WorkbenchError("invalid_rank_threshold",
                                 f"{name} must be a positive rank between 1 and 10000.",
                                 {name: value})
    get_run(run_id, database_path)
    where, params = _query_filters(run_id, query_id, category, failed_at, min_rank, max_rank, passed_at)
    connection = _connect(database_path)
    try:
        total = connection.execute(f"SELECT COUNT(*) FROM queries WHERE {where}", params).fetchone()[0]
        direction = "DESC" if order == "worst" else "ASC"
        rows = connection.execute(f"SELECT query_id, category, annotation_index, reference_id, target_id, raw_captions, model_input_text, target_rank FROM queries WHERE {where} ORDER BY target_rank {direction}, query_id LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
        ids = [row[0] for row in rows]
        top_results = _top_results(connection, run_id, ids)
    finally:
        connection.close()
    items = [{"query_id": item_id, "category": item_category, "annotation_index": annotation_index, "reference_id": reference_id, "target_id": target_id, "raw_captions": json.loads(raw_captions), "model_input_text": model_input_text, "target_rank": target_rank, "top_results": top_results.get(item_id, [])} for item_id, item_category, annotation_index, reference_id, target_id, raw_captions, model_input_text, target_rank in rows]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def get_query(run_id: str, query_id: str, database_path: Path | None = None) -> dict[str, Any]:
    page = query_page(run_id, query_id=query_id, limit=1, database_path=database_path)
    if not page["items"]:
        raise WorkbenchError("query_not_found", "Query was not found in run.", {"run_id": run_id, "query_id": query_id})
    return page["items"][0]


def _query_maps(run_ids: list[str], database_path: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    connection = _connect(database_path)
    try:
        sql = f"SELECT run_id, query_id, category, annotation_index, reference_id, target_id, raw_captions, model_input_text, target_rank FROM queries WHERE run_id IN ({_placeholders(run_ids)})"
        rows = connection.execute(sql, run_ids).fetchall()
    finally:
        connection.close()
    output: dict[str, dict[str, dict[str, Any]]] = {run_id: {} for run_id in run_ids}
    for run_id, query_id, category, annotation_index, reference_id, target_id, raw_captions, model_input_text, target_rank in rows:
        output[run_id][query_id] = {"query_id": query_id, "category": category, "annotation_index": annotation_index, "reference_id": reference_id, "target_id": target_id, "raw_captions": json.loads(raw_captions), "model_input_text": model_input_text, "target_rank": target_rank, "top_results": []}
    return output


def get_query_universe(run_ids: list[str], database_path: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    return _query_maps(run_ids, database_path)

# ---------------------------------------------------------------------------------------------
# CIRCUS composition cohorts
#
# A cohort used to be describable only in the per-query DuckDB index, and the CIRCUS release runs
# that this workbench actually serves live in `artifacts/composition`, not in that index. So the
# release was the only dataset with real numbers AND the only dataset whose cohorts could not be
# saved: `PUT /api/cohorts` answered `run_not_found` for every run the UI lists
# (`raw/verify/100-api-cache-and-ui-blocked.md` §9). These functions are the second arm.
#
# A composition reference is spelled `c:<run_id>`. The prefix is the disambiguator: a bare run id is
# a DuckDB index reference exactly as before, so every existing cohort and every existing caller of
# the DuckDB arm keeps working, and a composition run can never be mistaken for an index run whose
# id happens to collide.
# ---------------------------------------------------------------------------------------------

COMPOSITION_PREFIX = "c:"

def is_composition_reference(reference: str) -> bool:
    """Whether a selection entry names a CIRCUS composition run rather than an index run."""
    return isinstance(reference, str) and reference.startswith(COMPOSITION_PREFIX)

def composition_run_id(reference: str) -> str:
    return reference[len(COMPOSITION_PREFIX):]

def _composition_rank_artifact(reference: str) -> tuple[str, Path, dict[str, Any]]:
    """`(run_id, ranks_path, ranks)`, refusing a reference that names no usable artifact.

    Refusal is by NAME and by cause: an unknown run id says which id, and a missing artifact says
    which run has ranks nowhere. A cohort cannot be validated against a table that does not exist.
    """
    from workbench.backend.composition import resolve_composition_path
    from workbench.backend.operator_config import resolve_config

    run_id = composition_run_id(reference)
    if not run_id:
        raise WorkbenchError("query_alignment_mismatch",
                             "A composition reference must name a run id after the prefix.",
                             {"reference": reference})
    artifacts = resolve_config().WORKBENCH_ARTIFACTS_ROOT
    try:
        ranks_dir = resolve_composition_path(artifacts, f"runs/{run_id}/ranks")
    except ValueError as error:
        raise WorkbenchError("run_not_found", "Composition run was not found.",
                             {"run_id": run_id, "reference": reference}) from error
    files = sorted(ranks_dir.glob("*.json")) if ranks_dir.is_dir() else []
    if not files:
        raise WorkbenchError("run_not_found", "Composition run was not found.",
                             {"run_id": run_id, "reference": reference,
                              "composition_ranks_path": str(ranks_dir)})
    path = files[0]
    try:
        ranks = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise WorkbenchError("run_not_found",
                             "The composition rank artifact exists but could not be read.",
                             {"run_id": run_id, "rank_artifact": str(path)}) from error
    if not isinstance(ranks, dict) or not isinstance(ranks.get("ranks"), dict):
        raise WorkbenchError("run_not_found",
                             "The composition rank artifact carries no rank table.",
                             {"run_id": run_id, "rank_artifact": str(path)})
    return run_id, path, ranks

def composition_selection(references: list[str]) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Runs, query maps and provenance for a cohort made of composition runs.

    The map for each run is `{qid: {"target_rank": ..., "ranks": {...}}}`, built from the SAME
    artifact `cohort_metrics` reads, so a saved cohort and its reopened metrics cannot come from two
    different sources. The provenance block records the run id, protocol, subset, condition and the
    sha256 of the artifact the selection was validated against; a cohort whose artifact changes
    afterwards is refusable by comparing that digest, rather than silently re-scored.
    """
    from workbench.backend.composition import load_cached_release

    release = load_cached_release()
    if release is None:
        raise WorkbenchError("query_dataset_unavailable",
                             "The CIRCUS release is not loaded locally, so a composition cohort "
                             "cannot be validated.",
                             {"references": references})
    protocol_id = str(getattr(getattr(release, "gallery", None), "protocol_id", ""))
    release_qids = {query.qid for query in release.queries}
    runs: list[dict[str, Any]] = []
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    provenance_runs: list[dict[str, Any]] = []
    conditions: dict[str, int] = {}
    for reference in references:
        run_id, path, ranks = _composition_rank_artifact(reference)
        table = ranks["ranks"]
        for condition, values in table.items():
            if isinstance(values, Mapping):
                conditions[condition] = len(values)
        # The run's own universe: the qids its OWN artifact ranks. Comparing qid sets across runs is
        # then a statement about which queries both were actually able to score, not about which
        # queries the release happens to hold.
        run_map: dict[str, dict[str, Any]] = {}
        for condition, values in table.items():
            if not isinstance(values, Mapping):
                continue
            for qid, rank in values.items():
                if qid not in release_qids:
                    continue
                entry = run_map.setdefault(qid, {"query_id": qid, "ranks": {}, "target_rank": None})
                entry["ranks"][condition] = rank
                if condition == "multimodal":
                    entry["target_rank"] = rank
        maps[reference] = run_map
        digest = _sha256_of_file(path)
        runs.append({"run_id": run_id, "reference": reference, "protocol_id": protocol_id,
                     "data_kind": "circus_composition", "subset": "",
                     "top_k_saved": 0, "family": "composition"})
        provenance_runs.append({"reference": reference, "run_id": run_id,
                                "protocol_id": protocol_id,
                                "rank_artifact": str(path), "rank_artifact_sha256": digest})
    selected_qids = set().union(*maps.values()) if maps else set()
    protocols = {run["protocol_id"] for run in runs}
    if len(protocols) != 1 or not protocol_id:
        raise WorkbenchError("cross_protocol",
                             "Cross-protocol analysis is invalid. Select runs from one protocol.",
                             {"protocol_ids": sorted(protocols)})
    return runs, maps, {"family": "composition", "protocol_id": protocol_id,
                        # The release subsets the selected qids BELONG to (a qid can be in all
                        # three). Deliberately not named `subset`: the cohort's own scope is the
                        # caller's `definition.subset`, and conflating "these qids are also in `sf`"
                        # with "this cohort is scoped to `sf`" would state the wrong scope.
                        "qid_subsets": sorted({subset for query in release.queries
                                               if query.qid in selected_qids
                                               for subset in query.subsets}),
                        "conditions": dict(sorted(conditions.items())),
                        "runs": provenance_runs}

def _sha256_of_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def composition_cohort_metrics(references: list[str], query_ids: list[str], *,
                               expected: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Per-run cohort metrics from the composition rank artifacts.

    `expected` maps a reference to the artifact digest the cohort was SAVED against. A mismatch is a
    refusal (`cohort_artifact_changed`), never a silent re-score: the saved cohort describes a table,
    and a table that changed makes the saved numbers describe something no longer on disk.
    """
    _, maps, provenance = composition_selection(references)
    by_reference = {item["reference"]: item for item in provenance["runs"]}
    for reference in references:
        claimed = (expected or {}).get(reference)
        actual = by_reference[reference]["rank_artifact_sha256"]
        if claimed and claimed != actual:
            raise WorkbenchError(
                "cohort_artifact_changed",
                "The run artifact this cohort was saved against has changed, so its saved numbers "
                "no longer describe the artifact on disk. Re-save the cohort.",
                {"reference": reference, "run_id": by_reference[reference]["run_id"],
                 "saved_rank_artifact_sha256": claimed, "current_rank_artifact_sha256": actual})
    rows: list[dict[str, Any]] = []
    for reference in references:
        run_map = maps[reference]
        ranks = [run_map[qid]["target_rank"] for qid in query_ids
                 if qid in run_map and isinstance(run_map[qid].get("target_rank"), int)]
        rows.append({
            # `run_id`, not the `c:` reference: the export's column set is shared with the DuckDB
            # arm (CSV/Markdown are written from `metrics[0]`'s keys), and adding a column here
            # would silently change the shape of every cohort export. The reference is derivable
            # (`c:<run_id>`) and is recorded in the cohort's provenance.
            "run_id": by_reference[reference]["run_id"],
            "query_count": len(query_ids),
            "scored_count": len(ranks),
            **{f"r{k}": (100 * sum(rank <= k for rank in ranks) / len(ranks)) if ranks else None
               for k in (1, 5, 10, 20, 50, 100)},
            "mean_target_rank": mean(ranks) if ranks else None,
            "median_target_rank": median(ranks) if ranks else None,
        })
    return rows

def composition_unknown_query_ids(references: list[str], query_ids: list[str]) -> list[str]:
    """qids in `query_ids` that no selected composition run ranks at all.

    A qid that exists in the release but has no rank in the run is NOT a valid member of that run's
    cohort: the cohort would carry a query the run never scored, and its metrics would silently
    treat it as absent. Refused by name, with a bounded sample like the DuckDB arm.
    """
    _, maps, _ = composition_selection(references)
    universe = set().union(*(set(mapping) for mapping in maps.values())) if maps else set()
    return sorted(set(query_ids) - universe)

def cohort_selection(references: list[str], *, k: int | None = None,
                     require_top_k: bool = True, database_path: Path | None = None):
    """`validate_analysis_selection` for a selection that mixes both families.

    Split, not duplicated: an index reference goes through the DuckDB arm exactly as before, a
    `c:` reference through the composition arm, and the two results are merged. A selection that is
    ALL composition is therefore handled entirely by the composition arm, which is what makes a
    CIRCUS cohort saveable today without touching the index at all.
    """
    composition = [reference for reference in references if is_composition_reference(reference)]
    index = [reference for reference in references if not is_composition_reference(reference)]
    if not composition:
        runs, maps = validate_analysis_selection(references, k=k, require_top_k=require_top_k,
                                                 database_path=database_path)
        return runs, maps, None
    if index:
        raise WorkbenchError("mixed_data_kind",
                             "A cohort cannot mix per-query index runs and CIRCUS composition runs: "
                             "the two report different quantities.",
                             {"index_run_ids": index, "composition_references": composition})
    runs, maps, provenance = composition_selection(composition)
    if k is not None and not 1 <= k <= 200:
        raise WorkbenchError("insufficient_top_k_depth", "K must be between 1 and 200.",
                             {"requested_k": k})
    baseline_reference, baseline = runs[0]["reference"], maps[runs[0]["reference"]]
    baseline_ids = set(baseline)
    for run in runs[1:]:
        candidate = maps[run["reference"]]
        missing = sorted(baseline_ids - set(candidate))
        extra = sorted(set(candidate) - baseline_ids)
        if missing or extra:
            raise WorkbenchError(
                "query_alignment_mismatch",
                "Selected runs do not have an identical query universe.",
                {"baseline_run_id": runs[0]["run_id"], "run_id": run["run_id"],
                 "missing_query_count": len(missing), "missing_query_ids": missing[:20],
                 "extra_query_count": len(extra), "extra_query_ids": extra[:20]})
        for query_id, expected in baseline.items():
            actual = candidate[query_id]
            if expected["ranks"] != actual["ranks"]:
                raise WorkbenchError(
                    "canonical_query_mismatch",
                    "Canonical query fields differ for the same query_id.",
                    {"query_id": query_id, "baseline_run_id": runs[0]["run_id"],
                     "run_id": run["run_id"],
                     "mismatches": {"ranks": {"expected": expected["ranks"],
                                              "actual": actual["ranks"]}}})
    return runs, maps, provenance


def validate_analysis_selection(run_ids: list[str], *, k: int | None = None, require_top_k: bool = True, database_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    runs = get_runs(run_ids, database_path)
    protocols = {run["protocol_id"] for run in runs}
    if len(protocols) != 1:
        raise WorkbenchError("cross_protocol", "Cross-protocol analysis is invalid. Select runs from one FashionIQ protocol.", {"protocol_ids": sorted(protocols)})
    data_kinds = {run["data_kind"] for run in runs}
    if len(data_kinds) != 1:
        raise WorkbenchError("mixed_data_kind", "Mock and experiment runs cannot share an analysis, cohort, or export.", {"data_kinds": sorted(data_kinds), "run_ids": run_ids})
    if k is not None:
        if not 1 <= k <= 200:
            raise WorkbenchError("insufficient_top_k_depth", "K must be between 1 and 200.", {"requested_k": k})
        insufficient = [{"run_id": run["run_id"], "top_k_saved": run["top_k_saved"]} for run in runs if run["top_k_saved"] < k]
        if require_top_k and insufficient:
            raise WorkbenchError("insufficient_top_k_depth", "Selected runs do not store enough top-K retrievals.", {"requested_k": k, "runs": insufficient})
    maps = _query_maps(run_ids, database_path)
    baseline_id, baseline = run_ids[0], maps[run_ids[0]]
    baseline_ids = set(baseline)
    fields = ("category", "annotation_index", "reference_id", "target_id", "raw_captions")
    for run_id in run_ids[1:]:
        candidate = maps[run_id]
        missing, extra = sorted(baseline_ids - set(candidate)), sorted(set(candidate) - baseline_ids)
        if missing or extra:
            raise WorkbenchError("query_alignment_mismatch", "Selected runs do not have an identical query universe.", {"baseline_run_id": baseline_id, "run_id": run_id, "missing_query_count": len(missing), "missing_query_ids": missing[:20], "extra_query_count": len(extra), "extra_query_ids": extra[:20]})
        for query_id, expected in baseline.items():
            actual = candidate[query_id]
            mismatches = {field: {"expected": expected[field], "actual": actual[field]} for field in fields if expected[field] != actual[field]}
            if mismatches:
                raise WorkbenchError("canonical_query_mismatch", "Canonical query fields differ for the same query_id.", {"query_id": query_id, "baseline_run_id": baseline_id, "run_id": run_id, "mismatches": mismatches})
    return runs, maps


def get_analysis_input(run_ids: list[str], k: int, database_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    runs, maps = validate_analysis_selection(run_ids, k=k, database_path=database_path)
    connection = _connect(database_path)
    try:
        sql = f"SELECT run_id, query_id, rank, image_id, score FROM top_results WHERE run_id IN ({_placeholders(run_ids)}) AND rank <= ? ORDER BY run_id, query_id, rank"
        rows = connection.execute(sql, [*run_ids, k]).fetchall()
    finally:
        connection.close()
    for run_id, query_id, rank, image_id, score in rows:
        maps[run_id][query_id]["top_results"].append({"rank": rank, "image_id": image_id, "score": score})
    return runs, maps


def cohort_metrics(run_ids: list[str], query_ids: list[str], database_path: Path | None = None) -> list[dict[str, Any]]:
    if not query_ids:
        return [{"run_id": run_id, "query_count": 0, "scored_count": 0, **{f"r{k}": None for k in (1, 5, 10, 20, 50, 100)}, "mean_target_rank": None, "median_target_rank": None} for run_id in run_ids]
    connection = _connect(database_path)
    try:
        sql = f"SELECT run_id, target_rank FROM queries WHERE run_id IN ({_placeholders(run_ids)}) AND query_id IN ({_placeholders(query_ids)})"
        rows = connection.execute(sql, [*run_ids, *query_ids]).fetchall()
    finally:
        connection.close()
    ranks_by_run: dict[str, list[int]] = defaultdict(list)
    for run_id, rank in rows:
        ranks_by_run[run_id].append(rank)
    return [{"run_id": run_id, "query_count": len(query_ids), "scored_count": len(ranks := ranks_by_run[run_id]), **{f"r{k}": 100 * sum(rank <= k for rank in ranks) / len(ranks) if ranks else None for k in (1, 5, 10, 20, 50, 100)}, "mean_target_rank": mean(ranks) if ranks else None, "median_target_rank": median(ranks) if ranks else None} for run_id in run_ids]
