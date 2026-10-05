from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from workbench.backend.errors import WorkbenchError
from workbench.backend.index import DATABASE_PATH


def _database_path(database_path: Path | None) -> Path:
    path = database_path or DATABASE_PATH
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
        sql = "SELECT metadata FROM runs"
        params: list[str] = []
        if protocol_id is not None:
            sql += " WHERE protocol_id = ?"
            params.append(protocol_id)
        sql += " ORDER BY run_id"
        return [json.loads(metadata) for (metadata,) in connection.execute(sql, params).fetchall()]
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
        rows = connection.execute(f"SELECT run_id, metadata FROM runs WHERE run_id IN ({_placeholders(run_ids)})", run_ids).fetchall()
    finally:
        connection.close()
    records = {run_id: json.loads(metadata) for run_id, metadata in rows}
    missing = [run_id for run_id in run_ids if run_id not in records]
    if missing:
        raise WorkbenchError("run_not_found", "One or more run IDs were not found.", {"requested_run_ids": run_ids, "missing_run_ids": missing})
    return [records[run_id] for run_id in run_ids]


def _query_filters(run_id: str, query_id: str | None, category: str | None, failed_at: int | None, min_rank: int | None, max_rank: int | None) -> tuple[str, list[Any]]:
    filters = ["run_id = ?"]
    params: list[Any] = [run_id]
    for column, value in (("query_id", query_id), ("category", category)):
        if value is not None:
            filters.append(f"{column} = ?")
            params.append(value)
    if failed_at is not None:
        filters.append("target_rank > ?")
        params.append(failed_at)
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


def query_page(run_id: str, *, query_id: str | None = None, category: str | None = None, failed_at: int | None = None, min_rank: int | None = None, max_rank: int | None = None, order: str = "worst", limit: int = 50, offset: int = 0, database_path: Path | None = None) -> dict[str, Any]:
    if not 1 <= limit <= 200 or offset < 0:
        raise WorkbenchError("invalid_pagination", "limit must be 1..200 and offset must be non-negative.", {"limit": limit, "offset": offset})
    if order not in {"worst", "best"}:
        raise WorkbenchError("invalid_query_order", "order must be worst or best.", {"order": order})
    get_run(run_id, database_path)
    where, params = _query_filters(run_id, query_id, category, failed_at, min_rank, max_rank)
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


def validate_analysis_selection(run_ids: list[str], *, k: int | None = None, require_top_k: bool = True, database_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    runs = get_runs(run_ids, database_path)
    protocols = {run["protocol_id"] for run in runs}
    if len(protocols) != 1:
        raise WorkbenchError("cross_protocol", "Cross-protocol analysis is invalid. Select runs from one FashionIQ protocol.", {"protocol_ids": sorted(protocols)})
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
        return [{"run_id": run_id, "query_count": 0, **{f"r{k}": None for k in (1, 5, 10, 20, 50, 100)}, "mean_target_rank": None, "median_target_rank": None} for run_id in run_ids]
    connection = _connect(database_path)
    try:
        sql = f"SELECT run_id, target_rank FROM queries WHERE run_id IN ({_placeholders(run_ids)}) AND query_id IN ({_placeholders(query_ids)})"
        rows = connection.execute(sql, [*run_ids, *query_ids]).fetchall()
    finally:
        connection.close()
    ranks_by_run: dict[str, list[int]] = defaultdict(list)
    for run_id, rank in rows:
        ranks_by_run[run_id].append(rank)
    return [{"run_id": run_id, "query_count": len(ranks := ranks_by_run[run_id]), **{f"r{k}": 100 * sum(rank <= k for rank in ranks) / len(ranks) if ranks else None for k in (1, 5, 10, 20, 50, 100)}, "mean_target_rank": mean(ranks) if ranks else None, "median_target_rank": median(ranks) if ranks else None} for run_id in run_ids]
