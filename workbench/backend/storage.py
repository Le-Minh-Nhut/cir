from __future__ import annotations

import json
from pathlib import Path

from workbench.backend.index import DATABASE_PATH, load_runs
from workbench.backend.schemas.results import ResultRun


def indexed_runs(database_path: Path | None = None) -> list[ResultRun]:
    database_path = database_path or DATABASE_PATH
    if not database_path.is_file():
        return load_runs()
    import duckdb

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        records = []
        for (metadata,) in connection.execute("SELECT metadata FROM runs ORDER BY run_id").fetchall():
            run = json.loads(metadata)
            query_rows = connection.execute("SELECT query_id, category, annotation_index, reference_id, target_id, raw_captions, model_input_text, target_rank FROM queries WHERE run_id = ? ORDER BY annotation_index", [run["run_id"]]).fetchall()
            queries = []
            for query_id, category, annotation_index, reference_id, target_id, raw_captions, model_input_text, target_rank in query_rows:
                top = connection.execute("SELECT rank, image_id, score FROM top_results WHERE run_id = ? AND query_id = ? ORDER BY rank", [run["run_id"], query_id]).fetchall()
                queries.append({"query_id": query_id, "category": category, "annotation_index": annotation_index, "reference_id": reference_id, "target_id": target_id, "raw_captions": json.loads(raw_captions), "model_input_text": model_input_text, "target_rank": target_rank, "top_results": [{"rank": rank, "image_id": image_id, "score": score} for rank, image_id, score in top]})
            records.append(ResultRun.model_validate({"schema_version": 2, "run": run, "queries": queries}))
        return records
    finally:
        connection.close()
