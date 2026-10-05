from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from workbench.backend.errors import WorkbenchError
from workbench.backend.schemas.results import ResultRun

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "artifacts" / "results"
DATABASE_PATH = ROOT / "artifacts" / "workbench.duckdb"


def result_files(root: Path = RESULTS_ROOT) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.name != ".gitkeep")


def load_runs(root: Path = RESULTS_ROOT) -> list[ResultRun]:
    runs: list[ResultRun] = []
    sources: dict[str, Path] = {}
    for path in result_files(root):
        result = ResultRun.model_validate_json(path.read_text())
        prior = sources.get(result.run.run_id)
        if prior is not None:
            raise WorkbenchError("duplicate_run_id", "Duplicate run_id in result JSON files.", {"run_id": result.run.run_id, "first_source": str(prior), "second_source": str(path)})
        sources[result.run.run_id] = path
        runs.append(result)
    return runs


def _create_schema(connection) -> None:
    connection.execute("CREATE TABLE runs (run_id VARCHAR PRIMARY KEY, protocol_id VARCHAR NOT NULL, model_id VARCHAR NOT NULL, checkpoint_id VARCHAR NOT NULL, checkpoint_training_noise_pct INTEGER, data_kind VARCHAR NOT NULL, top_k_saved INTEGER NOT NULL, gallery_size INTEGER NOT NULL, r10 DOUBLE, r50 DOUBLE, mean DOUBLE, metadata JSON NOT NULL)")
    connection.execute("CREATE TABLE queries (run_id VARCHAR NOT NULL, query_id VARCHAR NOT NULL, category VARCHAR NOT NULL, annotation_index INTEGER NOT NULL, reference_id VARCHAR NOT NULL, target_id VARCHAR NOT NULL, raw_captions JSON NOT NULL, model_input_text VARCHAR NOT NULL, target_rank INTEGER NOT NULL, PRIMARY KEY(run_id, query_id))")
    connection.execute("CREATE TABLE top_results (run_id VARCHAR NOT NULL, query_id VARCHAR NOT NULL, rank INTEGER NOT NULL, image_id VARCHAR NOT NULL, score DOUBLE NOT NULL, PRIMARY KEY(run_id, query_id, rank))")
    connection.execute("CREATE INDEX queries_by_run_rank ON queries(run_id, target_rank)")
    connection.execute("CREATE INDEX queries_by_run_category ON queries(run_id, category)")
    connection.execute("CREATE INDEX top_results_by_query ON top_results(run_id, query_id, rank)")


def rebuild_index(results_root: Path = RESULTS_ROOT, database_path: Path = DATABASE_PATH) -> int:
    import duckdb

    runs = load_runs(results_root)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = database_path.with_name(f".{database_path.name}.{uuid4().hex}.tmp")
    connection = duckdb.connect(str(temporary))
    try:
        _create_schema(connection)
        for result in runs:
            run = result.run
            metadata = run.model_dump_json()
            connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [run.run_id, run.protocol_id, run.model_id, run.checkpoint_id, run.checkpoint_training_noise_pct, run.data_kind, run.top_k_saved, run.gallery_size, run.reproduced_metrics.r10, run.reproduced_metrics.r50, run.reproduced_metrics.mean, metadata])
            for query in result.queries:
                connection.execute("INSERT INTO queries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [run.run_id, query.query_id, query.category, query.annotation_index, query.reference_id, query.target_id, json.dumps(query.raw_captions), query.model_input_text, query.target_rank])
                for item in query.top_results:
                    connection.execute("INSERT INTO top_results VALUES (?, ?, ?, ?, ?)", [run.run_id, query.query_id, item.rank, item.image_id, item.score])
        connection.execute("SELECT COUNT(*) FROM runs").fetchone()
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    os.replace(temporary, database_path)
    return len(runs)
