from __future__ import annotations

from pathlib import Path

from workbench.backend.schemas.results import ResultRun

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "artifacts" / "results"
DATABASE_PATH = ROOT / "artifacts" / "workbench.duckdb"


def result_files(root: Path = RESULTS_ROOT) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.name != ".gitkeep")


def load_runs(root: Path = RESULTS_ROOT) -> list[ResultRun]:
    return [ResultRun.model_validate_json(path.read_text()) for path in result_files(root)]


def rebuild_index(results_root: Path = RESULTS_ROOT, database_path: Path = DATABASE_PATH) -> int:
    import duckdb

    runs = load_runs(results_root)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("DROP TABLE IF EXISTS runs")
        connection.execute("DROP TABLE IF EXISTS queries")
        connection.execute("CREATE TABLE runs (run_id VARCHAR PRIMARY KEY, protocol_id VARCHAR, model_id VARCHAR, checkpoint_training_noise_pct INTEGER, data_kind VARCHAR, r10 DOUBLE, r50 DOUBLE, mean DOUBLE, payload JSON)")
        connection.execute("CREATE TABLE queries (run_id VARCHAR, query_id VARCHAR, category VARCHAR, annotation_index INTEGER, reference_id VARCHAR, target_id VARCHAR, target_rank INTEGER, payload JSON, PRIMARY KEY(run_id, query_id))")
        for result in runs:
            run = result.run
            connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [run.run_id, run.protocol_id, run.model_id, run.checkpoint_training_noise_pct, run.data_kind, run.reproduced_metrics.r10, run.reproduced_metrics.r50, run.reproduced_metrics.mean, result.model_dump_json()])
            for query in result.queries:
                connection.execute("INSERT INTO queries VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [run.run_id, query.query_id, query.category, query.annotation_index, query.reference_id, query.target_id, query.target_rank, query.model_dump_json()])
    finally:
        connection.close()
    return len(runs)
