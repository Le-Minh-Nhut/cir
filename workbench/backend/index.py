from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from workbench.backend.errors import WorkbenchError
from workbench.backend.schemas.results import ResultRun

ROOT = Path(__file__).resolve().parents[1]


def database_path(config=None) -> Path:
    from workbench.backend.operator_config import resolve_config

    return (config or resolve_config()).WORKBENCH_ARTIFACTS_ROOT / "workbench.duckdb"

configured_database_path = database_path


def results_path(config=None) -> Path:
    from workbench.backend.operator_config import resolve_config

    return (config or resolve_config()).WORKBENCH_RESULTS_ROOT


DATABASE_PATH = configured_database_path()
RESULTS_ROOT = results_path()


def result_files(root: Path | None = None) -> list[Path]:
    from workbench.backend.operator_config import resolve_config

    root = root or results_path()
    files = []
    for path in sorted(root.rglob("*.json")):
        if path.name == ".gitkeep":
            continue
        try:
            artifact = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            files.append(path)
            continue
        if artifact.get("artifact_type") == "official_aggregate_evaluation_report":
            continue
        files.append(path)
    return files


def load_runs(root: Path | None = None) -> list[ResultRun]:
    return [result for _, result in load_runs_with_digests(root)]


def load_runs_with_digests(root: Path | None = None) -> list[tuple[str, ResultRun]]:
    """Every canonical run with the digest of the FILE it was read from.

    The digest is what an evidence receipt binds: a verdict that a run was verified is a statement
    about specific bytes, so promotion needs the digest of the artifact on disk, not a value the
    artifact claims about itself (that would be self-certification again). Computed here from the
    bytes actually read, so a later edit cannot silently keep a promotion it no longer deserves.
    """
    import hashlib

    runs: list[tuple[str, ResultRun]] = []
    sources: dict[str, Path] = {}
    for path in result_files(root):
        raw = path.read_bytes()
        result = ResultRun.model_validate_json(raw.decode("utf-8"))
        prior = sources.get(result.run.run_id)
        if prior is not None:
            raise WorkbenchError("duplicate_run_id", "Duplicate run_id in result JSON files.", {"run_id": result.run.run_id, "first_source": str(prior), "second_source": str(path)})
        sources[result.run.run_id] = path
        runs.append((hashlib.sha256(raw).hexdigest(), result))
    return runs


def evidence_status(result: ResultRun, verifier_verdict: dict | None = None,
                    artifact_digest: str | None = None) -> str:
    """The evidence state of a stored run, from the ONE admission policy (F-07).

    Previously this read `data_kind == "experiment"` and answered `PER_QUERY_VERIFIED`. That made a
    schema-valid JSON file certify its own scientific authenticity: `data_kind` is what the producer
    DECLARED about itself, and anyone able to write a file could write `"experiment"`.

    Now the state comes from `schemas.results.admission`, which asks what the artifact's own
    provenance can support, and is promoted to `PER_QUERY_VERIFIED` only by the authoritative
    verifier's verdict (evidence ledger + completion proof + re-extraction).
    """
    from workbench.backend.schemas.results import admission, promote

    state, _ = promote(result, *admission(result)[:1], verifier_verdict,
                       artifact_digest=artifact_digest)
    return state


def _create_schema(connection) -> None:
    connection.execute("CREATE TABLE runs (run_id VARCHAR PRIMARY KEY, protocol_id VARCHAR NOT NULL, model_id VARCHAR NOT NULL, checkpoint_id VARCHAR NOT NULL, checkpoint_training_noise_pct INTEGER, data_kind VARCHAR NOT NULL, top_k_saved INTEGER NOT NULL, gallery_size INTEGER NOT NULL, category_gallery_sizes JSON, category_eligible_counts JSON, evidence_status VARCHAR NOT NULL, r10 DOUBLE, r50 DOUBLE, mean DOUBLE, metadata JSON NOT NULL)")
    connection.execute("CREATE TABLE queries (run_id VARCHAR NOT NULL, query_id VARCHAR NOT NULL, category VARCHAR NOT NULL, annotation_index INTEGER NOT NULL, reference_id VARCHAR NOT NULL, target_id VARCHAR NOT NULL, raw_captions JSON NOT NULL, model_input_text VARCHAR NOT NULL, target_rank INTEGER NOT NULL, PRIMARY KEY(run_id, query_id))")
    connection.execute("CREATE TABLE top_results (run_id VARCHAR NOT NULL, query_id VARCHAR NOT NULL, rank INTEGER NOT NULL, image_id VARCHAR NOT NULL, score DOUBLE NOT NULL, PRIMARY KEY(run_id, query_id, rank))")
    connection.execute("CREATE INDEX queries_by_run_rank ON queries(run_id, target_rank)")
    connection.execute("CREATE INDEX queries_by_run_category ON queries(run_id, category)")
    connection.execute("CREATE INDEX top_results_by_query ON top_results(run_id, query_id, rank)")


def verifier_verdicts(config=None) -> dict[str, dict]:
    """Aggregate-report verification receipts, keyed by run_id, for admission promotion.

    This is what lets a stored per-query artifact reach ``PER_QUERY_VERIFIED``: the artifact's own
    contents can never do it, so the index asks the same authoritative receipt the API serves
    (``main._aggregate_reports`` → ``verify_run``). A run with no corroborated aggregate report
    stays unverified, which is the correct answer for `experiment` that nobody checked.
    """
    try:
        from workbench.backend.main import _aggregate_reports
    except Exception:
        # A cycle-free fallback: during early import the aggregate reader is not available yet, and
        # refusing to promote is the safe direction.
        return {}
    verdicts: dict[str, dict] = {}
    try:
        reports = _aggregate_reports() or []
    except Exception:
        # A ledger that cannot be read corroborates nothing. The index must still build: refusing to
        # index would hide every run from the API because of an unrelated ledger problem, and the
        # safe direction is to leave every experiment UNVERIFIED (never to promote one we could not
        # check). The API surfaces the ledger's own named error separately.
        return {}
    for report in reports:
        run_id = report.get("run_id")
        verification = report.get("verification")
        if run_id and isinstance(verification, dict):
            verdicts[run_id] = verification
    return verdicts


def rebuild_index(results_root: Path | None = None, database_path: Path | None = None) -> int:
    import duckdb

    database_path = database_path or configured_database_path()

    loaded = load_runs_with_digests(results_root)
    runs = [result for _, result in loaded]
    digests = {result.run.run_id: digest for digest, result in loaded}
    verdicts = verifier_verdicts()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = database_path.with_name(f".{database_path.name}.{uuid4().hex}.tmp")
    connection = duckdb.connect(str(temporary))
    try:
        _create_schema(connection)
        rows = {"runs": [], "queries": [], "top_results": []}
        for result in runs:
            run = result.run
            rows["runs"].append([run.run_id, run.protocol_id, run.model_id, run.checkpoint_id, run.checkpoint_training_noise_pct, run.data_kind, run.top_k_saved, run.gallery_size, json.dumps(run.category_gallery_sizes) if run.category_gallery_sizes else None, json.dumps(run.category_eligible_counts) if run.category_eligible_counts else None, evidence_status(result, verdicts.get(run.run_id), digests.get(run.run_id)), run.reproduced_metrics.r10, run.reproduced_metrics.r50, run.reproduced_metrics.mean, run.model_dump_json()])
            for query in result.queries:
                rows["queries"].append([run.run_id, query.query_id, query.category, query.annotation_index, query.reference_id, query.target_id, json.dumps(query.raw_captions), query.model_input_text, query.target_rank])
                rows["top_results"].extend([run.run_id, query.query_id, item.rank, item.image_id, item.score] for item in query.top_results)
        for table, table_rows in rows.items():
            if not table_rows:
                continue
            with tempfile.NamedTemporaryFile(mode="w", newline="", dir=database_path.parent, suffix=".csv", delete=False) as file:
                csv.writer(file).writerows(table_rows)
                source = Path(file.name)
            try:
                connection.execute(f"COPY {table} FROM ? (FORMAT CSV)", [str(source)])
            finally:
                source.unlink(missing_ok=True)
        connection.execute("SELECT COUNT(*) FROM runs").fetchone()
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    os.replace(temporary, database_path)
    return len(runs)
