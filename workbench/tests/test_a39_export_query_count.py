import json
from pathlib import Path
import duckdb
import pytest

from workbench.backend.storage import cohort_metrics


def _setup_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("""
    CREATE TABLE runs (
        run_id VARCHAR PRIMARY KEY,
        protocol_id VARCHAR,
        data_kind VARCHAR,
        top_k_saved INTEGER,
        metadata VARCHAR,
        evidence_status VARCHAR
    );
    """)
    con.execute("""
    CREATE TABLE queries (
        run_id VARCHAR,
        query_id VARCHAR,
        category VARCHAR,
        annotation_index INTEGER,
        reference_id VARCHAR,
        target_id VARCHAR,
        raw_captions VARCHAR,
        model_input_text VARCHAR,
        target_rank INTEGER
    );
    """)
    meta = json.dumps({"run_id": "run-1", "protocol_id": "p1", "data_kind": "mock", "top_k_saved": 50})
    con.execute("INSERT INTO runs VALUES ('run-1', 'p1', 'mock', 50, ?, 'PER_QUERY_VERIFIED')", [meta])
    # Insert only q1 for run-1; q2 and q3 are missing / unscored
    con.execute("INSERT INTO queries VALUES ('run-1', 'q1', 'dress', 0, 'ref1', 'tgt1', '[\"cap\"]', 'cap', 1)")
    con.close()


def test_cohort_metrics_exposes_both_requested_and_scored_query_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "workbench.duckdb"
    _setup_db(db_path)

    requested_queries = ["q1", "q2", "q3"]
    metrics = cohort_metrics(["run-1"], requested_queries, database_path=db_path)
    assert len(metrics) == 1
    item = metrics[0]

    # The defect: previously query_count was len(ranks) = 1, masking that 2 queries were unscored.
    # Required: expose both requested count (3) and scored count (1) so consumers can distinguish
    # "3 requested, 1 scored" from "1 requested, 1 scored".
    assert "scored_count" in item, "scored_count must be exposed"
    assert item["scored_count"] == 1, "scored_count must reflect only queries actually scored"
    assert item["query_count"] == 3, "query_count must reflect total cohort queries requested"
    # Recall denominator semantics remain over scored queries: 1 hit out of 1 scored = 100.0%
    assert item["r1"] == 100.0
