from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from workbench.backend import autopilot
from test_autopilot_config import fixture_config


def _config(tmp_path: Path, monkeypatch):
    path, _, registry = fixture_config(tmp_path, ["done", "blocked", "failed"])
    registry["models"] = [{"model_id": name} for name in ("done", "blocked", "failed")]
    monkeypatch.setattr(autopilot, "load_registry", lambda path: registry)
    config = autopilot.load_config(path)
    plan = {
        "schema_version": 1, "configuration_digest": "fixture", "solver_verified": True,
        "environment_plan": {"blocked": []},
        "models": [{"model_id": name, "status": "NOT_RUN", "blockers": [], "metrics": None}
                   for name in config.selected_model_ids],
    }
    return config, plan


def _records():
    return {
        "evaluation:done": {"model_id": "done", "name": "evaluation", "status": "COMPLETE"},
        "sync:blocked": {"model_id": "blocked", "name": "sync", "status": "FAILED", "return_code": 1},
        "evaluation:failed": {"model_id": "failed", "name": "evaluation", "status": "FAILED", "return_code": 1},
    }


def test_mixed_statuses_write_complete_reports_and_partial_exit(tmp_path, monkeypatch):
    config, plan = _config(tmp_path, monkeypatch)
    rows, blockers = autopilot._classify(plan, {"exit_code": 1, "stages": _records()}, config)
    result = autopilot._write_reports(config, plan, rows, blockers, {"exit_code": 1}, planned=False)
    paths = autopilot._artifact_paths(config)
    report = json.loads(paths["summary_json"].read_text())
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert result["exit_code"] == 2
    assert {row["model_id"]: row["status"] for row in report["models"]} == {
        "done": "COMPLETED", "blocked": "BLOCKED_SOURCE", "failed": "FAILED_EVALUATION"}
    assert {row["model_id"]: row["status"] for row in records} == {
        row["model_id"]: row["status"] for row in report["models"]}
    assert all(json.loads(row["metrics"]) is None for row in records)
    assert json.loads(paths["blockers"].read_text()) == blockers
    assert paths["summary_md"].is_file() and paths["manifest"].is_file()


def test_report_exit_distinguishes_full_and_failed_workflows(tmp_path, monkeypatch):
    config, plan = _config(tmp_path, monkeypatch)
    success = [{"model_id": name, "status": "COMPLETED", "blockers": [],
                "missing_permissions": [], "metrics": None, "run_id": None}
               for name in config.selected_model_ids]
    assert autopilot._write_reports(config, plan, success, [], {"exit_code": 0}, planned=False)["exit_code"] == 0
    failure = [{**row, "status": "FAILED_EVALUATION", "blockers": ["failed"]} for row in success]
    assert autopilot._write_reports(config, plan, failure, [], {"exit_code": 1}, planned=False)["exit_code"] == 1
