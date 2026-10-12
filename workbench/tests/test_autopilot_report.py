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

def test_mandatory_aggregate_report_proof_validation_and_mutation_rejection(tmp_path: Path, monkeypatch):
    import secrets
    import sys
    import uuid
    from workbench.backend.operator_config import WorkbenchConfig
    import workbench.scripts.evaluate_models as em

    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", secrets.token_hex(32))

    repo = tmp_path / "repo"
    repo.mkdir()
    FashionIQ = tmp_path / "FashionIQ"
    FashionIQ.mkdir()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    third_party = tmp_path / "third_party"
    third_party.mkdir()

    source = third_party / "CSMCIR"
    source.mkdir(parents=True)
    ckpt_file = checkpoints / "csmcir" / "model.pt"
    ckpt_file.parent.mkdir(parents=True)
    ckpt_file.write_bytes(b"csmcir-checkpoint-bytes")

    pin = "a" * 40

    wb_config = WorkbenchConfig(
        CIR_REPO_ROOT=repo,
        CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=FashionIQ,
        WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=checkpoints,
        WORKBENCH_RESULTS_ROOT=results,
        WORKBENCH_THIRD_PARTY_ROOT=third_party,
        WORKBENCH_ARTIFACTS_ROOT=artifacts,
    )

    eval_script = tmp_path / "fake_evaluator.py"
    eval_script.write_text("""import json
print(json.dumps({
    "dress_recall_at10": 60.0, "dress_recall_at50": 80.0,
    "shirt_recall_at10": 55.0, "shirt_recall_at50": 75.0,
    "toptee_recall_at10": 50.0, "toptee_recall_at50": 70.0,
    "average_recall_at10": 55.0, "average_recall_at50": 75.0, "average_recall": 65.0,
}))
""")

    eval_plan = em.EvaluationPlan(
        model_id="csmcir",
        checkpoint_id="ckpt_v1",
        protocol="fashioniq_original_split",
        dataset_root=FashionIQ,
        source=source,
        checkpoint=ckpt_file,
        cwd=source,
        command=[sys.executable, str(eval_script)],
        pin=pin,
        actual_source_commit=pin,
        paper_metrics={"r10": 55.0, "r50": 75.0, "mean": 65.0},
    )

    run_id = f"testrun_{uuid.uuid4().hex[:8]}"
    exit_code = em.execute(eval_plan, wb_config, run_id=run_id)
    assert exit_code == 0

    evidence = em.completion_validator("csmcir", "ckpt_v1", "fashioniq_original_split", pin, wb_config, ckpt_file)
    ok, reason, detail = evidence({"run_id": run_id})
    assert ok is True, f"completion proof must validate: {reason}"
    assert detail is not None
    assert detail.get("report_digest")

    registry_yaml = tmp_path / "registry.yaml"
    registry = {
        "models": [
            {
                "model_id": "csmcir",
                "method_name": "CSMCIR",
                "native_protocol": "fashioniq_original_split",
                "reported_r10": 55.0,
                "reported_r50": 75.0,
                "reported_mean": 65.0,
                "upstream_commit_sha": pin,
                "checkpoint_variants": [
                    {
                        "checkpoint_id": "ckpt_v1",
                        "filename": "model.pt",
                        "status": "URL_VERIFIED",
                    }
                ],
            }
        ]
    }
    registry_yaml.write_text(yaml.safe_dump(registry))

    ap_cfg_dict = {
        "schema_version": 1,
        "selected_models": ["csmcir"],
        "paths": {
            "registry": str(registry_yaml),
            "dataset_root": str(FashionIQ),
            "artifacts": str(artifacts),
            "checkpoints": str(checkpoints),
            "results": str(results),
            "third_party": str(third_party),
        },
        "budgets": {"max_download_bytes": 10737418240, "max_models": 10, "top_k": 200},
        "permissions": {k: True for k in autopilot.PERMISSIONS},
    }
    ap_cfg_path = tmp_path / "autopilot.yaml"
    ap_cfg_path.write_text(yaml.safe_dump(ap_cfg_dict))
    monkeypatch.setattr(autopilot, "load_registry", lambda path: registry)
    config = autopilot.load_config(ap_cfg_path)

    plan = {
        "schema_version": 1,
        "configuration_digest": "digest_test",
        "solver_verified": True,
        "environment_plan": {"groups": [{"environment_id": "csmcir_env", "members": ["csmcir"]}]},
        "models": [{"model_id": "csmcir", "status": "NOT_RUN", "blockers": [], "metrics": None}],
    }
    execution = {
        "exit_code": 0,
        "stages": {
            "evaluation:csmcir": {
                "model_id": "csmcir",
                "name": "evaluation",
                "status": "COMPLETE",
                "run_id": run_id,
                "evidence": detail,
            }
        },
        "missing_permissions": {},
    }

    rows, blockers = autopilot._classify(plan, execution, config)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "COMPLETED"
    assert row["run_id"] == run_id
    assert row["dress_r10"] == 60.0
    assert row["dress_r50"] == 80.0
    assert row["shirt_r10"] == 55.0
    assert row["shirt_r50"] == 75.0
    assert row["toptee_r10"] == 50.0
    assert row["toptee_r50"] == 70.0
    assert row["macro_r10"] == 55.0
    assert row["macro_r50"] == 75.0
    assert row["completion_proof"] is not None
    assert Path(row["completion_proof"]).is_file()
    assert row["aggregate_report"] is not None
    assert Path(row["aggregate_report"]).is_file()
    assert row["metrics"] is not None
    assert row["metrics"]["parity_status"] == "PARITY_VERIFIED"
    assert blockers == []

    rep = autopilot._write_reports(config, plan, rows, blockers, execution, planned=False)
    assert rep["exit_code"] == 0
    paths = autopilot._artifact_paths(config)
    summary_data = json.loads(paths["summary_json"].read_text())
    assert summary_data["models"][0]["run_id"] == run_id
    assert summary_data["models"][0]["dress_r10"] == 60.0

    # MUTATE the aggregate report after completion
    rep_path = Path(row["aggregate_report"])
    tampered_text = rep_path.read_text().replace("60.0", "99.0")
    rep_path.write_text(tampered_text)

    rows_tampered, blockers_tampered = autopilot._classify(plan, execution, config)
    assert len(rows_tampered) == 1
    t_row = rows_tampered[0]
    assert t_row["status"] == "FAILED_OUTPUT_VALIDATION"
    assert t_row["status"] != "COMPLETED"
    assert t_row["metrics"] is None
    assert t_row["dress_r10"] is None
    assert any("corrupted evidence rejected" in b for b in t_row["blockers"])
    assert len(blockers_tampered) == 1


def test_markdown_report_renders_synthetic_table_and_null_reasons(tmp_path: Path, monkeypatch):
    registry = {
        "models": [
            {
                "model_id": "synthetic_model",
                "method_name": "SyntheticModel",
                "synthetic_source": True,
                "native_protocol": "fashioniq_original_split",
                "reported_r10": 50.0,
                "reported_r50": 70.0,
            },
            {
                "model_id": "unparsed_model",
                "method_name": "UnparsedModel",
                "native_protocol": "fashioniq_original_split",
            },
        ]
    }
    registry_yaml = tmp_path / "registry.yaml"
    registry_yaml.write_text(yaml.safe_dump(registry))
    FashionIQ = tmp_path / "FashionIQ"
    FashionIQ.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    dirs = {name: tmp_path / name for name in ("checkpoints", "results", "third_party")}
    for d in dirs.values():
        d.mkdir()

    ap_cfg_dict = {
        "schema_version": 1,
        "selected_models": ["synthetic_model", "unparsed_model"],
        "paths": {
            "registry": str(registry_yaml),
            "dataset_root": str(FashionIQ),
            "artifacts": str(artifacts),
            "checkpoints": str(dirs["checkpoints"]),
            "results": str(dirs["results"]),
            "third_party": str(dirs["third_party"]),
        },
        "budgets": {"max_download_bytes": 10737418240, "max_models": 10, "top_k": 200},
        "permissions": {k: True for k in autopilot.PERMISSIONS},
    }
    ap_cfg_path = tmp_path / "autopilot.yaml"
    ap_cfg_path.write_text(yaml.safe_dump(ap_cfg_dict))
    monkeypatch.setattr(autopilot, "load_registry", lambda path: registry)
    config = autopilot.load_config(ap_cfg_path)

    plan = {
        "schema_version": 1,
        "configuration_digest": "fixture_digest",
        "solver_verified": True,
        "environment_plan": {"groups": []},
        "models": [
            {"model_id": "synthetic_model", "status": "NOT_RUN", "blockers": [], "metrics": None},
            {"model_id": "unparsed_model", "status": "NOT_RUN", "blockers": [], "metrics": None},
        ],
    }

    row_synthetic = {
        "model_id": "synthetic_model",
        "method_name": "SyntheticModel",
        "status": "COMPLETED",
        "blockers": [],
        "missing_permissions": [],
        "run_id": "run_syn_123",
        "checkpoint_id": "ckpt_syn",
        "checkpoint_provenance": "synthetic",
        "environment_group": "group_syn",
        "native_protocol": "fashioniq_original_split",
        "evaluation_kind": "SYNTHETIC",
        "dress_r10": 60.0,
        "dress_r50": 80.0,
        "shirt_r10": 55.0,
        "shirt_r50": 75.0,
        "toptee_r10": 50.0,
        "toptee_r50": 70.0,
        "macro_r10": 55.0,
        "macro_r50": 75.0,
        "paper_r10": 50.0,
        "paper_r50": 70.0,
        "paper_mean": 60.0,
        "measured_difference": {"r10_diff": 5.0, "r50_diff": 5.0, "label": "observed - paper (R@10: +5.00)"},
        "evaluation_duration_seconds": 12.34,
        "completion_proof": str(artifacts / "reports" / "synthetic_model_ckpt_syn_run_syn_123_completion.json"),
        "aggregate_report": str(artifacts / "reports" / "synthetic_model_ckpt_syn_run_syn_123_aggregate.json"),
        "aggregate_only": True,
        "per_query_export_available": False,
        "missing_metrics_reason": None,
        "metrics": {"extraction_status": "EXTRACTED_SYNTHETIC", "observed_metrics": {"r10": 55.0, "r50": 75.0}},
    }

    row_unparsed = {
        "model_id": "unparsed_model",
        "method_name": "UnparsedModel",
        "status": "FAILED_EVALUATION",
        "blockers": ["evaluation command failed"],
        "missing_permissions": [],
        "run_id": "run_unp_456",
        "checkpoint_id": None,
        "checkpoint_provenance": None,
        "environment_group": None,
        "native_protocol": "fashioniq_original_split",
        "evaluation_kind": "NOT_RUN",
        "dress_r10": None,
        "dress_r50": None,
        "shirt_r10": None,
        "shirt_r50": None,
        "toptee_r10": None,
        "toptee_r50": None,
        "macro_r10": None,
        "macro_r50": None,
        "paper_r10": None,
        "paper_r50": None,
        "paper_mean": None,
        "measured_difference": None,
        "evaluation_duration_seconds": None,
        "completion_proof": None,
        "aggregate_report": None,
        "aggregate_only": True,
        "per_query_export_available": False,
        "missing_metrics_reason": "aggregate parser unavailable for model",
        "metrics": None,
    }

    rows = [row_synthetic, row_unparsed]
    blockers = [{"model_id": "unparsed_model", "status": "FAILED_EVALUATION", "reasons": ["evaluation command failed"], "missing_permissions": []}]
    execution = {"exit_code": 1, "stages": {}}

    autopilot._write_reports(config, plan, rows, blockers, execution, planned=False)
    paths = autopilot._artifact_paths(config)
    md_text = paths["summary_md"].read_text()

    assert "Category Recall Evaluation" in md_text
    assert "Dress R@10" in md_text
    assert "Shirt R@10" in md_text
    assert "Toptee R@10" in md_text
    assert "Macro R@10" in md_text

    assert "**SYNTHETIC**" in md_text
    assert "60.00%" in md_text

    assert "null (aggregate parser unavailable for model)" in md_text
    unparsed_lines = [line for line in md_text.splitlines() if "unparsed_model" in line]
    assert len(unparsed_lines) >= 1
    cat_line = [l for l in unparsed_lines if "null" in l][0]
    assert "0.0" not in cat_line
    assert "0%" not in cat_line

    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))
    syn_csv = next(r for r in reader if r["model_id"] == "synthetic_model")
    unp_csv = next(r for r in reader if r["model_id"] == "unparsed_model")
    assert syn_csv["dress_r10"] == "60.00"
    assert unp_csv["dress_r10"] == "null"
    assert unp_csv["missing_metrics_reason"] == "aggregate parser unavailable for model"
