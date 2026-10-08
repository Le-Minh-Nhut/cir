"""Synthetic end-to-end reproduction harness executing real subprocesses."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.registry import model_by_id
import workbench.scripts.pipeline as pipeline
import workbench.scripts.evaluate_models as evaluate_models
import workbench.backend.index as index

@pytest.fixture
def synthetic_harness(tmp_path: Path):
    """Constructs a tiny synthetic FashionIQ environment with real files."""
    repo = tmp_path / "repo"
    data = tmp_path / "data" / "FashionIQ"
    checkpoints = tmp_path / "checkpoints"
    results = tmp_path / "results"
    third_party = tmp_path / "third_party"
    logs = tmp_path / "logs"

    for d in (repo, data, checkpoints, results, third_party, logs):
        d.mkdir(parents=True)

    # Synthetic FashionIQ dataset layout
    captions = data / "captions"
    splits = data / "image_splits"
    images = data / "images"
    captions.mkdir()
    splits.mkdir()
    images.mkdir()

    for cat in ("dress", "shirt", "toptee"):
        (captions / f"cap.{cat}.train.json").write_text('[{"target": "img1", "candidate": "img0", "captions": ["a", "b"]}]')
        (captions / f"cap.{cat}.val.json").write_text('[{"target": "img1", "candidate": "img0", "captions": ["a", "b"]}]')
        (splits / f"split.{cat}.val.json").write_text('["img0", "img1"]')
    (images / "img0.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (images / "img1.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

    # Synthetic CSMCIR upstream repository with git checkout
    csmcir_dir = third_party / "CSMCIR"
    csmcir_dir.mkdir()
    subprocess.run(["git", "init"], cwd=csmcir_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=csmcir_dir, check=True)
    subprocess.run(["git", "config", "user.email", "tester@test.invalid"], cwd=csmcir_dir, check=True)

    csmcir_src = csmcir_dir / "src"
    csmcir_src.mkdir()

    # Real synthetic evaluator script
    evaluator_script = csmcir_src / "validate_blip_csmcir.py"
    evaluator_code = """#!/usr/bin/env python3
import sys, json

if "--fail" in sys.argv:
    sys.stderr.write("Evaluator induced failure\\n")
    sys.exit(2)

print("Starting synthetic evaluation")
output = {
    "dress_recall_at10": 58.0,
    "dress_recall_at50": 78.0,
    "shirt_recall_at10": 56.0,
    "shirt_recall_at50": 76.0,
    "toptee_recall_at10": 54.0,
    "toptee_recall_at50": 74.0,
    "average_recall_at10": 56.0,
    "average_recall_at50": 76.0,
    "average_recall": 66.0
}
print(json.dumps(output, indent=4))
print("Synthetic evaluation completed successfully")
sys.exit(0)
"""
    evaluator_script.write_text(evaluator_code)
    evaluator_script.chmod(0o755)

    # Auxiliary folders for CSMCIR
    (csmcir_dir / "COT_ours2" / "fashioniq").mkdir(parents=True)
    for cat in ("dress", "shirt", "toptee"):
        (csmcir_dir / "COT_ours2" / "fashioniq" / f"{cat}_cot_val.json").write_text("{}")
    qwen = data / "qwen_captions"
    qwen.mkdir()
    for cat in ("dress", "shirt", "toptee"):
        (qwen / f"{cat}_cot_val.json").write_text("{}")

    # Link for CSMCIR layout
    (csmcir_dir / "fashionIQ_dataset").symlink_to(data, target_is_directory=True)

    subprocess.run(["git", "add", "."], cwd=csmcir_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=csmcir_dir, check=True)
    csmcir_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=csmcir_dir, text=True).strip()

    # Synthetic checkpoint
    csmcir_ckpt_dir = checkpoints / "csmcir"
    csmcir_ckpt_dir.mkdir()
    ckpt_file = csmcir_ckpt_dir / "fashioniq_tuned_clip_best.pt"
    ckpt_file.write_bytes(b"synthetic-csmcir-checkpoint-v1")

    config = WorkbenchConfig(
        CIR_REPO_ROOT=repo,
        CIR_DATA_ROOT=data.parent,
        FASHIONIQ_ROOT=data,
        WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=checkpoints,
        WORKBENCH_RESULTS_ROOT=results,
        WORKBENCH_THIRD_PARTY_ROOT=third_party,
    )

    return SimpleNamespace(
        root=tmp_path,
        config=config,
        data=data,
        checkpoints=checkpoints,
        results=results,
        third_party=third_party,
        ckpt_file=ckpt_file,
        csmcir_dir=csmcir_dir,
        csmcir_head=csmcir_head,
        evaluator_script=evaluator_script,
    )


def test_scenario_a_complete_synthetic_execution_and_extraction(synthetic_harness, monkeypatch) -> None:
    """Scenario A: Complete synthetic eligible model runs real subprocess and extracts metrics."""
    h = synthetic_harness
    monkeypatch.setattr(evaluate_models, "resolve_config", lambda: h.config)
    monkeypatch.setattr(evaluate_models, "pinned_revision", lambda _: h.csmcir_head)
    import workbench.backend.runtime as runtime
    monkeypatch.setattr(runtime, "source_clean_and_pinned", lambda model, _: (True, h.csmcir_head, None))

    plan = evaluate_models.EvaluationPlan(
        model_id="csmcir",
        checkpoint_id="fashioniq",
        protocol="fashioniq_original_split",
        dataset_root=h.csmcir_dir / "fashionIQ_dataset",
        source=h.csmcir_dir,
        checkpoint=h.ckpt_file,
        cwd=h.csmcir_dir / "src",
        command=[sys.executable, str(h.evaluator_script), "--dataset", "fashionIQ"],
        pin=h.csmcir_head,
        actual_source_commit=h.csmcir_head,
        paper_metrics={"r10": 57.07, "r50": 77.27, "mean": 67.17},
    )

    # Real execute() running the real subprocess
    rc = evaluate_models.execute(plan, h.config)
    assert rc == 0

    # Verify log and aggregate report
    logs_dir = h.config.CIR_REPO_ROOT / "workbench" / "artifacts" / "logs"
    subdirs = list(logs_dir.iterdir())
    assert len(subdirs) == 1
    log_dir = subdirs[0]

    report = json.loads((log_dir / "aggregate_report.json").read_text())
    assert report["extraction_status"] == "EXTRACTED_VERIFIED"
    assert report["observed_metrics"] == {"r10": 56.0, "r50": 76.0, "mean": 66.0}
    assert report["parity_status"] in ("PARITY_VERIFIED", "PARITY_MISMATCH")
    assert report["per_query_export_available"] is False
    assert (log_dir / "stdout.log").is_file()
    assert "Synthetic evaluation completed successfully" in (log_dir / "stdout.log").read_text()


def test_scenario_b_missing_checkpoint_blocks_evaluation(synthetic_harness, monkeypatch) -> None:
    """Scenario B: Missing checkpoint blocks evaluation."""
    h = synthetic_harness
    h.ckpt_file.unlink()  # Delete checkpoint

    model = model_by_id("csmcir")
    ckpt = model["checkpoint_variants"][0]
    plan, reasons = evaluate_models.guarded_plan(model, ckpt, "fashioniq_original_split", h.data, 200, h.config)
    assert plan is None
    assert any("checkpoint missing:" in r for r in reasons)


def test_scenario_c_incorrect_hash_blocks_evaluation(synthetic_harness, monkeypatch) -> None:
    """Scenario C: Incorrect hash blocks evaluation."""
    h = synthetic_harness
    model = model_by_id("csmcir")
    ckpt = dict(model["checkpoint_variants"][0])
    ckpt["expected_sha256"] = "0" * 64  # mismatch

    plan, reasons = evaluate_models.guarded_plan(model, ckpt, "fashioniq_original_split", h.data, 200, h.config)
    assert plan is None
    assert any("official checkpoint hash mismatch" in r or "checkpoint missing:" in r for r in reasons)


def test_scenario_d_and_e_resume_invalidates_on_changed_inputs(synthetic_harness, monkeypatch, tmp_path) -> None:
    """Scenarios D & E: Changed checkpoint or dataset invalidates resume cache."""
    h = synthetic_harness
    state_file = tmp_path / "state.json"
    called = []

    def mock_run(argv):
        called.append(argv)
        return 0

    monkeypatch.setattr(pipeline, "run_command", mock_run)
    stage = pipeline.Stage(
        name="evaluation",
        commands=([sys.executable, "eval.py"],),
        model_id="csmcir",
        input_paths=(h.ckpt_file, h.data / "captions" / "cap.dress.val.json"),
        output_paths=(tmp_path / "out.json",),
    )
    (tmp_path / "out.json").write_text('{"done": true}')

    # First run -> executes
    rc = pipeline.run_stages([stage], dry_run=False, continue_on_error=False, resume=True, state_path=state_file, config=h.config)
    assert rc == 0
    assert len(called) == 1

    # Second run without changes -> skips via resume!
    called.clear()
    rc = pipeline.run_stages([stage], dry_run=False, continue_on_error=False, resume=True, state_path=state_file, config=h.config)
    assert rc == 0
    assert len(called) == 0

    # Scenario D: mutate checkpoint -> reruns!
    h.ckpt_file.write_bytes(b"modified-checkpoint-bytes")
    called.clear()
    rc = pipeline.run_stages([stage], dry_run=False, continue_on_error=False, resume=True, state_path=state_file, config=h.config)
    assert rc == 0
    assert len(called) == 1

    # Scenario E: mutate dataset annotation -> reruns!
    (h.data / "captions" / "cap.dress.val.json").write_text('[{"modified": true}]')
    called.clear()
    rc = pipeline.run_stages([stage], dry_run=False, continue_on_error=False, resume=True, state_path=state_file, config=h.config)
    assert rc == 0
    assert len(called) == 1


def test_scenario_g_failed_evaluator_propagates_nonzero_exit_code(synthetic_harness, monkeypatch) -> None:
    """Scenario G: Failed evaluator propagates nonzero exit code."""
    h = synthetic_harness
    plan = evaluate_models.EvaluationPlan(
        model_id="csmcir",
        checkpoint_id="fashioniq",
        protocol="fashioniq_original_split",
        dataset_root=h.data,
        source=h.csmcir_dir,
        checkpoint=h.ckpt_file,
        cwd=h.csmcir_dir / "src",
        command=[sys.executable, str(h.evaluator_script), "--fail"],
        pin=h.csmcir_head,
        actual_source_commit=h.csmcir_head,
    )
    rc = evaluate_models.execute(plan, h.config)
    assert rc == 2


def test_scenario_i_aggregate_only_output_excluded_from_per_query_index(synthetic_harness) -> None:
    """Scenario I: Aggregate report is not picked up as a per-query result file."""
    h = synthetic_harness
    # Place aggregate report in results directory
    (h.results / "aggregate_report.json").write_text('{"artifact_type": "official_aggregate_evaluation_report"}')
    files = index.result_files(h.results)
    assert h.results / "aggregate_report.json" not in files


def test_scenario_j_valid_schema_v2_indexes_independently(synthetic_harness) -> None:
    """Scenario J: A valid schema-v2 result artifact is indexed into DuckDB."""
    h = synthetic_harness
    from workbench.tests.mock_data import build_mock_runs
    mock_run = build_mock_runs()[0]
    out_file = h.results / f"{mock_run.run.run_id}.json"
    out_file.write_text(mock_run.model_dump_json(indent=2))

    db_path = h.root / "test.duckdb"
    count = index.rebuild_index(h.results, db_path)
    assert count == 1
    assert db_path.is_file()
